import time
import cv2
import carla
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import math

# from torch.utils.tensorboard import SummaryWriter
from running_norm import RunningNorm

class CarlaRacingDreamerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, carla_client, route, fps=20, action_repeat=2, img_size=64):
        super().__init__()
        self.client = carla_client
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.route = route  # list[carla.Waypoint]
        self.fps = fps
        self.action_repeat = action_repeat
        self.img_size = img_size

        # Actions: steer[-1,1], throttle[0,1], brake[0,1]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0], np.float32),
                                       high=np.array([+1.0, 1.0, 1.0], np.float32),
                                       dtype=np.float32)

        # Observations: image (64x64x3) + vector of 6 scalars
        self.image_shape = (img_size, img_size, 3)
        self.vec_dim = 6  # speed, lat_err, head_err, progress, damage, on_track(0/1)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=self.image_shape, dtype=np.uint8),
            "vector": spaces.Box(-np.inf, np.inf, shape=(self.vec_dim,), dtype=np.float32),
        })

        self.episode_steps = 0
        self.max_steps = int(5 * 60 * fps)  # 5 minutes
        self._last_image = np.zeros(self.image_shape, np.uint8)
        self._stuck_ticks = 0
        self.route_idx = 0
        self.progress_m = 0.0
        self.prev_loc = None
        self.actors_to_destroy = []
        self._vec_norm = RunningNorm()

        # Track collisions
        self._collision_hist = []
        self._last_collision = None 

        # create TB log dir
        # self.writer = SummaryWriter(log_dir="logs_carla/tb_env")  

        # ensure synchronous sim with fixed step
        settings = self.world.get_settings()
        if not settings.synchronous_mode or (settings.fixed_delta_seconds or 0) != 1.0 / self.fps:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.fps
            self.world.apply_settings(settings)

    # ---------------- Gym API ----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self._setup_episode()
        # give sensors a couple ticks to warm up
        for _ in range(3):
            self.world.tick()
        obs = self._get_obs()
        self.episode_steps = 0
        self.damage = 0.0
        self._collision_hist = []
        self._last_collision = None
        self._stuck_ticks = 0

        # Warmup counter to trigger the vehicle at start
        self._warmup_steps = 20  

        return obs, {}


    def rescale_action(self, action):
        """Map Dreamer [-1,1] → CARLA ranges (steer[-1,1], throttle[0,1], brake[0,1])."""
        action = np.asarray(action, np.float32).squeeze()  # handle (1,3) or (3,)
        steer = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip((action[1] + 1) / 2, 0, 1))  # [-1,1] → [0,1]
        brake = float(np.clip((action[2] + 1) / 2, 0, 1))      # [-1,1] → [0,1]
        return np.array([steer, throttle, brake], np.float32)

    def step(self, action):
        # Warmup phase: apply gentle throttle for a few steps
        if hasattr(self, "_warmup_steps") and self._warmup_steps > 0:
            # Small proportional centering controller for a few ticks.
            lat_err, head_err = self._track_errors()
            lane_w = float(self.route[self.route_idx].lane_width) * 0.5 + 1e-6
            lat_n = np.clip(lat_err / lane_w, -2.0, 2.0)
            head_n = np.clip(head_err / np.pi, -1.0, 1.0)

            k_lat, k_head = 0.25, 0.15   # gentle gains (tune if needed)
            steer = np.clip(-k_lat * lat_n - k_head * head_n, -0.35, 0.35)
            throttle = 0.35              # small forward push
            brake = 0.0
            self._warmup_steps -= 1
        else:
            steer, throttle, brake = self.rescale_action(action)

        reward = 0.0
        continue_flag = True
        for _ in range(self.action_repeat):
            self._apply_control(steer, throttle, brake)
            self.world.tick()
            r_step, cont = self._compute_reward_and_continue()
            reward += float(r_step)
            continue_flag = bool(continue_flag and cont)
            if not continue_flag:
                break

        self.episode_steps += 1
        obs = self._get_obs()
        terminated = not continue_flag
        truncated = self.episode_steps >= self.max_steps

        info = {"continue": float(continue_flag)}
        if terminated or truncated:
            self._cleanup()
        return obs, float(reward), terminated, truncated, info


    # ---------------- Internals ----------------
    def _setup_episode(self):
        self._cleanup()

        # Pick a route waypoint to spawn at (e.g., route[0])
        wp = self.route[0]
        spawn_tf = carla.Transform(
            location=wp.transform.location + carla.Location(z=0.1),
            rotation=wp.transform.rotation
        )

        bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        bp.set_attribute("role_name", "hero")

        self.ego = self.world.try_spawn_actor(bp, spawn_tf)
        if self.ego is None:
            # fallback to any spawn point
            for sp in np.random.permutation(self.map.get_spawn_points()):
                ego = self.world.try_spawn_actor(bp, sp)
                if ego is not None:
                    self.ego = ego
                    break
        if self.ego is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        self.actors_to_destroy.append(self.ego)

        # (… rest of your sensors setup …)

        self.route_idx = 0
        self.progress_m = 0.0
        self.prev_loc = self.ego.get_transform().location
        self._stuck_ticks = 0

        # Initial gentle forward velocity kick (instead of brake)
        # transform = self.ego.get_transform()
        # forward = transform.get_forward_vector()
        # initial_speed = 5.0  # m/s ≈ 7 km/h
        # self.ego.set_target_velocity(
            # carla.Vector3D(forward.x * initial_speed,
                           # forward.y * initial_speed,
                           # forward.z * initial_speed))

        # (Optional) Instead apply throttle for a few steps
        # self.ego.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0, steer=0.0))



    def _cleanup(self):
        for actor in list(self.actors_to_destroy):
            try:
                if isinstance(actor, carla.Sensor):
                    actor.stop()
            except Exception:
                pass

        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "hero":  # or your ego tag
                actor.destroy()

        for actor in list(self.actors_to_destroy):
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors_to_destroy.clear()
        self._collision_hist.clear()

    def _on_cam(self, image: carla.Image):
        # BGRA -> RGB uint8
        a = np.frombuffer(image.raw_data, dtype=np.uint8)
        a = a.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self._last_image = a

    def _apply_control(self, steer, throttle, brake):
        self.ego.apply_control(carla.VehicleControl(
            steer=float(np.clip(steer, -1, 1)),
            throttle=float(np.clip(throttle, 0, 1)),
            brake=float(np.clip(brake, 0, 1)),
        ))

    def _grab_camera(self):
        return self._last_image

    def _resize_to(self, shape_hw3):
        h, w = shape_hw3[0], shape_hw3[1]
        if self._last_image.shape[0] != h or self._last_image.shape[1] != w:
            return cv2.resize(self._last_image, (w, h), interpolation=cv2.INTER_AREA)
        return self._last_image

    def _vehicle_speed_mps(self):
        v = self.ego.get_velocity()
        return float(np.linalg.norm([v.x, v.y, v.z]))

    # ------- Observations & reward -------

    def _get_obs(self):
        rgb = self._grab_camera()
        rgb = self._resize_to(self.image_shape)

        # --- Raw values ---
        speed = self._vehicle_speed_mps()
        lat_err, head_err = self._track_errors()
        progress = self._route_progress()
        damage = self._damage_norm()
        on_track = 1.0 if self._on_track() else 0.0

        raw_vec = np.array([
            speed,
            lat_err,
            head_err,
            progress,
            damage,
            on_track,
        ], dtype=np.float32)

        # --- Normalization ---
        self._vec_norm.update(raw_vec[None])  # update running stats
        norm_vec = self._vec_norm.normalize(raw_vec)

        # --- Return both ---
        return {
            "image": rgb,
            "vector": norm_vec,       # feed to Dreamer
            "vector_raw": raw_vec,    # for logging/debugging
            "progress_m": self.progress_m,  # absolute progress in meters
        }

    def _compute_reward_and_continue(self):
        # --- Raw signals ---
        steps = self.episode_steps
        speed = self._vehicle_speed_mps()
        ds_centerline = self._progress_delta()
        lat_err, head_err = self._track_errors()

        lane_w = float(self.route[self.route_idx].lane_width) * 0.5 + 1e-6
        lat_n = abs(lat_err) / lane_w     # normalized lateral error
        head_n = abs(head_err) / np.pi    # normalized heading error

        # --- Continuous curriculum (0→1 as steps increase) ---
        alpha = min(1.0, steps / 100_000.0)

        w_progress = 0.2 + 0.8 * alpha
        w_speed    = 0.05 + 0.45 * alpha
        w_lat      = 2.0 - 1.0 * alpha
        w_head     = 1.0 - 0.5 * alpha

        col_penalty   = -0.2 - 1.8 * alpha
        track_penalty = -0.2 - 0.8 * alpha

        # --- Reward calculation ---
        reward = 0.0

        # Progress shaping: nonlinear, gated by lane discipline
        progress_term = np.tanh(ds_centerline * 20)
        lane_factor   = np.exp(-5.0 * lat_n)  # sharper decay than before

        reward += w_progress * progress_term * (lane_factor ** 2)
        reward += w_speed * np.tanh(speed / 10.0) * lane_factor
        reward -= w_lat * (lat_n ** 2)
        reward -= w_head * (head_n ** 2)

        # --- Anti-camping shaping ---
        if speed < 0.5:
            reward -= 0.1
        if ds_centerline == 0:
            reward -= 0.2

        # --- Centering bonus (encourage staying near middle of lane) ---
        reward += 0.05 * (1.0 - min(lat_n, 1.0))

        # --- Recovery bonus (reward if error decreases) ---
        if hasattr(self, "_last_lat_err"):
            if abs(lat_err) < abs(self._last_lat_err):
                reward += 0.1
        self._last_lat_err = lat_err

        # --- Explicit off-lane penalty ---
        if lat_n > 1.0:  # more than one lane width off
            reward -= 2.0 * (lat_n - 1.0)

        # --- Collisions & off-track penalties ---
        if self._collision_happened():
            reward += col_penalty * (1.0 + self._damage_norm())
        if not self._on_track():
            reward += track_penalty * min(lat_n, 2.0)

        # --- Termination logic (stricter than before) ---
        continue_flag = True
        if self._collision_hard() or self._finished_lap():
            continue_flag = False
        elif self._stuck_too_long():
            continue_flag = False
        elif abs(lat_err) > 2.0 * lane_w:   # stricter cutoff (was 2.5–3.5)
            reward -= 5.0
            if abs(lat_err) > 2.5 * lane_w:
                continue_flag = False

        return reward, continue_flag


        # Nonlinear shaping so even small forward motion is rewarded
        progress_term = np.tanh(ds_centerline * 20)
        lane_factor   = np.exp(-2.0 * lat_n)   # smoothly decays off-lane reward

        reward += w_progress * progress_term * lane_factor
        reward += w_speed * np.tanh(speed / 10.0) * lane_factor
        reward -= w_lat * (lat_n ** 2)
        reward -= w_head * (head_n ** 2)

        # --- Anti-camping shaping ---
        if speed < 0.5:   # discourage standing still
            reward -= 0.1
        if ds_centerline == 0:   # no forward progress
            reward -= 0.2

        # --- Recovery bonus (reward if improving error vs. last step) ---
        if hasattr(self, "_last_lat_err"):
            if abs(lat_err) < abs(self._last_lat_err):
                reward += 0.1
        self._last_lat_err = lat_err

        # --- Collisions & off-track penalties ---
        if self._collision_happened():
            reward += col_penalty * (1.0 + self._damage_norm())
        if not self._on_track():
            reward += track_penalty * min(lat_n, 2.0)

        # --- Termination logic (soft, with chance to recover) ---
        continue_flag = True
        if self._collision_hard() or self._finished_lap():
            continue_flag = False
        elif self._stuck_too_long():
            continue_flag = False
        elif abs(lat_err) > 2.5 * lane_w:
            # Strong penalty but allow recovery unless it's really bad
            reward -= 5.0
            if abs(lat_err) > 3.5 * lane_w:
                continue_flag = False

        return reward, continue_flag


    # ------- Track geometry & errors -------

    def _nearest_segment(self):
        loc = self.ego.get_transform().location
        best_i, best_d = self.route_idx, 1e9
        for j in range(self.route_idx, self.route_idx + 30):
            i = j % len(self.route)
            d = self.route[i].transform.location.distance(loc)
            if d < best_d:
                best_d, best_i = d, i
        self.route_idx = best_i
        wp = self.route[self.route_idx]
        nxt = self.route[(self.route_idx + 1) % len(self.route)]
        return wp, nxt

    def _track_errors(self):
        wp, nxt = self._nearest_segment()
        loc = self.ego.get_transform().location
        p = np.array([loc.x, loc.y], np.float32)
        a = wp.transform.location; b = nxt.transform.location
        a = np.array([a.x, a.y], np.float32); b = np.array([b.x, b.y], np.float32)
        t = b - a
        t_norm = np.linalg.norm(t) + 1e-6
        t_hat = t / t_norm
        n_hat = np.array([-t_hat[1], t_hat[0]], np.float32)
        lat_err = float(np.dot(p - a, n_hat))
        yaw = np.deg2rad(self.ego.get_transform().rotation.yaw)
        seg_yaw = float(np.arctan2(t_hat[1], t_hat[0]))
        head_err = float(np.arctan2(np.sin(yaw - seg_yaw), np.cos(yaw - seg_yaw)))
        return lat_err, head_err

    def _route_length_m(self):
        if not hasattr(self, "_route_len"):
            L = 0.0
            for i in range(len(self.route)):
                a = self.route[i].transform.location
                b = self.route[(i+1) % len(self.route)].transform.location
                L += a.distance(b)
            self._route_len = float(L)
        return self._route_len

    def _route_progress(self):
        return float((self.progress_m % (self._route_length_m() + 1e-6)) / (self._route_length_m() + 1e-6))

    def _progress_delta(self):
        wp, nxt = self._nearest_segment()
        cur = self.ego.get_transform().location
        if self.prev_loc is None:
            self.prev_loc = cur
            return 0.0
        d = np.array([cur.x - self.prev_loc.x, cur.y - self.prev_loc.y], np.float32)
        t = np.array([nxt.transform.location.x - wp.transform.location.x,
                      nxt.transform.location.y - wp.transform.location.y], np.float32)
        t_hat = t / (np.linalg.norm(t) + 1e-6)
        ds = float(np.dot(d, t_hat))
        self.progress_m += max(0.0, ds)
        self.prev_loc = cur
        return max(0.0, ds)

    # ------- Status / penalties / termination -------

    def _on_track(self):
        lat_err, _ = self._track_errors()
        lane_w = float(self.route[self.route_idx].lane_width) * 0.5
        return abs(lat_err) < (lane_w + 0.5)
    
    def _on_collision(self, event):
        self._collision_hist.append(event)
        self._last_collision = event

    def _damage_norm(self):
        if not self._last_collision:
            return 0.0  # No collisions yet

        imp = getattr(self._last_collision, "normal_impulse", 0.0)

        if isinstance(imp, (int, float)):
            mag = imp
        elif hasattr(imp, "length"):  # Vector3D
            mag = imp.length()
        elif hasattr(imp, "x"):  # fallback
            mag = math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)
        else:
            mag = 0.0

        return float(np.tanh(mag / 1000.0))

    def _collision_happened(self):
        return len(self._collision_hist) > 0

    def _collision_hard(self):
        def magnitude(impulse):
            if isinstance(impulse, (int, float)):
                return impulse
            elif hasattr(impulse, "length"):
                return impulse.length()
            elif hasattr(impulse, "x"):  # Vector3D-like
                return math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            return 0.0

        return any(magnitude(getattr(e, "normal_impulse", 0.0)) > 1000.0 for e in self._collision_hist)


    def _stuck_too_long(self, thresh_s=4.0):
        v = self._vehicle_speed_mps()
        if v < 0.5:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        return self._stuck_ticks > thresh_s * self.fps

    def _finished_lap(self):
        return self.progress_m >= self._route_length_m()
    
    def close(self):
        """Close env and TensorBoard writer."""
        self._cleanup()