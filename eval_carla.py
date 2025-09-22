import os
import carla
import cv2
import time
import jax
import numpy as np
import elements
from dreamerv3.embodied.jax import agent as agent_module
from dreamer_env import CarlaRacingDreamerEnv
from train_carla import load_agent_cfg  # reuse cfg loader
from torch.utils.tensorboard import SummaryWriter


LOGDIR = "logs_carla"
CKPTDIR = f"{LOGDIR}/ckpt"
VIDEO_DIR = f"{LOGDIR}/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

def generate_route(map, steps=500, spacing=2.0):
    spawn = map.get_spawn_points()[0]
    start_wp = map.get_waypoint(spawn.location)
    route = []
    wp = start_wp
    for _ in range(steps):
        route.append(wp)
        next_wps = wp.next(spacing)
        if not next_wps:
            break
        wp = next_wps[0]
    return route


def main():
    # Connect to CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.load_world("Town10HD_Opt")  # same as training
    carla_map = world.get_map()
    route = generate_route(carla_map, steps=1000, spacing=2.0)
    env = CarlaRacingDreamerEnv(client, route)

    # Build obs/act spaces manually (as in training)
    obs_space = {
        "image": elements.Space(np.uint8, (64, 64, 3)),
        "vector": elements.Space(np.float32, (6,)),
        "reward": elements.Space(np.float32, ()),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    act_space = {
        "action": elements.Space(np.float32, (3,)),
    }

    # Load agent config (reuse your loader)
    args = elements.Config(
        logdir=LOGDIR,
        batch_size=16,
        batch_length=16,
        replay_context=1,
        report_length=100,
    )

    agent_cfg = load_agent_cfg(args, profile="loconav")

    device_type = "gpu" if any(d.platform == "gpu" for d in jax.devices()) else "cpu"

    # Correct: update returns a new Config
    agent_cfg = agent_cfg.update({"jax": {"platform": device_type}})
    print("Agent cfg JAX:", agent_cfg.jax)

    # Build agent + load checkpoint
    agent = agent_module.Agent(obs_space, act_space, agent_cfg)
    ckpt = elements.Checkpoint(directory=CKPTDIR)
    ckpt.agent = agent
    ckpt.load()
    print(f"Loaded checkpoint from {CKPTDIR}")

    num_episodes = 5
    results = []

    carry = agent.init_policy(batch_size=1) #recurrent state

    for ep in range(num_episodes):
        raw_obs, info = env.reset()
        obs = {
            "image": raw_obs["image"][None],
            "vector": raw_obs["vector"][None],
            "reward": np.array([0.0], np.float32),
            "is_first": np.array([True], bool),
            "is_last": np.array([False], bool),
            "is_terminal": np.array([False], bool),
        }
        done = False
        total_reward = 0
        start_time = time.time()
        collisions = 0

        video_path = os.path.join(VIDEO_DIR, f"episode_{ep+1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_path, fourcc, env.fps, (env.img_size, env.img_size)
        )

        while not done:
            # Query Dreamer policy
            carry, acts, outs = agent.policy(carry, obs, mode="eval")
            action = acts["action"].squeeze(0) # [-1,1]

            # Step env
            raw_obs, rew, term, trunc, info = env.step(action)

            # Build obs dict for Dreamer
            obs = {
                "image": raw_obs["image"][None],             # (1, 64, 64, 3)
                "vector": raw_obs["vector"][None],           # (1, 6)
                "reward": np.array([rew], np.float32),       # (1,)
                "is_first": np.array([False], bool),         # (1,)
                "is_last": np.array([term], bool),           # (1,)
                "is_terminal": np.array([term], bool),       # (1,)
            }

            total_reward += rew
            done = term or trunc

            if info.get("collision", False):
                collisions += 1

            # Save frame
            frame = raw_obs["image"]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        lap_time = time.time() - start_time

        results.append(
            {
                "episode": ep + 1,
                "reward": total_reward,
                "lap_time_sec": lap_time,
                "collisions": collisions,
            }
        )

        print(
            f"Episode {ep+1}: reward={total_reward:.2f}, "
            f"lap_time={lap_time:.1f}s, collisions={collisions}"
        )
        print(f"Saved video: {video_path}")

    # Print summary
    print("\n=== Evaluation Results ===")

    writer = SummaryWriter(log_dir=os.path.join(LOGDIR, "eval_tb"))
    for r in results:
        print(
            f"Ep {r['episode']}: reward={r['reward']:.2f}, "
            f"lap={r['lap_time_sec']:.1f}s, collisions={r['collisions']}"
        )
        # ✅ log evaluation metrics
        writer.add_scalar("eval/reward", r["reward"], r["episode"])
        writer.add_scalar("eval/lap_time_sec", r["lap_time_sec"], r["episode"])
        writer.add_scalar("eval/collisions", r["collisions"], r["episode"])

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
