# Minimal DreamerV3 + CARLA training entrypoint (single env).

import os
import numpy as np
import embodied
from dreamerv3.embodied.jax.agent import Agent as DreamerAgent
from dreamer_env import CarlaRacingDreamerEnv
from routes import build_loop_route
import carla
import elements
import yaml
import os
from val_carla import val_carla
from adapters import GymToEmbodied
import jax.numpy as jnp
import jax

os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

print("JAX devices in train_carla.py:", jax.devices())

# Global RNG key
global_key = jax.random.PRNGKey(0)

def curriculum_callback(env, counter):
    steps = int(counter)
    if steps < 200_000:
        env._penalty_scale = 0.5
    elif steps < 500_000:
        env._penalty_scale = 1.0
    else:
        env._penalty_scale = 2.0

def make_env(index=0, port=2000, counter=None):
    # client = carla.Client("localhost", port) # Train on local machine
    client = carla.Client("100.79.76.114", port)  # Train on Runpod
    client.set_timeout(5.0)
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1 / 20
    world.apply_settings(s)

    route = build_loop_route(world.get_map())
    print("Training map:", world.get_map().name)
    gym_env = CarlaRacingDreamerEnv(client, route)  # Gym env
    curriculum_callback(gym_env, counter)  # pass the raw env
    return GymToEmbodied(gym_env)


def make_replay(cfg):
    # A small replay to start; tune capacity later.
    return embodied.replay.Replay(length=cfg.batch_length, capacity=4e5)

import jax.numpy as jnp

def make_stream(replay, mode, args):
    while True:
        length = args.report_length if mode == "report" else args.batch_length
        batch = replay.sample(batch=args.batch_size, mode=mode)

        # Truncate or slice to the required length
        for key, value in batch.items():
            if value.ndim >= 2 and value.shape[1] > length:
                batch[key] = value[:, :length]

        if "consec" not in batch:
            batch["consec"] = np.zeros(
                (batch["is_first"].shape[0], length), dtype=np.int32
            )

        # add seed
        np_seed = np.random.randint(0, 2**31 - 1, size=(2,), dtype=np.uint32)
        batch["seed"] = jnp.asarray(np_seed, dtype=jnp.uint32)

        # convert everything to JAX arrays 
        batch = {k: jnp.asarray(v) for k, v in batch.items()}

        # safeguard for actions
        if "action" in batch and batch["action"].ndim == 4:
            batch["action"] = batch["action"].squeeze(2)  # (B,T,1,3) → (B,T,3)

        yield batch

def make_logger():
    counter = elements.Counter()
    logger = elements.Logger(
        counter,
        [
            elements.logger.TerminalOutput(),
            elements.logger.JSONLOutput(os.path.join("logs_carla", "metrics.jsonl")),
            elements.logger.TensorBoardOutput(os.path.join("logs_carla", "tb")),
        ],
    )
    return counter, logger

def load_agent_cfg(args, profile="loconav"):  # or "dmc_vision"
    import yaml, os
    cfg_path = os.path.join("dreamerv3", "configs.yaml")
    with open(cfg_path, "r") as f:
        all_cfg = yaml.safe_load(f)

    if "defaults" not in all_cfg:
        raise RuntimeError(f"'defaults' missing in {cfg_path}")
    if profile not in all_cfg:
        raise RuntimeError(f"Profile '{profile}' not found. Available: {list(all_cfg.keys())}")

    # Merge top-level: defaults ⊕ profile (profile overrides defaults)
    merged_top = {}
    merged_top.update(all_cfg["defaults"] or {})
    merged_top.update(all_cfg[profile] or {})

    # Agent sub-config (your fork places enc/dec/world-model here)
    agent_section = merged_top.get("agent")
    if not isinstance(agent_section, dict):
        raise RuntimeError(f"'agent' section missing after merge. Keys: {list(merged_top.keys())}")

    def to_plain(x):
        # elements.Config behaves like a mapping; we convert recursively
        if isinstance(x, dict):
            return {k: to_plain(v) for k, v in x.items()}
        # Fallback for mapping-like objects (e.g., elements.Config)
        try:
            items = x.items()
        except Exception:
            return x
        else:
            return {k: to_plain(v) for k, v in items}

    agent_plain = to_plain(agent_section)

    # Sanity: need enc + dec + (rssm or dyn)
    have_enc = "enc" in agent_plain
    have_dec = "dec" in agent_plain
    have_rssm = "rssm" in agent_plain
    have_dyn = "dyn" in agent_plain
    if not (have_enc and have_dec and (have_rssm or have_dyn)):
        raise RuntimeError(
            "Agent config missing required blocks. Need enc, dec, and one of {rssm, dyn}.\n"
            f"Top-level agent keys: {list(agent_plain.keys())}"
        )

    # ---- Coerce numeric strings to floats on the plain dict ----
    def coerce_float_at(d, path):
        cur = d
        for k in path[:-1]:
            if not isinstance(cur, dict) or k not in cur:
                return
            cur = cur[k]
        leaf = path[-1]
        if isinstance(cur, dict) and leaf in cur and isinstance(cur[leaf], str):
            try:
                cur[leaf] = float(cur[leaf])
            except ValueError:
                pass

    # Normalize sections that often have "1e-8" as a string
    for block in ["valnorm", "retnorm", "advnorm"]:
        for key in ["limit", "eps"]:
            coerce_float_at(agent_plain, (block, key))

    # (optional debug)
    for block in ["valnorm", "retnorm", "advnorm"]:
        if block in agent_plain:
            print(block, agent_plain[block])

    for p in [
        ("valnorm", "limit"),
        ("retnorm", "limit"),
        ("advnorm", "eps"),
        ("retnorm", "eps"),
        ("valnorm", "eps"),
    ]:
        coerce_float_at(agent_plain, p)

    coerce_float_at(agent_plain, ("opt", "eps"))
    coerce_float_at(agent_plain, ("opt", "lr"))

    # Inject runtime fields
    agent_plain["logdir"] = args.logdir
    agent_plain["batch_size"] = int(args.batch_size)
    agent_plain["batch_length"] = int(args.batch_length)
    agent_plain["replay_context"] = int(args.replay_context)
    agent_plain["report_length"] = int(args.report_length)
    agent_plain["seed"] = int(getattr(args, "seed", np.random.randint(0, 1e9)))
    agent_plain.setdefault("jax", {})
    if not isinstance(agent_plain["jax"], dict):
        agent_plain["jax"] = {}
    agent_plain["jax"]["jit"] = True
    agent_plain["jax"]["platform"] = "gpu"  # or "cpu"

    print(f"[config] merged profile: defaults + {profile} (using agent section)")
    return elements.Config(agent_plain)

def main():
    args = elements.Config(
        steps=1_000_000,
        envs=1,
        train_ratio=32.0,
        batch_size=16,
        batch_length=16,
        replay_context=0,
        report_length=8,
        report_batches=1,
        consec_report=1,
        logdir=os.path.join(os.getcwd(), "logs_carla"),
        log_every=10.0,
        report_every=30.0,
        save_every=300.0,
        from_checkpoint="",
        usage=dict(psutil=True),
        debug=False,
    )

    # Counter + Logger first
    counter, logger = make_logger()

    # Env + agent
    env = make_env(index=0, port=2000, counter=counter)
    obs_space, act_space = env.obs_space, env.act_space
    agent_cfg = load_agent_cfg(args, profile="loconav")
    agent = DreamerAgent(obs_space, act_space, agent_cfg)
    # Replay buffer
    replay = make_replay(args)

    ckpt_dir = os.path.join(args.logdir, "ckpt")
    cp = elements.Checkpoint(ckpt_dir)

    # Attach what you want to checkpoint
    cp.step = counter
    cp.agent = agent
    cp.replay = replay

    if args.from_checkpoint:
        cp.load(args.from_checkpoint)
    else:
        cp.load_or_save()

    # Reset env and init policy carry
    obs = env.reset()
    # Keep vector_raw, progress_m for logging
    obs_for_log = dict(obs) 
    for k in ["vector_raw", "progress_m"]:
        if k in obs:
            obs.pop(k)

    carry_act = agent.init_policy(batch_size=1)
    carry_train = agent.init_train(batch_size=args.batch_size)

    episode_reward = 0
    episode_len = 0
    episode_speeds = [] 

    # Training loop
    while counter.value < args.steps:
        # === Dreamer policy ===
        obs_for_policy = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        carry_act, acts, _ = agent.policy(carry_act, obs_for_policy, mode="explore")

        action = {"action": acts["action"]}
        obs = env.step(action)
        # Keep vector_raw, progress_m for logging
        obs_for_log = dict(obs) 
        for k in ["vector_raw", "progress_m"]:
            if k in obs:
                obs.pop(k)

        if "vector_raw" in obs_for_log:
            v = obs_for_log["vector_raw"]
            episode_speeds.append(float(v[0]))

        action_arr = np.array(acts["action"]).squeeze()
        action = {"action": action_arr}

        # Store transition
        replay.add({**obs, **action})

        # === Reward accumulation ===
        reward = float(np.array(obs["reward"]).item())
        done = bool(obs["is_last"])
        episode_reward += reward
        episode_len += 1

        # === Training updates ===
        if not hasattr(agent, "last_train_mets"):
            agent.last_train_mets = {}

        train_mets = {}
        for _ in range(int(args.train_ratio)):
            if len(replay) > args.batch_size * args.batch_length:
                carry_train, outs, mets = agent.train(
                    carry_train, next(make_stream(replay, "train", args))
                )
                for k, v in mets.items():
                    if "loss" in k and isinstance(v, (int, float, np.number)):
                        train_mets[k] = float(v)

        if train_mets:
            agent.last_train_mets = train_mets

        # Curriculum shaping
        curriculum_callback(env.base_env, counter)

        # === Step-level logging ===
        if counter.value % 50 == 0:  # log every 50 steps for smoother curves
            logger.scalar("train/steps", counter.value)
            logger.scalar("train/batch_size", args.batch_size)

            # Latest reward
            logger.scalar("env/reward", reward)

            # Keep vector_raw ONLY for logging
            if "vector_raw" in obs_for_log:
                v = obs_for_log["vector_raw"]
                # logger.scalar("env/speed_raw", float(v[0]))
                logger.scalar("env/lat_err_raw", float(v[1]))
                logger.scalar("env/head_err_raw", float(v[2]))
                logger.scalar("env/yaw_err_raw", float(v[3]))
                logger.scalar("env/progress_raw", float(v[4]))
                logger.scalar("env/damage_raw", float(v[5]))
                logger.scalar("env/on_track_raw", float(v[6]))
                logger.scalar("env/wp_x", float(v[7]))
                logger.scalar("env/wp_y", float(v[8]))

            if "progress_m" in obs_for_log:
                logger.scalar("env/progress_m", float(obs_for_log["progress_m"]))

            '''
            if hasattr(env.base_env, "_vec_norm"):
                rn = env.base_env._vec_norm
                if rn.mean is not None:
                    for i, name in enumerate(
                        ["speed", "lat_err", "head_err", "yaw_err",
                        "progress", "damage", "on_track", "wp_x", "wp_y"]
                    ):
                        logger.scalar(f"norm_mean/{name}", float(rn.mean[i]))
                        logger.scalar(f"norm_var/{name}", float(rn.var[i]))
            '''

            for k, v in agent.last_train_mets.items():
                logger.scalar(f"train/{k}", v)

            logger.write()  # flush once per logging block

        # === Episode end ===
        if bool(obs.get("is_last", False)):
            logger.scalar("episode/score", episode_reward)
            logger.scalar("episode/length", episode_len)

            if episode_speeds:
                logger.scalar("episode/mean_speed", np.mean(episode_speeds))
                logger.scalar("episode/max_speed", np.max(episode_speeds))

            logger.write()

            # Reset episode stats
            episode_reward = 0
            episode_len = 0
            episode_speeds = []

        # === Periodic evaluation ===
        if counter.value % 100000 == 0 and counter.value > 0:
            eval_env = make_env(port=2000)
            score = val_carla(eval_env, agent, episodes=5)
            logger.scalar("eval/score", score)
            logger.write()
            eval_env.close()

        counter.increment()

        if counter.value % 100 == 0:
            cp.save()

    env.close()

if __name__ == "__main__":
    main()
