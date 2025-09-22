from functools import partial as bind

import elements
import embodied
import numpy as np

import utils


class TestTrain:

  def test_run_loop(self, tmpdir):
    args = self._make_args(tmpdir)
    agent = self._make_agent()
    embodied.run.train(
        lambda: agent, bind(self._make_replay, args),
        self._make_env, self._make_logger, args)
    stats = agent.stats()
    print('Stats:', stats)
    replay_steps = args.steps * args.train_ratio
    assert stats['lifetime'] >= 1  # Otherwise decrease log and ckpt interval.
    assert np.allclose(stats['env_steps'], args.steps, 100, 0.1)
    assert np.allclose(stats['replay_steps'], replay_steps, 100, 0.1)
    assert stats['reports'] >= 1
    assert stats['saves'] >= 2
    assert stats['loads'] == 0
    args = args.update(steps=2 * args.steps)
    embodied.run.train(
        lambda: agent, bind(self._make_replay, args),
        self._make_env, self._make_logger, args)
    stats = agent.stats()
    assert stats['loads'] == 1
    assert np.allclose(stats['env_steps'], args.steps, 100, 0.1)

  def _make_agent(self):
    env = self._make_env(0)
    agent = utils.TestAgent(env.obs_space, env.act_space)
    env.close()
    return agent

  def _make_env(self, index):
    # --- Use CARLA instead of dummy ---
    import carla
    from routes import build_loop_route
    from dreamer_env import CarlaRacingDreamerEnv
    from adapters import GymToEmbodied

    # If you plan to run multiple envs later, launch multiple CARLA servers
    # on different ports and use: port = 2000 + int(index)
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1/20
    world.apply_settings(s)

    route = build_loop_route(world.get_map())
    gym_env = CarlaRacingDreamerEnv(client, route)   # Gym env
    env = GymToEmbodied(gym_env)                     # wrap to embodied.Env
    return env

  def _make_replay(self, args):
    kwargs = {'length': args.batch_length, 'capacity': 1e4}
    return embodied.replay.Replay(**kwargs)

  def _make_logger(self):
    return elements.Logger(elements.Counter(), [
        elements.logger.TerminalOutput(),
    ])

  def _make_args(self, logdir):
    return elements.Config(
        steps=1000,
        train_ratio=32.0,
        log_every=0.1,
        report_every=0.2,
        save_every=0.2,
        report_batches=1,
        from_checkpoint='',
        usage=dict(psutil=True),
        debug=False,
        logdir=str(logdir),
        envs=1,
        batch_size=8,
        batch_length=16,
        replay_context=0,
        report_length=8,
    )
