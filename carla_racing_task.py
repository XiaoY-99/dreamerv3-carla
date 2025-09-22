import carla
from dreamerv3.embodied.envs.from_gym import FromGym
from routes import build_loop_route
from dreamer_env import CarlaRacingDreamerEnv

def make_task():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # deterministic sim
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / 20.0
    world.apply_settings(s)

    route = build_loop_route(world.get_map())
    env = CarlaRacingDreamerEnv(client, route, fps=20, action_repeat=2, img_size=64)
    return FromGym(env)
