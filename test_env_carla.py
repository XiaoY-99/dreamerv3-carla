import carla
import numpy as np
from dreamer_env import CarlaRacingDreamerEnv
from routes import build_loop_route
from adapters import GymToEmbodied

def make_env(port=2000):
    client = carla.Client("localhost", port)
    client.set_timeout(5.0)
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1 / 20
    world.apply_settings(s)

    route = build_loop_route(world.get_map())
    print("Testing map:", world.get_map().name)
    gym_env = CarlaRacingDreamerEnv(client, route)
    return gym_env

def main():
    env = make_env(port=2000)

    # Reset env properly (Gym API usually returns (obs, info))
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs = reset_out
        info = {}

    print("Initial obs type:", type(obs))
    if isinstance(obs, dict):
        print("Initial obs keys:", obs.keys())
    else:
        print("Obs shape:", np.array(obs).shape)

    # Take a few random actions
    for step in range(5):
        action = env.action_space.sample()
        print(f"\nStep {step}, action: {action}")
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, done, truncated, info = step_out
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            obs = step_out
            reward, done, info = None, None, None
        print("Reward:", reward, "Done:", done)

    env.close()

if __name__ == "__main__":
    main()
