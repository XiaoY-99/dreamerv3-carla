import numpy as np

def val_carla(eval_env, agent, episodes=5):
    total = 0
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        carry = agent.init_policy(batch_size=1)
        is_first = True

        while not done:
            obs_dict = {
                "image": obs["image"][None],
                "vector": obs["vector"][None],
                "reward": np.array([0.0], np.float32),
                "is_first": np.array([is_first], bool),
                "is_last": np.array([False], bool),
                "is_terminal": np.array([False], bool),
            }
            carry, acts, _ = agent.policy(carry, obs_dict, mode="eval")
            action = acts["action"].squeeze(0)  # env expects (3,), so squeeze here
            obs, rew, term, trunc, _ = eval_env.step(action)
            total += rew
            done = term or trunc
            is_first = False
    return total / episodes