import elements
import embodied
import gymnasium as gym
import numpy as np

class GymToEmbodied(embodied.Env):
    """Wrap a Gym/Gymnasium env as an embodied.Env with reset flag."""
    def __init__(self, gym_env):
        self._env = gym_env
        self._done = True  # force reset on first step

        # Build obs_space from the gym dict
        gobs = self._env.observation_space
        spaces = gobs.spaces if hasattr(gobs, "spaces") else gobs
        self._obs_space = {
            k: elements.Space(sp.dtype, sp.shape) for k, sp in spaces.items()
        }
        # Add driver signals
        self._obs_space.update({
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        })

        # Action space (gym Box -> 'action'), plus 'reset' bool
        gact = self._env.action_space
        self._act_space = {
            "action": elements.Space(gact.dtype, getattr(gact, "shape", ())),
        }

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def act_space(self):
        return self._act_space
    
    @property
    def base_env(self):
        return self._env

    def step(self, action):
        if self._done:
            return self.reset()

        a = action.get("action", None)
        obs, rew, term, trunc, _ = self._env.step(a)
        self._done = bool(term or trunc)
        return self._pack(
            obs,
            reward=float(rew),
            is_last=self._done,
            is_terminal=bool(term),
        )

    def _pack(self, obs, reward=0.0, is_first=False, is_last=False, is_terminal=False):
        out = dict(obs)
        out.update(
            reward=np.float32(reward),
            is_first=bool(is_first),
            is_last=bool(is_last),
            is_terminal=bool(is_terminal),
        )
        return out
    
    def reset(self):
        obs, _ = self._env.reset()
        self._done = False
        return self._pack(obs, reward=0.0, is_first=True)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()