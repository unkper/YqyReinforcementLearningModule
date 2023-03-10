

class VizdoomEnvWrapper():
    def __init__(self, env):
        self._venv = env
        self.observation_space = self._venv.observation_space
        self.action_space = self._venv.action_space

    def step(self, action):
        observation, reward, done, truncated, info = self._venv.step(action)
        observation = observation['screen']
        return observation, reward, done, info

    def reset(self):
        observations, _ = self._venv.reset()
        observations = observations['screen']
        return observations

