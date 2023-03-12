from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv


class MyVizdoomEnv(VizdoomEnv):
    """
    scenario_file:"dense", "sparse", "verySparse"三个选项
    """
    def __init__(
            self, scenario_file, frame_skip=4, max_buttons_pressed=1, render_mode=None
    ):
        if scenario_file == "dense":
            scenario_file = r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\vizdoom" \
                            r"\doomFiles\wads\my_way_home_dense.wad"
        elif scenario_file == "sparse":
            scenario_file = r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\vizdoom" \
                            r"\doomFiles\wads\my_way_home_sparse.wad"
        elif scenario_file == "verySparse":
            scenario_file = r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\vizdoom" \
                            r"\doomFiles\wads\my_way_home_verySparse.wad"
        super(MyVizdoomEnv, self).__init__(
            scenario_file, frame_skip, max_buttons_pressed, render_mode
        )

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

    def render(self, mode):
        self._venv.render()

