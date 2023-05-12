import pprint

from ped_env.envs import PedsMoveEnv


class NMTransferWrapper:
    def __init__(self, env, n):
        """
        将场景中的N个智能体转换成可被M个智能体的多智能体强化学习算法所控制
        """
        self.env: PedsMoveEnv = env
        self.M = self.env.num_agents
        self.N = n
        assert self.M >= self.N
        assert self.M % self.N == 0

        self.control_list = []

    def reset(self):
        obs_dict = self.env.reset()
        return obs_dict

    def step(self, actions):
        next_o, r, done, truncated, info = self.env.step(actions)

        return next_o, r, done, truncated, info

    def render(self, mode="human", ratio=1.0):
        self.env.render(mode=mode, ratio=ratio)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)


def test_wrapper_api(debug=False):
    import time

    person_num = 20
    env = NMTransferWrapper(
        PedsMoveEnv(terrain="map_10", person_num=12, group_size=(1, 1), discrete=True),
        4
    )

    for epoch in range(1):
        start_time = time.time()
        step = 0
        is_done = {"0": False}
        obs = env.reset()

        def get_single_action(agent):
            return env.env.action_space(agent).sample()

        obs_arr = []

        while not all(is_done.values()):
            action = {agent:get_single_action(agent) for agent in env.env.agents}
            obs, reward, is_done, trunc, info = env.step(action)
            env.render()
            # print("is_done:")
            # pprint.pprint(state)
            step += env.env.frame_skipping
            time.sleep(0.01)
    env.close()


if __name__ == '__main__':
    test_wrapper_api()
