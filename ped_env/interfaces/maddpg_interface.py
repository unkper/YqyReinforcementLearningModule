from ped_env.envs import PedsMoveEnv
from ped_env.interfaces.base import BaseWrapper


class MADDPG_Wrapper(BaseWrapper):
    def __init__(self, env, single_mode=False):
        """
        将场景中的N个智能体转换成可被M个智能体的多智能体强化学习算法所控制
        """
        self.env: PedsMoveEnv = env

    def reset(self):
        obs_dict = self.env.reset()
        return [obs_dict[idx] for idx in self.env.possible_agents]

    def step(self, actions):
        ac = {agent:actions[i] for i, agent in enumerate(self.env.possible_agents)}
        next_o, r, done, truncated, info = self.env.step(ac)
        next_o = [next_o[idx] for idx in self.env.possible_agents]
        r = [r[idx] for idx in self.env.possible_agents]
        done = all(done.values())

        return next_o, r, done, info

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)
