import copy
import logging
import pprint

from collections import namedtuple
from typing import List, Dict

from ped_env.envs import PedsMoveEnv
from ped_env.interfaces.base import BaseWrapper
from ped_env.interfaces.maicm_interface import PedEnvWrapper


class MNTransferWrapper(BaseWrapper):
    def __init__(self, env, n, control_gene_type="seq"):
        """
        将场景中的M个智能体转换成可被N个智能体的多智能体强化学习算法所控制，
        control_gene_type用于生成控制序列
        """
        self.env: PedsMoveEnv = env
        self.M = self.env.num_agents
        self.N = n
        assert self.M >= self.N
        assert self.M % self.N == 0
        self.cur_state: List = []
        self.now_gived_action_count = 0
        self.control_list: List[str] = []
        self.control_list_seq = []
        self.now_control_index = 0
        self.cur_action_list = {}

    def reset_in_env(self):
        obs_dict = self.env.reset()
        return obs_dict

    def reset(self):
        obs_list = self.reset_in_env()
        # 进行随机一步
        acts = {}
        for ag in self.env.possible_agents:
            acts[ag] = self.env.action_space.sample()
        self.step_in_env(acts)

        self.now_gived_action_count = 0
        self.gene_control_list()
        self.update_control_list()
        return obs_list

    def gene_control_list(self):
        self.control_list_seq = []
        for i in range(self.M // self.N):
            k = self.N * i
            self.control_list_seq.append([self.env.possible_agents[j] for j in range(k, k + self.N)])

    def update_control_list(self):
        self.control_list = self.control_list_seq[self.now_control_index]
        self.now_control_index += 1
        self.now_control_index %= (self.M // self.N)
        #print(self.now_control_index)

    def step_in_env(self, actions):
        next_o, r, done, truncated, info = self.env.step(actions)
        self.cur_state = [next_o, r, done, truncated, info]
        self.now_gived_action_count = 0

    def step(self, actions):
        """
        这里给出的输出
        """
        for idx in actions.keys():
            self.cur_action_list[idx] = actions[idx]
        self.now_gived_action_count += self.N
        if self.now_gived_action_count == self.M:
            acts = {}
            for ag in self.env.possible_agents:
                if self.cur_action_list[ag] is None:
                    logging.error("错误的控制序列")
                acts[ag] = self.cur_action_list[ag]

            self.step_in_env(acts)
            self.now_gived_action_count = 0
            self.now_control_index = 0
        else:
            self.update_control_list()
        next_o, r, done, truncated, info = self.cur_state
        return next_o, r, done, truncated, info

    def render(self, mode="human", ratio=1.0):
        self.env.render(mode=mode, ratio=ratio)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)


def create_ped_env_mn(map="map_09", agent_num=4, leader_num=4, group_size=1, maxStep=10000, disable_reward=False,
                      frame_skip=8,
                      seed=None, use_adv_net=False, use_concat_obs=False, with_force=True):
    if (leader_num * group_size) % agent_num != 0:
        raise Exception(u"错误的输入参数!")
    env = PedEnvWrapper(
        MNTransferWrapper(
            PedsMoveEnv(terrain=map, person_num=leader_num * group_size, group_size=(group_size, group_size),
                        maxStep=maxStep, disable_reward=disable_reward, discrete=True, frame_skipping=frame_skip,
                        with_force=with_force),
            agent_num),
        use_adv_network=use_adv_net, use_concat_obs=use_concat_obs)
    env.seed(seed)
    return env


def test_wrapper_api(debug=False):
    import time

    M = 20
    N = 4
    env = PedEnvWrapper(MNTransferWrapper(
        PedsMoveEnv(terrain="map_10", person_num=M, group_size=(1, 1), discrete=True),
        N
    ), use_adv_network=False, use_concat_obs=True)

    for epoch in range(1):
        start_time = time.time()
        step = 0
        is_done = False
        gobs, obs = env.reset()

        def get_single_action(agent):
            return env.env.action_space.sample()

        obs_arr = []

        while not is_done:
            for _ in range(M // N):
                # action = {agent: get_single_action(agent) for agent in env.now_control_list}
                action = [get_single_action(agent) for agent in env.env.agents]
                state, obs, reward, is_done, info = env.step(action)
            env.render()
            if len(obs) != 0:
                pprint.pprint(reward)
            # print("is_done:")
            # pprint.pprint(state)
            step += env.env.frame_skipping
            time.sleep(0.01)
    env.close()


if __name__ == '__main__':
    test_wrapper_api()
