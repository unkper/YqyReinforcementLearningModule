import numpy as np
import gym

from numpy import inf

from ped_env.envs import PedsMoveEnv
from ped_env.run import save_video


class PedEnvWrapper:
    def __init__(self, env, joint_count=False, use_adv_network=False):
        self.env: PedsMoveEnv = env
        self.row, self.col = self.env.terrain.width, self.env.terrain.height
        self.num_agents = self.env.num_agents
        if use_adv_network:
            self.state_space = gym.spaces.Box(0, 9, [self.env.terrain.width, self.env.terrain.height])
        else:
            self.state_space = gym.spaces.Box(0, 9, [self.env.terrain.width*self.env.terrain.height])
        self.observation_space = self.env.observation_space("0")
        self.action_space = self.env.action_space("0")
        self.joint_count = joint_count
        self.use_adv_net = use_adv_network

        if self.joint_count:
            self.visit_counts = np.zeros(self.num_agents * [self.row * 2, self.col * 2])
        else:
            self.visit_counts = np.zeros((self.num_agents, self.row * 2, self.col * 2))

        self._prv_state = None
        self._prv_obs = None

    def reset(self):
        obs_dict = self.env.reset()
        _obs = []
        for agent in self.env.possible_agents:
            _obs.append(np.array(obs_dict[agent]))

        global_obs = self.get_global_obs()

        if self.joint_count:
            visit_inds = tuple(sum([[int(a.x), int(a.y)] for a in self.env.possible_agents], []))
            self.visit_counts[visit_inds] += 1
        else:
            for a in self.env.possible_agents:
                idx = int(a)
                agent = self.env.agents_dict[a]
                self.visit_counts[idx, int(agent.x), int(agent.y)] += 1
        self._prv_state = global_obs
        self._prv_obs = _obs
        return global_obs, _obs

    def get_st_obs(self):
        return self._prv_state, self._prv_obs

    def get_global_obs(self):
        wall_symbol = {'1', '2', 'dw', 'lw', 'rw', 'uw', 'cluw', 'cruw', 'cldw', 'crdw'}
        exit_symbol = {str(i) for i in range(3, 9)}
        h, w = self.env.terrain.height, self.env.terrain.width
        g_obs = np.zeros([w, h])
        for i in range(w):
            for j in range(h):
                if self.env.terrain.map[i][j] in wall_symbol:
                    g_obs[i, j] = 1  # 障碍物统一设置为1
                elif self.env.terrain.map[i][j] in exit_symbol:
                    g_obs[i, j] = int(self.env.terrain.map[i][j])
        for ped in self.env.leaders:  # 原来是peds
            g_obs[int(ped.x), int(ped.y)] = ped.exit_type  # 将有人的设置为2
        return g_obs.ravel() if not self.use_adv_net else g_obs

    def step(self, actions):
        acts = {}
        for i, agent in enumerate(self.env.agents):
            if isinstance(actions[i], np.ndarray):
                actions[i] = np.argmax(actions[i])
            acts[agent] = actions[i]
        # pprint.pprint(actions)
        next_o, r, done, truncated, info = self.env.step(acts)
        _obs = []
        _rew = []
        _done = []
        _info = {}
        for agent in self.env.possible_agents:
            _obs.append(np.array(next_o[agent]))
            _rew.append(r[agent])
            _done.append(done[agent])

        _info['visit_count_lookup'] = []
        _info['n_found_treasures'] = []
        for a in self.env.possible_agents:
            agent = self.env.agents_dict[a]
            _info['visit_count_lookup'].append([int(agent.x), int(agent.y)])
            _info['n_found_treasures'].append(1 if agent.is_done else 0)
        global_obs = self.get_global_obs()

        if self.joint_count:
            visit_inds = tuple(sum([[int(a.x), int(a.y)] for a in self.env.possible_agents], []))
            self.visit_counts[visit_inds] += 1
        else:
            for a in self.env.possible_agents:
                idx = int(a)
                agent = self.env.agents_dict[a]
                self.visit_counts[idx, int(agent.x), int(agent.y)] += 1
        self._prv_state = global_obs
        self._prv_obs = _obs
        # 注意该接口返回的rew的是所有agent奖励的加和值，all代表所有智能体都为done后
        return global_obs, _obs, np.mean(_rew).tolist(), all(_done), _info

    def render(self, mode="human", ratio=1.0):
        self.env.render(mode=mode, ratio=ratio)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)


def create_ped_env(map="map_09", leader_num=4, group_size=1, maxStep=10000, disable_reward=False, frame_skip=8,
                   seed=None, use_adv_net=False):
    env = PedEnvWrapper(
        PedsMoveEnv(terrain=map, person_num=leader_num * group_size, group_size=(group_size, group_size),
                    maxStep=maxStep, disable_reward=disable_reward, discrete=True, frame_skipping=frame_skip)
        , use_adv_network=use_adv_net)
    env.seed(seed)
    return env


def test_wrapper_api(debug=False):
    import time

    person_num = 20
    env = create_ped_env("map_10", 4, 5, maxStep=2000)
    # env = Env(map_simple, person_num, group_size=(1, 1), frame_skipping=8, maxStep=10000, debug_mode=False,
    #           random_init_mode=True, person_handler=PedsRLHandlerWithForce)

    for epoch in range(1):
        start_time = time.time()
        step = 0
        is_done = False
        env.reset()

        def get_single_action(agent):
            return env.env.action_space(agent).sample()

        obs_arr = []

        while not is_done:
            action = [get_single_action(agent) for agent in env.env.agents]
            state, obs, reward, is_done, info = env.step(action)
            env.render()
            # print("is_done:")
            # pprint.pprint(info)
            step += env.env.frame_skipping
            # time.sleep(0.01)
            obs_arr.append(state)
        save_video(obs_arr, "123")
    env.close()


if __name__ == '__main__':
    for i in range(1):
        test_wrapper_api()
