import copy
import os
import pickle
import time

import gym
import random
import numpy as np
import torch
from torch import nn
from multiprocessing import Pipe, Process

import ped_env
from departed_rl.utils.functions import flatten_data, process_maddpg_experience_data
from departed_rl.utils.model.functions import set_rollout_length, set_model_train_freq
from departed_rl.utils.updates import hard_update


class Transition():
    counter = 0

    def __init__(self, s0, a0, reward: float, is_done: bool, s1):
        self.id = Transition.counter
        Transition.counter += 1
        self.data = [s0, a0, reward, is_done, s1]

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        return "s0:{};a0:{};reward:{:.3f};is_done:{};s1:{};".format(
            self.data[0], self.data[1], self.data[2], self.data[3],
            self.data[4]
        )

    def __repr__(self):
        return "id=" + str(self.id)

    @property
    def s0(self):
        return self.data[0]

    @property
    def a0(self):
        return self.data[1]

    @property
    def reward(self):
        return self.data[2]

    @property
    def is_done(self) -> bool:
        return self.data[3]

    @property
    def s1(self):
        return self.data[4]


class Experience():
    '''
    该类是用来存储智能体的相关经历的，它由一个列表所组成，
    该类可以通过调用方法来随机返回几个不相关的序列
    '''

    def __init__(self, capacity: int = 20000):
        capacity = int(capacity)
        self.capacity = capacity  # 容量：指的是trans总数量
        self.transitions = np.ndarray([self.capacity], dtype=np.object)
        self.next_id = 0  # 下一个episode的Id
        self.total_trans = 0

    def push(self, trans: Transition):
        if self.capacity <= 0:
            return
        self.transitions[self.next_id] = trans
        self.next_id = (self.next_id + 1) % self.capacity  # 循环队列
        if self.total_trans < self.capacity:  # 如果超过就丢弃掉最开始的trans
            self.total_trans += 1
        return trans

    def resize(self, capacity):
        capacity = int(capacity)
        if self.capacity == capacity:
            return
        if self.capacity < capacity:  # $
            new_arr = np.ndarray([capacity], dtype=np.object)
            new_arr[:self.total_trans] = self.transitions[:self.total_trans]
            self.capacity = capacity
            self.next_id = self.total_trans % self.capacity
            self.transitions = new_arr
        else:
            new_arr = np.ndarray([capacity], dtype=np.object)
            if self.total_trans <= capacity:  # $
                self.capacity = capacity
                self.total_trans = 0
                self.next_id = 0
                self.transitions = new_arr
            else:
                self.capacity = capacity
                self.total_trans = 0
                self.next_id = 0
                self.transitions = new_arr

    def sample(self, batch_size=1):  # sample transition
        '''randomly sample some transitions from agent's experience.abs
        随机获取一定数量的状态转化对象Transition
        args:
            number of transitions need to be sampled
        return:
            list of Transition.
        '''
        return random.sample(self.transitions[:self.total_trans].tolist(), batch_size)

    def sample_and_shuffle(self):
        idx = np.random.permutation(self.total_trans)
        return self.transitions[idx]

    def last_n_trans(self, N):
        if self.len >= N:
            return self.transitions[self.total_trans - N: self.total_trans].tolist()
        return None

    @property
    def last_trans(self):
        if self.len > 0:
            return self.transitions[self.total_trans - 1].tolist()
        return None

    @property
    def len(self):
        return self.total_trans

    def __str__(self):
        return "exp info:{0:5} trans, memory usage {1}/{2}". \
            format(self.len, self.total_trans, self.capacity)

    def __len__(self):
        return self.len


class Noise():
    '''
    用于连续动作空间的噪声辅助类，输出具有扰动的一系列值
    '''

    def __init__(self, action_dim, policy_noise=0.4, noise_clip=1.0):
        '''
        动作
        :param action_dim:动作空间的维数
        :param mu:
        :param theta:
        :param sigma:
        '''
        self.action_dim = action_dim
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def sample(self):
        return np.clip(np.random.randn(self.action_dim) * self.policy_noise, -self.noise_clip, self.noise_clip)


class SaveNetworkMixin():
    def save(self, sname: str, name: str, network: nn.Module, step: int):
        p = os.path.join("./", sname)
        if not os.path.exists(p):
            os.mkdir(p)
        save_name = os.path.join("./", sname, "./model/{}.pkl".format(name))
        torch.save(network.state_dict(), save_name)
        desc_txt_file = open(os.path.join(sname, "desc.txt"), "w+")
        desc_txt_file.write("algorithm:" + str(self) + "\n")
        desc_txt_file.write("batch_size:" + str(self.batch_size) + "\n")
        desc_txt_file.write("update_freq:" + str(self.update_frequent) + "\n")
        desc_txt_file.write("lr:" + str(self.learning_rate) + "\n")
        desc_txt_file.write("gamma:" + str(self.gamma) + "\n")
        desc_txt_file.write("envName:" + str(self.env_name) + "\n")
        desc_txt_file.write("agent_count:" + str(self.env.agent_count) + "\n")
        desc_txt_file.write("actor_hidden_dim:" + str(self.actor_hidden_dim) + "\n")
        desc_txt_file.write("critic_hidden_dim:" + str(self.critic_hidden_dim) + "\n")
        if isinstance(self.env, ped_env.envs.PedsMoveEnv) or isinstance(self.env,
                                                                        SubprocEnv) and self.env.type == ped_env.envs.PedsMoveEnv:
            if isinstance(self.env, SubprocEnv):
                a1, a2, a3, a4 = self.env.get_env_attr()
            else:
                a1, a2, a3, a4 = self.env.person_num, self.env.group_size, self.env.maxStep, self.env.terrain.name
            desc_txt_file.write("person_num:" + str(a1) + "\n")
            desc_txt_file.write("group_size:" + str(a2) + "\n")
            desc_txt_file.write("max_step:" + str(a3) + "\n")
            desc_txt_file.write("map_name:" + str(a4) + "\n")
        if self.info_handler != None:
            self.info_handler.save(self.log_dir)
        return save_name

    def load(self, savePath, network: nn.Module):
        network.load_state_dict(torch.load(savePath))


class SaveDictMixin():
    def save_obj(self, obj, name):
        save_name = os.path.join("./", name + '.pkl')
        with open(save_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return save_name

    def load_obj(self, savePath):
        with open(savePath, 'rb') as f:
            return pickle.load(f)


class MAAgentMixin():
    def get_exploitation_action(self, state):
        """
        得到给定状态下依据目标演员网络计算出的行为，不探索
        :param state: numpy数组
        :return: 动作 numpy数组
        """
        action_list = []
        for i in range(self.env.agent_count):
            s = flatten_data(state[i], self.state_dims[i], self.device)
            action = self.agents[i].step(s, False).detach().cpu().numpy()
            action_list.append(action)
        action_list = np.array(action_list, dtype=np.float)
        return action_list

    def get_exploration_action(self, state, epsilon=0.1):
        '''
        得到给定状态下根据演员网络计算出的带噪声的行为，模拟一定的探索
        :param state: numpy数组
        :return: action numpy数组
        '''
        action_list = []
        value = random.random()
        for i in range(self.env.agent_count):
            s = flatten_data(state[i], self.state_dims[i], self.device)
            action = self.agents[i].step(s, True if value < epsilon else False).detach().cpu().numpy()
            action_list.append(action)
        action_list = np.array(action_list, dtype=np.float)
        return action_list

    def play_init(self, savePath, s0):
        import os
        for i in range(self.env.agent_count):
            saPath = os.path.join(savePath, "Actor{}.pkl".format(i))
            self.load(saPath, self.agents[i].actor)
            hard_update(self.agents[i].target_actor, self.agents[i].actor)
        return self.get_exploitation_action(s0)

    def play_step(self, savePath, s0):
        return self.get_exploitation_action(s0)

    def step_in_network(self, s0, explore, epsilon):
        if self.n_rol_threads == 1:
            if explore:
                a0 = self.get_exploration_action(s0, epsilon)
            else:
                a0 = self.get_exploitation_action(s0)
            return a0
        else:
            a0 = []
            for i in range(self.n_rol_threads):
                s = s0[i]
                if explore:
                    a = self.get_exploration_action(s, epsilon)
                else:
                    a = self.get_exploitation_action(s)
                a0.append(a)
            return np.stack(a0)

    def policy_init_step(self):
        self.loss_critic, self.loss_actor = 0.0, 0.0

    def policy_update_step(self, step, epsilon=0.0):
        if self.total_trans > self.batch_size and step % self.update_frequent == 0:
            loss_c, loss_a = 0, 0
            for i in range(self.n_steps_train):
                trans_pieces = self.experience.sample(self.batch_size)
                lc, la = self._learn_from_memory(trans_pieces)

                if self.demo_experience:
                    demo_trans_pieces = self.demo_experience.sample(self.batch_size_d)
                    self._learn_from_memory(demo_trans_pieces, True)
                loss_c += lc
                loss_a += la

            self.train_update_count += 1
            loss_c /= self.n_steps_train
            loss_a /= self.n_steps_train

            self.loss_critic += loss_c
            self.loss_actor += loss_a

    def policy_end_step(self, time_in_episode):
        self.loss_critic /= time_in_episode
        self.loss_actor /= time_in_episode
        return [self.loss_critic, self.loss_actor]

    def learning_method(self, epsilon=0.2, explore=True, display=False,
                        wait=False, waitSecond: float = 0.01):
        self.state = self.env.reset()
        time_in_episode = 0
        total_reward = [0.0 for _ in range(self.env.agent_count)]
        is_done = np.array([[False]])
        s0 = self.state
        self.policy_init_step()
        # is_done此时已经为数组
        while not is_done.any():
            a0 = self.step_in_network(s0, explore, epsilon)
            s1, r1, is_done, info = self.act(a0)
            if isinstance(is_done, list):
                is_done = np.array(is_done)
            if self.n_rol_threads == 1:
                for i in range(self.env.agent_count):
                    total_reward[i] += np.mean(r1[i])
            else:
                for i in range(self.env.agent_count):
                    total_reward[i] += np.mean(r1[:, i])
            if display:
                self.env.render()
            self.policy_update_step(int(self.total_steps_in_train / self.n_rol_threads), epsilon)
            time_in_episode += 1
            self.total_steps_in_train += self.n_rol_threads
            s0 = s1
            if wait:
                time.sleep(waitSecond)
        if self.info_callback_ != None and info != None:
            if self.n_rol_threads == 1:
                self.info_callback_(info, self.info_handler, True)
            else:
                self.info_callback_(info[0], self.info_handler, True)
        loss = self.policy_end_step(time_in_episode)

        if self.total_episodes_in_train > 0:
            rewards = {}
            sum = 0
            last_episodes = np.array(self.episode_rewards[-1:])
            for i in range(self.env.agent_count):
                me = np.mean(last_episodes[:, i])
                rewards['Agent{}'.format(i)] = me
                sum += me
            self.writer.add_scalars("agents/reward", rewards, self.total_steps_in_train)
            self.writer.add_scalar("agents/mean_reward", sum / self.env.agent_count, self.total_steps_in_train)

        if self.total_episodes_in_train > 0 \
                and self.total_episodes_in_train % (self.log_frequent) == 0:
            rewards = []
            last_episodes = np.array(self.episode_rewards[-self.log_frequent:])
            for i in range(self.env.agent_count):
                rewards.append(np.mean(last_episodes[:, i]))
            print("average rewards in last {} episodes:{}".format(self.log_frequent, rewards))
            print("{}".format(self.experience.__str__()))
            for i, agent in enumerate(self.agents):
                print("Agent{}:{}".format(i, agent.count))
                agent.count = [0 for _ in range(agent.action_dim)]
        return time_in_episode, total_reward, loss


from math import cos, sin


class ModelBasedMAAgentMixin():
    def policy_init_step(self):
        self.loss_critic, self.loss_actor, self.loss_model = 0.0, 0.0, 0.0

    def policy_update_step(self, step, epsilon=0.0):
        loss_c, loss_a, loss_m = 0, 0, 0
        if self.experience.len > self.model_batch_size and step % self.model_train_freq == 0:  # 修改
            # self.n_step_model = set_model_train_freq(self.total_episodes_in_train, self.n_steps_model_range[0],
            #                                         self.n_steps_model_range[1], self.rollout_epoch_range[0],
            #                                         self.rollout_epoch_range[1])

            losses = self._learn_simulate_world()
            loss_m = losses[0]
            self.model_trained = True

        loss_c, loss_a = 0.0, 0.0
        if self.model_trained and self.total_trans > self.batch_size and step % self.update_frequent == 0:
            # rollout model
            new_rollout_length = set_rollout_length(self.total_episodes_in_train, self.rollout_length_range[0],
                                                    self.rollout_length_range[1], self.rollout_epoch_range[0],
                                                    self.rollout_epoch_range[1])
            if self.rollout_length != new_rollout_length:
                self.rollout_length = new_rollout_length
                rollouts_per_epoch = self.rollout_batch_size * self.episode_length / self.model_train_freq  # 此处应该是一个episode总的step数,MAMBPO里设定为固定值2000
                model_steps_per_epoch = int(self.rollout_length * rollouts_per_epoch)
                self.model_experience.resize(self.model_retain_epochs * model_steps_per_epoch)

            if self.experience.len >= self.rollout_batch_size and self.real_ratio < 1.0:
                self._rollout_model(self.rollout_length, epsilon)

                # if self.demo_experience:
                #     self._rollout_model(self.rollout_length, epsilon, True)
            # Gradient update n steps
            for i in range(self.n_steps_train):
                real_batch_size = int(self.real_ratio * self.batch_size)  # 从环境中采集经验数
                model_batch_size = self.batch_size - real_batch_size  # 从模型中采集经验数

                if self.real_ratio < 1.0 and model_batch_size > 0 and len(self.model_experience) >= model_batch_size:
                    trans_pieces = self.experience.sample(real_batch_size)
                    model_trans = self.model_experience.sample(model_batch_size)
                    trans_pieces += model_trans
                else:
                    trans_pieces = self.experience.sample(self.batch_size)

                lc, la = self._learn_from_memory(trans_pieces)

                if self.demo_experience:
                    demo_trans_pieces = self.demo_experience.sample(self.batch_size_d)
                    self._learn_from_memory(demo_trans_pieces, True)

                loss_c += lc
                loss_a += la
            loss_c /= self.n_steps_train
            loss_a /= self.n_steps_train
            self.train_update_count += 1

        if loss_m != 0.0:
            self.writer.add_scalar("step_loss/model", loss_m, self.total_steps_in_train)
        self.loss_critic += loss_c
        self.loss_actor += loss_a
        self.loss_model += loss_m

    def policy_end_step(self, time_in_episode):
        self.loss_critic /= time_in_episode
        self.loss_actor /= time_in_episode
        self.loss_model /= time_in_episode
        return [self.loss_critic, self.loss_actor, self.loss_model]

    def transform_discrete_a(self, a0_idx):
        global DIRECTIONS
        a0 = np.zeros([a0_idx.shape[0], a0_idx.shape[1] * 2])
        for i in range(a0_idx.shape[0]):
            for j in range(a0_idx.shape[1]):
                a0[i, j * 2:(j + 1) * 2] = DIRECTIONS[a0_idx[i, j]]
        return torch.from_numpy(a0).to(self.device)

    def _learn_simulate_world(self):
        global DIRECTIONS
        mean_losses = [0.0 for _ in range(3)]
        for i in range(self.n_steps_model):
            trans_pieces = self.experience.sample(self.model_batch_size)
            s0, temp_a0, r1, is_done, s1, s0_critic_in, s1_critic_in = \
                process_maddpg_experience_data(trans_pieces, self.state_dims, self.env.agent_count, self.device)
            if self.discrete:
                a0_idx = np.argmax(np.array([x.a0 for x in trans_pieces]), axis=2).astype(int)  # 将One-hot形式转换为索引
                a0 = self.transform_discrete_a(a0_idx)
            else:
                a0 = temp_a0
            delta_state = s1_critic_in - s0_critic_in
            # world model输入为(s,a),输出为(s',r,is_done)
            inputs = torch.cat([s0_critic_in, a0], dim=-1).detach().cpu().numpy()
            labels = torch.cat([torch.reshape(r1, (r1.shape[0], -1)), delta_state], dim=-1).detach().cpu().numpy()
            # 输入x = (state,action),y = (r,delta_state)
            eval_loss, var_loss, mse_loss = self.predict_env.model.train_r_network_with_collector(inputs, labels, batch_size=256,
                                                                                                  holdout_ratio=0.2)

            mean_losses[0] += eval_loss.mean().item()
            mean_losses[1] += var_loss
            mean_losses[2] += mse_loss.mean().item()
        for i in range(mean_losses.__len__()):
            mean_losses[i] /= self.n_steps_model
        print("model learn finished,eval_loss:{},var_loss:{},mse_loss:{}.".format(mean_losses[0], mean_losses[1],
                                                                                  mean_losses[2]))
        return mean_losses

    def init_random_step(self, state, step):
        a0 = self.step_in_network(state, True, 1.0)
        return self.act(a0)

    def _rollout_model(self, rollout_length, epsilon, use_a_star_policy=False):
        trans_pieces = self.experience.sample(self.rollout_batch_size)
        s0 = np.array([x.s0 for x in trans_pieces])
        r1 = np.array([x.reward for x in trans_pieces])
        # is_done = np.array([x.is_done for x in trans_pieces])
        s1 = np.array([x.s1 for x in trans_pieces])

        state = s0
        for i in range(rollout_length):
            state_in = np.reshape(state, [state.shape[0], state.shape[1] * state.shape[2]])  # 打平数组以便输入
            if use_a_star_policy:
                raw_action = []
                for s in state:
                    raw_action.append(self.a_star_policy.step(s))
            else:
                raw_action = []
                for s in state:
                    raw_action.append(self.get_exploration_action(s, epsilon))
            if self.discrete:
                action_idx = np.argmax(np.array(raw_action), axis=2)  # 将One-hot形式转换为索引
                action = self.transform_discrete_a(action_idx).cpu().numpy()
            else:
                action = np.array(raw_action).astype(np.float)
                action = np.reshape(action, [action.shape[0], action.shape[1] * action.shape[2]])
            next_states, rewards, terminals, info = self.predict_env.step(state_in, action)
            if i == 0:
                delta_s1 = abs(s1 - next_states)
                delta_r = abs(r1 - rewards)
                s1_data = {"max": np.max(delta_s1), "min": np.min(delta_s1), "mean": np.mean(delta_s1)}
                r_data = {"max": np.max(delta_r), "min": np.min(delta_r), "mean": np.mean(delta_r)}
                self.writer.add_scalars("step_loss/state", s1_data, self.total_steps_in_train)
                self.writer.add_scalars("step_loss/reward", r_data, self.total_steps_in_train)
                # log_string = "s1_delta,max:{}%%%min:{}%%%mean:{},".format(str(np.max(delta_s1)), str(np.min(delta_s1)), str(np.mean(delta_s1))) + \
                #              "r_delta,max:{}%%%min:{}%%%mean:{},".format(str(np.max(delta_r)), str(np.min(delta_r)), str(np.mean(delta_r))) + \
                #              "loss_model:{},".format(self.loss_model)
                # max_delta_s1_idx = np.unravel_index(np.argmax(delta_s1, axis=None), delta_s1.shape)
                # log_string2 = "max_delta_s1_idx:{}{}".format(max_delta_s1_idx, max_delta_s1_idx[1] % 16)
                # print("Step:{},{},{}".format(self.total_steps_in_train, log_string, log_string2))

            for j in range(state.shape[0]):
                tran = Transition(state[j], raw_action[j], rewards[j], terminals[j], next_states[j])
                if use_a_star_policy:
                    self.demo_experience.push(tran)
                else:
                    self.model_experience.push(tran)
            nonterm_mask = np.ones([terminals.shape[0]], dtype=np.bool)
            for idx in range(terminals.shape[0]):
                if terminals[idx].all():  # 这里将any改为all
                    nonterm_mask[idx] = False
            if nonterm_mask.sum() == 0:
                break
            state = next_states[nonterm_mask]  # 去掉终止态的状态


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_count':
            remote.send(env.agent_count)
        elif cmd == 'render':
            env.render()
        elif cmd == 'get_type':
            remote.send(type(env))
        elif cmd == 'get_attr':
            if isinstance(env, ped_env.envs.PedsMoveEnv):
                remote.send((env.person_num, env.group_size, env.maxStep, env.terrain.name))
            else:
                remote.send(None)
        else:
            raise NotImplementedError


def make_parallel_env(ped_env, n_rollout_threads):
    def get_env_fn(rank):
        def init_env():
            env = copy.deepcopy(ped_env)
            return env

        return init_env

    if n_rollout_threads == 1:
        return get_env_fn(0)
    else:
        return SubprocEnv([get_env_fn(i) for i in range(n_rollout_threads)])


# https://github.com/openai/baselines
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


# https://github.com/shariqiqbal2810/maddpg-pytorch
class SubprocEnv(gym.Env):
    def __init__(self, env_fns, spaces=None):
        """
        env_fns: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_count', None))
        self.agent_count = self.remotes[0].recv()
        self.remotes[0].send(('get_type', None))
        self.type = self.remotes[0].recv()
        self.remotes[0].send(('get_attr', None))
        self.extra_data = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, action):
        self.step_async(action)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self, mode="human"):
        for remote in self.remotes:
            remote.send(('render', None))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_env_attr(self):
        return self.extra_data


class PedsMoveInfoDataHandler:
    def __init__(self, terrain, agent_count, save_l_pos=True):
        self.info_data = {
            "evacuation_time": [],
            "collision_wall_agent": [],
            "collision_agent_agent": [],
            "leader_pos": []
        }
        self.agent_count = agent_count
        self.save_l_pos = save_l_pos
        self.map_shape = terrain.map.shape
        self.leader_pos_arr = np.zeros([agent_count, self.map_shape[0], self.map_shape[1]], dtype=int)

    def step(self, info):
        if not self.save_l_pos:
            return
        time, c_wa, c_aa, l_pos = info
        l_pos_idx = np.array(l_pos).astype(int)
        for i in range(self.leader_pos_arr.shape[0]):
            self.leader_pos_arr[i, l_pos_idx[i, 0], l_pos_idx[i, 1]] += 1

    def reset(self, info):
        time, c_wa, c_aa, l_pos = info
        self.info_data["evacuation_time"].append(time)
        self.info_data["collision_wall_agent"].append(c_wa)
        self.info_data["collision_agent_agent"].append(c_aa)
        self.info_data["leader_pos"].append(self.leader_pos_arr)
        self.leader_pos_arr = np.zeros([self.agent_count, self.map_shape[0], self.map_shape[1]], dtype=np.int)

    def save(self, dir):
        pa = os.path.join(dir, "extra_data")
        if not os.path.exists(pa):
            os.mkdir(pa)
        for key, value in self.info_data.items():
            file = open(os.path.join(pa, str(key) + ".npy"), "wb+")
            arr = np.array(value, dtype=np.float32)
            np.save(file, arr)
            file.close()


if __name__ == "__main__":
    cap = 20
    exp = Experience(capacity=cap)
    for i in range(int(cap)):
        trans = Transition(i, i, i, i, i)
        exp.push(trans)
    # for i in range(8):
    #     print(exp.sample(int(cap / 2)))
    # print(exp.sample_and_shuffle())
    exp.resize(10)
    for i in range(20):
        exp.push(Transition(0, 0, 0, 0, 0))
    print(exp.sample(8))
    print(exp.sample_and_shuffle())
