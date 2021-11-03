
import abc
import numpy as np



from math import inf
from typing import List
from multiprocessing import Process, Pipe

from gym.spaces import Box, Discrete
from ped_env.functions import parse_discrete_action, calculate_nij, normalized
from ped_env.objects import Person, Group

ACTION_DIM = 9

class PedsHandlerInterface(abc.ABC):
    def __init__(self, env):
        pass

    @abc.abstractmethod
    def step(self, peds:List[Person]):
        pass

    @abc.abstractmethod
    def get_observation(self, ped:Person):
        pass

    @abc.abstractmethod
    def set_action(self, ped:Person, action):
        pass

    @abc.abstractmethod
    def set_follower_action(self, ped:Person, action, group:Group, exit_pos):
        pass

    @abc.abstractmethod
    def get_reward(self, ped:Person, ped_index:int):
        pass

class PedsRLHandler(PedsHandlerInterface):
    '''
    非合作的奖励机制
    '''
    def __init__(self, env, r_arrival=15, r_approaching=2.5, r_collision=-15, r_cost_time=-0.1):
        super().__init__(env)
        self.env = env

        person_num_sum = self.env.person_num
        reminder = person_num_sum % len(self.env.terrain.start_points)
        person_num_in_every_spawn = person_num_sum // len(self.env.terrain.start_points) \
            if person_num_sum >= len(self.env.terrain.start_points) else 1
        person_num = [person_num_in_every_spawn
                      for _ in range(len(self.env.terrain.start_points))]
        person_num[-1] += reminder
        self.agent_count = sum([int(num / int(sum(self.env.group_size) / 2)) for num in person_num])

        # 强化学习MDP定义区域
        # 定义观察空间为[id,8个方向的传感器,智能体当前位置(x,y),智能体当前速度(dx,dy)]一共12个值
        self.observation_space = [Box(-inf, inf, (12,)) for _ in range(self.agent_count)]
        # 定义动作空间为[不动，向左，向右，向上，向下]施加1N的力
        self.action_space = [Discrete(ACTION_DIM) for _ in range(self.agent_count)]

        self.r_arrival = r_arrival
        self.r_approach = r_approaching
        self.r_collision = r_collision
        self.r_cost_time = r_cost_time

    def step(self, peds:List[Person]):
        '''
        根据当前所有行人的状态，评估得到它们的奖励
        :param peds:
        :return: s',r
        '''
        obs, rewards = [], []
        for idx, ped in enumerate(peds):
            obs.append(self.get_observation(ped))
            rewards.append(self.get_reward(ped, idx))
        return obs, rewards

    def get_observation(self, ped:Person):
        observation = []
        if ped.is_done:
            #根据论文中方法，给予一个零向量
            observation.extend([0.0 for _ in range(8)]) #5个方向上此时都不应该有障碍物
            observation.extend([0.0, 0.0]) #将智能体在出口的位置赋予
            observation.extend([0.0, 0.0]) #智能体的速度设置为0
            return observation
        #依次得到8个方向上的障碍物,在回调函数中体现，每次调用该函数都会给observation数组中添加值，分别代表该方向上最近的障碍物有多远（5米代表不存在）
        for i in range(8):
            temp = ped.raycast(self.env.world, ped.directions[i], ped.view_length)
            observation.append(temp)
        #给予智能体当前位置
        observation.append(ped.getX)
        observation.append(ped.getY)
        #给予智能体当前速度
        vec = ped.body.linearVelocity
        observation.append(vec.x)
        observation.append(vec.y)
        return observation

    def set_action(self, ped:Person, action):
        ped.self_driven_force(parse_discrete_action(action))
        ped.fij_force(self.env.not_arrived_peds, self.env.group_dic)
        ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)

    def get_reward(self, ped:Person, ped_index:int):
        reward = 0.0
        if ped.is_done and ped.has_removed:
            pass
        else:
            if ped.collide_with_agent:  # 如果智能体其他智能体相撞，奖励减一
                reward += self.r_collision
            if ped.is_done and not ped.has_removed:
                pass
            else:
                last_pos = self.env.points_in_last_step[ped_index]
                now_pos = (ped.getX, ped.getY)
                last_dis = self.env.distance_to_exit[ped_index]
                now_dis = self.env.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
                if not (last_pos[0] - 0.001 <= now_pos[0] <= last_pos[0] + 0.001 and last_pos[1] - 0.001 <= now_pos[1] <= last_pos[1] + 0.001) :
                    reward += self.r_approach * (last_dis - now_dis)  # 给予(之前离出口距离-目前离出口距离)的差值
                    reward = max(0, reward) #一次尝试，以避免两个智能体对撞的现象发生
                    self.env.distance_to_exit[ped_index] = now_dis
                    self.env.points_in_last_step[ped_index] = now_pos
                else:
                    reward += self.r_collision  # 给予停止不动的行人以碰撞惩罚
        return reward

    # def get_reward(self, ped:Person, ped_index:int):
    #     reward = 0.0
    #     if ped.is_done and ped.has_removed:
    #         pass
    #     else:
    #         reward += self.r_cost_time
    #         if ped.is_done and not ped.has_removed:
    #             pass
    #         else:
    #             last_pos = self.env.points_in_last_step[ped_index]
    #             now_pos = (ped.getX, ped.getY)
    #             if not (last_pos[0] - 0.001 <= now_pos[0] <= last_pos[0] + 0.001 and last_pos[1] - 0.001 <= now_pos[1] <= last_pos[1] + 0.001):
    #                 self.env.points_in_last_step[ped_index] = now_pos
    #             else:
    #                 reward += self.r_collision  # 给予停止不动的行人以碰撞惩罚
    #     return reward

class PedsRLHandlerWithCooper(PedsHandlerInterface):
    '''
    合作的奖励机制
    '''
    def __init__(self, env, r_arrival=15, r_approaching=1, r_collision=-1, r_wait = -15):
        super().__init__(env)
        self.env = env

        person_num_sum = self.env.person_num
        reminder = person_num_sum % len(self.env.terrain.start_points)
        person_num_in_every_spawn = person_num_sum // len(self.env.terrain.start_points) \
            if person_num_sum >= len(self.env.terrain.start_points) else 1
        person_num = [person_num_in_every_spawn
                      for _ in range(len(self.env.terrain.start_points))]
        person_num[-1] += reminder
        self.agent_count = sum([int(num / int(sum(self.env.group_size) / 2)) for num in person_num])

        # 强化学习MDP定义区域
        # 定义观察空间为[id,8个方向的传感器,智能体当前位置(x,y),智能体当前速度(dx,dy),相对目标的位置(rx,ry)]一共15个值
        self.observation_space = [Box(-inf, inf, (15,)) for _ in range(self.agent_count)]
        # 定义动作空间为[不动，向左，向右，向上，向下]施加1N的力
        self.action_space = [Discrete(ACTION_DIM) for _ in range(self.agent_count)]

        self.r_arrival = r_arrival
        self.r_approach = r_approaching
        self.r_collision = r_collision
        self.r_wait = r_wait

    def step(self, peds:List[Person]):
        '''
        根据当前所有行人的状态，评估得到它们的奖励
        :param peds:
        :return: s',r
        '''
        obs = []
        rewards = []
        global_reward = 0.0
        for idx, ped in enumerate(peds):
            if ped.is_leader:
                obs.append(self.get_observation(ped))
                gr, lr = self.get_reward(ped, idx)
                global_reward += gr
                rewards.append(lr)
        for i in range(len(rewards)):
            rewards[i] += global_reward
        return obs, rewards

    def get_observation(self, ped:Person):
        observation = []
        if ped.is_done:
            #根据论文中方法，给予id和一个零向量
            observation.append(ped.id)
            observation.extend([0.0 for _ in range(8)]) #5个方向上此时都不应该有障碍物
            observation.extend([0.0, 0.0]) #将智能体在出口的位置赋予
            observation.extend([0.0, 0.0]) #智能体的速度设置为0
            observation.extend([0.0, 0.0]) #相对距离为0
            return observation
        #给与智能体id
        observation.append(ped.id)
        #依次得到8个方向上的障碍物,在回调函数中体现，每次调用该函数都会给observation数组中添加值，分别代表该方向上最近的障碍物有多远（5米代表不存在）
        for i in range(8):
            temp = ped.raycast(self.env.world, ped.directions[i], ped.view_length)
            observation.append(temp)
        #给予智能体当前位置
        observation.append(ped.getX)
        observation.append(ped.getY)
        #给予智能体当前速度
        vec = ped.body.linearVelocity
        observation.append(vec.x)
        observation.append(vec.y)
        #给予智能体相对目标的位置
        rx, ry = self.env.get_ped_rel_pos_to_exit((ped.getX, ped.getY), ped.exit_type)
        observation.append(rx)
        observation.append(ry)
        return observation

    def set_action(self, ped:Person, action):
        ped.self_driven_force(parse_discrete_action(action))
        # if not ped.exam_leader_moved(ped.body):
        ped.fij_force(self.env.not_arrived_peds, self.env.group_dic)
        ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)

    def set_follower_action(self, ped:Person, action, group:Group, exit_pos):
        if not group.leader.is_done:
            control_dir = parse_discrete_action(action)
            leader_dir = calculate_nij(group.leader, ped)
            mix_dir = ped.alpha * control_dir + (1 - ped.alpha) * leader_dir
        else:
            pos_i = exit_pos
            pos_j = ped.pos
            mix_dir = normalized(pos_i - pos_j)
        ped.self_driven_force(mix_dir) #跟随者的方向为alpha*control_dir + (1-alpha)*leader_dir
        ped.fij_force(self.env.not_arrived_peds, self.env.group_dic)
        ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)
        #ped.ij_group_force(group)

        # 是follower用操纵力加社会力模型来控制
        # if ped.exam_leader_moved(leader.body):
        #     ped.leader_follow_force(leader.body)
        #     ped.fij_force(self.not_arrived_peds, self.leader_follower_dic)
        # #ped.fraction_force()
        # ped.fiw_force(self.walls + self.obstacles + self.exits)
        # ped.evade_controller(leader.body)

    def get_reward(self, ped:Person, ped_index:int):
        gr, lr = 0.0, 0.0
        if ped.is_done and ped.has_removed:
            pass
        else:
            if ped.collide_with_agent:
                gr += self.r_collision
            if ped.is_done and not ped.has_removed:
                lr += self.r_arrival
            else:
                last_pos = self.env.points_in_last_step[ped_index]
                now_pos = (ped.getX, ped.getY)
                last_dis = self.env.distance_to_exit[ped_index]
                now_dis = self.env.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
                if not (last_pos[0] - 0.001 <= now_pos[0] <= last_pos[0] + 0.001 and last_pos[1] - 0.001 <= now_pos[1] <= last_pos[1] + 0.001) :
                    lr += self.r_approach * (last_dis - now_dis)  # 给予(之前离出口距离-目前离出口距离)的差值
                    self.env.distance_to_exit[ped_index] = now_dis
                    self.env.points_in_last_step[ped_index] = now_pos
                else:
                    lr += self.r_wait  # 给予停止不动的行人以惩罚
        return gr, lr

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
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError

#https://github.com/shariqiqbal2810/maddpg-pytorch
class SubprocEnv():
    def __init__(self, env_fns, spaces=None):
        """
        env_fns: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

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





