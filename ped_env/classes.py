from math import inf

import abc
from typing import List

from Box2D import b2QueryCallback, b2Fixture, b2RayCastCallback
from gym.spaces import Box, Discrete
from ped_env.functions import parse_discrete_action
from ped_env.objects import Person
from ped_env.utils.misc import ObjectType

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
        ped.fraction_force()
        if not ped.exam_leader_moved(ped.body):
            ped.fij_force(self.env.not_arrived_peds, self.env.leader_follower_dic)
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
        ped.fraction_force()
        if not ped.exam_leader_moved(ped.body):
            ped.fij_force(self.env.not_arrived_peds, self.env.leader_follower_dic)
        ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)

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

