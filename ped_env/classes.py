
import abc

from math import inf
from typing import List, Dict, DefaultDict
from multiprocessing import Process, Pipe

from gym.spaces import Box, Discrete
from ped_env.functions import parse_discrete_action, calculate_nij, normalized
from ped_env.objects import Person, Group, BoxWall

ACTION_DIM = 9

class PedsHandlerInterface(abc.ABC):
    def __init__(self, env):
        pass

    @abc.abstractmethod
    def step(self, peds:List[Person], group_dic:Dict[Person, Group], time):
        pass

    @abc.abstractmethod
    def get_observation(self, ped:Person, group:Group, time):
        pass

    @abc.abstractmethod
    def set_action(self, ped:Person, action):
        pass

    @abc.abstractmethod
    def set_follower_action(self, ped:Person, action, group:Group, exit_pos):
        pass

    @abc.abstractmethod
    def get_reward(self, ped:Person, ped_index:int, time):
        pass

class PedsRLHandler(PedsHandlerInterface):
    '''
    合作的奖励机制
    '''
    def __init__(self, env, r_arrival=0, r_move = -0.1, r_wait = -0.5, r_collision=-1):
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
        # 定义观察空间为[智能体当前位置(x,y),智能体当前速度(dx,dy),相对目标的位置(rx,ry),其他跟随者的位置(不存在为(0,0))*5]一共16个值
        self.observation_space = [Box(-inf, inf, (16,)) for _ in range(self.agent_count)]
        if self.env.discrete:
            # 定义动作空间为[不动，向左，左上，向上，...]施加相应方向的力
            self.action_space = [Discrete(ACTION_DIM) for _ in range(self.agent_count)]
        else:
            #定义连续动作空间为[分量x，分量y]施加相应方向的力
            self.action_space = [Box(-1, 1, (2,)) for _ in range(self.agent_count)]

        self.r_arrival = r_arrival
        self.r_collision = r_collision
        self.r_wait = r_wait
        self.r_move = r_move

        self.last_observation = {}

    def step(self, peds:List[Person], group_dic:Dict[Person, Group], time):
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
                obs.append(self.get_observation(ped, group_dic[ped], time))
                gr, lr = self.get_reward(ped, idx, time)
                global_reward += gr
                rewards.append(lr)
        for i in range(len(rewards)):
            rewards[i] += global_reward
        return obs, rewards

    def get_observation(self, ped:Person, group:Group, time):
        observation = []
        if ped.is_done:
            # #为了防止模型预测时的loss过大，这里返回完成前的上一步观察状态加将智能体速度与距离出口的位置置为0
            # self.last_observation[ped.id][2:6] = [0.0, 0.0, 0.0, 0.0]
            # 由于采用上述方式无法走出出口，改为全部为0
            # self.last_observation[ped.id][:] = [0.0 for _ in range(16)]
            return [0.0 for _ in range(16)]
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
        for follower in group.followers:
            observation.append(follower.getX)
            observation.append(follower.getY)
        fill_num = 5 - len(group.followers)
        observation.extend([0.0 for _ in range(fill_num * 2)])
        self.last_observation[ped.id] = observation
        return observation

    def set_action(self, ped:Person, action):
        ped.self_driven_force(parse_discrete_action(action) if self.env.discrete else action)
        ped.fij_force(self.env.not_arrived_peds, self.env.group_dic)
        ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)

    def set_follower_action(self, ped:Person, action, group:Group, exit_pos):
        if not group.leader.is_done:
            control_dir = parse_discrete_action(action) if self.env.discrete else action
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

    def get_reward(self, ped:Person, ped_index:int, time):
        gr, lr = 0.0, 0.0
        if ped.is_done and ped.has_removed:
            pass
        else:
            if len(ped.collide_agents) > 0:
                lr += self.r_collision
            if ped.is_done and not ped.has_removed:
                lr += self.r_arrival
            else:
                last_pos = self.env.points_in_last_step[ped_index]
                now_pos = (ped.getX, ped.getY)
                last_dis = self.env.distance_to_exit[ped_index]
                now_dis = self.env.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
                if not (last_pos[0] - 0.001 <= now_pos[0] <= last_pos[0] + 0.001 and last_pos[1] - 0.001 <= now_pos[1] <= last_pos[1] + 0.001) :
                    lr += self.r_move  # 给予-0.1以每步
                    self.env.distance_to_exit[ped_index] = now_dis
                    self.env.points_in_last_step[ped_index] = now_pos
                else:
                    lr += self.r_wait  # 给予停止不动的行人以惩罚
        return gr, lr
