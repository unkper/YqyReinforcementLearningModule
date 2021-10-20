from math import inf

import abc

from gym.spaces import Box, Discrete
from ped_env.functions import parse_discrete_action
from ped_env.objects import Person

ACTION_DIM = 9

class PedsHandlerInterface(abc.ABC):
    def __init__(self, env):
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
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # 强化学习MDP定义区域
        # 定义观察空间为[智能体id,8个方向的传感器,智能体当前位置(x,y),智能体当前速度(dx,dy)]一共13个值
        self.observation_space = [Box(-inf, inf, (13,)) for _ in range(self.env.person_num)]
        # 定义动作空间为[不动，向左，向右，向上，向下]施加1N的力
        self.action_space = [Discrete(ACTION_DIM) for _ in range(self.env.person_num)]
        self.agent_count = self.env.person_num

    def get_observation(self, ped:Person):
        observation = []
        if ped.is_done:
            #根据论文中方法，给予一个零向量加智能体的id
            observation.append(ped.id)
            observation.extend([0.0 for _ in range(8)]) #5个方向上此时都不应该有障碍物
            observation.extend([0.0, 0.0]) #将智能体在出口的位置赋予
            observation.extend([0.0, 0.0]) #智能体的速度设置为0
            return observation
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
        return observation

    def set_action(self, ped:Person, action):
        ped.self_driven_force(parse_discrete_action(action))
        ped.fraction_force()
        ped.fij_force(self.env.world)
        ped.fiw_force(self.env.world)

    def get_reward(self, ped:Person, ped_index:int):
        reward = 0.0
        if ped.collide_with_agent:  # 如果智能体其他智能体相撞，奖励减一
            reward += self.env.r_collision
        if ped.is_done and not ped.has_removed:
            reward += self.env.r_arrival  # 智能体到达出口获得10的奖励
        elif ped.is_done:
            pass
        else:
            last_dis = self.env.distance_to_exit[ped_index]
            now_dis = self.env.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
            if last_dis != now_dis:
                reward += self.env.r_approach * (last_dis - now_dis)  # 给予(之前离出口距离-目前离出口距离)的差值
                self.env.distance_to_exit[ped_index] = now_dis
            # else:
            #     reward += self.env.r_collision  # 给予停止不动的行人以碰撞惩罚
        return reward
