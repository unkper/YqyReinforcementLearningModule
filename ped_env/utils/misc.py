import enum

class ObjectType(enum.Enum):
    Agent = 1
    Wall = 2
    Obstacle = 3
    Exit = 4
    Sensor = 5

class FixtureInfo():
    def __init__(self, id:int, model:object, type:ObjectType):
        self.id = id
        self.model = model
        self.type = type

    def __str__(self):
        return str(self.type) + str(self.id)

if __name__ == '__main__':
    class obj:
        def __init__(self, v):
            self.val = v

        def __repr__(self):
            return "obj:"+str(self.val)

        def str(self):
            return "obj:" + str(self.val)
    li = []
    for i in range(10):
        li.append(obj(i))
    li_3 = li[2]
    del li[2]
    del li_3
    print(li)
    print(li_3)



# counter = 0
# vecs = []
# for ped in self.peds:
#     if ped.body.linearVelocity.length > 1.6_map11_use:
#         counter += 1
#         vecs.append(ped.body.linearVelocity.length)
# if counter > 0: print("目前有{}智能体超速，其速度为{}!".format(counter, vecs))

# for i,ped in enumerate(self.peds):
#     self.vec[i] += ped.body.linearVelocity.length
# if self.step_in_env % 200 == 0:
#     print("智能体平均速度为{}".format([x/200 for x in self.vec]))
#     self.vec = [0.0 for _ in range(len(self.peds))]



# def fraction_force(self):
#     #给行人施加摩擦力，力的大小为-self.mass * velocity / self.tau
#     vec = self.body.linearVelocity
#     self.total_force += (-self.mass * vec / self.tau)
#
# SLOW_DOWN_DISTANCE = 0.6_map11_use
# def arrive_force(self, target):
#     now_point = np.array([self.getX, self.getY])
#     target_point = np.array(target)
#     now_vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])
#
#     to_target = target_point - now_point
#     distance = np.linalg.norm(to_target)
#     if distance > self.SLOW_DOWN_DISTANCE:
#         vec = normalized(to_target) * self.desired_velocity
#         applied_force = vec - now_vec
#     else:
#         vec = to_target - now_vec
#         applied_force = vec - now_vec
#     applied_force = applied_force * self.desired_velocity * self.mass / self.tau
#     self.total_force += applied_force
#     #self.body.ApplyForceToCenter(applied_force, wake=True)
#
# def seek_force(self, target):
#     now_point = np.array([self.getX, self.getY])
#     target_point = np.array(target)
#     now_vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])
#
#     vec = np.linalg.norm(target_point - now_point)
#     applied_force = vec - now_vec
#     applied_force = applied_force * self.desired_velocity * self.mass / self.tau
#     self.total_force += applied_force
#     #self.body.ApplyForceToCenter(applied_force, wake=True)
#
# LEADER_BEHIND_DIST = 0.25
# def leader_follow_force(self, leader_body:b2Body):
#     #计算目标点，并驱使arrive_force到达该点
#     leader_vec = np.array([leader_body.linearVelocity.x, leader_body.linearVelocity.y])
#     leader_pos = np.array([leader_body.position.x, leader_body.position.y])
#
#     target = leader_pos + self.LEADER_BEHIND_DIST * normalized(-leader_vec)
#     self.arrive_force(target)
#
# def evade_force(self, target_body:b2Body):
#     now_point = np.array([self.getX, self.getY])
#     vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])
#     target_vec = np.array([target_body.linearVelocity.x, target_body.linearVelocity.y])
#     target_point = np.array([target_body.position.x, target_body.position.y])
#     to_target = target_point - now_point
#     #计算向前预测的时间
#     lookahead_time = np.linalg.norm(to_target) / (self.desired_velocity + np.linalg.norm(target_vec))
#     #计算预期速度
#     applied_force = normalized(now_point - (target_point + target_vec * lookahead_time)) - vec
#     applied_force = applied_force * self.desired_velocity * self.mass / self.tau
#     self.total_force += applied_force
#     #self.body.ApplyForceToCenter(applied_force, wake=True)
#
# def evade_controller(self, leader:b2Body, evade_distance=0.5):
#     '''
#     :param leader:
#     :param evade_distance_sqr: 躲避距离的平方值
#     :return:
#     '''
#     #计算领队前方的一个点
#     leader_pos = np.array([leader.position.x, leader.position.y])
#     leader_vec = np.array([leader.linearVelocity.x, leader.linearVelocity.y])
#     pos = np.array([self.getX, self.getY])
#     leader_ahead = leader_pos + normalized(leader_vec) * self.LEADER_BEHIND_DIST
#     #计算角色当前位置与领队前方某点的位置，如果小于某个值，就需要躲避
#     dist = pos - leader_ahead
#     if np.linalg.norm(dist) < evade_distance:
#         self.evade_force(leader)
#
# leader_last_pos = None
# timer = 0
# def exam_leader_moved(self, leader:b2Body):
#     moved = False
#     if self.timer > 0:
#         self.timer -= 1
#         return moved
#     if self.leader_last_pos is None:
#         self.leader_last_pos = np.array([leader.position.x, leader.position.y])
#         self.timer = 2
#     else:
#         now_pos = np.array([leader.position.x, leader.position.y])
#         diff = 0.01
#         if self.leader_last_pos[0] - diff < now_pos[0] < self.leader_last_pos[1] + diff \
#             and self.leader_last_pos[1] - diff < now_pos[1] < self.leader_last_pos[1] + diff:
#             self.timer = 2
#         else:
#             moved = True
#         #if not is_done: self.leader_last_pos = now_pos
#     return moved

# class PedsRLHandler(PedsHandlerInterface):
#     '''
#     非合作的奖励机制
#     '''
#     def __init__(self, env, r_arrival=15, r_approaching=2.5, r_collision=-15, r_cost_time=-0.1):
#         super().__init__(env)
#         self.env = env
#
#         person_num_sum = self.env.person_num
#         reminder = person_num_sum % len(self.env.terrain.start_points)
#         person_num_in_every_spawn = person_num_sum // len(self.env.terrain.start_points) \
#             if person_num_sum >= len(self.env.terrain.start_points) else 1
#         person_num = [person_num_in_every_spawn
#                       for _ in range(len(self.env.terrain.start_points))]
#         person_num[-1] += reminder
#         self.agent_count = sum([int(num / int(sum(self.env.group_size) / 2)) for num in person_num])
#
#         # 强化学习MDP定义区域
#         # 定义观察空间为[id,8个方向的传感器,智能体当前位置(x,y),智能体当前速度(dx,dy)]一共12个值
#         self.observation_space = [Box(-inf, inf, (12,)) for _ in range(self.agent_count)]
#         # 定义动作空间为[不动，向左，向右，向上，向下]施加1N的力
#         self.action_space = [Discrete(ACTION_DIM) for _ in range(self.agent_count)]
#
#         self.r_arrival = r_arrival
#         self.r_approach = r_approaching
#         self.r_collision = r_collision
#         self.r_cost_time = r_cost_time
#
#     def step(self, peds:List[Person]):
#         '''
#         根据当前所有行人的状态，评估得到它们的奖励
#         :param peds:
#         :return: s',r
#         '''
#         obs, rewards = [], []
#         for idx, ped in enumerate(peds):
#             obs.append(self.get_observation(ped))
#             rewards.append(self.get_reward(ped, idx))
#         return obs, rewards
#
#     def get_observation(self, ped:Person):
#         observation = []
#         if ped.is_done:
#             #根据论文中方法，给予一个零向量
#             observation.extend([0.0 for _ in range(8)]) #5个方向上此时都不应该有障碍物
#             observation.extend([0.0, 0.0]) #将智能体在出口的位置赋予
#             observation.extend([0.0, 0.0]) #智能体的速度设置为0
#             return observation
#         #依次得到8个方向上的障碍物,在回调函数中体现，每次调用该函数都会给observation数组中添加值，分别代表该方向上最近的障碍物有多远（5米代表不存在）
#         for i in range(8):
#             temp = ped.raycast(self.env.world, ped.directions[i], ped.view_length)
#             observation.append(temp)
#         #给予智能体当前位置
#         observation.append(ped.getX)
#         observation.append(ped.getY)
#         #给予智能体当前速度
#         vec = ped.body.linearVelocity
#         observation.append(vec.x)
#         observation.append(vec.y)
#         return observation
#
#     def set_action(self, ped:Person, action):
#         ped.self_driven_force(parse_discrete_action(action))
#         ped.fij_force(self.env.not_arrived_peds, self.env.group_dic)
#         ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)
#
#     def get_reward(self, ped:Person, ped_index:int):
#         reward = 0.0
#         if ped.is_done and ped.has_removed:
#             pass
#         else:
#             if ped.collide_with_agent:  # 如果智能体其他智能体相撞，奖励减一
#                 reward += self.r_collision
#             if ped.is_done and not ped.has_removed:
#                 pass
#             else:
#                 last_pos = self.env.points_in_last_step[ped_index]
#                 now_pos = (ped.getX, ped.getY)
#                 last_dis = self.env.distance_to_exit[ped_index]
#                 now_dis = self.env.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
#                 if not (last_pos[0] - 0.001 <= now_pos[0] <= last_pos[0] + 0.001 and last_pos[1] - 0.001 <= now_pos[1] <= last_pos[1] + 0.001) :
#                     reward += self.r_approach * (last_dis - now_dis)  # 给予(之前离出口距离-目前离出口距离)的差值
#                     reward = max(0, reward) #一次尝试，以避免两个智能体对撞的现象发生
#                     self.env.distance_to_exit[ped_index] = now_dis
#                     self.env.points_in_last_step[ped_index] = now_pos
#                 else:
#                     reward += self.r_collision  # 给予停止不动的行人以碰撞惩罚
#         return reward
#
#     # def get_reward(self, ped:Person, ped_index:int):
#     #     reward = 0.0
#     #     if ped.is_done and ped.has_removed:
#     #         pass
#     #     else:
#     #         reward += self.r_cost_time
#     #         if ped.is_done and not ped.has_removed:
#     #             pass
#     #         else:
#     #             last_pos = self.env.points_in_last_step[ped_index]
#     #             now_pos = (ped.getX, ped.getY)
#     #             if not (last_pos[0] - 0.001 <= now_pos[0] <= last_pos[0] + 0.001 and last_pos[1] - 0.001 <= now_pos[1] <= last_pos[1] + 0.001):
#     #                 self.env.points_in_last_step[ped_index] = now_pos
#     #             else:
#     #                 reward += self.r_collision  # 给予停止不动的行人以碰撞惩罚
#     #     return reward

# class PedsRLHandlerWithCooper(PedsHandlerInterface):
#     '''
#     合作的奖励机制
#     '''
#     def __init__(self, env, r_arrival=15, r_approaching=1, r_collision=-1, r_wait = -15, r_cost_time=-0.001):
#         super().__init__(env)
#         self.env = env
#
#         person_num_sum = self.env.person_num
#         reminder = person_num_sum % len(self.env.terrain.start_points)
#         person_num_in_every_spawn = person_num_sum // len(self.env.terrain.start_points) \
#             if person_num_sum >= len(self.env.terrain.start_points) else 1
#         person_num = [person_num_in_every_spawn
#                       for _ in range(len(self.env.terrain.start_points))]
#         person_num[-1] += reminder
#         self.agent_count = sum([int(num / int(sum(self.env.group_size) / 2)) for num in person_num])
#
#         # 强化学习MDP定义区域
#         # 定义观察空间为[智能体当前位置(x,y),智能体当前速度(dx,dy),相对目标的位置(rx,ry),其他跟随者的位置(不存在为(0,0))*4]一共14个值
#         self.observation_space = [Box(-inf, inf, (14,)) for _ in range(self.agent_count)]
#         # 定义动作空间为[不动，向左，向右，向上，向下]施加1N的力
#         self.action_space = [Discrete(ACTION_DIM) for _ in range(self.agent_count)]
#
#         self.r_arrival = r_arrival
#         self.r_approach = r_approaching
#         self.r_collision = r_collision
#         self.r_wait = r_wait
#         self.r_cost_time = r_cost_time
#
#     def step(self, peds:List[Person], group_dic:Dict[Person, Group], time):
#         '''
#         根据当前所有行人的状态，评估得到它们的奖励
#         :param peds:
#         :return: s',r
#         '''
#         obs = []
#         rewards = []
#         global_reward = 0.0
#         for idx, ped in enumerate(peds):
#             if ped.is_leader:
#                 obs.append(self.get_observation(ped, group_dic[ped], time))
#                 gr, lr = self.get_reward(ped, idx)
#                 global_reward += gr
#                 rewards.append(lr)
#         for i in range(len(rewards)):
#             rewards[i] += global_reward
#         return obs, rewards
#
#     def get_observation(self, ped:Person, group:Group, time):
#         observation = []
#         if ped.is_done:
#             #根据论文中方法，给予一个零向量
#             observation.extend([0.0, 0.0]) #将智能体在出口的位置赋予0
#             observation.extend([0.0, 0.0]) #智能体的速度设置为0
#             observation.extend([0.0, 0.0]) #相对距离为0
#             observation.extend([0.0 for _ in range(8)])  # 其他follower为0
#             return observation
#         #给予智能体当前位置
#         observation.append(ped.getX)
#         observation.append(ped.getY)
#         #给予智能体当前速度
#         vec = ped.body.linearVelocity
#         observation.append(vec.x)
#         observation.append(vec.y)
#         #给予智能体相对目标的位置
#         rx, ry = self.env.get_ped_rel_pos_to_exit((ped.getX, ped.getY), ped.exit_type)
#         observation.append(rx)
#         observation.append(ry)
#         for follower in group.followers:
#             observation.append(follower.getX)
#             observation.append(follower.getY)
#         fill_num = 4 - len(group.followers)
#         assert fill_num >= 0
#         observation.extend([0.0 for _ in range(fill_num * 2)])
#         return observation
#
#     def set_action(self, ped:Person, action):
#         ped.self_driven_force(parse_discrete_action(action))
#         # if not ped.exam_leader_moved(ped.body):
#         ped.fij_force(self.env.not_arrived_peds, self.env.group_dic)
#         ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)
#
#     def set_follower_action(self, ped:Person, action, group:Group, exit_pos):
#         if not group.leader.is_done:
#             control_dir = parse_discrete_action(action)
#             leader_dir = calculate_nij(group.leader, ped)
#             mix_dir = ped.alpha * control_dir + (1 - ped.alpha) * leader_dir
#         else:
#             pos_i = exit_pos
#             pos_j = ped.pos
#             mix_dir = normalized(pos_i - pos_j)
#         ped.self_driven_force(mix_dir) #跟随者的方向为alpha*control_dir + (1-alpha)*leader_dir
#         ped.fij_force(self.env.not_arrived_peds, self.env.group_dic)
#         ped.fiw_force(self.env.walls + self.env.obstacles + self.env.exits)
#
#     def get_reward(self, ped:Person, ped_index:int, time):
#         gr, lr = 0.0, 0.0
#         if ped.is_done and ped.has_removed:
#             pass
#         else:
#             if ped.collide_with_agent:
#                 lr += self.r_collision
#             if ped.is_done and not ped.has_removed:
#                 gr += self.r_arrival
#             else:
#                 last_pos = self.env.points_in_last_step[ped_index]
#                 now_pos = (ped.getX, ped.getY)
#                 last_dis = self.env.distance_to_exit[ped_index]
#                 now_dis = self.env.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
#                 if not (last_pos[0] - 0.001 <= now_pos[0] <= last_pos[0] + 0.001 and last_pos[1] - 0.001 <= now_pos[1] <= last_pos[1] + 0.001) :
#                     gr += self.r_approach * (last_dis - now_dis)  # 给予(之前离出口距离-目前离出口距离)的差值
#                     self.env.distance_to_exit[ped_index] = now_dis
#                     self.env.points_in_last_step[ped_index] = now_pos
#                 else:
#                     lr += self.r_wait  # 给予停止不动的行人以惩罚
#         return gr, lr

