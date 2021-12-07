import datetime
import os
import time
import gym
import random
import numpy as np

from gym import Env
from random import random,choice
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from rl.utils.classes import Experience,Transition
from rl.utils.classes import make_parallel_env

class Agent():
    '''
    个体基类，没有学习能力
    '''
    def __init__(self,
                 env: Env = None,
                 capacity = 10000,
                 env_name=None,
                 lambda_=0.9,
                 gamma=0.9,
                 alpha=0.5,
                 n_rol_threads = -1,
                 explore_step = 30000,
                 init_train_steps:int = 0,
                 log_dir = "./"
                 ):
        #保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env if n_rol_threads == 1 else make_parallel_env(env, n_rol_threads)
        self.env_name = env_name
        self.name = "Agent"
        assert n_rol_threads > 0 ,"not properly initialized n_rol_threads!"
        self.n_rol_threads = n_rol_threads
        if type(env) is list:
            self.obs_space = env[0].observation_space
            self.action_space = env[0].action_space
        else:
            self.obs_space = env.observation_space if env is not None else None
            self.action_space = env.action_space if env is not None else None
        if type(self.obs_space) is gym.spaces.Discrete:
            self.S = [str(i) for i in range(self.obs_space.n)]
        elif type(self.obs_space) is gym.spaces.Box:
            self.S = self.obs_space
        else:
            self.S = None

        if type(self.action_space) is gym.spaces.Discrete:
            self.A = [str(i) for i in range(self.action_space.n)]
        elif type(self.action_space) is gym.spaces.Box:
            self.A = self.action_space
        else:
            self.A = None

        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.explore_step = explore_step

        self.experience = Experience(capacity=capacity)
        #记录当前agent的状态
        self.state = None

        #为了保存方便而使用
        self.init_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        log_dir = os.path.join(log_dir, self.init_time_str + "_" + self.env_name)
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.loss_callback_ = None
        self.save_callback_ = None
        self.early_stop_callback_ = None

        self.init_train_steps = int(init_train_steps)
        self.total_steps_in_train = 0
        self.total_episodes_in_train = 0


    def policy(self,A ,s = None,Q = None, epsilon = None):
        '''
        智能体遵循的行动策略\n
        :param A: 环境的行动空间\n
        :param s: 当前所处的状态\n
        :param Q: 当前的价值函数\n
        :param epsilon: 实施epsilon-贪心策略需要\n
        :return: 需要执行的动作a0\n
        '''
        return random.sample(self.A,k=1)[0]

    def perform_policy(self, s, Q = None,epsilon = 0.05):
        '''
        执行自定义的policy后返回动作，通常用于下一状态的估值使用\n
        :param s:
        :param Q:
        :param epsilon:
        :return:
        '''
        action = self.policy(self.A,s,Q,epsilon)
        return int(action)

    def act(self,a0)->tuple:
        '''
        给出行动a0，执行相应动作并返回参数
        :param a0:
        :return: s1(下一状态),r1(执行该动作环境给出的奖励),is_done(是否为终止状态),info,total_reward(目前episode的总奖励)
        '''
        s0 = self.state
        s1, r1, is_done, info = self.env.step(a0)

        if self.n_rol_threads == 1:
            trans = Transition(s0, a0, r1, is_done, s1)
            self.experience.push(trans)
        else:
            for i in range(self.n_rol_threads):
                trans = Transition(s0[i], a0[i], r1[i], is_done[i], s1[i])
                self.experience.push(trans)
        self.state = s1
        return s1, r1, is_done, info

    def learning_method(self,
                        epsilon = 0.2,
                        display = False,
                        wait = False,
                        waitSecond:float = 0.01):
        '''
        一个没有学习能力的学习方法，用于后续agent的重载
        用于一个完整的Episode的更新方法\n
        :param wait:是否在每次循环时等待
        :param waitSecond:等待的秒数
        :param lambda_:
        :param gamma:用于td更新时乘以往后看的Q值,即 update = r + gamma*Q'
        :param alpha:用于td更新时乘以累进更新值,即 old_q = old_q + alpha * (update - old_q)
        :param epsilon:用于epsilon-贪心策略使用
        :param display:
        :return:
        '''
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0,epsilon)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            s1, r1, is_done, info, reward = self.act(a0)
            total_reward += np.mean(reward)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1,epsilon)
            s0, a0 = s1, a1
            time_in_episode += 1
            self.total_trans_in_train += 1
            if wait:
                time.sleep(waitSecond)
        print("reward in last episode:{}".format(total_reward))
        return time_in_episode, total_reward, 0.0

    def learning(self,
                 epsilon_high = 1.0,
                 epsilon_low = 0.01,
                 p = 0.6,
                 decaying_epsilon = True,
                 explore_episodes_percent = 1.0,
                 max_episode_num = 800,
                 display = False,
                 display_in_episode = 0,
                 wait = False,
                 waitSecond = 0.01)->tuple:
        '''
        agent的主要循环在该方法内，用于整个学习过程，通过设置相应超参数，其会调用自己的learning_method进行学习
        同时返回（经历的总次数，每个经历的奖励，以及经历的编号）\n
        :param lambda_:
        :param epsilon_high:
        :param epsilon_low:
        :param p:
        :param decaying_epsilon:
        :param gamma:
        :param alpha:
        :param max_episode_num:
        :param display:
        :return:
        '''
        self.max_episode_num = max_episode_num #用于内部的方法使用
        if display_in_episode > 0:
            display = False
            wait = False
        total_time, episode_reward, num_episode = 0,0,0
        total_times,self.episode_rewards,num_episodes = [],[],[]
        max_explore_num = int(max_episode_num * explore_episodes_percent)
        self.init_train()
        for i in tqdm(range(0, max_episode_num, 1)):
            #用于ε-贪心算法中ε随着经历的递增而逐级减少
            if decaying_epsilon:
                #epsilon = epsilon_low + (epsilon_high - epsilon_low) * ((max_explore_num - i) / max_explore_num) if i < max_explore_num else 0
                epsilon = epsilon_low + (epsilon_high - epsilon_low) * np.power(np.e, -4/p*i/max_explore_num) if i < max_explore_num else 0
            else:
                epsilon = epsilon_high
            time_in_episode,episode_reward,loss = self.learning_method(
                epsilon = epsilon,
                display = display,wait = wait,
                waitSecond = waitSecond)
            total_time += time_in_episode
            num_episode += 1
            self.total_episodes_in_train += 1
            if display_in_episode > 0 and num_episode >= display_in_episode:
                display = True
                wait = True
            total_times.append(total_time)
            self.episode_rewards.append(episode_reward)
            num_episodes.append(num_episode)
            if self.loss_callback_ and loss:
                self.loss_callback_(self, loss)
            if self.save_callback_:
                self.save_callback_(self, num_episode)
            if self.early_stop_callback_ != None and self.early_stop_callback_(self, episode_reward, i):
                print("Early stop in {} episode!".format(i))
                break

        #在训练完成后关闭训练环境
        self.env.close()
        self.writer.close()
        return total_times,self.episode_rewards,num_episodes

    def init_random_step(self, state, step): #用于初始化探索更新使用
        pass

    def init_train(self):
        if self.init_train_steps == 0:
            return
        print("Init random train start,total step:{}!".format(self.init_train_steps))
        self.state = self.env.reset()
        is_done = np.array([[False]])
        for i in tqdm(range(0, self.init_train_steps, 1)):
            if is_done.any():
                self.state = self.env.reset()
                is_done = np.array([[False]])
            else:
                s1, r1, is_done, info = self.init_random_step(self.state, i)
                self.state = s1

    def play(self,savePath:str,load_E:int,episode:int=5,display:bool=True,wait:bool=True,waitSecond:float=0.01):
        ep = 0
        while ep < episode:
            self.state = self.env.reset()
            s0 = self.state
            if display:
                self.env.render()
            a0 = self.play_init(savePath, s0)
            time_in_episode = 0

            is_done = [False]
            ep += 1
            while not is_done[0]:
                s1, r1, is_done, info = self.act(a0)
                if type(is_done) is bool:is_done = [is_done]
                if display:
                    self.env.render()
                a1 = self.play_step(savePath,s0)
                s0, a0 = s1, a1
                time_in_episode += 1
                if wait:
                    time.sleep(waitSecond)
        self.env.close()

    def play_init(self, savePath, s0, step):
        '''
        用于play函数的初始化操作，该函数一定要返回一个初始动作a0\n
        :param savePath:
        :param s0:
        :return:
        '''
        return random.sample(self.A, k = 1)[0]

    def play_step(self,savePath,s0):
        '''
        用于play函数的更新操作，该函数一定要返回一个动作a\n
        :param savePath:
        :param s0:
        :return:
        '''
        return random.sample(self.A, k = 1)[0]

    def sample(self,batch_size = 64):
        return self.experience.sample(batch_size)

    def __del__(self):
        self.env.close()

    @property
    def total_trans(self) -> int:
        return self.experience.total_trans