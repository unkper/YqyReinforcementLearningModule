import os
import sys
curPath = os.path.abspath("../" + os.path.curdir)
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, './rl/env'))
sys.path.append(os.path.join(curPath, './rl/env/multiagent_particle_envs'))

import ped_env.envs as my_env

from ped_env.utils.maps import *
from rl.agents.Matd3Agent import MATD3Agent
from rl.config import PedsMoveConfig, Config, DebugConfig
from rl.env.mpe import SimpleSpread_v3
from rl.run import test1, test2

def eval(useEnv,fileName,episode=5, AgentType=MATD3Agent,
          config:Config=None, rep_action_num:int=1):
    env = useEnv
    agent = AgentType(env,actor_network=config.actor_network, critic_network=config.critic_network,
                        actor_hidden_dim=config.actor_hidden_dim, critic_hidden_dim=config.critic_hidden_dim, log_dir=None)
    agent.play(os.path.join("../data/models/",fileName,"model"), episode=episode,
               waitSecond=0.05, rep_action_num=rep_action_num, press=True)

def testEnv():
    env = SimpleSpread_v3()
    state = env.reset()
    is_done = [False]
    for i in range(1000):
        while sum(is_done) == 0:
            obs, reward, is_done, info = env.step(np.random.randn(3,2))
            env.render()
            print(obs)
            print(reward)

def debug():
    config = DebugConfig()
    dir = "./debug"
    os.mkdir(dir)
    config.log_dir = dir
    env = my_env.PedsMoveEnv(terrain=map_05, person_num=32 ,group_size=(4, 4), maxStep=3000, train_mode=False)
    test2(env, "PedsMoveEnv", config)

if __name__ == '__main__':
    # debug()
    # eval_model()
    config = PedsMoveConfig(n_rol_threads=1, max_episode=20)
    eval_env = my_env.PedsMoveEnv(terrain=map_12, person_num=80, group_size=(8, 8), maxStep=250, train_mode=False, random_init_mode=True)
    eval(eval_env, "2022_01_09_14_08_29_PedsMoveEnv", 10, config=config)