import os
import sys
curPath = os.path.abspath("../" + os.path.curdir)
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, './rl/env'))
sys.path.append(os.path.join(curPath, './rl/env/multiagent_particle_envs'))

import ped_env.envs as my_env

from ped_env.utils.maps import *
from rl.agents.MAMBPOAgent import MAMBPOAgent
from rl.agents.Matd3Agent import MATD3Agent
from rl.config import PedsMoveConfig, Config, DebugConfig
from rl.env.mpe import SimpleSpread_v3
from rl.run import test1, test2
from rl.utils.model.predict_env import PredictEnv

def eval(useEnv,fileName,load_E,episode=5, AgentType=MATD3Agent,
          config:Config=None):
    env = useEnv
    agent = AgentType(env,actor_network=config.actor_network, critic_network=config.critic_network,
                        actor_hidden_dim=config.actor_hidden_dim, critic_hidden_dim=config.critic_hidden_dim, log_dir=None)
    agent.play(os.path.join("../data/models/",fileName,"model"), load_E, episode=episode, waitSecond=0.05)

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
    env = my_env.PedsMoveEnv(terrain=map_05, person_num=30, group_size=(5, 5), maxStep=3000, train_mode=False)
    test2(env, "PedsMoveEnv", config)

if __name__ == '__main__':
    #debug()
    # eval_model()
    config = PedsMoveConfig(n_rol_threads=1, max_episode=20)
    eval_env = my_env.PedsMoveEnv(terrain=map_05, person_num=30, group_size=(5, 5), maxStep=3000, train_mode=False, random_init_mode=True)
    eval(eval_env, "2021_12_23_14_11_55_PedsMoveEnv", 10, config=config)