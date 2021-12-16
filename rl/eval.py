import os
import sys
curPath = os.path.abspath("../" + os.path.curdir)
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, './rl/env'))
sys.path.append(os.path.join(curPath, './rl/env/multiagent_particle_envs'))

import ped_env.envs as my_env

from ped_env.utils.maps import *
from rl.agents.Matd3Agent import MATD3Agent
from rl.config import PedsMoveConfig, Config
from rl.env.mpe import SimpleSpread_v3


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

if __name__ == '__main__':
    config = PedsMoveConfig(n_rol_threads=1, max_episode=20)
    eval_env = my_env.PedsMoveEnv(terrain=map_08, person_num=30, group_size=(5, 5), maxStep=3000, train_mode=False)
    eval(eval_env, "2021_12_10_12_36_PedsMoveEnv", 10, config=config)