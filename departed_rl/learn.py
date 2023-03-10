from departed_rl.agents.QAgents import DQNAgent
from departed_rl.wrapper_env import SinglePedEnv

if __name__ == '__main__':
    env = SinglePedEnv()
    maxStep = 20000
    agent = DQNAgent(env, ddqn=False)
    agent.learning(explore_episodes_percent=0.85, max_episode_num=maxStep)
    agent.save(agent.log_dir, "Q_Network", agent.behavior_Q, maxStep)
