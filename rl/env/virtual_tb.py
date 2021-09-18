import gym
import virtualTB












if __name__ == '__main__':
    env = gym.make('VirtualTB-v0')
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.observation_space.high)
    state = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if done: break
    env.render()