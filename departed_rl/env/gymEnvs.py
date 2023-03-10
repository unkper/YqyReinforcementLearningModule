import gym
import copy


class Pendulum(gym.Env):
    def __init__(self):
        self.wrappedEnv = gym.make("Pendulum-v0")
        print(self.wrappedEnv.observation_space)
        print(self.wrappedEnv.action_space)
        self.observation_space = self.wrappedEnv.observation_space
        self.action_space = self.wrappedEnv.action_space
        self.action_lim = 2.0

    def step(self, action):
        _action = copy.copy(action)
        _action = _action.detach().cpu().numpy() * self.action_lim
        s1, r1, is_done, info = self.wrappedEnv.step(_action)
        return s1, r1, is_done, info

    def reset(self):
        s0 = self.wrappedEnv.reset()
        return s0

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

if __name__ == '__main__':
    env = Pendulum()