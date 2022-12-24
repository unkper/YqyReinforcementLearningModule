import numpy as np


class RandomPolicy:
    def __init__(self, action_space:np.ndarray):
        self.action_space = action_space
    
    def action(self, obs):
        return np.random.uniform(-1, 1, self.action_space)

