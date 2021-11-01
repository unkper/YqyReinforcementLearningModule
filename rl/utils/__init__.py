import math

import numpy as np
import matplotlib.pyplot as plt

def func4():
    epsilon_high = 1.0
    epsilon_low = 0.15
    p = 0.8
    max_episode_num = 100
    x = [x for x in range(max_episode_num)]
    y = []
    for i in range(max_episode_num):
        epsilon = epsilon_low +( epsilon_high - epsilon_low) * np.power(np.e,-4/p*i/max_episode_num)
        y.append(epsilon)
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    func4()