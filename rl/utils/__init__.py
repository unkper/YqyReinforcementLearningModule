import numpy as np
import matplotlib.pyplot as plt





def func4():
    epsilon_high = 0.9
    epsilon_low = 0.1
    p = 0.5
    max_episode_num = 200
    x = [x for x in range(max_episode_num)]
    y = []
    for i in range(max_episode_num):
        epsilon = epsilon_low + (epsilon_high - epsilon_low) * np.power(np.e, -4 / p * i / max_episode_num)
        y.append(epsilon)
    plt.plot(x,y)
    plt.show()
