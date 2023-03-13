import pprint

import gym

import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from wrapper import WalkerEnvWrapper, create_walker_env
from matplotlib.animation import FuncAnimation

env = create_walker_env()
pprint.pprint(env.observation_space)

# Reset the environment
obs = env.reset()
obs_arr = []

done = False
for i in range(250):
    # Select a random action
    action = env.action_space.sample()

    # Take the selected action and get the new state, reward, and done flag
    obs, reward, done, info = env.step(action)

    # Render the current state of the environment
    obs_arr.append(obs)

fig, ax = plt.subplots()

now_obs_idx = 0

def update(frame):
    global now_obs_idx, obs_arr
    ax.clear()

    ax.imshow(obs_arr[now_obs_idx])
    # ax.imshow(np.transpose(obs_arr[now_obs_idx], (1, 2, 0)))
    now_obs_idx += 1
    now_obs_idx %= len(obs_arr)
    # 隐藏坐标轴
    ax.axis('off')

ani = FuncAnimation(fig, update)
ani.save('./animation.mp4', writer='ffmpeg')

# Close the environment window
env.close()