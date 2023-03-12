import pprint
import time

import gym
from vizdoom import gym_wrapper  # noqa

from rl_platform.tianshou_case.vizdoom.vizdoom_env_wrapper import MyVizdoomEnv

if __name__ == "__main__":
    #env = gym.make("VizdoomMyWayHome-v0", render_mode="human")
    env = MyVizdoomEnv("sparse", render_mode="human")

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs = env.reset()

        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            #
            pprint.pprint(rew)
            done = terminated or truncated
            #time.sleep(0.01)
            env.render()