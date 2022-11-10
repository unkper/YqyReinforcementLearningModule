import copy
import os.path
import pickle
import time
from datetime import datetime
from typing import List

import gym
import numpy as np
from d3rlpy.datasets import MDPDataset
from tqdm import tqdm
from multiprocessing import Process, Manager


class DataRecorder:
    def __init__(self, env: gym.Env, policy, agent_count: int):
        self.env = env
        self.policy = policy
        self.dataset = None
        self.agent_count = agent_count

    def collect(self, episodes: int):
        observations = []
        all_actions = []
        next_observations = []
        all_rewards = []
        is_terminal = []
        for episode in tqdm(range(episodes)):
            now_obs = self.env.reset()
            is_done = [False]
            while not all(is_done):
                actions = []
                for obs in now_obs:
                    actions.append(self.policy.action(obs))
                next_obs, rewards, is_done, info = self.env.step(actions)

                for n in range(self.agent_count):
                    observations.append(now_obs[n])
                    all_actions.append(actions[n].tolist())
                    next_observations.append(next_obs[n])
                    all_rewards.append(rewards[n])
                    is_terminal.append(is_done[n])
                now_obs = next_obs
        observations = np.array(observations)
        all_actions = np.array(all_actions)
        next_observations = np.array(next_observations)
        all_rewards = np.array(all_rewards)
        is_terminal = np.array(is_terminal)
        return MDPDataset(observations, all_actions, all_rewards, is_terminal)

    def save(self, path="./"):
        if self.dataset is None:
            raise Exception()
        now = datetime.now()
        if not os.path.exists("./datasets"):
            os.mkdir("./datasets")
        save_path = path + "datasets/dataset_" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".h5"
        self.dataset.dump(save_path)


def wrapper_func(recorder: DataRecorder, episode, manager):
    manager.append(recorder.collect(episode))



class MultiProcessRecorder:
    def __init__(self, data_recorder, num_of_process):
        self.recorders = []
        self.data_recorder = data_recorder
        self.num_of_process = num_of_process

    def collect(self, episode):
        with Manager() as manager:
            datasets = manager.list()
            for i in range(self.num_of_process):
                p = Process(target=wrapper_func, args=(copy.deepcopy(self.data_recorder), episode, datasets))
                p.start()
                self.recorders.append(p)
            for process in self.recorders:
                process.join()
            ans = []
            for ele in datasets:
                ans.append(ele)
        return ans

    def save(self, datasets: List[MDPDataset], path="./"):
        if len(datasets) > 1:
            dataset = datasets[0]
            for i in range(1, len(datasets)):
                dataset.extend(datasets[i])
        else:
            dataset = datasets[0]
        now = datetime.now()
        if not os.path.exists("./datasets"):
            os.mkdir("./datasets")
        with open(path + "datasets/dataset_" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".pkl", "wb") as f:
            pickle.dump(dataset, f)


def load_dataset(path) -> MDPDataset:
    return MDPDataset.load(path)
