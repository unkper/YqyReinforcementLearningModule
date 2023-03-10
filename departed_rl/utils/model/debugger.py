import datetime
import os
import pickle
import ped_env.envs as my_env

import torch
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter

from ped_env.utils.maps import map_05, map_02
from departed_rl.utils.functions import process_maddpg_experience_data, back_specified_dimension
from departed_rl.utils.model.model import EnsembleDynamicsModel
from departed_rl.utils.model.predict_env import PredictEnv
from departed_rl.utils.classes import DIRECTIONS

class Debugger:
    def __init__(self, env, exp_file, n_steps_model=1, model_batch_size=1024, model_hidden_dim=200, use_batch_size=True, discrete=True):
        self.env = env
        self.discrete = discrete
        self.state_dims = []
        for obs in env.observation_space:
            self.state_dims.append(back_specified_dimension(obs))
        self.action_dims = []
        for action in env.action_space:
            if self.discrete:
                self.action_dims.append(action.n)
            else:
                self.action_dims.append(back_specified_dimension(action))

        self.n_steps_model = n_steps_model
        self.model = EnsembleDynamicsModel(10, 10, sum(self.state_dims),
                                           sum(self.action_dims) if not self.discrete else self.env.agent_count * 2,
                                           self.env.agent_count, model_hidden_dim, True)
        self.predict_env = PredictEnv(self.model, "PedsMoveEnv", "pytorch")
        file = open(exp_file, "rb")
        self.experience = pickle.load(file)
        self.model_batch_size = model_batch_size if use_batch_size else len(self.experience)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.init_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.summary_dir = os.path.join("data/debug", self.init_time_str)
        self.writer = SummaryWriter(self.summary_dir)
        self.step = 0

    def run(self, count=500):
        for i in tqdm.tqdm(range(count)):
            self._learn_simulate_world()

    def transform_discrete_a(self, a0_idx):
        global DIRECTIONS
        a0 = np.zeros([a0_idx.shape[0], a0_idx.shape[1] * 2])
        for i in range(a0_idx.shape[0]):
            for j in range(a0_idx.shape[1]):
                a0[i, j * 2:(j + 1) * 2] = DIRECTIONS[a0_idx[i, j]]
        return torch.from_numpy(a0).to(self.device)

    def _learn_simulate_world(self):
        mean_losses = [0.0 for _ in range(3)]
        for i in range(self.n_steps_model):
            trans_pieces = self.experience.sample(self.model_batch_size)
            s0, _, r1, is_done, s1, s0_critic_in, s1_critic_in = \
                process_maddpg_experience_data(trans_pieces, self.state_dims, self.env.agent_count, self.device)
            if self.discrete:
                a0_idx = np.argmax(np.array([x.a0 for x in trans_pieces]), axis=2).astype(int)  # 将One-hot形式转换为索引
                a0 = self.transform_discrete_a(a0_idx)
            else:
                a0 = _
            delta_state = s1_critic_in - s0_critic_in
            # world model输入为(s,a),输出为(s',r,is_done)
            inputs = torch.cat([s0_critic_in, a0], dim=-1).detach().cpu().numpy()
            labels = torch.cat([torch.reshape(r1, (r1.shape[0], -1)), delta_state], dim=-1).detach().cpu().numpy()
            # 输入x = (state,action),y = (r,delta_state)
            eval_loss, var_loss, mse_loss = self.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

            mean_losses[0] += eval_loss.mean().item()
            mean_losses[1] += var_loss
            mean_losses[2] += mse_loss.mean().item()
            self.step += 1
        for i in range(mean_losses.__len__()):
            mean_losses[i] /= self.n_steps_model
        print("model learn finished,eval_loss:{},var_loss:{},mse_loss:{}.".format(mean_losses[0], mean_losses[1], mean_losses[2]))
        self.writer.add_scalar("model/eval_loss", mean_losses[0], self.step)
        self.writer.add_scalar("model/var_loss", mean_losses[1], self.step)
        self.writer.add_scalar("model/mse_loss", mean_losses[2], self.step)
        return mean_losses

if __name__ == '__main__':
    exp_file = "../data/exp/map_05_exp_random/experience.pkl"
    env = my_env.PedsMoveEnv(terrain=map_05, person_num=30, group_size=(5, 5), maxStep=1000, train_mode=False, discrete=True)
    debugger = Debugger(env, exp_file, model_batch_size=4096, use_batch_size=True, discrete=True)
    debugger.run(2000)