import numpy as np

from departed_rl.utils.model.model import EnsembleDynamicsModel

class PredictEnv:
    def __init__(self, model:EnsembleDynamicsModel, env_name, model_type):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type

    def _termination_fn(self, env_name, obs, act, next_obs):
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "PedsMoveEnv":
            state_dim = int(next_obs.shape[1] / self.model.reward_size)
            done = np.ones([next_obs.shape[0], self.model.reward_size], dtype=np.bool)
            for i in range(self.model.reward_size):
                rel = next_obs[:,i:i+state_dim][:, 4:6] #得到智能体i相对出口的位置,速度可能预测不准
                dis = np.sqrt(np.power(rel[:,0],2)+np.power(rel[:,1],2))
                idx = np.where(dis > 1.2) #1.2是指离出口的距离
                done[idx[0], i] = False #一旦出现大于0.1的就将终态置为False
            return done # np.zeros([next_obs.shape[0], self.model.reward_size], dtype=np.bool)
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        else:
            return np.zeros([next_obs.shape[0], self.model.reward_size], dtype=np.bool)

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=True):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:, :, self.model.reward_size:] += obs #因为预测的是delta_state，故而要加上state
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        #model_means = ensemble_model_means[model_idxes, batch_idxes]
        #model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        #log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :self.model.reward_size], samples[:, self.model.reward_size:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        # i = 0
        # for re in rewards[terminals]:
        #     if i > 3:
        #         break
        #     i += 1
        #     print("rewards:{}".format(re))
        # print("#####################################")
        # if self.env_name == "PedsMoveEnv":#判断为结束状态时直接将奖励置为0
        #     rewards[terminals] = 0

        #batch_size = model_means.shape[0]
        #return_means = np.concatenate((model_means[:, :self.model.reward_size], terminals, model_means[:, self.model.reward_size:]), axis=-1)
        #return_stds = np.concatenate((model_stds[:, :self.model.reward_size], np.zeros((batch_size, self.model.reward_size)), model_stds[:, self.model.reward_size:]), axis=-1) #修改为多智能体形式

        if return_single:
            next_obs = next_obs[0]
            #return_means = return_means[0]
            #return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        next_obs = next_obs.reshape([next_obs.shape[0], self.model.reward_size,
                                     int(next_obs.shape[1] / self.model.reward_size)])

        #info = {'mean': return_means, 'std': return_stds}
        info = {}
        return next_obs, rewards, terminals, info

    def load_model(self, dir):
        self.model.load(dir)
