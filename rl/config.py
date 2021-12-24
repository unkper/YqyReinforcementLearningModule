from rl.utils.networks.maddpg_network import MLPNetworkActor, MLPNetworkCritic, DoubleQNetworkCritic



class Config:
    def __init__(self):
        self.max_episode = 100
        self.capacity = 1e6
        self.n_rol_threads = 1
        self.batch_size = 1024
        self.learning_rate = 1e-4
        self.update_frequent = 50
        self.debug_log_frequent = 10
        self.gamma = 0.95
        self.tau = 0.01
        self.actor_network = MLPNetworkActor
        self.critic_network = MLPNetworkCritic
        self.actor_hidden_dim = 64
        self.critic_hidden_dim = 64
        self.n_steps_train = 5
        self.env_name = "training_env"
        #Matd3 Parameters
        self.K = 1
        #Masac Parameters

        #Model Parameters
        self.init_train_steps = 0
        self.model_hidden_dim = 200
        self.network_size = 7
        self.elite_size = 5
        self.use_decay = False
        self.model_batch_size = 2048
        self.model_train_freq = 1000
        self.n_steps_model = 50
        self.rollout_length_range = (1, 1)
        self.rollout_epoch_range = (60, 1500)
        self.rollout_batch_size = 256
        self.real_ratio = 0.3
        self.model_retain_epochs = 1

        #init bc learn parameter
        self.use_init_bc = True
        self.init_bc_steps = 0
        self.batch_size_d = 128

    def update_parameter(self, alg_type):
        self.rollout_epoch_range = (int(self.max_episode * 0.1), int(self.max_episode * 0.15))
        if alg_type == "MATD3" or alg_type == "GD_MAMBPO" or alg_type == "MAMBPO":
            self.actor_network = MLPNetworkActor
            self.critic_network = DoubleQNetworkCritic
        else:
            self.actor_network = MLPNetworkActor
            self.critic_network = MLPNetworkCritic

#从0.1-0.8时刻使用rollout
class MPEConfig(Config):
    def __init__(self, n_rol_threads=8, max_episode=100, test=100):
        super().__init__()
        self.max_episode = max_episode
        self.n_rol_threads = n_rol_threads

        self.batch_size = 256
        self.learning_rate = 0.003
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 256
        self.tau = 0.01
        self.gamma = 0.95
        self.update_frequent = 15
        self.debug_log_frequent = 100
        self.n_steps_train = 10

        self.real_ratio = 0.1
        self.rollout_length_range = (1, 10)
        self.rollout_epoch_range = (int(max_episode*0.1), int(max_episode*0.15))
        self.model_batch_size = int(512 / 0.8) #因为有0.2作为验证集
        self.model_train_freq = 250#250
        self.n_steps_model = 500
        self.network_size = 10
        self.elite_size = 10
        self.use_decay = True

        #init bc learn parameter
        self.use_init_bc = True
        self.init_bc_steps = 500
        self.batch_size_d = 128

        self.test = test

    def update_parameter(self, alg_type):
        super(MPEConfig, self).update_parameter(alg_type)
        if alg_type == "g_maddpg" or alg_type == "GD_MAMBPO" or alg_type == "MAMBPO":
            self.init_train_steps = 5 * self.max_episode * 2 * 25 #25指的是1episode=25step
        else:
            self.init_train_steps = 0

class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.max_episode = 20
        self.n_rol_threads = 2

        self.batch_size = 5
        self.learning_rate = 0.003
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 256
        self.tau = 0.01
        self.gamma = 0.95
        self.update_frequent = 50
        self.debug_log_frequent = 1
        self.n_steps_train = 10

        self.real_ratio = 0.1
        self.rollout_length_range = (1, 10) #(1,10)
        self.rollout_epoch_range = (int(20*0.1), int(20*0.15))
        self.model_batch_size = int(10 / 0.8) #因为有0.2作为验证集
        self.model_train_freq = 250 # 25#250
        self.n_steps_model = 1
        self.network_size = 10
        self.elite_size = 10
        self.use_decay = True
        # init bc learn parameter
        self.use_init_bc = False
        self.init_bc_steps = 0
        self.batch_size_d = 6  # demo replay buffer batch_size

    def update_parameter(self, alg_type):
        super(DebugConfig, self).update_parameter(alg_type)
        if alg_type == "g_maddpg" or alg_type == "GD_MAMBPO" or alg_type == "MAMBPO":
            self.init_train_steps = 5 * self.max_episode * 2 * 25 #25指的是1episode=25step
        else:
            self.init_train_steps = 0

class PedsMoveConfig(Config):
    '''
    model的train freq一定要多，不然会出现model bias
    '''
    def __init__(self, n_rol_threads=8, max_episode=100):
        super().__init__()
        self.max_episode = max_episode
        self.n_rol_threads = n_rol_threads

        self.batch_size = 1024
        self.learning_rate = 0.003
        self.update_frequent = 25 #10 25 50
        self.debug_log_frequent = 10
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 256
        self.gamma = 0.99 #0.99
        self.tau = 0.01
        self.n_steps_train = 10
        #model learn parameter
        self.real_ratio = 0.1
        self.rollout_length_range = (1, 1)#(0, 8) (1, 10)
        self.ROLLOUT_EPOCH_RANGE = (int(max_episode * 0.05), int(max_episode * 0.5))
        self.rollout_epoch_range = self.ROLLOUT_EPOCH_RANGE
        self.model_batch_size = int(1024 / 0.8)  # 因为有0.2作为验证集 #512
        self.model_train_freq = 250#500  # 250
        self.n_steps_model = 20
        self.model_hidden_dim = 400
        self.network_size = 10
        self.elite_size = 10
        self.use_decay = True
        #init bc learn parameter
        self.use_init_bc = True
        self.init_bc_steps = 0
        self.batch_size_d = 512 #demo replay buffer batch_size

    def update_parameter(self, alg_type):
        super(PedsMoveConfig, self).update_parameter(alg_type)
        self.rollout_epoch_range = self.ROLLOUT_EPOCH_RANGE
        if alg_type == "g_maddpg" or alg_type == "GD_MAMBPO" or alg_type == "MAMBPO":
            self.init_train_steps = int(0 / self.n_rol_threads)#int(5000 * 5 * 5 / self.n_rol_threads) #25指的是1episode平均500step
        else:
            self.init_train_steps = 0
