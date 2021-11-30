from rl.utils.networks.maddpg_network import MLPNetworkActor, MLPNetworkCritic


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

        #Masac Parameters

        #Model Parameters
        self.model_hidden_dim = 400
        self.network_size = 7
        self.elite_size = 5
        self.use_decay = True
        self.model_batch_size = 2048
        self.model_train_freq = 1000
        self.n_steps_model = 50
        self.rollout_length_range = (1, 1)
        self.rollout_epoch_range = (60, 1500)
        self.rollout_batch_size = 256
        self.real_ratio = 0.3
        self.model_retain_epochs = 300


#从0.1-0.8时刻使用rollout
class MPEConfig(Config):
    def __init__(self, n_rol_threads=8, max_episode=100):
        super().__init__()
        self.max_episode = max_episode
        self.n_rol_threads = n_rol_threads

        self.learning_rate = 0.003
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 256
        self.tau = 0.01
        self.gamma = 0.95

        self.real_ratio = 0.1
        self.rollout_epoch_range = (int(max_episode*0.1), int(max_episode*0.8))
        self.model_batch_size = 512
        self.model_train_freq = 250
        self.n_steps_model = 5
        self.network_size = 10
        self.elite_size = 10

class PedsMoveConfig(Config):
    def __init__(self, n_rol_threads=8, max_episode=100):
        super().__init__()
        self.max_episode = max_episode
        self.n_rol_threads = n_rol_threads

        self.batch_size = 1024
        self.learning_rate = 3e-4
        self.update_frequent = 50
        self.debug_log_frequent = 20
        self.n_rol_threads = 20
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 256

        self.model_train_freq = 400
        self.rollout_epoch_range = (int(max_episode * 0.1), int(max_episode * 0.8))
        self.n_steps_model = 10
        self.network_size = 7
        self.elite_size = 5


# n_rol_threads=n_rol_threads,capacity=1e6,batch_size=1024,learning_rate=3e-4
#                         ,update_frequent=50,debug_log_frequent=20,gamma=0.99,tau=0.01,
#                         env_name=envName,actor_network=actor_network,critic_network=critic_network,actor_hidden_dim=actor_hidden_dim,critic_hidden_dim=critic_hidden_dim,
#                         real_ratio=0.5, num_train_repeat=5, model_train_freq=400, n_steps_model=5, model_batch_size=1024,
#                         model_hidden_dim=500, rollout_epoch_range=(400, 8000)