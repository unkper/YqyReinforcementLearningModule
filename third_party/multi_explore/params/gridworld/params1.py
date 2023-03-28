import pprint

from easydict import EasyDict

model_name = "test"
env_type = "gridworld"
map_ind = 1  # Index of map to use (only for gridworld)
num_agents = 2  # for 1-4 gridworld, for 1-2 vizdoom
task_config = 1  # for gridworld task config
frame_skip = 2
intrinsic_reward = 1  # 0 for no intrinsic reward, 1 using visit counts
"""
    Type of exploration, can provide multiple\n" + \
     "0: Independent exploration\n" + \
     "1: Minimum exploration\n" + \
     "2: Covering exploration\n" + \
     "3: Burrowing exploration\n" + \
     "4: Leader-Follower exploration\n" 
"""
explr_types = [0, 1, 2, 3, 4]
uniform_heads = True  # Meta-policy samples all heads uniformly
beta = 0.1  # Weighting for intrinsic reward
decay = 0.7  # Decay rate for state-visit counts in intrinsic reward, f(n) = 1 / N ^ decay
n_rollout_threads = 12  # 启用的总线程数，用于环境经验的收集工作
buffer_length = 1e6  # "Set to 5e5 for ViZDoom (if memory limited)"
train_time = int(1e6)
max_episode_length = 500  # 一集的最大长度
steps_per_update = 100
"""
"Number of episodes to rollout before updating the meta-policy " +
 "(policy selector). Better if a multiple of n_rollout_threads"
"""
metapol_episodes = 12
steps_before_update = 0
num_updates = 50  # Number of SAC updates per cycle
metapol_updates = 100  # Number of updates for meta-policy per turn
batch_size= 1024 # set 128 for vizdoom
save_interval = 20000
pol_hidden_dim = 32
critic_hidden_dim = 128 # set 256 for vizdoom
nonlinearity = "relu" # relu or leaky_relu
pi_lr = 0.001 # 0.0005 for vizdoom
q_lr = 0.001 # 0.0005 for vizdoom
phi_lr = 0.04
adam_eps = 1e-8
q_decay = 1e-3
phi_decay = 1e-3
tau = 0.005
hard_update = None # int , if hard step is not None, use hard update instead of soft update
gamma_e = 0.99
gamma_i = 0.99
reward_scale = 100.
head_reward_scale = 5.
use_gpu = True
"""
 Use GPU for rollouts (more useful for lots of
 parallel envs or image-based observations
"""
gpu_rollout = True


args = EasyDict(globals())

exp_count = 0

def different_explore_type_exp():
    global explr_types, exp_count, train_time, args
    train_time = int(1e6 / 4)
    exp_count_list = [[0], [1], [2], [3], [4], [0, 1, 2, 3, 4]]
    explr_types = exp_count_list[exp_count]
    args = EasyDict(globals())
    exp_count += 1
    return args


def debug_mode():
    global train_time, buffer_length, n_rollout_threads, steps_per_update, steps_before_update, num_updates, save_interval
    train_time = 500
    buffer_length = 100
    n_rollout_threads = 2
    steps_before_update = 0
    steps_per_update = 20
    save_interval = 20
    num_updates = 5
    args = EasyDict(globals())
    return args

if __name__ == '__main__':
    for i in range(6):
        different_explore_type_exp()
        pprint.pprint(args.explr_types)

















