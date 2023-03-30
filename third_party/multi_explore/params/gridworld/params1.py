import pprint
import types

from easydict import EasyDict


class Params:
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
    buffer_length = int(1e6)  # "Set to 5e5 for ViZDoom (if memory limited)"
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
    batch_size = 1024  # set 128 for vizdoom
    save_interval = 20000
    pol_hidden_dim = 32
    critic_hidden_dim = 128  # set 256 for vizdoom
    nonlinearity = "relu"  # relu or leaky_relu
    pi_lr = 0.001  # 0.0005 for vizdoom
    q_lr = 0.001  # 0.0005 for vizdoom
    phi_lr = 0.04
    adam_eps = 1e-8
    q_decay = 1e-3
    phi_decay = 1e-3
    tau = 0.005
    hard_update = None  # int , if hard step is not None, use hard update instead of soft update
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

    def __init__(self):
        filtered_dict = {k: v for k, v in vars(Params).items() if not k.startswith("__")}
        filtered_dict = {k: v for k, v in filtered_dict.items() if not isinstance(v, classmethod)}
        self.args = EasyDict(filtered_dict)


def debug_mode(args):
    args.train_time = 500
    args.buffer_length = 100
    args.n_rollout_threads = 2
    args.steps_before_update = 0
    args.steps_per_update = 20
    args.save_interval = 20
    args.num_updates = 5
    args.batch_size = 16
    return args


exp_count = 0


def change_explore_type_exp(args):
    global exp_count
    ways = [[0], [1], [2], [3], [4], [0, 1, 2, 3, 4]]
    args.temp_value = args.train_time
    args.train_time = int(args.temp_value / 3)
    args.explr_types = ways[exp_count]
    exp_count += 1
    return args


if __name__ == '__main__':
    p = Params()
    args = debug_mode(p.args)
    pprint.pprint(args)
    for i in range(6):
        args = change_explore_type_exp(p.args)
        pprint.pprint(args.explr_types)