import math

from easydict import EasyDict
from dizoo.gfootball.envs.gfootball_env import GfootballEnv

cfg = {'manager': {'episode_num': math.inf, 'max_retry': 1, 'retry_type': 'reset', 'auto_reset': True, 'step_timeout': None, 'reset_timeout': None, 'retry_waiting_time': 0.1, 'cfg_type': 'BaseEnvManagerDict', 'type': 'base', 'shared_memory': False}, 'stop_value': 999, 'type': 'gfootball', 'import_names': ['dizoo.gfootball.envs.gfootball_env'], 'evaluator_env_num': 3, 'n_evaluator_episode': 3, 'env_name': '11_vs_11_easy_stochastic', 'save_replay_gif': False, 'save_replay': False}
easy_dict = EasyDict(cfg)

env = GfootballEnv(easy_dict)
env.reset()

