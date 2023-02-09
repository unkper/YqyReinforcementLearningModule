from typing import Optional, Tuple

import torch
from tianshou.policy import BasePolicy


def _get_agent(
        agent_learn: Optional[BasePolicy] = None,
        agent_count: int = 1,
        optim: Optional[torch.optim.Optimizer] = None,
        file_path=None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    if agent_learn is None:
        # model
        agent_learn, optim = get_policy(env, optim)
    if file_path is not None:
        state_dict = torch.load(file_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        agent_learn.load_state_dict(state_dict["agent"])

    return agent_learn, optim, None

env_test = False

def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    global env_test
    def wrapped_mario_env():
        wrappers = [
            lambda env: MaxAndSkipWrapper(env, skip=4),
            lambda env: WarpFrameWrapper(env, size=84),
            lambda env: ScaledFloatFrameWrapper(env),
            lambda env: FrameStackWrapper(env, n_frames=4),
            lambda env: EvalEpisodeReturnEnv(env),
        ]
        if not env_test:
            wrappers.append(lambda env: MarioRewardWrapper(env))  # 为了验证ICM机制的有效性而加

        return DingEnvWrapper(
            JoypadSpace(gym_super_mario_bros.make(env_name), action_type),
            cfg={
                'env_wrapper': wrappers
            }
        )
    return wrapped_mario_env()