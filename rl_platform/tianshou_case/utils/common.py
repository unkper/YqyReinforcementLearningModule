from typing import Optional, Tuple, Callable

import pettingzoo as pet
import torch
from tianshou.env import PettingZooEnv
from tianshou.policy import BasePolicy, MultiAgentPolicyManager

from ped_env.envs import PedsMoveEnv


def _get_agents(
        env,
        agent_learn: Optional[BasePolicy] = None,
        agent_count: int = 1,
        optim: Optional[torch.optim.Optimizer] = None,
        file_path=None,
        get_policy: Callable = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    if agent_learn is None:
        # model
        agent_learn, optim = get_policy(env, optim)

    agents = [agent_learn for _ in range(agent_count)]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents