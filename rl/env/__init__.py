import torch

from rl.utils.functions import onehot_from_logits,gumbel_softmax

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as ped_env. This can be used similar to a gym
    environment by calling ped_env.reset() and ped_env.step().
    Use ped_env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful ped_env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

if __name__ == '__main__':
    # ped_env = make_env("simple")
    # s0 = ped_env.reset()
    # for i in range(10000):
    #     a0 = []
    #     for a in ped_env.action_space:
    #         a0.append(onehot_from_logits(a.sample(),a.n))
    #     s1,r1,is_done,info = ped_env.step(a0)
    #     print("s1:{},r1:{},is_done:{}".format(s1,r1,is_done))
    #     ped_env.render()
    data = torch.rand(1,4)
    ret = gumbel_softmax(data,hard=True)
    print(ret)
