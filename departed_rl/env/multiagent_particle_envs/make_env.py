"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
import gym

def done_callback(env):
    if env.time > 15:
        env.time = 0
        return True
    return False


def make_env(scenario_name, benchmark=False)->gym.Env:
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    env = None
    from multiagent.environment import MultiAgentEnv
    print("HELLO FROM CUSTOMIZED MULTIAGENT ENV!")
    if scenario_name in ["half_cheetah_multi"]:
        if scenario_name == "half_cheetah_multi":
            from multiagent.envs import MultiAgentHalfCheetah
            env = MultiAgentHalfCheetah()
    else:
        import multiagent.scenarios as scenarios

        # load scenario from script
        scenario = scenarios.load(scenario_name + ".py").Scenario()

        # branch off! if scenario in mujoco then use custom env!

        # create world
        world = scenario.make_world()
        # create multiagent environment
        if benchmark:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=done_callback)
    return env
