from mpe import SimpleSpread_v2



env = SimpleSpread_v2()
print(env.wrapperEnv.action_space, env.wrapperEnv.observation_space)