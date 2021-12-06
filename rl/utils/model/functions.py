
def set_rollout_length(epoch_step, rollout_min_length, rollout_max_length, rollout_min_epoch, rollout_max_epoch):
    run_percent = (epoch_step - rollout_min_epoch) / (rollout_max_epoch - rollout_min_epoch)
    rollout_length = (min(max(rollout_min_length + run_percent * (rollout_max_length - rollout_min_length),
                              rollout_min_length), rollout_max_length))
    return int(rollout_length)

def scale_action(env, agent_id, action):
    return (env.action_space[agent_id].high - env.action_space[agent_id].low) * action * 0.5