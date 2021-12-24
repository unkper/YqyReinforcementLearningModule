
def set_rollout_length(epoch_step, rollout_min_length, rollout_max_length, rollout_min_epoch, rollout_max_epoch):
    if epoch_step < rollout_min_epoch:#如果当前步数小于rollout变化的最小步数，则直接返回0
        return 0
    run_percent = (epoch_step - rollout_min_epoch) / (rollout_max_epoch - rollout_min_epoch)
    rollout_length = (min(max(rollout_min_length + run_percent * (rollout_max_length - rollout_min_length),
                              rollout_min_length), rollout_max_length))
    return int(rollout_length)

def set_model_train_freq(epoch_step, step_min_length, step_max_length, rollout_min_epoch, rollout_max_epoch):
    if epoch_step < rollout_min_epoch:
        return step_max_length
    run_percent = (epoch_step - rollout_min_epoch) / (rollout_max_epoch - rollout_min_epoch)
    step_length = (min(max(step_min_length + run_percent * (step_max_length - step_min_length),
                              step_min_length), step_max_length))
    return step_length


def scale_action(env, agent_id, action):
    return (env.action_space[agent_id].high - env.action_space[agent_id].low) * action * 0.5