
def set_rollout_length(epoch_step, rollout_min_length, rollout_max_length, rollout_min_epoch, rollout_max_epoch):
    if epoch_step < rollout_min_epoch or epoch_step > rollout_max_epoch:
        return 0
    rollout_length = (min(max(rollout_min_length + (epoch_step - rollout_min_epoch)
                              / (rollout_max_epoch - rollout_min_epoch) * (rollout_max_length - rollout_min_length),
                              rollout_min_length), rollout_max_length))
    return int(rollout_length)