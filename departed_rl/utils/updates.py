

def soft_update(target, source, tau):
    '''
    使用下式将source网络(x)参数软更新至target网络(y)参数：
    y = tau * x + (1 - tau) * y
    :param target: 目标网络
    :param source: 源网络 network
    :param tau: 更新比率
    :return: None
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    '''
    直接将source网络(x)参数更新到target网络(y)参数：
    y = x
    :param target:
    :param source:
    :return:
    '''
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(param.data)