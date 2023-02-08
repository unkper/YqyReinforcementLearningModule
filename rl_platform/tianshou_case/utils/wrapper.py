import gym


class DisableRewardWrapper(gym.RewardWrapper):
    """
    Overview:
        Disable Env reward to Zero.
    Interface:
        ``__init__``, ``reward``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.

    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)

    def reward(self, reward):
        """
        Overview:
            Disable Env reward to Zero.
        Arguments:
            - reward(:obj:`Float`): Raw Reward
        Returns:
            - reward(:obj:`Float`): Zero Reward
        """
        return 0