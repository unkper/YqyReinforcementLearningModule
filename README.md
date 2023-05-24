# PedestrainSimlationModule
## 该模块是我开发的行人集结疏散仿真模块，目前期望能够利用强化学习算法结合社会力模型等人群控制方法来实现出行人集结疏散效果。
<div align="center">
<img src="https://github.com/unkper/PedestrainSimulationModule/blob/main/pic/environments.gif" height=300 width=300 alt="疏散场景演示"/>
</div>

代码结构
---
        ped_env: 嵌入了社会力模型(Social Force Model,SFM)的行人粒子仿真系统, 使用 _pybox2d_ 作为物理引擎, 
                 使用 _pyglet_ 做渲染 并实现了 Gym 接口以供强化学习训练.下面是具体的代码结构：
                 --envs:实现了行人仿真环境的主要逻辑，包括了对渲染引擎，物理引擎的更新，接入行人仿真环境需要详细了解这部分内容，可以参考gym接口的用法。
                 --mdp:实现了几种不同的马尔可夫模型，如果想要修改行人环境的状态空间，动作空间，奖励函数等内容，可以重点参考这部分代码。
                 --run:包含了几个测试类，用于使用随机策略或者A*算法来控制行人在空间内移动。
                 --interfaces:包含了几个接口类，支持了行人建模仿真平台对maddpg,maicm等算法的接口转换工作，可以仿照这两个接口来接入新的强化学习算法
        departed_rl: 自行实现的算法，基于pytorch，支持同时收集多个训练环境的数据以供加速训练，支持在训练过程中使用
                Tensorboard查看目前训练效果，目前实现的算法有（注意随着平台接口的升级，有些代码可能已经老旧，需要替换最新的接口！）:
                1.Q-Learning
                2.SARSA,SARSA(\lambda)
                3.DQN,DDQN
                4.DDPG,TD3
                5.MADDPG
                6.MATD3
                7.MAMBPO(MATD3的model-based版本),GD-MAMBPO(加入模仿学习的model-based版本)
    
                如果需要自行实现算法并结合行人仿真系统进行训练，可以重点参考这部分的代码内容。

        rl_platform: 这部分主要将行人建模仿真环境接入了两个强化学习平台（注意随着平台接口的升级，有些代码可能已经老旧，需要替换最新的接口！），分别是d3rl和tianshou平台，其中tianshou平台
                     的案例比较完善，包含了对单智能体下的ppo，sac以及sac+icm的接入工作，可以重点参考这部分内容（在rl_platform/ped_env内）
        
        third_party: 这部分主要接入了github上相关论文的源代码，包括了EC和MAICM的代码，以及maddpg的代码。

EC: [EPISODIC CURIOSITY THROUGH REACHABILITY](https://arxiv.org/abs/1810.02274)

MAICM: [Coordinated Exploration via Intrinsic Rewards for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.12127)

MADDPG: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)

安装
---
        Python 3.6.2, Box2d 2.3.10, gym 0.19.0, kdtree 0.16, matplotlib 3.3.4, numpy 1.19.5, pyglet 1.5.19
        torch 1.9.0+cu102, tqdm 4.62.1, tensorboard 2.7.0, seaborn 0.8.1, scipy 1.5.4, pandas 1.1.5, all 
        packages must be installed to run the project. The version of these packages must be correct.

