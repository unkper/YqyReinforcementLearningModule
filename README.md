# PedestrainSimlationModule
## 该模块是我开发的行人集结疏散仿真模块，目前期望能够利用强化学习算法结合社会力模型等人群控制方法来实现出行人集结疏散效果。
<div align="center">
<img src="https://github.com/unkper/PedestrainSimulationModule/blob/main/pic/environments.gif" height=300 width=300 alt="疏散场景演示"/>
</div>

代码结构

        ped_env: 耦合了社会力模型(Social Force Model,SFM)的行人粒子仿真系统, 使用 _pybox2d_ 作为物理引擎, 
                 使用 _pyglet_ 做渲染 并实现了 Gym 接口以供强化学习训练.
        rl: 目前实现的算法，基于pytorch，支持同时收集多个训练环境的数据以供加速训练，支持在训练过程中使用
            Tensorboard查看目前训练效果，目前实现的算法有:
            1.Q-Learning
            2.SARSA,SARSA(\lambda)
            3.DQN,DDQN
            4.DDPG,TD3
            5.MADDPG
            6.MATD3
            7.MAMBPO(MATD3的model-based版本),GD-MAMBPO(加入模仿学习的model-based版本)
            8.MASAC(尚待完善...)
安装

        Python 3.6.2, Box2d 2.3.10, gym 0.19.0, kdtree 0.16, matplotlib 3.3.4, numpy 1.19.5, pyglet 1.5.19
        torch 1.9.0+cu102, tqdm 4.62.1, tensorboard 2.7.0, seaborn 0.8.1, scipy 1.5.4, pandas 1.1.5, all 
        packages must be installed to run the project. The version of these packages must be correct.
运行

        1.Use do_experiment.sh in rl/utils/ to generate demo and pre_training for model replay buffer.
        2.Use do_experiment.sh and do_experiement2.sh in rl/ to train, generate files A and B.
        3.Move the files A and B above to rl/analyse/data/.
        4.Change path to A and B locations and use draw_plot.py in rl/analyse/ to generate statistical charts.
        5.Put file A and B in rl/, and you can use eval.py in rl/ to watch simuations based on trained policy.
           But the map and person num must be correct.
#结果
<img src="https://github.com/unkper/PedestrainSimulationModule/blob/main/pic/heatmap.jpg" height=300 width=300 alt="探索空间频次热度图"/>

#GD-MAMBPO框架简图
<img src="https://github.com/unkper/PedestrainSimulationModule/blob/main/pic/framework.jpg" height=300 width=300 alt="GD_MAMBPO"/>


