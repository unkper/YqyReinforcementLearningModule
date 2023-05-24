## 算法接入

核心思想是利用强化学习算法在每个时间步给每个对应的智能体一个自驱动力方向，从而让算法在学习到有效策略后
能引导智能体合理的进行路径规划，该环境支持gym接口，请在接入前详细了解一下gym接口的含义（[https://gymnasium.farama.org/]()）。

注意最新的接口都统一改成如下的形式：一个字典，键是智能体的ID，值是该智能体的相关内容（包括obs，action，done，truncated等）
输入输出都需采用如下配置。

## 马尔可夫模型修改

环境的马尔可夫模型可以参考mdp.py里面的内容，如果需要替换只需重新创建一个继承自**PedsHandlerInterface**的类，
然后在环境创建时把新的类赋值给环境类。一个具体的例子可以参考mdp.py里的**PedsRLHandlerWithForce**类。

## 社会力参数调节

社会力的相关实现均在object.py的Person类内，相关社会力参数也需在这里调节，注意社会力的添加需要在mdp类的set_action方法内加入，
具体依然可以参考**PedsRLHandlerWithForce**类。

## 地图生成

创建一张地图，需要包含起始点信息（当点生成模式时，行人以此为圆心生成），
出口点信息（出口的中心，用于估算行人距离出口的位置），生成半径，
出口指定（每一个数组代表一种类型的行人可以离开的出口类型），
地图和生成地图。重点参考ped_env/maps.py里面的内容，里面已经包含了几张可以使用的地图。
其中地图含有数种元素，代表的意思是：
### 注意：地图必须是ndarray的str类型！
* '1':方形1m*1m的墙
* '2':外围的墙
* '3'-'9':代表一种类型的智能体与其对应的出口
* 'lw': left wall,代表0.2m宽，占方格左边的围墙
* 'rw': right wall
* 'uw': up wall
* 'dw': down wall
* 'mrw': mid-row wall,代表0.2m宽，占方格中间的横向的墙
* 'mcw': mid-column wall
* 'cluw': corner left up wall,代表0.2m宽，占左边和上边的形成一个拐角的墙
* 'cldw': corner left down wall
* 'cruw': corner right up wall
* 'crdw': corner right down wall

