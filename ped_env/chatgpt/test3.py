import pprint

import dill



# 反序列化字典
with open(r'D:\projects\python\PedestrainSimulationModule\third_party\multi_explore\models\pedsmove\map_09_6agents_taskleave\test\run31\data\agent_pos.pkl', 'rb') as f:
    d2 = dill.load(f)

# 打印字典
pprint.pprint(d2)