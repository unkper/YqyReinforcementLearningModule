import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos')

# 设置图例位置，并将它们放在折线图正下方
plt.legend(loc='lower center', ncol=2)

# 显示图形
plt.show()