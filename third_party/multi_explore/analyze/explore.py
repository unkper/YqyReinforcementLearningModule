import matplotlib.pyplot as plt
import numpy as np

phi = 0.7
# 创建数据点
x = np.linspace(1, 10, 100)
y = 1 / np.power(x, phi)

# 绘制图像
plt.plot(x, y)

# 添加标题和轴标签
plt.title('1/N^{}'.format(phi))
plt.xlabel('N')
plt.ylabel('1/N^{}'.format(phi))

# 显示图像
plt.show()