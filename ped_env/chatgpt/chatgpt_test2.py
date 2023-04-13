import pandas as pd
import matplotlib.pyplot as plt

# 创建一个包含3列数据的数据集
data = {'name': ['Tom', 'Jerry', 'Mickey'],
        'age': [25, 30, 35],
        'score': [80.0, 90.5, 75.0]}

# 将数据集转换为DataFrame
df = pd.DataFrame(data)
df.to_csv("./123.csv", index=False)
# 创建一个包含3列数据的数据集
data = {'name': ['Tom', 'Jerry', 'Mickey'],
        'age': [25, 30, 3123],
        'score': [80.0, 90.5, 75.0]}

# 将数据集转换为DataFrame
df = pd.DataFrame(data)
df.to_csv("./123.csv", index=False)
df.to_csv("./123.csv", index=False)

# 创建Figure对象和Axes对象
fig, ax = plt.subplots()

# 绘制折线图
ax.plot(df['name'], df['score'], marker='o')
ax.plot(df['name'], df['age'], marker='x')

# 添加标题和标签
ax.set_title('Score vs. Name')
ax.set_xlabel('Name')
ax.set_ylabel('Score')

# 显示网格线
ax.grid(True)

# 显示图形
plt.show()