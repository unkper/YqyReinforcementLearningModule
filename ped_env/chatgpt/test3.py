import numpy as np
import pandas as pd

# 生成10个随机浮点数作为浮动数
fluctuations = np.random.uniform(-1, 1, size=10)

# 创建示例DataFrame
df = pd.DataFrame({'numbers': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]})

# 将浮动数添加到指定列中
df['numbers'] = df['numbers'] + fluctuations

print(df)