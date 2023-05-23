import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 输入数据（一组二维点）
data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# 创建KMeans对象并指定聚类簇的数量
kmeans = KMeans(n_clusters=2)

# 执行聚类
kmeans.fit(data)

# 获取聚类结果（每个样本点所属的簇标签）
labels = kmeans.labels_

# 获取聚类中心点坐标
centroids = kmeans.cluster_centers_

# 绘制原始数据点
plt.scatter(data[:, 0], data[:, 1], c=labels)
# 绘制聚类中心点
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=100)
plt.show()