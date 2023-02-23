import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans

X = np.array([[5, 3],
              [10, 15],
              [15, 12],
              [24, 10],
              [30, 30],
              [85, 70],
              [71, 80],
              [60, 78],
              [70, 55],
              [80, 91], ])
print(X)

# Поcтроение дендограммы
linked = linkage(X, 'single')
labelList = range(1, 11)

plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Dendogram')

# Поиск оптимального количества кластеров на основании метода локтя
data = list(X)
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.subplot(1, 3, 2)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# Отрисовка диаграммы рассеивания с количеством кластеров

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.subplot(1, 3, 3)
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.title('Cluster Diagram')
plt.xlabel('X0')
plt.ylabel('X1')


plt.show()

