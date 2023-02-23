import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import AgglomerativeClustering

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
plt.subplot(1, 2, 1)
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Dendogram')


# Добавление таргетов на основании метода Иерархического кластеринга
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
classes = hierarchical_cluster.fit_predict(X)

plt.subplot(1, 2, 2)
plt.scatter(X[:,0], X[:,1], c=classes)
plt.title('Hierarchical')

plt.show()