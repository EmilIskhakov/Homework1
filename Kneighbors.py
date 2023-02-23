import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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


# Добавление таргетов на основании методла К средних
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
classes=kmeans.labels_

plt.subplot(1, 3, 2)
plt.scatter(X[:,0], X[:,1], c=classes)
plt.title('Diagram')




knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, classes)
knn_classes=knn.predict(X)
print(knn_classes)

confusion_matrix = metrics.confusion_matrix(classes, knn_classes)

Accuracy = metrics.accuracy_score(classes, knn_classes)
print(confusion_matrix)
print(Accuracy)

plt.subplot(1, 3, 3)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()

plt.show()