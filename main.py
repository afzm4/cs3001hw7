# -*- coding: utf-8 -*-
"""
Andrew Floyd
November 14th, 2018
CS3001: Intro to Data Science
Dr. Fu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nltk.cluster import KMeansClusterer, euclidean_distance
from scipy.spatial.distance import cityblock

#print(football)
football = np.array([[3,5],
                     [3,4],
                     [2,8],
                     [2,3],
                     [6,2],
                     [6,4],
                     [7,3],
                     [7,4],
                     [8,5],
                     [7,6]],np.int64)

init = [[4, 6], [5, 4]]
#kmeans = KMeans(n_clusters=2, init=init, n_init=1).fit(football)

#print(kmeans.labels_)

#plt.scatter(football[:,0],football[:,1], c=kmeans.labels_, cmap='rainbow')

clusterer = KMeansClusterer(2, distance=cityblock, initial_means=init)
clusters = clusterer.cluster(football, True, trace=True)
print('Clustered:', football)
print('As:', clusters)
print('Means:', clusterer.means())
print()
plt.figure(0)
plt.title('Manhattan Distance with Centroids of (4, 6), (5, 4)')
plt.xlabel('# of wins 2016')
plt.ylabel('# of wins 2017')
plt.scatter(football[:,0],football[:,1], c=clusters, cmap='rainbow')

clusterer = KMeansClusterer(2, distance=euclidean_distance, initial_means=init)
clusters = clusterer.cluster(football, True, trace=True)
print('Clustered:', football)
print('As:', clusters)
print('Means:', clusterer.means())
print()
plt.figure(1)
plt.title('Euclidean Distance with Centroids of (4, 6), (5, 4)')
plt.xlabel('# of wins 2016')
plt.ylabel('# of wins 2017')
plt.scatter(football[:,0],football[:,1], c=clusters, cmap='rainbow')

init = [[3, 3], [8, 3]]
clusterer = KMeansClusterer(2, distance=cityblock, initial_means=init)
clusters = clusterer.cluster(football, True, trace=True)
print('Clustered:', football)
print('As:', clusters)
print('Means:', clusterer.means())
print()
plt.figure(2)
plt.title('Manhattan Distance with Centroids of (3, 3), (8, 3)')
plt.xlabel('# of wins 2016')
plt.ylabel('# of wins 2017')
plt.scatter(football[:,0],football[:,1], c=clusters, cmap='rainbow')

init = [[3, 2], [4, 8]]
clusterer = KMeansClusterer(2, distance=cityblock, initial_means=init)
clusters = clusterer.cluster(football, True, trace=True)
print('Clustered:', football)
print('As:', clusters)
print('Means:', clusterer.means())
print()
plt.figure(3)
plt.title('Manhattan Distance with Centroids of (3, 2), (4, 8)')
plt.xlabel('# of wins 2016')
plt.ylabel('# of wins 2017')
plt.scatter(football[:,0],football[:,1], c=clusters, cmap='rainbow')