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
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from nltk.cluster import KMeansClusterer, euclidean_distance
from scipy.spatial.distance import cityblock, cosine, jaccard
def sse_ed(clusterer, data):
    sse = 0
    for i in range(0, 150):
        #print(clusterer.classify_vectorspace(football[i]))
        dist = euclidean_distance(data[i], clusterer.means()[clusterer.classify_vectorspace(data[i])])
        sse = sse + dist
    return sse

#1
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

kmeans = KMeans(n_clusters=2).fit(football)
print(kmeans.inertia_)
clusterer = KMeansClusterer(2, distance=euclidean_distance)
clusters = clusterer.cluster(football, True, trace=True)
sse = 0
for i in range(0, 10):
    #print(clusterer.classify_vectorspace(football[i]))
    dist = euclidean_distance(football[i], clusterer.means()[clusterer.classify_vectorspace(football[i])])
    sse = sse + dist
print("SSE: ", sse)
print('Clustered:', football)
print('As:', clusters)
print('Means:', clusterer.means())
print()
plt.figure(0)
plt.title('Manhattan Distance with Centroids of (4, 6), (5, 4)')
plt.xlabel('# of wins 2016')
plt.ylabel('# of wins 2017')
plt.scatter(football[:,0],football[:,1], c=clusters, cmap='rainbow')
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

'''sse = 0.0
for c in clusters:
    temp = 0.0
    c_centroid = c.centroid.coords
    for p in c.points:
        temp += euclidean_distance(p, c.centroid)
    sse += temp
print("SSE: ", sse)'''

#2

iris = load_iris()
data = iris.data
target = iris.target

#Q1
clusterer = KMeansClusterer(4, distance=euclidean_distance)
clusters = clusterer.cluster(data, True, trace=True)
#print('Clustered:', data)
print('As:', clusters)
print('Means:', clusterer.means())
#sse = clusterer._sum_distances()
#print("SSE: ", sse)
plt.figure(4)
plt.title('Euclidean Distance')
plt.scatter(data[:,0],data[:,1], c=clusters, cmap='rainbow')
sse = sse_ed(clusterer, data)
print("SSE: ", sse)
print()

clusterer = KMeansClusterer(4, distance=cosine)
clusters = clusterer.cluster(data, True, trace=True)
#print('Clustered:', data)
print('As:', clusters)
print('Means:', clusterer.means())
plt.figure(5)
plt.title('Cosine Distance')
plt.scatter(data[:,0],data[:,1], c=clusters, cmap='rainbow')
sse = sse_ed(clusterer, data)
print("SSE: ", sse)
print()

clusterer = KMeansClusterer(8, distance=jaccard, avoid_empty_clusters=True)
clusters = clusterer.cluster(data, True, trace=True)
#print('Clustered:', data)
print('As:', clusters)
print('Means:', clusterer.means())
plt.figure(6)
plt.title('Jaccard Distance')
plt.scatter(data[:,0],data[:,1], c=clusters, cmap='rainbow')
sse = sse_ed(clusterer, data)
print("SSE: ", sse)
print()
dist = euclidean_distance(data[0], clusterer.means()[clusterer.classify_vectorspace(data[0])])
print(dist)