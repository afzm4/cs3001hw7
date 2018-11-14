# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

football = pd.read_csv('football.csv')
football = football.values
football = np.delete(football, [0], axis=1)
#print(football)

kmeans = KMeans(n_clusters=2, random_state=0).fit(football)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(football[:,0],football[:,1], c=kmeans.labels_, cmap='rainbow')