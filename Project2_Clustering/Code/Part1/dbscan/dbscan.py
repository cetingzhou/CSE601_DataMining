# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:08:50 2017

@author: xuanhan
"""

import numpy
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from matplotlib import pyplot as plt
import PCA

def DBSCAN(D, eps, MinPts):
    C = 0
    mark = [0]*len(D) # 0 - unvisited; 0.1 - visited; -1 - noise; 1,2,... - cluster
    for P in range(0, len(D)):
        if (mark[P] == 0): # if unvisited
            mark[P] = 0.1 # mark as visited
            neighbors = regionQuery(D, P, eps)
            if len(neighbors) < MinPts:
                mark[P] = -1 # mark as noise
            else:
                C += 1 # new cluster
                expandCluster(D, P, mark, neighbors, C, eps, MinPts)
    return mark
        
def regionQuery(D, P, eps):
    neighbor = []
    for p in range(0, len(D)):
        if numpy.linalg.norm(D[P] - D[p]) <= eps: # if distance of P and p <= eps
            neighbor.append(p) # then p is P's neighbor in radius of eps, add p to P's neighbors
    return neighbor

def expandCluster(D, P, mark, neighbors, C, eps, MinPts):
    mark[P] = C # mark as cluster C
    n = 0
    while n < len(neighbors):
        if (mark[neighbors[n]] == 0): # if unvisited
               mark[neighbors[n]] = 0.1 # mark as visited
               neighbor = regionQuery(D, neighbors[n], eps) # find neighbors of n, n is P's neighbor
               if len(neighbor) >= MinPts:
                   neighbors = neighbors + neighbor # expand P's neighbors
        if (mark[neighbors[n]] < 1): # if holds when mark is 0, 0.1 or -1, not assigned
            mark[neighbors[n]] = C # mark as cluster C, C is expanding
        n += 1

import time
start_time = time.time()
# Reading data
#data = pd.read_csv('cho.txt', header=None, sep='\t')
data = pd.read_csv('iyer.txt', header=None, sep='\t')
#data = pd.read_csv('new_dataset_1.txt', header=None, sep='\t')

# Extracting features and ground truth
data = data.values
data_ground_truth = data[:, 1]
data_features = data[:, 2:]


# Setting the parameters and run function DBSCAN
"""
Before running, determine parameters using 'parameters_dbscan.py';
To run directly, we have recorded parameters for the given datasets:
'cho.txt'           eps: 1.3 MinPts: 16
'iyer.txt'          eps: 1.1 MinPts: 7
'new_dataset_1.txt' eps: 0.5 MinPts: 11
You can manually change eps and MinPts for different datasets.
"""
eps = 1.1
MinPts = 7

data_id = DBSCAN(data_features, eps, MinPts)
print("--- %s seconds ---" % (time.time() - start_time))

# Calculating rand index
ARI = adjusted_rand_score(data_ground_truth, data_id)
print ('The Rand Index is', ARI)


# visualization
unique_label = numpy.unique(data_id)
unique_label_gt = numpy.unique(data_ground_truth)

# using PCA to reduce the dimension of the clustered data from k-means and plot
dim2 = PCA.PCA(data_features, 2)
dim2_dbscan = pd.DataFrame(data = dim2, index = data_id)

# using PCA to reduce the dimension plot the ground truth
dim2_ground_truth = pd.DataFrame(data = dim2, index = data_ground_truth)

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)
a = fig.add_subplot(1, 2, 1)
img_dbscan = PCA.plot_pca_dim2(dim2_dbscan, unique_label)
a.set_title('iyer Clusters from DBSCAN')
a = fig.add_subplot(1, 2, 2)
img_ground = PCA.plot_pca_dim2(dim2_ground_truth, unique_label_gt)
a.set_title('iyer Clusters from Ground Truth')


#iyer data withour outliers
'''
import time
start_time = time.time()
#iyer without outliers
iyer_df = pd.DataFrame(data = data_features, index = data_ground_truth)
iyer_noOutliers = iyer_df[iyer_df.index != -1]
iyer_noOutliers_data = iyer_noOutliers.values

# implementing KMeans
data_id_nooutliers = DBSCAN(iyer_noOutliers_data, eps, MinPts)
print("--- %s seconds ---" % (time.time() - start_time))

iyer_noOutliers_unique_label = numpy.unique(data_id_nooutliers)

# PCA
iyer_noOutliers_dim2 = PCA.PCA(iyer_noOutliers_data, 2)
iyer_noOutliers_dim2_dbscan = \
        pd.DataFrame(data = iyer_noOutliers_dim2, index = data_id_nooutliers)

# using PCA to reduce the dimension plot the ground truth
iyer_noOutliers_dim2_ground_truth = \
        pd.DataFrame(data = iyer_noOutliers_dim2, index = iyer_noOutliers.index)

# implement Rand Index to compare the clustering results and the ground truth
ARI_iyer_noOutliers = \
        adjusted_rand_score(numpy.array(iyer_noOutliers.index), data_id_nooutliers)
print ('The iyer without outliers Rand Index is', ARI_iyer_noOutliers)

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)
a = fig.add_subplot(1, 2, 1)
img_dbscan = PCA.plot_pca_dim2(iyer_noOutliers_dim2_dbscan, iyer_noOutliers_unique_label)
a.set_title('iyer data without outliers: Clusters from DBSCAN')
a = fig.add_subplot(1, 2, 2)
img_ground = PCA.plot_pca_dim2(iyer_noOutliers_dim2_ground_truth, list(range(1,11)))
a.set_title('iyer data without outliers: Clusters from Ground Truth')
'''