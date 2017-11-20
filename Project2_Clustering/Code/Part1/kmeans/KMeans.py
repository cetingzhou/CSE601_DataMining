#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:43:17 2017

@author: Jeremy
"""
import numpy as np
import pandas as pd
#from scipy.spatial import distance
import PCA
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
#import matplotlib.image as mpimg

def KMeans(data, k, centroids, n_iter):
    '''
    @data    is the input data (m x n) with the rows are observations and 
             the columns are the attributes.
    @k       is the predefined number of clusters of the data.
    @n_iter  the algorithm will stop until it converges or reach the n_iter.
    '''
    # get the dimension of the data
    m = data.shape[0]
    #n = data.shape[1]
    
    # create a m by 1 array to record the cluster of each observation belong to
    X_clusters = np.empty([m, 1])
    dist_temp = np.empty([1, k])
    '''
    Calculate the Euclidean distance between the observation and each centroid.
    Assigning the observation to the centroid responds to the closest distance.
    '''
    # create an array to record the result of each iteration
    X_clusters_iter = np.empty([m, n_iter])
    iter_count = 0
    for t in range(n_iter):
        for i in range(m):
            for j in range(k):
                dist_temp[0, j] = dist_euclidean(data[i, :], centroids[j, :])
                #dist_temp[0, j] = distance.euclidean(data[i, :], centroids[j, :])
            X_clusters[i] = np.argmin(dist_temp)
        # updating the new centroids respect to the new clusters
        centroids = computeCentroids(data, X_clusters, k)
        # recording the clusters for each iteration
        X_clusters_iter[:, t] = X_clusters[:, 0]
        # if the results do not change anymore, end the iterations
        iter_count += 1
        #print ('Converge yet?', np.all(X_clusters_iter[:, t-1] == X_clusters_iter[:, t]),\
        #       'iteration:', iter_count)
        if t > 0 and np.all(X_clusters_iter[:, t-1] == X_clusters_iter[:, t]):
            break
        if t == n_iter-1:
            print ('WARNING: The K-Means algorithm not converge yet, more iterations needed')
    return X_clusters.flatten(), centroids
        
def Multi_KMeans(data, k, n_iter, n_init):
    '''
    Run the KMeans multiple time with different initial centroids and pick the
    result with smallest SSE.
    @n_init  set the number of time the k-means algorithm will be run with
             different centroid seeds. The final result will be the best
             output of n_init consecutive runs in terms of SSE.
    '''
    m = data.shape[0]
    n = data.shape[1]
    # each column of multi_X_clusters is the cluster labels result from one
    # time of running KMeans
    multi_X_clusters = np.empty([m, n_init])
    # multi_centroids is a 3-D numpy array and each level is the centroid result
    # from one time of running KMeans 
    multi_centroids = np.empty([n_init, k, n])
    # sse is to record the SSE of each run of KMeans
    sse = np.empty([n_init, 1])
    KMeans_count = 0
    for i in range(n_init):
        KMeans_count += 1
        print ('Try K-Means with different initial centroids:', KMeans_count)
        # choose k obeservations(rows) at random from data for initial centroids
        init_centroid = data[np.random.randint(m, size=k), :]
        multi_X_clusters[:, i], multi_centroids[i, :, :] = KMeans(data, k, init_centroid, n_iter)
        sse[i] = SSE(data, multi_centroids[i, :, :], multi_X_clusters[:, i])
    # take the run with smallest SSE
    best_run = np.argmin(sse)
    # get the best result from the multiple run of k-means
    best_X_clusters = multi_X_clusters[:, best_run] + 1
    best_centroids = multi_centroids[best_run, :, :]
    # using a disctionary to record the observations of each cluster
    clusters = {}
    for i in range(k):
        cluster_i = np.where(best_X_clusters == i)
        clusters[i+1] = data[cluster_i, :]
        clusters[i+1] = clusters[i+1][0, :, :]
    #print ('The best result is from the #{} try.'.format(best_run+1))
    return clusters, best_X_clusters, best_centroids

def dist_euclidean(vector1, vector2):
    '''
    @vector1, vector2 are two same size either column or row vectors
                      and they should be two numpy arrays
    '''
    return np.math.sqrt(np.sum(np.power(vector1 - vector2, 2)))

def computeCentroids(data, X_clusters, k):
    '''
    Given the cluster label of each observations of the data, we need to compute
    the new centroid of the cluster.
    @data         is the input data (m x n) with the rows are observations and 
                  the columns are the attributes.
    @X_clusters   is a m by 1 numpy array and the X_clusters[i] represents the
                  the cluster the ith observation belong to.
    @k            is the predefined number of clusters.
    '''
    # centroids records the centroids respect to the X_clusters
    centroids = np.empty([k, data.shape[1]])
    for i in range(k):
        # cluster_i records the row index of data in the ith cluster
        cluster_i, col = np.where(X_clusters == i)
        # compute the new centroids
        centroids[i, :] = np.sum(data[cluster_i, :], axis=0) / len(cluster_i)
    return centroids

def SSE(data, centroids, X_clusters):
    '''
    Sum of Square Error
    @data         is the input data (m x n) with the rows are observations and 
                  the columns are the attributes.
    @centroids    is a numpy array with dimension k by n which records the
                  centroid of each cluster.
    @X_clusters   is a m by 1 numpy array and the X_clusters[i] represents the
                  the cluster the ith observation belong to.
    '''
    k = centroids.shape[0]
    sse = 0
    for i in range(k):
        # cluster_i are the obeservations beling to the ith cluster.
        cluster_i = data[np.where(X_clusters == i), :]
        sse += np.sum(np.power(cluster_i - centroids[i, :], 2))
    return sse

################################## cho ########################################
cho = pd.read_csv('cho.txt', header = None, sep = '\t')
#print (cho.info())
cho = cho.values
cho_ground_truth = cho[:, 1]
cho_data = cho[:, 2:]

# implementing KMeans
cho_clusters, cho_labels, cho_centroids = Multi_KMeans(cho_data, 5, 100, 10)

cho_unique_label = list(cho_clusters.keys())
cho_unique_label_gt = np.unique(cho[:, 1])

# using PCA to reduce the dimension of the clustered data from k-means and plot
cho_dim2 = PCA.PCA(cho_data, 2)
cho_dim2_kmeans = pd.DataFrame(data = cho_dim2, index = cho_labels)

# using PCA to reduce the dimension plot the ground truth
cho_cluster_ground_truth = cho[:, 1]
cho_dim2_ground_truth = pd.DataFrame(data = cho_dim2, index = cho_cluster_ground_truth)

# implement Rand Index to compare the clustering results and the ground truth
ARI_cho = adjusted_rand_score(cho_cluster_ground_truth, cho_labels)
print ('The cho Rand Index is', ARI_cho)

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)
a = fig.add_subplot(1, 2, 1)
img_kmeans = PCA.plot_pca_dim2(cho_dim2_kmeans, cho_unique_label)
a.set_title('cho Clusters from KMeans')
a = fig.add_subplot(1, 2, 2)
img_ground = PCA.plot_pca_dim2(cho_dim2_ground_truth, cho_unique_label_gt)
a.set_title('cho Clusters from Ground Truth')

################################## iyer #######################################
iyer = pd.read_csv('iyer.txt', header = None, sep = '\t')
#print (cho.info())
iyer = iyer.values
iyer_ground_truth = iyer[:, 1]
iyer_data = iyer[:, 2:]

# implementing KMeans
iyer_clusters, iyer_labels, iyer_centroids = Multi_KMeans(iyer_data, 10, 100, 10)

iyer_unique_label = list(iyer_clusters.keys())
iyer_unique_label_gt = np.unique(iyer[:, 1])
# using PCA to reduce the dimension of the clustered data from k-means and plot
iyer_dim2 = PCA.PCA(iyer_data, 2)
iyer_dim2_kmeans = pd.DataFrame(data = iyer_dim2, index = iyer_labels)

# using PCA to reduce the dimension plot the ground truth
iyer_cluster_ground_truth = iyer[:, 1]
iyer_dim2_ground_truth = pd.DataFrame(data = iyer_dim2, index = iyer_cluster_ground_truth)

# implement Rand Index to compare the clustering results and the ground truth
ARI_iyer = adjusted_rand_score(iyer_cluster_ground_truth, iyer_labels)
print ('The iyer Rand Index is', ARI_iyer)

################## do k-means to iyer data without outliers ###################
iyer_df = pd.DataFrame(data = iyer_data, index = iyer_ground_truth)
iyer_noOutliers = iyer_df[iyer_df.index != -1]
iyer_noOutliers_data = iyer_noOutliers.values

# implementing KMeans
iyer_noOutliers_clusters, iyer_noOutliers_labels, iyer_noOutliers_centroids = \
        Multi_KMeans(iyer_noOutliers_data, 10, 100, 10)

iyer_noOutliers_unique_label = list(iyer_noOutliers_clusters.keys())

# PCA
iyer_noOutliers_dim2 = PCA.PCA(iyer_noOutliers_data, 2)
iyer_noOutliers_dim2_kmeans = \
        pd.DataFrame(data = iyer_noOutliers_dim2, index = iyer_noOutliers_labels)

# using PCA to reduce the dimension plot the ground truth
iyer_noOutliers_dim2_ground_truth = \
        pd.DataFrame(data = iyer_noOutliers_dim2, index = iyer_noOutliers.index)

# implement Rand Index to compare the clustering results and the ground truth
ARI_iyer_noOutliers = \
        adjusted_rand_score(np.array(iyer_noOutliers.index), iyer_noOutliers_labels)
print ('The iyer without outliers Rand Index is', ARI_iyer_noOutliers)

############################# plot iyer results ###############################
fig2 = plt.figure()
fig2.set_figheight(10)
fig2.set_figwidth(12)
b = fig2.add_subplot(2, 2, 1)
iyer_img_kmeans = PCA.plot_pca_dim2(iyer_dim2_kmeans, iyer_unique_label)
b.set_title('iyer Clusters from KMeans')
b = fig2.add_subplot(2, 2, 2)
iyer_img_ground = PCA.plot_pca_dim2(iyer_dim2_ground_truth, iyer_unique_label_gt)
b.set_title('iyer Clusters from Ground Truth')
b = fig2.add_subplot(2, 2, 3)
iyer_img_kmeans_noOutliers = PCA.plot_pca_dim2(iyer_noOutliers_dim2_kmeans, iyer_noOutliers_unique_label)
b.set_title('iyer Clusters without Outliers from KMeans')
b = fig2.add_subplot(2, 2, 4)
iyer_img_ground_noOutliers = PCA.plot_pca_dim2(iyer_noOutliers_dim2_ground_truth, iyer_noOutliers_unique_label)
b.set_title('iyer Clusters without Outliers from Ground Truth')
