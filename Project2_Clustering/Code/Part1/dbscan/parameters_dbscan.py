#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:05:57 2017

@author: xuanhan
"""

import numpy
import dbscan
from matplotlib import pyplot as plt
import pandas as pd

def epsOpt(i, D):
    iDistance = [0]*len(D) # build a vector to stores all points' distance to their ith nearest neighbor
    matrix = distanceMatrix(D) # build a distance matrix to store all pair-distances and sort distances for each point 
    for a in range(0, len(D)):
        iDistance[a] = matrix[i, a] # return each point's ith nearest distance by extracting row-i col-point in dist matrix
    iDis = numpy.sort(iDistance) # sort all the ith nearest distances
    # plot sorted distance of every point to its kth nearest neighbor
    px =list(range(len(iDis))) 
    py = iDis
    plt.scatter(px, py)
    plt.xlabel('Points')
    plt.ylabel('ith nearest distance')
    plt.show
    
def distanceMatrix(D): 
    distance = numpy.zeros((len(D), len(D)))
    for p in range(0, len(D)):
        for q in range(0, len(D)):
            distance[p, q] = numpy.linalg.norm(D[p] - D[q])
    return numpy.sort(distance, axis = 0)

# Reading and extracting data
data = pd.read_csv('cho.txt', header=None, sep='\t')
#data = pd.read_csv('iyer.txt', header=None, sep='\t')
#data = pd.read_csv('new_dataset_1.txt', header=None, sep='\t')
data = data.values
data_ground_truth = data[:, 1]
data_features = data[:, 2:]

# Determining eps
for i in range(3, 20, 1): # i - MinPts, we consider an representative range 3 to 20 all the time
    epsOpt(i-1, data_features) # obtain sorted distance plot by running epsOpt
# determing eps by taking the average of best eps for each MinPts from 3 to 20 by the plot,
# the best eps for each MinPts is the gap point

# Determining MinPts by iteration
# after determining eps by first running, subsititute eps in the following DBSCAN function and then run again
for j in range(3, 20, 1): 
        data_id = dbscan.DBSCAN(data_features, 1.3, j)
        ARI = dbscan.adjusted_rand_score(data_ground_truth, data_id)
        print ('The Rand Index of eps {} MinPts {} is {}'.format(1.3, j, ARI)) 
# choose the MinPts-eps pair with the largest rand index
    