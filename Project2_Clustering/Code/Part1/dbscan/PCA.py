#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:07:27 2017

@author: Jeremy
"""
import numpy as np
from numpy.linalg import eig
from matplotlib import pyplot as plt

# @X is the input data; @k is the target dimension
def PCA(X, k):
    # column index should be the data feature
    featureMean = X.mean(axis = 0)
    
    # adjust the original data by the mean of each feature
    X_adj = np.subtract(X, featureMean)
    
    # compute the covariance matrix
    m = X.shape[0]     # m is the number of examples
    S = (1 / m) * np.dot(X_adj.T, X_adj)
    
    # get the eigenvectors sorted from largest to smallest eigenvalues
    eigenValues, eigenVectors = eig(S)
    idx = eigenValues.argsort()[::-1]
    eigenVectors = eigenVectors[:, idx]
    
    # get the eigenvectors respect to the k largest eigenvalues
    eigenVectors = eigenVectors[:, :k]
    
    # return the data after dim-reduction
    return np.dot(X, eigenVectors)
    
# @X_pca is the 2-dim pandas.DataFrame (the index should be the label);
# @labels is the unique label array respect to the data
def plot_pca_dim2(X_pca, labels):
    colors = ['red', 'blue', 'green', 'black', 'purple', 'lime',\
              'cyan', 'orange', 'yellow', 'brown', 'olive', 'gray', 'pink']
    for i in range(len(labels)):
        #px = X_pca.loc[labels[i]][0].values
        #py = X_pca.loc[labels[i]][1].values
        x = X_pca.loc[labels[i]][0]
        y = X_pca.loc[labels[i]][1]
        if x.size > 1:
            px = X_pca.loc[labels[i]][0].values
        else:
            px = x
        if y.size > 1:
            py = X_pca.loc[labels[i]][1].values
        else:
            py = y
        plt.scatter(px, py, c = colors[i])
        
    plt.legend(labels)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show
