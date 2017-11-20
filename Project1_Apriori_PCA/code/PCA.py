#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:59:24 2017

@author: Jeremy
"""

import numpy as np
import pandas as pd
from numpy.linalg import eig
from numpy.linalg import svd
from sklearn.manifold import TSNE
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

def SVD(X, k):
    featureMean = X.mean(axis = 0)
    X_adj = np.subtract(X, featureMean)
    
    # use SVD to do dimensionality reduction
    U, S, V = svd(X_adj, full_matrices = True)
    V = V[:k, :].T
    
    return np.dot(X, V)
    
# @X_pca is the 2-dim pandas.DataFrame (the index should be the label);
# @labels is the unique label array respect to the data
def plot_pca_dim2(X_pca, labels):
    colors = ['red', 'blue', 'green', 'black', 'purple', 'lime', 'cyan', 'orange', 'yellow']
    for i in range(len(labels)):
        px = X_pca.loc[labels[i]][0].values
        py = X_pca.loc[labels[i]][1].values
        plt.scatter(px, py, c = colors[i])
        
    plt.legend(labels)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show


#--------------------------------------data_a-----------------------------------#
# load the data as pandas dataframe
pca_a = pd.read_csv('/Users/Jeremy/Desktop/Pre-CS-Courses/601DataMining/project1/pca_a.txt', header = None, sep = '\t')

# check the information of the data
print (pca_a.info())

# convert dataframe to numpy ndarray
pca_a = pca_a.values

# use PCA function to do dim-reduction for data "pca_a.txt"
X_a = pca_a[:, :(pca_a.shape[1] - 1)].astype(float)

X_a_label = pca_a[:, pca_a.shape[1]-1].reshape(pca_a.shape[0], 1)
label_a = np.unique(X_a_label)
print ("labels:", label_a)   # check the labels of the data

X_a_dim2 = PCA(X_a, 2)
X_a_dim2 = np.concatenate((X_a_dim2, X_a_label), axis=1)
X_a_dim2 = pd.DataFrame(data = X_a_dim2[:, :2], index = X_a_dim2[:, 2])

plot_pca_dim2(X_a_dim2, label_a)

# use SVD to reduce dimension
X_a_dim2_svd = SVD(X_a, 2)
X_a_dim2_svd = np.concatenate((X_a_dim2_svd, X_a_label), axis=1)
X_a_dim2_svd = pd.DataFrame(data = X_a_dim2_svd[:, :2], index = X_a_dim2_svd[:, 2])

plot_pca_dim2(X_a_dim2_svd, label_a)

# use t-SNE to reduce dimension
X_a_dim2_tSNE = TSNE(n_components = 2).fit_transform(X_a)
X_a_dim2_tSNE = np.concatenate((X_a_dim2_tSNE, X_a_label), axis=1)
X_a_dim2_tSNE = pd.DataFrame(data = X_a_dim2_tSNE[:, :2], index = X_a_dim2_tSNE[:, 2])
plot_pca_dim2(X_a_dim2_tSNE, label_a)

#--------------------------------------data_b-----------------------------------#
pca_b = pd.read_csv('/Users/Jeremy/Desktop/Pre-CS-Courses/601DataMining/project1/pca_b.txt', header = None, sep = '\t')
print (pca_b.info())

# convert dataframe to numpy ndarray
pca_b = pca_b.values

# use PCA function to do dim-reduction for data "pca_b.txt"
X_b = pca_b[:, :(pca_b.shape[1] - 1)].astype(float)

X_b_label = pca_b[:, pca_b.shape[1]-1].reshape(pca_b.shape[0], 1)
label_b = np.unique(X_b_label)
print ("labels:", label_b)   # check the labels of the data


X_b_dim2 = PCA(X_b, 2)
X_b_dim2 = np.concatenate((X_b_dim2, X_b_label), axis=1)
X_b_dim2 = pd.DataFrame(data = X_b_dim2[:, :2], index = X_b_dim2[:, 2])

plot_pca_dim2(X_b_dim2, label_b)

# use SVD to reduce dimension
X_b_dim2_svd = SVD(X_b, 2)
X_b_dim2_svd = np.concatenate((X_b_dim2_svd, X_b_label), axis=1)
X_b_dim2_svd = pd.DataFrame(data = X_b_dim2_svd[:, :2], index = X_b_dim2_svd[:, 2])

plot_pca_dim2(X_b_dim2_svd, label_b)

# use t-SNE to reduce dimension
X_b_dim2_tSNE = TSNE(n_components = 2).fit_transform(X_b)
X_b_dim2_tSNE = np.concatenate((X_b_dim2_tSNE, X_b_label), axis=1)
X_b_dim2_tSNE = pd.DataFrame(data = X_b_dim2_tSNE[:, :2], index = X_b_dim2_tSNE[:, 2])
plot_pca_dim2(X_b_dim2_tSNE, label_b)

#--------------------------------------data_c-----------------------------------#
pca_c = pd.read_csv('/Users/Jeremy/Desktop/Pre-CS-Courses/601DataMining/project1/pca_c.txt', header = None, sep = '\t')
print (pca_c.info())

# convert dataframe to numpy ndarray
pca_c = pca_c.values

# use PCA function to do dim-reduction for data "pca_b.txt"
X_c = pca_c[:, :(pca_c.shape[1] - 1)].astype(float)

X_c_label = pca_c[:, pca_c.shape[1]-1].reshape(pca_c.shape[0], 1)
label_c = np.unique(X_c_label)
print ("labels:", label_c)   # check the labels of the data


X_c_dim2 = PCA(X_c, 2)
X_c_dim2 = np.concatenate((X_c_dim2, X_c_label), axis=1)
X_c_dim2 = pd.DataFrame(data = X_c_dim2[:, :2], index = X_c_dim2[:, 2])

plot_pca_dim2(X_c_dim2, label_c)

# use SVD to reduce dimension
X_c_dim2_svd = SVD(X_c, 2)
X_c_dim2_svd = np.concatenate((X_c_dim2_svd, X_c_label), axis=1)
X_c_dim2_svd = pd.DataFrame(data = X_c_dim2_svd[:, :2], index = X_c_dim2_svd[:, 2])

plot_pca_dim2(X_c_dim2_svd, label_c)

# use t-SNE to reduce dimension
X_c_dim2_tSNE = TSNE(n_components = 2).fit_transform(X_c)
X_c_dim2_tSNE = np.concatenate((X_c_dim2_tSNE, X_c_label), axis=1)
X_c_dim2_tSNE = pd.DataFrame(data = X_c_dim2_tSNE[:, :2], index = X_c_dim2_tSNE[:, 2])

plot_pca_dim2(X_c_dim2_tSNE, label_c)