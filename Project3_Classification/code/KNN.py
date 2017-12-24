#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:13:11 2017

@author: Jeremy
"""
import numpy as np

class KNearestNeighbors:
    def __init__(self):
        pass
    
    def train(self, X, y):
        '''
        @X_train: the training data set whose each row is an example;
                  the dimension is N x D;
        @y_train: the labels of the training dataset;
        '''
        # the training for K-NN is just to remember the training data and labels
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k, dist_metric):
        '''
        @X: the data need to be classified whose each row is an example;
            the dimension is N x D;
        @k: the number of nearest neighbor;
        @dist_metric: 
            the distance metric type:  "l1", "l2"
        '''
        N = X.shape[0];
        # the output prediction should have same length as the number of the 
        # predicted examples and same type with the y_train
        y_pred = np.zeros(N, self.y_train.dtype)
        
        # iterate over all test rows
        for i in range(N):
            # for "l1" distance
            if dist_metric == "l1":
                distances = np.sum(np.abs(self.X_train - X[i, :]), axis = 1)
            # for "l2" distance
            if dist_metric == "l2":
                distances = np.sqrt(np.sum(np.power(self.X_train - X[i, :], 2), axis = 1))
            # pick out the indices of the k-th smallest distance
            min_indices = distances.argsort()[:k]
            y_nearest = self.y_train[min_indices]
            # count number of occurrences of each label of the nearest neighbors
            y_nearest = y_nearest.astype(int)
            count = np.bincount(y_nearest)
            # if the most frequent label is not unique, mark the example as -1(unclassified)
            if (count == np.max(count)).sum() > 1:
                y_pred[i] = -1
            # predict the label as the most frequent label in the nearest neighbors
            y_pred[i] = np.argmax(count)
            
        return y_pred

if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn import preprocessing
    import performance_metrics as pm
    from matplotlib import pyplot as plt
    data = pd.read_csv('../../project3/project3_dataset1.txt', header = None, sep = '\t')
    #data = pd.read_csv('../../project3/project3_dataset2.txt', header = None, sep = '\t')
    data = data.values
    
    # convert nominal attribute to numerical
    if data.dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(['Present', 'Absent'])
        data[:, 4] = le.transform(data[:, 4])
        data = data.astype(float)
    
    # normalize the data
    data[:, :-1] = preprocessing.scale(data[:, :-1])
    
    num_examples = data.shape[0]
    kfold = KFold(n_splits = 10)
    K = 50
    accuracy_hist = np.zeros(K)
    precision_hist = np.zeros(K)
    recall_hist = np.zeros(K)
    F_measure_hist = np.zeros(K)
    
    for k in range(K):
        accuracy = 0
        precision = 0
        recall = 0
        F_measure = 0  
        for train, val in kfold.split(data):
            X_train = data[train, :-1]
            y_train = data[train, -1]
            X_val = data[val, :-1]
            y_val = data[val, -1]
            knn = KNearestNeighbors()
            knn.train(X_train, y_train)
            y_pred = knn.predict(X_val, k+1, "l2")
            accuracy += pm.accuracy(y_val, y_pred)
            precision += pm.precision(y_val, y_pred)
            recall += pm.recall(y_val, y_pred)
            F_measure += pm.F_measure(y_val, y_pred)
            
        accuracy = accuracy/10
        precision = precision/10
        recall = recall/10
        F_measure = F_measure/10    
        
        accuracy_hist[k] = accuracy
        precision_hist[k] = precision
        recall_hist[k] = recall
        F_measure_hist[k] = F_measure
    
    '''
    import PCA
    data_2dim = PCA.PCA(data, 2)
    data_df = pd.DataFrame(data_2dim, index = data[:,-1])
    PCA.plot_pca_dim2(data_df, [0, 1])
    '''
    # plot the results from different hyperparamters
    px = list(range(1, K+1))
    py1 = accuracy_hist
    py2 = precision_hist
    py3 = recall_hist
    py4 = F_measure_hist
    plt.scatter(px, py1)
    plt.scatter(px, py2)
    plt.scatter(px, py3)
    plt.scatter(px, py4)
    plt.legend(['accuracy', 'precision', 'recall', 'F_measure'])
    plt.title('KNN with L2 distance')
    plt.xlabel('k')
    plt.ylabel('accuracy')
            
                