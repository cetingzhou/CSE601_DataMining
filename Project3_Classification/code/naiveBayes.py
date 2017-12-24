#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 21:53:28 2017

@author: Jeremy
"""

import numpy as np
import math
import pandas as pd
import performance_metrics as pm
from sklearn import preprocessing
from sklearn.model_selection import KFold

class naiveBayes:
    ''' for binary classification '''
    log_pos_prior = 0
    log_neg_prior = 0
    log_pos_phi = {}
    log_neg_phi = {}
    featType = []
    
    def __init__(self):
        pass
    
    def normpdf(self, x, mean, var):
        pi = 3.1415926
        denom = (2*pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom

    def train(self, X, y):
        ''' assuming the continuous feature belong to gaussian distribution '''
        # get the number example (m) and feature (n)
        m, n = X.shape
        
        # compute priors
        self.log_pos_prior = sum(y) / m  # y = {0, 1}
        self.log_neg_prior = 1 - self.log_pos_prior
        self.log_pos_prior = np.log(self.log_pos_prior)
        self.log_neg_prior = np.log(self.log_neg_prior)
        
        # record if a feature is categorical or numerical
        for i in range(n):
            checkType = X[:, i].dtype
            if (checkType == str or checkType == object):
                self.featType.append('categoric')
            else:
                self.featType.append('numeric')
        
        # split the training data based on the labels
        pos = np.where(y == 1)
        neg = np.where(y == 0)
        X_pos = X[pos, :]#[0, :, :]
        X_neg = X[neg, :]#[0, :, :]
        
        # compute p(x|y)
        for i in range(n):
            if self.featType[i] == 'numeric':
                # for postive examples
                pos_mean_i = np.mean(X_pos[:, i])
                pos_var_i = np.var(X_pos[:, i])
                self.log_pos_phi[i] = lambda x: np.log(self.normpdf(x, pos_mean_i, pos_var_i))
                
                # for negative examples
                neg_mean_i = np.mean(X_neg[:, i])
                neg_var_i = np.var(X_neg[:, i])
                self.log_neg_phi[i] = lambda x: np.log(self.normpdf(x, neg_mean_i, neg_var_i))
            else:
                X_pos = X[pos, :][0, :, :]
                X_neg = X[neg, :][0, :, :]
                pos_countFeat = {}
                for t in X_pos[:, i]:
                    if t not in pos_countFeat:
                        pos_countFeat[t] = 1
                    else:
                        pos_countFeat[t] += 1
                # divide the number of positive example
                for key in pos_countFeat:
                    pos_countFeat[key] = np.log(pos_countFeat[key] / X_pos.shape[0])
                # put the uniform distribution into the phi dictionary
                self.log_pos_phi[i] = pos_countFeat
                
                neg_countFeat = {}
                for t in X_neg[:, i]:
                    if t not in neg_countFeat:
                        neg_countFeat[t] = 1
                    else:
                        neg_countFeat[t] += 1
                # divide the number of negtive example
                for key in neg_countFeat:
                    neg_countFeat[key] = np.log(neg_countFeat[key] / X_neg.shape[0])
                # put the uniform distribution into the phi dictionary
                self.log_neg_phi[i] = neg_countFeat
                
    def classify(self, x):
        ''' classify an example, @x is a 1D array '''
        prob_pos = self.log_pos_prior
        prob_neg = self.log_neg_prior
        for i in range(len(x)):
            if self.featType[i] == 'numeric':
                prob_pos += self.log_pos_phi[i](x[i])
                prob_neg += self.log_neg_phi[i](x[i])
            else:
                prob_pos += self.log_pos_phi[i][x[i]]
                prob_neg += self.log_neg_phi[i][x[i]]
        if prob_pos > prob_neg:
            #print ('The test example should be marked as 1\n')
            #print ('The log posterior p(H_0|x) is {}\n'.format(prob_neg))
            #print ('The log posterior p(H_1|x) is {}\n'.format(prob_pos))
            return 1
        else:
            #print ('The test example should be marked as 0\n')
            #print ('The log posterior p(H_0|x) is {}\n'.format(prob_neg))
            #print ('The log posterior p(H_1|x) is {}\n'.format(prob_pos))
            return 0
        
    def predict(self, X):
        ''' @X is ndarray '''
        predLabels = []
        if X.ndim == 1:
            return self.classify(X)
        else:
            for example in X:
                label = self.classify(example)
                predLabels.append(label)
        return predLabels

if __name__ == '__main__':
    data = pd.read_csv('../../project3/project3_dataset1.txt', header = None, sep = '\t')
    dataset = data.values
    
    # convert nominal attribute to numerical
    if dataset.dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(['Present', 'Absent'])
        dataset[:, 4] = le.transform(dataset[:, 4])
        dataset = dataset.astype(float)
        # normalize the dataset2
        dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    
    # cross validation
    kfold = KFold(n_splits = 10)

    accuracy = 0
    precision = 0
    recall = 0
    F_measure = 0  
    for train, val in kfold.split(dataset):
        X_train = dataset[train, :-1]
        y_train = dataset[train, -1]
        X_val = dataset[val, :-1]
        y_val = dataset[val, -1]
        model = naiveBayes()
        model.train(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy += pm.accuracy(y_val, y_pred)
        precision += pm.precision(y_val, y_pred)
        recall += pm.recall(y_val, y_pred)
        F_measure += pm.F_measure(y_val, y_pred)
                
    accuracy = accuracy/10
    precision = precision/10
    recall = recall/10
    F_measure = F_measure/10    
                    
                    