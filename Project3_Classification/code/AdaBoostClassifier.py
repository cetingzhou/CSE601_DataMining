#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:22:35 2017

@author: Jeremy
"""
import numpy as np
import pandas as pd
import tree_C45
import performance_metrics as pm
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn import preprocessing

class AdaBoost:
    def __init__(self, M):
        """ @M: int, the number of weak classifier """
        self.M = M
    
    def bootstrap(self, W):
        """ input the example weights W, which is the probability of each example
            to be chosen; output indices of the bag examples and out of bag examples;
            input:
                @W: 1D list, sample length as the number of training examples;
            output:
                @bag: 1D list, indices of bag examples;
                @oob: 1D list, indices of out of bag examples;
        """ 
        N = len(W)
        # W contains the probabilities of each example to be picked
        bag = np.random.choice(N, size=N, replace=True, p=W)
        oob = [i for i in range(N) if i not in bag]
        return bag, oob
    
    def trainAdaBoost(self, X, y):
        """
        input:
            @X: NxD numpy array, where each row is an example;
            @y: 1D numpy array, where each element is a label of the example;
        output:
            @models: 1D list, each element is a weak classifier;
            @alphas: 1D list, each element is a weight of the classifier;
        """
        
        self.models = []
        self.alphas = []
        self.oobs = []
        
        # the number of training examples
        N = X.shape[0]
        # initialize the example weights 
        W = np.ones(N) / N
        
        while len(self.models) < self.M:
            bag, oob = self.bootstrap(W)
            X_bag = X[bag]; y_bag = y[bag];
            
            # train the tree on the bagging samples and apply it on the original
            # training set
            tree = tree_C45.trainTree(X_bag, y_bag, maxDepth=2, minLeafSize=1, randomFeature=False)
            pred = tree_C45.predict(X, tree)
            
            err = W.dot(pred != y)
            # only when error < 0.5, the tree is a weak classifier or star over
            if err < 0.5:
                alpha = 0.5 * (np.log(1 - err) - np.log(err))
            
                W = W * np.exp(-alpha * y * pred)
                W = W / sum(W) # normalize W
            
                self.models.append(tree)
                self.alphas.append(alpha)
                self.oobs.append(oob)
    
    def predict(self, X):
        """ input test examples X; output prediction pred """
        N = X.shape[0]
        FX = np.zeros(N)
        for (tree, alpha) in zip(self.models, self.alphas):
            FX = FX + alpha * np.array(tree_C45.predict(X, tree))
        return np.sign(FX), FX
    
    def score(self, X, y):
        """
        check the performance of the model;
        @X: NxD numpyarray, the data whose each row is an example;
        @y: 1D numpyarray, labels;
        """
        y_pred, _  = self.predict(X)
        y_pred[y_pred == -1] = 0
        
        accuracy  = pm.accuracy(y, y_pred)
        precision = pm.precision(y, y_pred)
        recall    = pm.recall(y, y_pred)
        Fmeasure  = pm.F_measure(y, y_pred)

        return (accuracy, precision, recall, Fmeasure)

if __name__ == '__main__':
    # input data
    #data = pd.read_csv('../../project3/project3_dataset1.txt', header = None, sep = '\t')
    data = pd.read_csv('../../project3/project3_dataset2.txt', header = None, sep = '\t')
    dataset = data.values
    
    if dataset.dtype == object:
        # convert nominal attribute to numerical
        le = preprocessing.LabelEncoder()
        le.fit(['Present', 'Absent'])
        dataset[:, 4] = le.transform(dataset[:, 4])
        dataset = dataset.astype(float)
        
    # train the adaboost
    M = list(range(10, 110, 10))
    acc_hist = []; prec_hist = []; recall_hist = []; Fmeas_hist = [];
     # cross validation
    kfold = KFold(n_splits = 10)

    for m in M:
        accuracy = 0 ; precision = 0
        recall = 0   ; F_measure = 0  
        for train, val in kfold.split(dataset):
            # preprocess dataset
            X_train = dataset[train, :-1]
            y_train = dataset[train, -1]
            X_val = dataset[val, :-1]
            y_val = dataset[val, -1]
            
            y_train = y_train.astype(float)
            y_val   = y_val.astype(float)
            
            # adaboost require the labels to be {-1, 1}
            y_train[y_train == 0] = -1
            
            # train the adaboost model
            model = AdaBoost(m)
            model.trainAdaBoost(X_train, y_train)
            
            acc, prec, rec, Fmeas = model.score(X_val, y_val)
            accuracy += acc ; precision += prec
            recall += rec   ; F_measure += Fmeas
            
        accuracy = accuracy/10
        precision = precision/10
        recall = recall/10
        F_measure = F_measure/10    
        
        print ('----------Adaboost with {} Weak Classifiers:----------'.format(m))
        print ('accuracy: {}'.format(accuracy))
        print ('precision: {}'.format(precision))
        print ('recall: {}'.format(recall))
        print ('F-measure: {}'.format(F_measure))
        
        acc_hist.append(accuracy)
        prec_hist.append(precision)
        recall_hist.append(recall)
        Fmeas_hist.append(F_measure)
    
    # plot the results from different hyperparamters
    px = M
    py1 = acc_hist
    py2 = prec_hist
    py3 = recall_hist
    py4 = Fmeas_hist
    plt.scatter(px, py1)
    plt.scatter(px, py2)
    plt.scatter(px, py3)
    plt.scatter(px, py4)
    plt.legend(['accuracy', 'precision', 'recall', 'F_measure'])
    plt.title('AdaBoost')
    plt.xlabel('Number of Weak Classifier')
    plt.ylabel('results')

            