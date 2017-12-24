#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:59:00 2017

@author: Jeremy
"""

import numpy as np
import pandas as pd
import tree_C45
import performance_metrics as pm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold

def bagging(n):
    """ 
    select n random sample with replacement of the training set with n examples;
    this function returns the indices of bag and out of bag;
    @n: int, the number of examples in training set
    """
    bag = np.random.choice(n, size=n, replace=True)
    oob = [i for i in range(n) if i not in bag]
    return bag, oob

def trainForest(X, y, forestSize=50, maxDepth=20, minLeafSize=5):
    """
    each tree is trained by a bag of examples, use a list to store the trees and
    use a list to store out of bags;
    @X: NxD numpyarray, the data whose each row is an example;
    @y: 1D numpyarray, labels;
    @forestSize: int, the number of tree you want in the forest;
    @maxDepth: int, the maximum of depth of each tree;
    @minLeafSize: int, the minimum of leaf size of each tree;
    """
    forest = []
    oobs = []
    
    for i in range(forestSize):
        # generate indices of bag and out of bag
        bag, oob = bagging(len(y))
        # take the examples of the bag
        X_i = X[bag]
        y_i = y[bag]
        # use the examples in bag to train a tree
        tree_i = tree_C45.trainTree(X_i, y_i, maxDepth=maxDepth, 
                                    minLeafSize=minLeafSize, randomFeature=True)
        forest.append(tree_i)
        oobs.append(oob)

    return forest, oobs

def classify(example, forest):
    """ 
    predict a single example using the random forest;
    @example: an 1D numpy array;
    @forest: list, store the trees
    """
    # use a list to record each prediction from a tree in the forest
    pred = []
    for i in range(len(forest)):
        tree_i = forest[i]
        pred_i = tree_C45.classify(example, tree_i)
        pred.append(pred_i)        
    return tree_C45.majority(pred)

def predict(X, forest):
    """ 
    This function calls classify() to classify multiple examples and ouput
    a list whose each element is a label;
    """    
    predLabels = []
    if X.ndim == 1:
        return classify(X, forest)
        
    for example in X:
        classLabel = classify(example, forest)
        predLabels.append(classLabel)
    
    return predLabels

def oobTest(X, y, oobs, forest):
    """
    use the out of bags to test the randorm forest classifier;
    @X: NxD numpyarray, the data whose each row is an example;
    @y: 1D numpyarray, labels;
    @oobs: list, a 2D list to store the list of oob;
    @forest: list, store the trees
    """
    
    # for binary classification
    if len(np.unique(y)) == 2:
        accuracy  = 0
        precision = 0
        recall    = 0
        Fmeasure  = 0
    
        for i in range(len(forest)):
            X_i = X[oobs[i]]
            y_i = y[oobs[i]]
            y_pred = predict(X_i, forest)
            accuracy += pm.accuracy(y_i, y_pred)
            recall += pm.recall(y_i, y_pred)
            precision += pm.precision(y_i, y_pred)
            Fmeasure += pm.F_measure(y_i, y_pred)

        return (accuracy/len(forest), precision/len(forest), 
                recall/len(forest), Fmeasure/len(forest))
    
    else:
        accuracy  = 0
        for i in range(len(forest)):
            X_i = X[oobs[i]]
            y_i = y[oobs[i]]
            y_pred = predict(X_i, forest)
            accuracy += pm.accuracy(y_i, y_pred)
            
        return accuracy/len(forest)

if __name__ == '__main__':
    # input data
    data = pd.read_csv('../../project3/project3_dataset1.txt', header = None, sep = '\t')
    #data = pd.read_csv('../../project3/project3_dataset2.txt', header = None, sep = '\t')
    dataset = data.values
    
    # convert nominal attribute to numerical
    if dataset.dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(['Present', 'Absent'])
        dataset[:, 4] = le.transform(dataset[:, 4])
        dataset = dataset.astype(float)
    
    # train the adaboost
    forestSize = list(range(10, 55, 5))
    acc_hist = []; prec_hist = []; recall_hist = []; Fmeas_hist = [];
    # cross validation
    kfold = KFold(n_splits = 10)

    for size in forestSize:
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
            
            # train the random forest
            forest, _ = trainForest(X_train, y_train, forestSize=size, \
                                    maxDepth=8, minLeafSize=10)
            y_pred = predict(X_val, forest)
            acc = pm.accuracy(y_val, y_pred)
            prec = pm.precision(y_val, y_pred)
            rec = pm.recall(y_val, y_pred)
            Fmeas = pm.F_measure(y_val, y_pred)
            
            accuracy += acc ; precision += prec
            recall += rec   ; F_measure += Fmeas
            
        # averaging the results from the cross validation
        accuracy = accuracy/10
        precision = precision/10
        recall = recall/10
        F_measure = F_measure/10    
        
        print ('----------Random Forest with tree size {}:----------'.format(size))
        print ('accuracy: {}'.format(accuracy))
        print ('precision: {}'.format(precision))
        print ('recall: {}'.format(recall))
        print ('F-measure: {}'.format(F_measure))
        
        acc_hist.append(accuracy)
        prec_hist.append(precision)
        recall_hist.append(recall)
        Fmeas_hist.append(F_measure)
    
    # plot the results from different hyperparamters
    px = forestSize
    py1 = acc_hist
    py2 = prec_hist
    py3 = recall_hist
    py4 = Fmeas_hist
    plt.scatter(px, py1)
    plt.scatter(px, py2)
    plt.scatter(px, py3)
    plt.scatter(px, py4)
    plt.legend(['accuracy', 'precision', 'recall', 'F_measure'])
    plt.title('Random Forest')
    plt.xlabel('The Number of Tree')
    plt.ylabel('results')
    
    
    