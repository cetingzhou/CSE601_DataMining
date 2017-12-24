#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:27:21 2017

@author: Jeremy
"""
import pandas as pd
import numpy as np
import math
from uuid import uuid4
from sklearn.model_selection import KFold
import performance_metrics as pm
from matplotlib import pyplot as plt
from graphviz import Digraph
from sklearn import preprocessing

class Node:
    """ create Node to store the information of a Node """
    prediction = None
    feature = None
    splitBranch = None
    left = None
    right = None
    entropy = None
    nCount = None
    distribution = None
    
def calcEntropy(labels):
    """ calculate Shannon entropy of the classes 
        @labels: a 1D numpy array whose each element is a class label 
    """
    labelCounts = {}
    for l in labels:
        if l not in labelCounts.keys():
            labelCounts[l] = 0
        labelCounts[l] += 1
        
    entropy = 0
    numEntries = len(labels)
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        entropy -= prob * math.log(prob, 2)
            
    return entropy

def calcInfoGain(y_parent, y_left, y_right):
    """ calculate the information gain after splitting """
    H_mother = calcEntropy(y_parent)
    H_left = calcEntropy(y_left)
    H_right = calcEntropy(y_right)
    H_children = (len(y_left) * H_left + len(y_right) * H_right) / len(y_parent)
    return H_mother - H_children

def isCategorical(X, feature):
    """ 
    check the data type of the feature;
    @X: NxD numpyarray the data whose each row is an example;
    @feature: the column index;
    """
    checkType = X[:, feature].dtype
    return (checkType==str or checkType==object)

def findBranches(X, y, feature):
    """
    find all possible splitable braches of a feature
    @X: NxD numpyarray, the data whose each row is an example;
    @y: 1D numpyarray, labels;
    @feature: int, the column index;
    """
    # if the feature is categorical, return the unique conditions of the feature
    if isCategorical(X, feature):
        return np.unique(X[:, feature])
    
    # if the feature is numeric or continuous, return the thresholds where change
    # of the classes on y occurs
    thresholds = []
    X_feature = X[:, feature]
    df_feature_y = pd.DataFrame(y, index = X_feature)
    df_feature_y.sort_index(inplace = True)
    df_feature_y.reset_index(inplace=True)
    for i in df_feature_y.index:
        try:
            if df_feature_y[0][i] != df_feature_y[0][i+1]:
                thresholds.append((df_feature_y['index'][i] + \
                                   df_feature_y['index'][i+1])/2)
        except: pass
    thresholds = np.unique(thresholds)
    return thresholds

def compare(X, feature, splitBranch):
    """ determine which exmaples fall into the splitted brach w.r.t. the feature """
    if isCategorical(X, feature):
        return X[:, feature] == splitBranch
    # if the feature is numeric
    return X[:, feature] <= splitBranch

def chooseBestSplitBranch(X, y, feature):
    """
    determine the best split brach for a feature based on the information gain
    @X: NxD numpyarray, the data whose each row is an example;
    @y: 1D numpyarray, labels;
    @feature: int, the column index;
    """
    bestInfoGain = 0
    bestSplitBranch = None
    
    # loop for all possible splitable branches of a feature
    thresholds = findBranches(X, y, feature)
    for splitBranch in thresholds:
        y_left = y[compare(X, feature, splitBranch)]
        y_right = y[np.logical_not(compare(X, feature, splitBranch))]
        infoGain = calcInfoGain(y, y_left, y_right)
        # update information gain
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestSplitBranch = splitBranch
            
    # return None if no information gain at all
    return (bestSplitBranch, bestInfoGain)

def chooseBestFeature(X, y, randomFeature=False):
    """
    choose the best feature to split a node
    @X: NxD numpyarray, the data whose each row is an example;
    @y: 1D numpyarray, labels;
    @randomFeature: boolean, reserved for random forest (default=False)
    """
    bestFeature = None
    bestSplitBranch = None
    bestInfoGain = 0
    
    # for random forest, randomly choose sqrt(n) features to test
    if randomFeature:
        n = int(math.sqrt(X.shape[1]))
        featureIndex = np.random.choice(X.shape[1], n, replace = False)
    else:
        # otherwise, test all features
        featureIndex = np.array(range(X.shape[1]))
        
    for feature in featureIndex:
        # for each feature, find its best split brach and information gain
        splitBranch, infoGain = chooseBestSplitBranch(X, y, feature)
        # test if the current information gain is the largest
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = feature
            bestSplitBranch = splitBranch
            
    # return None if no information gain at all      
    return (bestFeature, bestSplitBranch)

def splitDataset(X, y, feature, splitBranch):
    """
    split dataset given a feature and the split branch of this feature
    @X: NxD numpyarray, the data whose each row is an example;
    @y: 1D numpyarray, labels;
    @feature: int, the column index;
    @splitBranch: float/int/str/obj, an element of the thresholds
    """
    X_left = X[compare(X, feature, splitBranch)]
    y_left = y[compare(X, feature, splitBranch)]
    X_right = X[np.logical_not(compare(X, feature, splitBranch))]
    y_right = y[np.logical_not(compare(X, feature, splitBranch))]
    return (X_left, y_left, X_right, y_right)

def majority(y):
    """ This function works to find the most frequent value in a 1D array """
    count = {}
    for i in y:
        if i not in count.keys():
            count[i] = 0
        count[i] += 1
    majorClass = max(count, key = count.get)
    return majorClass

def trainTree(X, y, depth=1, maxDepth=50, minLeafSize=5, randomFeature=False):
    """
    train the tree recursively
    @X: NxD numpyarray, the data whose each row is an example;
    @y: 1D numpyarray, labels;
    @maxDepth: int, specify the depth of the tree to prevent overfitting;
    @minLeafSize: int, specify the minimum leaf size to prevent overfitting;
    @randomFeature: boolean, reserve for random forest
    """
    tree = Node()
    tree.entropy = calcEntropy(y)
    tree.nCount = len(y)
    unique, tree.distribution = np.unique(y, return_counts=True)
    
    # stop splitting when node size is below minLeafSize
    if len(y) <= minLeafSize:
        tree.prediction = majority(y)
        return tree
    
    # stop splitting when the node depth reaches to the maxDepth
    if depth == maxDepth:
        tree.prediction = majority(y)
        return tree
    
    # stop splitting when data have same label
    if len(np.unique(y)) == 1:
        tree.prediction = y[0]
        return tree
    
    # find the best feature and the best split branch of this feature
    feature, splitBranch = chooseBestFeature(X, y, randomFeature)
    
    # stop splitting when no information gain for any splitting
    if feature == None:
        tree.prediction = majority(y)
        return tree
    
    tree.feature = feature
    tree.splitBranch = splitBranch
    
    # split the node through recursion
    X_left, y_left, X_right, y_right = splitDataset(X, y, feature, splitBranch)
    depth += 1
    tree.left = trainTree(X_left, y_left, depth, maxDepth=maxDepth, 
                          minLeafSize=minLeafSize, randomFeature=randomFeature)
    tree.right = trainTree(X_right, y_right, depth, maxDepth=maxDepth, 
                          minLeafSize=minLeafSize, randomFeature=randomFeature)
    
    return tree

def classify(example, inputTree):
    """ 
    predict a single example using the decision tree;
    @example: an 1D numpy array;
    """
    # convert 1D (n,) numpy array to 2D (1, n)
    example = example.reshape(1, -1)
    if inputTree.prediction != None:
        return inputTree.prediction
    if compare(example, inputTree.feature, inputTree.splitBranch):
        return classify(example, inputTree.left)
    return classify(example, inputTree.right)

def predict(X, inputTree):
    """ 
    This function calls classify() to classify multiple examples and ouput
    a list whose each element is a label;
    """
    predLabels = []
    if X.ndim == 1:
        return classify(X, inputTree)
        
    for example in X:
        classLabel = classify(example, inputTree)
        predLabels.append(classLabel)
    
    return predLabels
'''
def drawGraph(graph, tree):
    node_id = uuid4().hex
    if tree.prediction != None:        
        graph.node(node_id, shape="box", 
                 label="%s\nentropy = %.4f\nsampels = %d\ny %s" 
                 % (tree.prediction, tree.entropy, tree.nCount, 
                    tree.distribution))
        return node_id
    
    graph.node(node_id, shape="box", 
             label="%s | %s\nentropy = %.4f\nsamples = %d\ny %s" 
             % (tree.feature, tree.splitBranch, tree.entropy, tree.nCount, 
                tree.distribution))
    left_id = drawGraph(graph, tree.left)
    graph.edge(node_id, left_id, label="left")
    right_id = drawGraph(graph, tree.right)    
    graph.edge(node_id, right_id, label="right")
    return node_id
'''
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

    # cross validation
    kfold = KFold(n_splits = 10)
    
    depth = 10; start = 2;
    accuracy_hist = []
    precision_hist = []
    recall_hist = []
    F_measure_hist = []
    
    # train the decision tree using C4.5
    for k in range(start,depth):
        accuracy = 0
        precision = 0
        recall = 0
        F_measure = 0  
        for train, val in kfold.split(dataset):
            X_train = dataset[train, :-1]
            y_train = dataset[train, -1]
            X_val = dataset[val, :-1]
            y_val = dataset[val, -1]
            myTree = trainTree(X_train, y_train, maxDepth=k, minLeafSize=20)
            y_pred = predict(X_val, myTree)
            accuracy += pm.accuracy(y_val, y_pred)
            precision += pm.precision(y_val, y_pred)
            recall += pm.recall(y_val, y_pred)
            F_measure += pm.F_measure(y_val, y_pred)
            
        accuracy = accuracy/10
        precision = precision/10
        recall = recall/10
        F_measure = F_measure/10    
        
        accuracy_hist.append(accuracy)
        precision_hist.append(precision)
        recall_hist.append(recall)
        F_measure_hist.append(F_measure)
    
    # plot the results from different hyperparamters
    px = list(range(start, depth))
    py1 = accuracy_hist
    py2 = precision_hist
    py3 = recall_hist
    py4 = F_measure_hist
    plt.scatter(px, py1)
    plt.scatter(px, py2)
    plt.scatter(px, py3)
    plt.scatter(px, py4)
    plt.legend(['accuracy', 'precision', 'recall', 'F_measure'])
    plt.title('C4.5 decision tree')
    plt.xlabel('depth')
    plt.ylabel('results')
    
    # plot the decision tree
    #graph = Digraph(comment = "DecisionTree")
    #drawGraph(graph, myTree)
    #graph
