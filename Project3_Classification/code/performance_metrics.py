#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:54:03 2017

@author: Jeremy
"""
import numpy as np

'''
precision, recall, F_measure: only for binary classification (0, 1)
@y_gt: the ground truth labels
@y_pred: the predicted labels
'''

def accuracy(y_gt, y_pred):
    return 1.*np.sum(y_gt == y_pred) / len(y_gt)

def precision(y_gt, y_pred):
    ''' TP / (TP + FP) '''
    TP = 0.
    FP = 0.
    num = len(y_gt)
    for i in range(num):
        if y_gt[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_gt[i] == 0 and y_pred[i] == 1:
            FP += 1
    if TP == 0.:
        return 0.
    return TP / (TP + FP)

def recall(y_gt, y_pred):
    ''' TP / (TP + FN) '''
    TP = 0.
    FN = 0.
    num = len(y_gt)
    for i in range(num):
        if y_gt[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_gt[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP / (TP + FN)

def F_measure(y_gt, y_pred):
    ''' 2*TP / (2*TP + FN + FP) '''
    TP = 0.
    FN = 0.
    FP = 0.
    num = len(y_gt)
    for i in range(num):
        if y_gt[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_gt[i] == 1 and y_pred[i] == 0:
            FN += 1
        if y_gt[i] == 0 and y_pred[i] == 1:
            FP += 1
    return 2*TP / (2*TP + FN + FP)

    