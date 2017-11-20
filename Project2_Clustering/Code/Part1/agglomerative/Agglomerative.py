# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:30:20 2017

@author: laiyunwu
"""

import numpy as np
import time
from sklearn.metrics.cluster import adjusted_rand_score
import PCA
import pandas as pd
from matplotlib import pyplot as plt

# reading data from *.txt file. Change the file name for data.
def readdata():
    with  open('iyer.txt', 'r') as f:
        data =f.read().split('\n')
    
    for k in range (0,len(data)):
        data[k] = data[k].split('\t')
    
    data.pop()
    return data

#calculating distance between two records of data
def distanceofcols(col1,col2):
    d=0
    for k in range(2,len(col1)):
        d+=np.power(float(col1[k])-float(col2[k]),2)
    d=np.sqrt(d)
    return d

#to generate the first distance matrix/distance dictionary recording the distance between every two records of data
def distancematrix(data):
    distance=[]
    for i in range(0,len(data)):
        distance.append([])
        for j in range(0,len(data)):
            distance[i].append(distanceofcols(data[i],data[j]))
    return distance

#to generate the single link(min distance) between two clusters
def finddisindisdict(cluster1,cluster2,disdict):
    minimum=float('inf')
    for i in cluster1:
        for j in cluster2:
            if disdict[int(float(i[0]))-1][int(float(j[0]))-1]<minimum:
                minimum=disdict[int(float(i[0]))-1][int(float(j[0]))-1]
    return minimum

#to update distance matrix at each iteration
def updatedismatrix(distance,objecti,objectj,newdata,disdict):
    distance.remove(distance[objectj])
    for k in range(0,len(distance)):
        distance[k].remove(distance[k][objectj])
        
    for h in range(0,len(distance[objecti])):
        distance[objecti][h]= finddisindisdict(newdata[objecti],newdata[h],disdict)
    for m in range(0,len(distance)):
        distance[m][objecti]=finddisindisdict(newdata[objecti],newdata[m],disdict)


#to merge two closest clusters and generate new dendrogram
def generatenewdata(newdata,objecti,objectj):
    for rows in newdata[objectj]:
        newdata[objecti].append(rows)
    newdata.remove(newdata[objectj])

#to find out the two clusters which has the cloest distance
def findmin(distance):
    minimum=float("inf")
    for i in range(0,len(distance)):
        for j in range(i+1,len(distance)):
            if distance[i][j]<minimum :
                minimum=distance[i][j]
                objecti=i
                objectj=j
    return minimum,objecti,objectj


#First of all, it will test how many clusters are there in the original data
#Secondly, this function will generate a array "clustersmin": row number + 1 is the number of clusters this row has, and for each row, the cluster each point belongs has been labeled             
#Finally, the output is a list that indicates the hierarchical clustering result based on the number of clusters in the original data
def hierarchical():
    start_time = time.time()
    data=readdata()
    newdata=[]
    for i in range(0,len(data)):
        newdata.append([])
        newdata[i].append(data[i])
    counter=len(newdata)
    c=len(newdata)
    
    clusters={}
    distance=distancematrix(data)
    disdict=distancematrix(data)
    while len(newdata)>=1:
        cluster=[]
        for cl in newdata:
            elements=[]
            for j in cl:
                elements.append(j[0])
            cluster.append(elements)
        clusters[counter]=cluster
        if len(newdata)<=1: break
        minimum,objecti,objectj=findmin(distance)
        minimum,objecti,objectj
        generatenewdata(newdata,objecti,objectj)
        updatedismatrix(distance,objecti,objectj,newdata,disdict)
        counter-=1
        print('Iteration %d/%d ' %(c-counter,len(data)-1))
    
    #to change the dictionary-like result(with clusters and data point names) into array-like results with all labels
    clustersmin=np.zeros((len(data),len(data)),dtype=np.int)
    for k in clusters:
        for i in range(0,len(clusters[k])):
            for j in clusters[k][i]:
                clustersmin[k-1][int(float(j))-1]=i+1    
    
    values = set(map(lambda x:int(x[1]), data))
    num=max(values)
    
    print("There are %d clusters in the original data" %(num))
    print("---Running Time: %s seconds ---" % (time.time() - start_time))
    return clustersmin[num-1].tolist()




#to generate rand index 
def rand():
    clusters=hierarchical()
    data=readdata()
            
    oldmatrix=np.zeros((len(data),len(data)),)
    for i in range(0,len(data)):
        for j in range(0,len(data)):
            if data[i][1]==data[j][1]:
                oldmatrix[i][j]=1
    
    values = set(map(lambda x:int(x[1]), data))
    num=max(values)
    newmatrix=np.zeros((len(data),len(data)))
    for i in clusters[num]:
        for j in i:
            for k in i:
                newmatrix[int(float(j))-1][int(float(k))-1]=1
    np.logical_xor(oldmatrix,newmatrix)
    count=0.
    for x in np.nditer(np.logical_xor(oldmatrix,newmatrix)): 
        if x ==False:
            count+=1
    rand=count/(len(data)*len(data))
    print ('Rand Index is: ', rand)

def adrand():
    newlist=[]
    for i in data:
        newlist.append(i[1])
    adjusted_rand_score(hierarchical(),newlist)


#data = pd.read_csv('cho.txt', header=None, sep='\t')
data = pd.read_csv('iyer.txt', header=None, sep='\t')

# Extracting features and ground truth
data = data.values
data_ground_truth = data[:, 1]
data_features = data[:, 2:]

data_id = hierarchical()

# Calculating rand index
ARI = adjusted_rand_score(data_ground_truth, data_id)
print ('The Rand Index is', ARI)


# visualization
unique_label = np.unique(data_id)
unique_label_gt = np.unique(data_ground_truth)

# using PCA to reduce the dimension of the clustered data from k-means and plot
dim2 = PCA.PCA(data_features, 2)
dim2_agg = pd.DataFrame(data = dim2, index = data_id)

# using PCA to reduce the dimension plot the ground truth
dim2_ground_truth = pd.DataFrame(data = dim2, index = data_ground_truth)

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)
a = fig.add_subplot(1, 2, 1)
img_agg = PCA.plot_pca_dim2(dim2_agg, unique_label)
a.set_title('iyer Clusters from Agglomerative')
a = fig.add_subplot(1, 2, 2)
img_ground = PCA.plot_pca_dim2(dim2_ground_truth, unique_label_gt)
a.set_title('iyer Clusters from Ground Truth')