import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math


df=pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\dataset",header=None,sep="[\t ]+")
df=df.iloc[:,1:]
df=np.array(df)
# print(df)

a=[1,2,3,4,5,6,7,8,9]

def k_means(data,k,max_iterations=100):
    # Step 1: Initialize the centroids

        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        centroids_old=np.zeros_like(centroids)


        for i in range(max_iterations):
        # Step 2: Assign each data point to the nearest centroid
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
        
        # Step 3: Update the centroids
            for j in range(k):
                centroids[j] = np.mean(data[labels == j], axis=0)
        
            if np.allclose(centroids,centroids_old):
                break
            else:
                centroids_old=centroids
    
    # Step 4: Calculate the Silhouette Coefficient
        if k==1:
            silhouette_coefficient=0
        else:
            distances = np.sqrt(((data[:, np.newaxis, :] - data)**2).sum(axis=2))
            s = np.zeros(len(data))
            for i in range(len(data)):
                a_i = np.mean(distances[i, labels == labels[i]])
                b_i = np.min(np.mean(distances[i, labels != labels[i]]))
                s[i] = (b_i - a_i) / max(a_i, b_i)
            silhouette_coefficient = np.mean(s)
    
        return labels, centroids, silhouette_coefficient,k

def k_means_plus(data,k,max_iterations=100):

  # Randomly select k data points as initial cluster centers.
    centroids = [data[np.random.randint(data.shape[0])]]
    centroids_old = np.zeros_like(centroids)
    for i in range(k):
        distances = np.sqrt(np.sum((data[:, np.newaxis, :] - np.array(centroids))**2, axis=2))
        # Find the minimum distance for each point
        min_dist = np.min(distances, axis=1)
        probs = min_dist*2 / np.sum(min_dist*2)
        index = np.random.choice(np.arange(data.shape[0]), p= probs)
        centroids.append(data[index])

  # Assign each data point to the nearest cluster center. 
    for i in range(max_iterations):
        distances = np.sum((data[:, np.newaxis, :] - centroids)**2, axis=2)
        labels = np.argmin(distances, axis=1)
  
  # Recalculate the cluster centers as the mean of all the data points assigned to that cluster.
        for j in range(k):
            centroids[j] = np.mean(data[labels == j], axis=0)

  # Check for convergence.
        if np.allclose(centroids, centroids_old):
            break
        else:
            centroids_old = centroids

    # Step 4: Calculate the Silhouette Coefficient
    if k==1:
        silhouette_coefficient=0
    else:
            distances = np.sqrt(((data[:, np.newaxis, :] - data)**2).sum(axis=2))
            s = np.zeros(len(data))
            for i in range(len(data)):
                a_i = np.mean(distances[i, labels == labels[i]])
                b_i = np.min(np.mean(distances[i, labels != labels[i]]))
                s[i] = (b_i - a_i) / max(a_i, b_i)
            silhouette_coefficient = np.mean(s)
    
    return labels, centroids, silhouette_coefficient,k

def bisecting_kmeans(X, k):
    clusters = [X]
    
    while len(clusters) < k:
        # Select cluster with maximum SSE (sum of squared errors)
        sse_list = [np.sum((cluster - np.mean(cluster, axis=0))**2) for cluster in clusters]
        idx_max_sse = np.argmax(sse_list)
        cluster_to_split = clusters[idx_max_sse]
        
        # Perform K-means clustering with k=2
        centroid1 = cluster_to_split[np.random.randint(cluster_to_split.shape[0])]
        centroid2 = cluster_to_split[np.random.randint(cluster_to_split.shape[0])]
        while np.array_equal(centroid1, centroid2):
            centroid2 = cluster_to_split[np.random.randint(cluster_to_split.shape[0])]
            
        cluster1 = []
        cluster2 = []
        for x in cluster_to_split:
            dist1 = np.linalg.norm(x - centroid1)
            dist2 = np.linalg.norm(x - centroid2)
            if dist1 < dist2:
                cluster1.append(x)
            else:
                cluster2.append(x)
        
        # Remove the original cluster and add the two new clusters
        clusters.pop(idx_max_sse)
        clusters.append(np.array(cluster1))
        clusters.append(np.array(cluster2))
    
    labels = np.concatenate([np.full(len(cluster), i) for i, cluster in enumerate(clusters)])

        # Step 4: Calculate the Silhouette Coefficient
    if k==1:
        silhouette_coefficient=0
    else:
            distances = np.sqrt(((X[:, np.newaxis, :] - X)**2).sum(axis=2))
            s = np.zeros(len(X))
            for i in range(len(X)):
                a_i = np.mean(distances[i, labels == labels[i]])
                b_i = np.min(np.mean(distances[i, labels != labels[i]]))
                s[i] = (b_i - a_i) / max(a_i, b_i)
            silhouette_coefficient = np.mean(s)
    
    
    return clusters,k,silhouette_coefficient,labels


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 2))

silhouette_coefficient_means=[]
kmeans=[]
for k in a:
    labels, centroids, silhouette_coefficient,k = k_means(df,k)
    silhouette_coefficient_means.append(silhouette_coefficient)
    kmeans.append(k)

ax[0].plot(kmeans, silhouette_coefficient_means, '-o')
ax[0].set_xlabel('Number of clusters (K)')
ax[0].set_ylabel('Silhouette Coefficient for k means')
ax[0].set_title('k means vs K')

silhouette_coefficient_plus=[]
kplus=[]
for k in a:
    labels, centroids, silhouette_coefficient,k = k_means_plus(df,k)
    silhouette_coefficient_plus.append(silhouette_coefficient)
    kplus.append(k)

ax[1].plot(kplus, silhouette_coefficient_plus, '-o')
ax[1].set_xlabel('Number of clusters (K)')
ax[1].set_ylabel('Silhouette Coefficient for k means plus')
ax[1].set_title('k means plus vs K')


silhouette_coefficient_bisecting=[]
kbisecting=[]
for k in a:
    labels, centroids, silhouette_coefficient,k = bisecting_kmeans(df,k)
    silhouette_coefficient_bisecting.append(silhouette_coefficient)
    kbisecting.append(k)

ax[2].plot(kbisecting, silhouette_coefficient_bisecting, '-o')
ax[2].set_xlabel('Number of clusters (K)')
ax[2].set_ylabel('Silhouette Coefficient for bisecting k means')
ax[2].set_title('bisecting k means vs K')


plt.subplots_adjust(wspace=0.5)
plt.show()