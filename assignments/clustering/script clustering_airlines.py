# -*- coding: utf-8 -*-
"""clustering airlines.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G31X31rSRlcnc85eHXJYryict9iCk420
"""

from google.colab import files
uploaded=files.upload()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

df=pd.read_excel("EastWestAirlines.xlsx")
df

df1 = df.copy()
df1.head()

df1_norm = preprocessing.scale(df1)
df1_norm = pd.DataFrame(df1_norm) #standardize the data to normal distribution
df1_norm.head()

"""3 - Finding out the optimal number of clusters"""

plt.figure(figsize=(10, 8))
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df1_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""As seen from the elbow graph, the slope changes at 2. However, since spltting the dataset into 2 groups would not be very beneficial, we further evaluate clusters for higher values of k.

4 - H Clustering
4.1 - Euclidean distance, Ward
"""

dendrogram = sch.dendrogram(sch.linkage(df1_norm, method='ward'))

"""From the Ward method, we see that as the height increases the clusters get grouped together

We decided to cut the tree at height 85 to obtain 3 clusters and then assigned each cluster with its respective observations
"""

X = df1_norm.values
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
h_cluster = model.fit(X)
labels = model.labels_
plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')

"""5 - K Means"""

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
k_means = kmeans.fit_predict(df1_norm)
k_means

k_means1=k_means+1
k_cluster = list(k_means1)
df1['k_cluster'] = k_cluster
kmeans_mean_cluster = pd.DataFrame(round(df1.groupby('k_cluster').mean(),1))
kmeans_mean_cluster

pd.DataFrame(round(df1.groupby('k_cluster').count(),1))

plt.scatter(X[:, 0], X[:, 1], c=k_means, s=50, cmap='viridis')

"""5 - Conclusion
From the above data generated from K-Means clustering, we can see Cluster-1 has around 63% total travelers and cluster 2 has 33% of the travelers. We will target cluster 1 & 2. Cluster 1 contains less frequent or first time travellers, by giving them discount provided they travel more than twice or thrice and introduce more offer if they register or take the membership.
"""


