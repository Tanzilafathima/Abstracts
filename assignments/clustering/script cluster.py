#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\assignmentsexcelr\\crimecluster.csv")
df


# In[3]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Unnamed: 0']=le.fit_transform(df['Unnamed: 0'])
df


# In[4]:


x=df.iloc[:,1:]


# In[5]:


import scipy.cluster.hierarchy as shc
#construction of dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("customer dendogram")
dend=shc.dendrogram(shc.linkage(x,method='complete'))


# In[6]:


#forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
y=cluster.fit_predict(x)
y=pd.DataFrame(y)
y.value_counts()


# In[7]:


y=pd.DataFrame(cluster.labels_,columns=['assignclustering'])
y


# In[8]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.scatter(x.iloc[:,0],x.iloc[:,1],c=cluster.labels_,cmap='rainbow')


# In[9]:


df1=pd.concat([df,y],axis=1)
df1


# In[10]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss=ss.fit_transform(df1)
df1


# In[12]:


x=df1.iloc[:,1:5]
x


# In[18]:


y=df1['assignclustering']
y


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=24)


# In[20]:


from sklearn.svm import SVC
clf = SVC(kernel='linear',C=1.0)
clf.fit(x_train, y_train)
y_pred_train = clf.predict(x_train)
y_pred_test  = clf.predict(x_test)

from sklearn import metrics
print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(2))
print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred_test).round(2))


# In[21]:


get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.iloc[:, 0], x.iloc[:, 1], x.iloc[:, 2])
plt.show()


# In[22]:


# Initializing KMeans
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
# Fitting with inputs
kmeans = kmeans.fit(x)
# Predicting the clusters
y = kmeans.predict(x)
y = pd.DataFrame(y)
y.value_counts()


# In[23]:


p1 = kmeans.inertia_
int(p1)
clust = []
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(x)
    clust.append(int(kmeans.inertia_))
print(clust)


# In[24]:


plt.plot(range(1, 11), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()


# In[30]:


df1.drop(['Murder','Assault'],axis=1,inplace=True)
DBSCAN=df1.values
DBSCAN


# In[33]:


from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(DBSCAN)
X = stscaler.transform(DBSCAN)
X


# In[34]:


from sklearn.cluster import DBSCAN
DBSCAN()
# dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan = DBSCAN(eps=1, min_samples=3)
dbscan.fit(X)


# In[35]:


#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl.value_counts()


# In[36]:


df_new = pd.concat([df,cl],axis=1)
noisedata = df_new[df_new['cluster']==-1]
print(noisedata)
finaldata = df_new[df_new['cluster']==0]


# In[ ]:




