#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
sns.set()


# In[3]:


df = pd.read_csv('C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\glass.csv')
df


# In[4]:


df1 = df.copy()
df1.loc[df['Type'] == 1, 'Type'] = 'building_windows_float_processed'
df1.loc[df['Type'] == 2, 'Type'] = 'building_windows_non_float_processed' 
df1.loc[df['Type'] == 3, 'Type'] = 'vehicle_windows_float_processed' 
df1.loc[df['Type'] == 4, 'Type'] = 'vehicle_windows_non_float_processed' 
df1.loc[df['Type'] == 5, 'Type'] = 'containers' 
df1.loc[df['Type'] == 6, 'Type'] = 'tableware' 
df1.loc[df['Type'] == 7, 'Type'] = 'headlamps' 
df1.head()


# In[5]:


df1.describe()


# In[6]:


sns.factorplot('Type', data=df1, kind="count",size = 5,aspect = 2)


# As shown in the graphs above, majority of the glass types are building_windows_float_processed and building_windows_non_float_processed, followed by headlamps

# In[7]:


df1.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()


# In[8]:


df1.plot(kind='box', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()


# # 3 - Finding correlation between the variables in the data

# In[9]:


cor = df1.corr(method='pearson')
cor.style.background_gradient(cmap='coolwarm')


# # As seen in the above graph, there is a high correlation exists between some of the variables. We can use PCA to reduce the hight correlated variables

# # 4 - KNN
# 

# 4.1 Finding optimal number of K
# 

# In[10]:


X = np.array(df1.iloc[:,3:5])
y = np.array(df1['Type'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []
for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.show()


# we can see that k=4 produces the most accurate results

# # 4.2 Applying the algorithm
# 

# In[11]:


knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred_KNeighborsClassifier = knn.predict(X_test)
scores = []
cv_scores = []
score = accuracy_score(y_pred_KNeighborsClassifier,y_test)
scores.append(score)
score_knn=cross_val_score(knn, X,y, cv=10)


# In[12]:


score_knn.mean()


# In[13]:


score_knn.std()


# In[14]:


cv_score = score_knn.mean()


# In[15]:


cv_scores.append(cv_score)
cv_scores


# # 5 - Conclusion
# Support Vector Machine Accuracy: 0.60 (+/- 0.21)

# In[ ]:




