#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[3]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\bookrecommendation.csv")
df


# In[3]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df=ss.fit_transform(df)


# In[4]:


from sklearn.decomposition import PCA
pca=PCA()
pc=pca.fit_transform(df)
pc


# In[5]:


pd.DataFrame(pc)


# In[4]:


df['Book.Title'].unique()


# In[5]:


len(df['Book.Title'].unique())


# In[7]:


book = df.drop(['Unnamed: 0'], axis = 1)
df.tail(2)


# In[8]:


df1 = book.rename({'User.ID':'UserId','Book.Title':'Title','Book.Rating':'Rating'}, axis=1)
df1


# In[10]:


df1['Rating'].value_counts()


# In[12]:


len(df1['UserId'].unique())


# In[13]:


df1['Rating'].value_counts().sort_index().plot(kind='bar')


# In[15]:


df1.sort_values('UserId')


# In[16]:


books_new = df1.pivot_table(index='UserId', columns='Title', values='Rating')
books_new


# In[18]:


#Impute those NaNs with 0 values
books_new.fillna(0, inplace=True)
books_new.index = df1['UserId'].unique()
books_new


# # Calculating Cosine Similarity between Users
# 

# In[21]:


user1 = 1 - pairwise_distances( books_new.values, metric='cosine')
user1


# In[22]:


np.fill_diagonal(user1, 0)
#Store the results in a dataframe
user = pd.DataFrame(user1)
user


# In[24]:


#Set the index and column names to user ids 
user.index = df1.UserId.unique()
user.columns = df1.UserId.unique()
user


# In[25]:


user.iloc[100:120, 100:120]


# In[26]:


user.idxmax(axis=1)


# In[27]:


user.idxmax(axis=1)[20:40]


# In[28]:


user.loc[8].idxmax()


# In[29]:


df1[(df1['UserId']==276813) | (df1['UserId']==8)]


# In[31]:


id1=df1[df1['UserId']==276813]
id1


# In[32]:


id2=df1[df1['UserId']==8]
id2


# In[33]:


id1.Title


# In[34]:


pd.merge(id1,id2,on='Title',how='outer')


# # Considering the most similar customer

# In[36]:


def recommend(custID):
    simID = user.loc[custID].idxmax()
    simID_books = df1[df1['UserId'] == simID].Title
    custID_books = df1[df1['UserId'] == custID].Title
    return set(simID_books) - set(custID_books)
recommend(276813)


# In[37]:


recommend(507)


# In[38]:


recommend(3462)


# In[ ]:




