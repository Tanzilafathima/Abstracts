#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\Zoo.csv")
df


# In[5]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[7]:


x=df.iloc[:,2:17]
x


# In[9]:


y=df['type']
y


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,p=2)
knn.fit(x_train,y_train)
y_pred_train=knn.predict(x_train)
y_pred_test=knn.predict(x_test)


# In[12]:


from sklearn.metrics import accuracy_score
trainining_accuracy=accuracy_score(y_train,y_pred_train)
test_accuracy=accuracy_score(y_test,y_pred_test)
print("knn trainining_accuracy",trainining_accuracy.round(4))
print("knn test_accuracy",test_accuracy.round(4))


# In[ ]:




