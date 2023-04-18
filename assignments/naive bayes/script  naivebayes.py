#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\SalaryData_Test.csv")
df


# In[4]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[5]:


for i in range(0,13):
    df.iloc[:,i]=le.fit_transform(df.iloc[:,i])
df


# In[6]:


x=df.iloc[:,1:13]
x


# In[7]:


y=df['Salary']
y


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)


# In[9]:


from sklearn.naive_bayes import MultinomialNB
MNB= MultinomialNB()
MNB.fit(x_train,y_train)


# In[10]:


y_pred_train=MNB.predict(x_train)
y_pred_test=MNB.predict(x_test)


# In[11]:


from sklearn.metrics import accuracy_score
trainining_accuracy=accuracy_score(y_train,y_pred_train)
test_accuracy=accuracy_score(y_test,y_pred_test)
print("naive_bayes trainining_accuracy",trainining_accuracy.round(4))
print("naive_bayes test_accuracy",test_accuracy.round(4))


# In[17]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\assignmentsexcelr\\SalaryData_Train (2).csv")
df


# In[18]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[19]:


for i in range(0,13):
    df.iloc[:,i]=le.fit_transform(df.iloc[:,i])
df


# In[20]:


x=df.iloc[:,1:13]
y=df['Salary']


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)


# In[23]:


from sklearn.naive_bayes import MultinomialNB
MNB= MultinomialNB()
MNB.fit(x_train,y_train)
y_pred_train=MNB.predict(x_train)
y_pred_test=MNB.predict(x_test)


# In[24]:


from sklearn.metrics import accuracy_score
trainining_accuracy=accuracy_score(y_train,y_pred_train)
test_accuracy=accuracy_score(y_test,y_pred_test)
print("naive_bayes trainining_accuracy",trainining_accuracy.round(4))
print("naive_bayes test_accuracy",test_accuracy.round(4))


# In[ ]:




