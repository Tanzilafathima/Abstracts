#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\assignmentsexcelr\\book.csv")
df


# In[3]:


from apyori import apriori


# In[6]:


trans = []
for i in range(0, 2000):
  trans.append([str(df.values[i,j]) for j in range(0, 11)])
trans


# In[7]:


trans[0]
trans[1]
trans[2]


# In[13]:


rules = apriori(transactions = trans, min_support = 0.001, min_confidence = 0.01, min_lift = 1, min_length = 2, max_length = 2)


# In[14]:


results = list(rules)
results


# In[17]:


baseitem=[]
additem =[]
support=[]
confidence = []
lift = []

for i in range(0,3):
    baseitem.append(results[i][2][0][0])
    additem.append(results[i][2][0][1])
    support.append(results[i][1])
    confidence.append(results[i][2][0][2])
    lift.append(results[i][2][0][3])


# In[18]:


d1= pd.DataFrame(baseitem)
d2= pd.DataFrame(additem)
d3= pd.DataFrame(support)
d4= pd.DataFrame(confidence)
d5= pd.DataFrame(lift)


# In[19]:


df1 = pd.concat([d1,d2,d3,d4,d5],axis=1)
df1


# In[47]:


df1.columns["baseitem" ,"additem","support","confidence","lift"]


# In[24]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\assignmentsexcelr\\my_movies.csv")
df.shape


# In[26]:


trans = []
for i in range(0, 10):
  trans.append([str(df.values[i,j]) for j in range(0, 15)])
trans


# In[36]:


rules = apriori(transactions = trans, min_support = 0.001, min_confidence = 0.03, min_lift = 3, min_length = 2, max_length = 2)


# In[37]:


results = list(rules)
results


# In[38]:


baseitem=[]
additem =[]
support=[]
confidence = []
lift = []

for i in range(0,3):
    baseitem.append(results[i][2][0][0])
    additem.append(results[i][2][0][1])
    support.append(results[i][1])
    confidence.append(results[i][2][0][2])
    lift.append(results[i][2][0][3])


# In[39]:


d1= pd.DataFrame(baseitem)
d2= pd.DataFrame(additem)
d3= pd.DataFrame(support)
d4= pd.DataFrame(confidence)
d5= pd.DataFrame(lift)


# In[40]:


df1 = pd.concat([d1,d2,d3,d4,d5],axis=1)
df1


# In[ ]:




