#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\bank-full.csv")
df


# In[3]:


#step3:data transformation
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
LE


# In[18]:


for i in range(0,17):
    df.iloc[:,i]=LE.fit_transform(df.iloc[:,i])
print(df)


# In[19]:


df['y'].value_counts()


# In[20]:


df['y']=LE.fit_transform(df['y'])#it transforms the data in o and 1 s benign is 0 and malignant is 1
df['y']


# In[21]:


y=df['y']
y


# In[22]:


x=df.iloc[:,1:16]
list(x)


# In[23]:


from sklearn.linear_model import LogisticRegression
LogReg=LogisticRegression()
LogReg.fit(x,y)
y_pred=LogReg.predict(x)
y_pred


# In[24]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
cm=confusion_matrix(y,y_pred)
cm


# In[25]:


ac=accuracy_score(y,y_pred)
print("accuracy_score:",ac.round(4))
print("sensitivity_score:",recall_score(y,y_pred).round(4))
print("specificity_score:",precision_score(y,y_pred).round(4))
print("f1_score:",f1_score(y,y_pred).round(4))


# In[26]:


fp=cm[0,1]
fp


# In[27]:


tn=cm[0,0]
tn


# In[28]:


print("specificity score:",(tn/(tn+fp)).round(4))


# In[29]:


LogReg.predict_proba(x).shape
y_proba=LogReg.predict_proba(x)[:,1]
y_proba


# In[30]:


from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,z=roc_curve(y,y_proba)


# In[34]:


import matplotlib.pyplot as plt
plt.scatter(fpr,tpr)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC")
plt.show()
print("Area under curve:",roc_auc_score(y,y_proba).round(5))


# In[33]:


import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC")
plt.show()
print("Area under curve:",roc_auc_score(y,y_proba).round(5))


# In[ ]:




