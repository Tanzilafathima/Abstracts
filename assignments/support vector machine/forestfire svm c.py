#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions


# In[2]:


df = pd.read_csv('C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\forestfires.csv')
df


# In[4]:


df.shape


# In[5]:


df[df.columns[0:11]].describe().T


# In[6]:


df[df.columns[0:11]].isnull().sum()


# # Finding Correlation

# In[8]:


corr = df[df.columns[0:11]].corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)


# In[10]:


#Outlier Check
ax = sns.boxplot(df['area'])


# There are 3 Outlier instances in our data
# 

# In[13]:


plt.rcParams["figure.figsize"] = 9,5
plt.figure(figsize=(16,5))
print("Skew: {}".format(df['area'].skew()))
print("Kurtosis: {}".format(df['area'].kurtosis()))
ax = sns.kdeplot(df['area'],shade=True,color='g')
plt.xticks([i for i in range(0,1200,50)])
plt.show()


# The Data is highly skewed and has large kurtosis value
# Majority of the forest fires do not cover a large area, most of the damaged area is under 100 hectares of land

# In[17]:


dfa = df[df.columns[0:10]]
month_colum = dfa.select_dtypes(include='object').columns.tolist()
plt.figure(figsize=(16,10))
for i,col in enumerate(month_colum,1):
    plt.subplot(2,2,i)
    sns.countplot(data=dfa,y=col)
    plt.subplot(2,2,i+2)
    df[col].value_counts(normalize=True).plot.bar()
    plt.ylabel(col)
    plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show() 


# Majority of the fire accors in the month Aug and Sep
# For Days Sun and Fri have recoreded the most cases

# In[18]:


num_columns = dfa.select_dtypes(exclude='object').columns.tolist()
plt.figure(figsize=(18,40))
for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(df[col],color='g',shade=True)
    plt.subplot(8,4,i+10)
    df[col].plot.box()
plt.tight_layout() 
plt.show()
num_data = df[num_columns]
pd.DataFrame(data=[num_data.skew(),num_data.kurtosis()],index=['skewness','kurtosis'])


# # 3 - SVM
# 

# In[20]:


X = df.iloc[:,2:30]
y = df.iloc[:,30]
mapping = {'small': 1, 'large': 2}
y = y.replace(mapping)


# In[21]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20, stratify = y)


# # 3.1 Linear
# 

# In[22]:


model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred_test_linear))


# # 3.2 Poly
# 

# In[23]:


model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred_test_poly))


# # 3.3 RBF
# 

# In[24]:


model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred_test_rbf))


# # 3.4 Sigmoid
# 

# In[25]:


model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(x_train,y_train)
pred_test_sigmoid = model_sigmoid.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred_test_sigmoid))


# # 4 - Conclusion
# Linear Model gives the best accuracy
# Below is an exmaple on how we can plot the data. I used PCA to select only 2 variables

# In[26]:


ytt = y_train.to_numpy()
pca = PCA(n_components = 2)
x_train2 = pca.fit_transform(x_train)
model_linear.fit(x_train2,ytt)


# In[27]:


plot_decision_regions(x_train2,ytt, clf=model_linear)
plt.show()


# In[ ]:




