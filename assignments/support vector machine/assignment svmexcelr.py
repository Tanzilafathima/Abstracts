#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[7]:


salary_train = pd.read_csv('C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\SalaryData_Train (2).csv')
salary_train


# In[8]:


salary_test = pd.read_csv('C:\\Users\\Tannu\\OneDrive\\Desktop\\excelr\\excelr\\assignmentsexcelr\\SalaryData_Test.csv')
salary_test


# In[9]:


salary_train.info()


# # Let's Visualize the data for better understanding

# In[10]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(12,5))
salary_train.workclass.value_counts().plot.bar();


# In[11]:


plt.figure(figsize=(12,5))
salary_train.education.value_counts().plot.bar(color='purple');


# In[12]:


plt.figure(figsize=(12,5))
salary_train.maritalstatus.value_counts().plot.bar(color='yellow');


# In[13]:


plt.figure(figsize=(12,5))
salary_train.occupation.value_counts().plot.bar(color='violet')


# In[14]:


plt.figure(figsize=(12,5))
salary_train.relationship.value_counts().plot.bar(color='brown')


# In[15]:


plt.figure(figsize=(12,5))
salary_train.race.value_counts().plot.bar(color='green');


# In[16]:


plt.figure(figsize=(12,5))
salary_train.sex.value_counts().plot.bar(color='orange');


# In[17]:


plt.figure(figsize=(12,5))
salary_train.Salary.value_counts().plot.bar(color='gray');


# In[18]:


# Since the Salary column is Y variable here, seperating it from the data set and applying the dummies on rest of the data. 

# For train data set 
salary_train1 = salary_train.iloc[:,0:13]

salary_train1 = pd.get_dummies(salary_train1)
salary_train1


# In[19]:


# For test data set 
salary_test1 = salary_test.iloc[:,0:13]

salary_test1 = pd.get_dummies(salary_test1)
salary_test1


# # PCA needs to apply here as the no. of columns are more

# Applyting Dimentionality Reduction technique PCA

# In[20]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Scaling the train dataset

sc.fit(salary_train1)
salary_train_norm = sc.transform(salary_train1)
salary_train_norm      


# In[21]:


#Scaling the test dataset

sc.fit(salary_test1)
salary_test_norm = sc.transform(salary_test1)
salary_test_norm #Normalised dataset


# In[22]:


from sklearn.decomposition import PCA

# For train dataset

salary_train_pca = PCA(n_components = 102)
salary_train_pca_values = salary_train_pca.fit_transform(salary_train_norm)
salary_train_pca_values


# In[23]:


# For test dataset 

salary_test_pca = PCA(n_components = 102)
salary_test_pca_values = salary_test_pca.fit_transform(salary_test_norm)
salary_test_pca_values


# In[24]:


# The amount of variance that each PCA explains is 
var = salary_train_pca.explained_variance_ratio_
var


# In[25]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[26]:


# Variance plot for PCA components obtained
plt.figure(figsize=(12,4))
plt.plot(var1,color="red");


# In[27]:


finaltrain = pd.concat([pd.DataFrame(salary_train_pca_values[:,0:90]),
                     salary_train[['Salary']]], axis = 1)
finaltrain


# In[28]:


finaltest = pd.concat([pd.DataFrame(salary_test_pca_values[:,0:90]),
                     salary_test[['Salary']]], axis = 1)
finaltest


# In[29]:


# Since the training dataset is huge, we'll use some part of it for the training purpose, to reduce time consumed.

# For train dataset
array = finaltrain.values
X = array[0:1000,0:90]
Y = array[0:1000,90]


# In[30]:


# For test dataset
x = finaltest.values[0:1000,0:90]
y = finaltest.values[0:1000,90]


# Since the training and test datasets are separately given in the problem, we don't need to split the data into train and test here.

# # SVM Classification

# Let's use Grid search CV to find out best value for params

# In[32]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[33]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[0.9,0.5,0.1],'C':[1,10,100] },
             {'kernel':['linear'],'C':[1,10,100]}]
gsv = GridSearchCV(clf,param_grid,cv=10,n_jobs=-1)
gsv.fit(X,Y)

gsv.best_params_ , gsv.best_score_


# In[35]:


#SVM Clasification
clf = SVC(C=10, kernel='linear')
clf.fit(x,y)
results = clf.score(x,y)
print(np.round(results, 4))


# The Model accuracy by SVM classification is 85%

# In[ ]:





# In[ ]:




