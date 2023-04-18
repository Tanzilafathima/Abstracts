#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets,tree
from sklearn.tree import export_graphviz 
from sklearn import externals
from io import StringIO
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv('C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/Company_Data.csv')
df


# In[6]:


df1 = df.copy()
df1.head()


# In[7]:


df1.describe().T


# In[8]:


df1.dtypes


# In[9]:


df1.isnull().sum()


# # Outlier Check

# In[10]:


ax = sns.boxplot(df1['Sales'])


# The data has 2 outlier instances

# In[11]:


plt.rcParams["figure.figsize"] = 9,5
plt.figure(figsize=(16,5))
print("Skew: {}".format(df1['Sales'].skew()))
print("Kurtosis: {}".format(df1['Sales'].kurtosis()))
ax = sns.kdeplot(df1['Sales'],shade=True,color='g')
plt.xticks([i for i in range(0,20,1)])
plt.show()


# The data is Skwed on the right
# The data has negative Kurtosis

# In[12]:


obj_colum = df1.select_dtypes(include='object').columns.tolist()
plt.figure(figsize=(16,10))
for i,col in enumerate(obj_colum,1):
    plt.subplot(2,2,i)
    sns.countplot(data=df1,y=col)
    plt.subplot(2,2,i+1)
    df1[col].value_counts(normalize=True).plot.bar()
    plt.ylabel(col)
    plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show()  


# In[13]:


num_columns = df1.select_dtypes(exclude='object').columns.tolist()
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


# In[14]:


corr = df1.corr()
df1 = pd.get_dummies(df1, columns = ['ShelveLoc','Urban','US'])
corr = df1.corr()
corr


# In[15]:


plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)


# # 3 - Random Forest Model
# Since the target variable is continious, we create a class of the value based on the mean
# <= 7.49 == "Small" and > 7.49 == "large"

# In[16]:


df1["sales"]="small"
df1.loc[df1["Sales"]>7.49,"sales"]="large"
df1.drop(["Sales"],axis=1,inplace=True)
X = df1.iloc[:,0:14]
y = df1.iloc[:,14]


# In[17]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
y_train.value_counts()


# In[18]:


model =RF(n_jobs=4,n_estimators = 150, oob_score =True,criterion ='entropy') 
model.fit(x_train,y_train)
model.oob_score_


# In[19]:


pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)


# In[20]:


confusion_matrix(y_train,pred_train)


# In[21]:


pred_test = model.predict(x_test)
accuracy_score(y_test,pred_test)


# In[22]:


confusion_matrix(y_test,pred_test)


# In[23]:


df_t=pd.DataFrame({'Actual':y_test, 'Predicted':pred_test})
df_t


# In[26]:


pip install pydotplus


# In[27]:


import pydotplus


# In[39]:


pip install python-graphviz


# In[38]:


pip install jurigged


# In[41]:


cols = list(df1.columns)
predictors = cols[0:14]
target = cols[14]
tree1 = model.estimators_[20]
dot_data = StringIO()
export_graphviz(tree1, out_file = dot_data, feature_names =predictors, class_names = target, filled =True,rounded=True,impurity =False,proportion=False,precision =2)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# # 4 - Conclusion
# Since the accuracy of the Training set is 100% we test the accurancy on the test data which is 76%
# As seen in the confusion matrix of Test data 61 instances are presdected correctly and 19 instances are not

# In[33]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[40]:


rf_small = RF(n_estimators=10, max_depth = 3)
rf_small.fit(x_train,y_train)
RandomForestClassifier(max_depth=3, n_estimators=10)
tree_small = rf_small.estimators_[5]
export_graphviz(tree_small, out_file = dot_data, feature_names = predictors, rounded = True, precision = 1)
graph_small = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[46]:


model.feature_importances_


# In[47]:


fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi


# As seen in the above table Price is most important feature

# In[ ]:




