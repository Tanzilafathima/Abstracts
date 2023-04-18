#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn import externals
from io import StringIO
import pydotplus
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import tree


# In[2]:


df = pd.read_csv('C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/Company_Data.csv')
df


# In[4]:


df1 = df.copy()
df1.head()


# In[5]:


df1.isnull().sum()


# # Outlier Check

# In[6]:


ax = sns.boxplot(df1['Sales'])


# The data has 2 outlier instances
# 

# In[8]:


plt.rcParams["figure.figsize"] = 9,5
plt.figure(figsize=(16,5))
print("Skew: {}".format(df1['Sales'].skew()))
print("Kurtosis: {}".format(df1['Sales'].kurtosis()))
ax = sns.kdeplot(df1['Sales'],shade=True,color='g')
plt.xticks([i for i in range(0,20,1)])
plt.show()


# The data is Skwed on the right
# The data has negative Kurtosis

# In[9]:


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


# In[10]:


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


# In[11]:


corr = df1.corr()
df1 = pd.get_dummies(df1, columns = ['ShelveLoc','Urban','US'])
corr = df1.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)


# # 3 - Decision Tree - Model
# Since the target variable is continious, we create a class of the value based on the mean
# <= 7.49 == "Small" and > 7.49 == "large"

# In[12]:


df1["sales"]="small"
df1.loc[df1["Sales"]>7.49,"sales"]="large"
df1.drop(["Sales"],axis=1,inplace=True)
X = df1.iloc[:,0:14]
y = df1.iloc[:,14]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, stratify = y)
y_train.value_counts()


# In[13]:


model = DT(criterion='entropy') 
model.fit(x_train,y_train)


# In[14]:


pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)


# In[15]:


confusion_matrix(y_train,pred_train)


# In[16]:


pred_test = model.predict(x_test)
accuracy_score(y_test,pred_test)


# In[17]:


confusion_matrix(y_test,pred_test)


# In[18]:


df_t=pd.DataFrame({'Actual':y_test, 'Predicted':pred_test})
df_t


# In[20]:


cols = list(df1.columns)
predictors = cols[0:14]
target = cols[14]
dot_data = StringIO()
export_graphviz(model,out_file = dot_data, filled =True, rounded = True, feature_names =predictors,class_names = target, impurity = False )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# # 4 - Conclusion
# Since the accuracy of the Training set is 100% we test the accurancy on the test data which is 70%
# As seen in the confusion matrix of Test data 56 instances are presdected correctly and 24 instances are not

# In[ ]:


img = mpimg.imread('C:/Users/SohailShaikh/OneDrive - tiqets.com/Tiqets/Adhoc/DS/Decision Tree/company_full.png') 
plt.imshow(img)


# In[24]:


tree.plot_tree(model)
fn=['CompPrice','Income','Advertising','Population','Price','Age','Education',
    'ShelveLoc_Bad','ShelveLoc_Good','ShelveLoc_Medium','Urban_No','Urban_Yes','US_No','US_Yes']
cn=['Low/Mid', 'High']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=900)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);
plt.title('Decision tree using Entropy Criteria',fontsize=5)


# In[25]:


model.feature_importances_


# In[26]:


fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi


# As seen in the above table Price is most important feature

# In[27]:


df = pd.read_csv('C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/Fraud_check.csv')
df


# In[28]:


df1 = df.copy()
df1.head()


# In[30]:


ax = sns.boxplot(df1['Taxable.Income'])


# In[31]:


plt.rcParams["figure.figsize"] = 9,5
plt.figure(figsize=(16,5))
print("Skew: {}".format(df1['Taxable.Income'].skew()))
print("Kurtosis: {}".format(df1['Taxable.Income'].kurtosis()))
ax = sns.kdeplot(df1['Taxable.Income'],shade=True,color='g')
plt.xticks([i for i in range(10000,100000,10000)])
plt.show()


# In[32]:


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


# In[34]:


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


# In[35]:


df1 = pd.get_dummies(df1, columns = ['Undergrad','Marital.Status','Urban'])
corr = df1.corr()
corr = df1.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)


# # 3 - Decision Tree
# Since the target variable is continious, we create a class of taxable_income <= 30000 as "Risky" and others are "Good"

# In[36]:


df1['Taxable.Income']=pd.cut(df1['Taxable.Income'],bins=[0,30000,100000],labels=['risky','good'])
X = df1.iloc[:,1:10]
y = df1.iloc[:,0]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
y_train.value_counts()


# In[37]:


model = DT(criterion='entropy') 
model.fit(x_train,y_train)


# In[38]:


pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)


# In[39]:


confusion_matrix(y_train,pred_train)


# In[40]:


pred_test = model.predict(x_test)
accuracy_score(y_test,pred_test)


# In[41]:


confusion_matrix(y_test,pred_test)


# In[42]:


df_t=pd.DataFrame({'Actual':y_test, 'Predicted':pred_test})
df_t


# In[43]:


cols = list(df1.columns)
predictors = cols[1:10]
target = cols[0]
dot_data = StringIO()
export_graphviz(model, out_file = dot_data ,filled = True,rounded =True,feature_names = predictors,class_names = target, impurity = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# # 4 - Conclusion
# Since the accuracy of the Training set is 100% we test the accurancy on the test data which is 69%
# As seen in the confusion matrix of Test data 82 instances are presdected correctly and 38 instances are not

# In[47]:


tree.plot_tree(model);
fn=['Taxable.Income','City.Population','Work.Experience','Undergrad_NO','Undergrad_YES','Marital.Status_Divorced',
    'Marital.Status_Married','Marital.Status_Single','Urban_NO','Urban_YES','Category']
cn=['Good', 'Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=900)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);
plt.title('Decision tree using Entropy Criteria',fontsize=5)


# In[45]:


model.feature_importances_


# In[46]:


fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi


# As seen in the above table Price is most important feature

# In[ ]:




