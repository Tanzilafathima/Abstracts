#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.polynomial.polynomial import polyfit
from sklearn.linear_model import LinearRegression


# In[4]:


# import dataset
df=pd.read_csv('C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr//delivery_time.csv')
df


# In[5]:


#scatterplot
x = df['Sorting Time']
y = df['Delivery Time']


# In[6]:


b, m = polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.title('Scatter plot Delivery Time')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


# In[7]:


#Correlation Analysis
corr = np.corrcoef(x, y)


# Corr
# array([[1. , 0.82599726], [0.82599726, 1. ]])
# 
# The correlation between delivery time and sorting Time is high (83%)

# # 3 - Regression Model
# 1 - No transformation

# In[8]:


model = sm.OLS(y, x).fit()
predictions = model.predict(x)
model.summary()


# 
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 2 - Log Transformation of X

# In[9]:


x_log = np.log(df['Sorting Time'])
model = sm.OLS(y, x_log).fit()
predictions = model.predict(x_log)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 3 - Log Transformation of Y

# In[10]:


y_log = np.log(df['Delivery Time'])
model = sm.OLS(y_log, x).fit()
predictions = model.predict(x)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 4 - Log Transformation of X & Y

# In[11]:


model = sm.OLS(y_log, x_log).fit()
predictions = model.predict(x_log)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 5 - Sq Root Transformation of X

# In[12]:


x_sqrt = np.sqrt(df['Sorting Time'])
model = sm.OLS(y, x_sqrt).fit()
predictions = model.predict(x_sqrt)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# 

# # 6 - Square Root Transformation of Y

# In[13]:


y_sqrt = np.sqrt(df['Delivery Time'])
model = sm.OLS(y_sqrt, x).fit()
predictions = model.predict(x)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 7 - Square Root Transformation of X & Y

# In[14]:


model = sm.OLS(y_sqrt, x_sqrt).fit()
predictions = model.predict(x_sqrt)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 4 - Output Interpretation

# We will use Model 7 as it has the best R square value
# 
# 1 - p-value < 0.01
# Thus the model is accepted
# 
# 2 - coefficient == 1.64
# Thus if the value of Sorting Time is increased by 1, the predicted value of Delivery Time will increase by 1.64
# 
# 3 - Adj. R-sqared == 0.987
# Thus the model explains 98.7% of the variance in dependent variable

# In[22]:


df=pd.read_csv('C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr//Salary_Data.csv')
df


# In[23]:


b, m = polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.title('Scatter plot Salary Hike')
plt.xlabel('Years of Experience')
plt.ylabel('Salary Hike')
plt.show()


# As displayed in the scatter plot,but there is positive correlation between Salary Hike and Years of experience

# # Correlation Analysis

# In[24]:


corr = np.corrcoef(x, y)


# Corr
# array([[1. , 0.97824162], [0.97824162, 1. ]])
# 
# The correlation between Salary Hike and Years of experience is high (98%)

# # 3 - Regression Model

# # 1 - No transformation

# In[25]:


model = sm.OLS(y, x).fit()
predictions = model.predict(x)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 2 - Log Transformation of X

# In[27]:


x_log = np.log(x)
model = sm.OLS(y, x_log).fit()
predictions = model.predict(x_log)
model.summary()

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# # 3 - Log Transformation of Y

# In[29]:


y_log = np.log(y)
model = sm.OLS(y_log, x).fit()
predictions = model.predict(x)
model.summary()


# # Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 4 - Log Transformation of X & Y

# In[30]:


model = sm.OLS(y_log, x_log).fit()
predictions = model.predict(x_log)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 5 - Sq Root Transformation of X

# In[31]:


x_sqrt = np.sqrt(x)
model = sm.OLS(y, x_sqrt).fit()
predictions = model.predict(x_sqrt)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 6 - Square Root Transformation of Y

# In[33]:


y_sqrt = np.sqrt(y)
model = sm.OLS(y_sqrt, x).fit()
predictions = model.predict(x)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified

# # 7 - Square Root Transformation of X & Y

# In[34]:


model = sm.OLS(y_sqrt, x_sqrt).fit()
predictions = model.predict(x_sqrt)
model.summary()


# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# # 4 - Output Interpretation
# We will use Model 5 as it has the best R square value
# 
# 1 - p-value < 0.01
# Thus the model is accepted
# 
# 2 - coefficient == 3.48e+04 Thus if the value of years of experience is increased by 1, the predicted value of Salary hike will increase by 3.48e+04
# 
# 3 - Adj. R-sqared == 0.989
# Thus the model explains 98.9% of the variance in dependent variable

# In[ ]:




