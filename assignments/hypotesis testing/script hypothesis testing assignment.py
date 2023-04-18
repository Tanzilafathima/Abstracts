#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import stat
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[3]:


df=pd.read_csv("C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/BuyerRatio.csv")
df


# In[4]:


# Make dimensional array
obs=np.array([[50,142,131,70],[435,1523,1356,750]])
obs


# In[5]:


# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)


# In[14]:


# without Yates’ correction for continuity
chi_val, p_val, dof, expected =  chi2_contingency(obs, correction=False)
chi_val, p_val, dof, expected


# In[16]:


# for log-likelihood method 
chi_val, p_val, dof, expected =  chi2_contingency(obs, lambda_="log-likelihood")
chi_val, p_val, dof, expected


# Final Statement : as we can see above pvalue is greater than alpha hence we will fail to reject H0 (Null hypothesis) here.
# 
# pvalue > alpha => (0.66 > 0.05)
# 
# Accept H0 => The male-female buyer ratios are similar across the regions / All proportions are equal

# In[22]:


df1=pd.read_csv("C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/Costomer+OrderForm.csv")

df1


# In[24]:


df1.describe()


# In[25]:


df1.info()


# In[27]:


df1.dtypes


# In[29]:


#The count of Error Free and Defective for given 4 centers
table = [[271,267,269,280],[29,33,31,20]]
table


# In[ ]:





# In[30]:


data1 = pd.DataFrame([
    [271,267,269,280],
    [29,33,31,20]], 
    index = ['Error Free', 'Defective'], columns = ['Phillippines', 'Indonesia', 'Malta', 'India'])
data1


# In[39]:


# Make a contingency table
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs


# In[41]:


# Chi2 contengency independence test
chi2_contingency(obs)


# In[43]:


# without Yates’ correction for continuity
chi_val, p_val, dof, expected =  chi2_contingency(obs, correction=False)
chi_val, p_val, dof, expected


# In[44]:


# for log-likelihood method 
chi_val, p_val, dof, expected =  chi2_contingency(obs, lambda_="log-likelihood")
chi_val, p_val, dof, expected


# Final Statement : Now if we analyse pvalue = 0.277 at 5% significance level (alpha = 0.05) we get pvlaue greater than alpha. Hence we will fail to reject H0 (Null Hypothesis)
# 
# pvalue > alpha (0.277 > 0.05)
# 
# Accept H0 => the defective % does not vary by centre / the variables are independent

# In[45]:


df2=pd.read_csv("C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/LabTAT.csv")
df2


# In[48]:


#Calculating pvalue with the help of F-ratio
stats.f_oneway(df2.iloc[:,0], df2.iloc[:,1], df2.iloc[:,2], df2.iloc[:,3])


# Final Statement : as we can see above pvalue = 2.11 x 10 raise to -57 which is almost 0 and lesser than alpha value hence we reject H0 (Null hypothesis)
# 
# pvalue < alpha ( 2.11 x 10 raise to -57 < 0.05)
# Accept Ha => At least 1 Lab's Average TAT is different // Not all the averages are same

# In[51]:


df3=pd.read_csv("C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr/Cutlets.csv")
df3


# In[53]:


#Calculating pvalue with t-test
stats.ttest_ind(df3['Unit A'],df3['Unit B'])


# Final Statement : as we can see above the pvalue we have got is 0.4722 which is greater then alpha valve we will fail to reject the Null hypothesis here
# 
# pvalue > alpha (0.4722 > 0.05)
# 
# Accept H0 => There is no difference in the diameter of the cutlets of Unit A and B
