#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr//ToyotaCorolla.csv',encoding = 'unicode_escape')
df


# In[3]:


df.columns


# Price, Age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, Weight,
# let's eliminate rest of the columns

# In[5]:


df.drop(df.columns[0:2], inplace=True,axis = 1)
df.head()


# In[6]:


df.drop(df.columns[2:4], inplace=True,axis = 1)
df.head()


# In[7]:


df.drop(df.columns[3:4], inplace=True,axis = 1)
df.head()


# In[8]:


df.drop(df.columns[4:7], inplace=True,axis = 1)
df.head()


# In[9]:


df.drop(df.columns[6:7], inplace=True,axis = 1)
df.head()


# In[10]:


df.drop(df.columns[9:39], inplace=True,axis = 1)
df.head()


# In[12]:


df.info()


# In[14]:


#check for missing values
df.isna().sum()


# In[15]:


df.corr()


# # Scatterplot between variables along with histograms

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[18]:


#Format the plot background and scatter plots for all the variables
sns.set_style(style='darkgrid')
sns.pairplot(df)


# # Preparing a model

# In[22]:


#Build model
import statsmodels.formula.api as smf 
model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',df).fit()


# In[23]:


#Coefficients
model.params


# In[24]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)


# In[82]:


#R squared and AIC values
(model.rsquared,model.rsquared_adj,model.aic)


# Looking at the pvalue above we get Multicollinearity issue in cc and Doors varibale as their pvalues (cc: 0.179, Doors: 0.968) are greater than alpha (0.05) Hence we need to check their performance in the model by SLR

# # Simple Linear Regression Models

# In[84]:


ml_cc =smf.ols('Price~cc',data=df).fit()
#t and p-Values
print(ml_cc.tvalues, '\n', ml_cc.pvalues)


# In[85]:


ml_dr =smf.ols('Price~Doors',data=df).fit()
#t and p-Values
print(ml_dr.tvalues, '\n', ml_dr.pvalues)


# In[86]:


ml_ccdr =smf.ols('Price~cc+Doors',data=df).fit()
#t and p-Values
print(ml_ccdr.tvalues, '\n', ml_ccdr.pvalues)


# Since we are not finding any issue in SLR as all the pvalues are lesser than alpha let's try using Variance Influation Factor

# # VIF Values

# In[88]:


rsq_age = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_age = 1/(1-rsq_age)
rsq_km = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared  
vif_km = 1/(1-rsq_km)

rsq_hp = smf.ols('HP~KM+Age_08_04+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared  
vif_hp = 1/(1-rsq_hp) 

rsq_cc = smf.ols('cc~HP+KM+Age_08_04+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared  
vif_cc = 1/(1-rsq_cc) 

rsq_dr = smf.ols('Doors~cc+HP+KM+Age_08_04+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared  
vif_dr = 1/(1-rsq_dr) 

rsq_gear = smf.ols('Gears~cc+HP+KM+Age_08_04+Doors+Quarterly_Tax+Weight',data=df).fit().rsquared  
vif_gear = 1/(1-rsq_gear) 

rsq_qt = smf.ols('Quarterly_Tax~cc+HP+KM+Age_08_04+Doors+Gears+Weight',data=df).fit().rsquared  
vif_qt = 1/(1-rsq_qt) 

rsq_wt = smf.ols('Weight~cc+HP+KM+Age_08_04+Doors+Gears+Quarterly_Tax',data=df).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

# Storing vif values in a data frame
T1 = {'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],'VIF':[vif_age,vif_km,vif_hp,vif_cc,vif_dr,vif_gear,vif_qt,vif_wt]}
Vif_frame = pd.DataFrame(T1)  
Vif_frame


# Even in VIF values we don't find any Multicollinearity issue hence let's go for

# # Residual Analysis
# Test for Normality of Residuals (Q-Q Plot)

# In[89]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()
# help(sm.qqplot) to play with other parameters


# Looking at the Q-Q plot we can say that Data points are following the Normal distribution however we can also detect the ouliers there which we will find out below

# In[90]:


list(np.where(model.resid>6000))


# As we can see above there are two observations which are outliers however we will confirm those by methods like Cook's Distance, Hat Point

# # Model Validation Technique
# Residual Plot - Fitted Vs Residuals (Ei Vs Y^)

# In[91]:


def get_standardized_values( values ):
    return (values - values.mean())/values.std()
plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# Since we can see a pattern or a data concentration at one place we can say that this is not the perfect model we should go for, as it has ouliers detected which we haven't yet treated. Let's start treating them,

# # Model Deletion Diagnostics
# Detecting Influencers/Outliers
# 1) Cook’s Distance

# In[92]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance
model_influence.cooks_distance


# In[94]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[95]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# # 2) High Influence points or Hat Points

# In[96]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[97]:


k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff 


# From the above plot, it is evident that data point 80, 221, 960, and 991 are the influencers

# In[98]:


df[df.index.isin([80, 221, 960, 991])]


# In[100]:


#Let's check the differences in Price, CC and other variable values
df.head(20)


# So comparing the 80th Observation with others we can see 80th observation's CC is way higher than other and Comparing 221st, 960th, 991st observation with other we understand that their age is more than rest all the observations in the daa set
# 
# Let's remove the above outliers and

# # Improve the model

# In[102]:


#Discard the data points which are influencers and reasign the row number (reset_index())
TC1=df.drop(df.index[[80, 221, 960, 991]],axis=0).reset_index()
TC1


# In[103]:


#Drop the original index
TC1=TC1.drop(['index'],axis=1)
TC1


# # Now Let's build a model

# In[104]:


final_model1 =smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=TC1).fit()
#Generating R-Squared and AIC values
(final_model1.rsquared,final_model1.rsquared_adj,final_model1.aic)


# In[105]:


print(final_model1.tvalues, '\n', final_model1.pvalues)


# # Cook's Distance
# 

# In[106]:


model_influence_f1 = final_model1.get_influence()
(c_f1, _) = model_influence_f1.cooks_distance
fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(TC1)),np.round(c_f1,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[107]:


#index of the data points where c is more than .5
(np.argmax(c_f1),np.max(c_f1))


# Since the C value (0.321) is < 1 , we can stop the diagnostic process and finalize the model
# Also comapring the 1st model's and final model's R square and AIC values we understand that R square value has increased and AIC value has decreased in Final model, which states that the Final model is good and ready to predict for new data points.

# # Predicting for new data

# In[108]:


#New data for prediction
new_data1=pd.DataFrame({'Age_08_04':33,'KM':22000,'HP':80,"cc":1200,"Doors":4,"Gears":5,'Quarterly_Tax':200,'Weight':800},index=[1])
final_model1.predict(new_data1)


# In[109]:


new_data2=pd.DataFrame({'Age_08_04':23,'KM':46986,'HP':90,"cc":2000,"Doors":3,"Gears":5,'Quarterly_Tax':210,'Weight':1165},index=[2])


# In[110]:


final_model1.predict(new_data2)


# In[26]:


df1=pd.read_csv('C:/Users/Tannu/OneDrive/Desktop/excelr/excelr/assignmentsexcelr//50_Startups.csv',encoding = 'unicode_escape')
df1


# In[27]:


df1.info()


# In[28]:


#check for missing values
df1.isna().sum()


# In[29]:


import numpy as np
import matplotlib.pyplot as plt


# In[31]:


#Since the column state is catogorical we need to change it to numerical type by One Hot encoding method
df2=pd.get_dummies(df1,columns=['State'])
df2


# In[32]:


# Dropping the first column of Dummy Variables (State) in terms to avoid Dummy Trap 
df3=df2.drop('State_California', axis=1)
df3.head()


# In[33]:


#Changing the column names 
df4=df3.rename({'R&D Spend': 'rnd_Spend', 'Marketing Spend':'Marketing_Spend','State_New York':'State_NewYork'}, axis=1)
df4.head()


# # Scatterplot between variables along with histograms

# In[34]:


import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf


# In[35]:


#Format the plot background and scatter plots for all the variables
sns.set_style(style='darkgrid')
sns.pairplot(df4)


# In[37]:


#Build model
model1 = smf.ols('Profit~rnd_Spend+Administration+Marketing_Spend+State_Florida+State_NewYork', data=df4).fit()


# In[38]:


#Coefficients
model1.params


# In[39]:


#t and p-Values
print(model1.tvalues, '\n', model1.pvalues)


# In[40]:


#R squared and AIC values
(model1.rsquared,model1.rsquared_adj,model1.aic)


# Looking at the pvalue above we get Multicollinearity issue in Administration (0.607), Marketing_Spend (0.122), State_Florida (0.953), and State_NewYork (0.989) varibale as their pvalues are greater than alpha (0.05) Hence we need to check their performance in the model by SLR

# # Simple Linear Regression Models

# In[42]:


ml_adm =smf.ols('Profit~Administration',data=df4).fit()
#t and p-Values
print(ml_adm.tvalues, '\n', ml_adm.pvalues)


# In[43]:


ml_mrk =smf.ols('Profit~Marketing_Spend',data=df4).fit()
#t and p-Values
print(ml_mrk.tvalues, '\n', ml_mrk.pvalues)


# In[44]:


ml_sf =smf.ols('Profit~State_Florida',data=df4).fit()
#t and p-Values
print(ml_sf.tvalues, '\n', ml_sf.pvalues)


# In[45]:


ml_sny =smf.ols('Profit~State_NewYork',data=df4).fit()
#t and p-Values
print(ml_sny.tvalues, '\n', ml_sny.pvalues)


# In[46]:


ml_amss =smf.ols('Profit~Administration+Marketing_Spend+State_Florida+State_NewYork',data=df4).fit()
#t and p-Values
print(ml_amss.tvalues, '\n', ml_amss.pvalues)


# # VIF Values

# In[47]:


rsq_rnd = smf.ols('rnd_Spend~Administration+Marketing_Spend+State_Florida+State_NewYork',data=df4).fit().rsquared
vif_rnd = 1/(1-rsq_rnd)

rsq_adm = smf.ols('Administration~rnd_Spend+Marketing_Spend+State_Florida+State_NewYork',data=df4).fit().rsquared
vif_adm = 1/(1-rsq_adm)

rsq_mar = smf.ols('Marketing_Spend~rnd_Spend+Administration+State_Florida+State_NewYork',data=df4).fit().rsquared
vif_mar = 1/(1-rsq_mar)

rsq_sf = smf.ols('State_Florida~rnd_Spend+Administration+Marketing_Spend+State_NewYork',data=df4).fit().rsquared
vif_sf = 1/(1-rsq_sf)
 
rsq_sny = smf.ols('State_NewYork~rnd_Spend+Administration+Marketing_Spend+State_Florida',data=df4).fit().rsquared
vif_sny = 1/(1-rsq_sny)

# Storing vif values in a data frame
T1 = {'Variables':['rnd_Spend','Administration','Marketing_Spend','State_Florida','State_NewYork'],'VIF':[vif_rnd,vif_adm,vif_mar,vif_sf,vif_sny]}
Vif_frame = pd.DataFrame(T1)  
Vif_frame


# In VIF values we don't find any Multicollinearity issue hence let's go for

# # Residual Analysis
# Test for Normality of Residuals (Q-Q Plot)

# In[48]:


import statsmodels.api as sm
qqplot=sm.qqplot(model1.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()


# Looking at the Q-Q plot we can say that Data points are following the Normal distribution however we can also detect the ouliers there which we will find out below

# In[49]:


list(np.where(model1.resid>15000))


# As we can see above there are two observations which are outliers however we will confirm those by methods like Cook's Distance, Hat Point

# # Model Validation Technique
# Residual Plot - Fitted Vs Residuals (Ei Vs Y^)

# In[50]:


def get_standardized_values( values ):
    return (values - values.mean())/values.std()


# In[51]:


plt.scatter(get_standardized_values(model1.fittedvalues),
            get_standardized_values(model1.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# Since we can see a pattern we can say that this is not the perfect model we should go for, as it has ouliers detected which we haven't yet treated. Let's start treating them,

# # Model Deletion Diagnostics
# Detecting Influencers/Outliers

# # 1) Cook’s Distance
# 

# In[52]:


model1_influence = model1.get_influence()
(c, _) = model1_influence.cooks_distance


# In[53]:


model1_influence.cooks_distance


# In[54]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# In[56]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df4)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# # 2) High Influence points or Hat Points

# In[57]:


influence_plot(model1)
plt.show()


# From the above plot, it is evident that data point 14, 19, 27, 36, 38, 45, 46, 48, and 49 are the influencers

# In[60]:


df4[df4.index.isin([14, 19, 27, 36, 38, 45, 46, 48, 49])]
df4


# #Let's check the differences in Price, CC and other variable values
# Let's remove the above outliers and

# # Improve the model

# In[61]:


#Discard the data points which are influencers and reasign the row number (reset_index())
df5=df4.drop(df4.index[[14, 19, 27, 36, 38, 45, 46, 48, 49]],axis=0).reset_index()
df5


# # Now Let's build a model

# In[66]:


final_model1=smf.ols('Profit~rnd_Spend+Administration+Marketing_Spend+State_Florida+State_NewYork', data=df5).fit()


# In[67]:


#Generating R-Squared and AIC values
(final_model1.rsquared,final_model1.rsquared_adj,final_model1.aic)


# # Cook's Distance for 2nd model
# 

# In[68]:


model_influence_f1 = final_model1.get_influence()
(c_f1, _) = model_influence_f1.cooks_distance


# In[69]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df5)),np.round(c_f1,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[70]:


#index of the data points where c is more than .5
(np.argmax(c_f1),np.max(c_f1))


# In[71]:


df5[df5.index.isin([14,38,40])]


# In[73]:


df6=df5.drop(df4.index[[14, 38, 40]],axis=0).reset_index()
df6


# In[74]:


df6=df6.drop(['index'],axis=1)
df6


# In[75]:


final_model2=smf.ols('Profit~rnd_Spend+Administration+Marketing_Spend+State_Florida+State_NewYork', data=df6).fit()


# In[76]:


#Generating R-Squared and AIC values
(final_model2.rsquared,final_model2.rsquared_adj,final_model2.aic)


# # Cook's Distance for 3rd model
# 

# In[77]:


model_influence_f2 = final_model2.get_influence()
(c_f2, _) = model_influence_f2.cooks_distance
fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df6)),np.round(c_f2,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[78]:


#index of the data points where c is more than .5
(np.argmax(c_f2),np.max(c_f2))


# # Since the C value (0.1) is < 1 , we can stop the diagnostic process and finalize the model

# Also comapring the 1st model's and final model's R square and AIC values we understand that R square value has increased and AIC value has decreased in Final model, which states that the Final model is good and ready to predict for new data points.

# # Predicting for new data

# In[81]:


#New data for prediction
new_data1=pd.DataFrame({'rnd_Spend':142107.34,'Administration':91391.77,'Marketing_Spend':366168.42,'State_Florida':1,"State_NewYork":0},index=[1])
final_model2.predict(new_data1)


# In[ ]:




