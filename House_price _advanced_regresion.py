#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
import scipy.stats
from sklearn.preprocessing import StandardScaler
from pycaret.regression import setup,compare_models
from sklearn.model_selection import KFold,cross_val_score

from sklearn.linear_model import BayesianRidge,Ridge,OrthogonalMatchingPursuit
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor,ExtraTreesRegressor




# In[2]:


path="C:\\Users\\admin\\Desktop\\DS Docs\\house-prices-advanced-regression-techniques\\train.csv"
path1="C:\\Users\\admin\\Desktop\\DS Docs\\house-prices-advanced-regression-techniques\\test.csv"
path2="C:\\Users\\admin\\Desktop\\DS Docs\\house-prices-advanced-regression-techniques\\sample_submission.csv"

train=pd.read_csv(path)
test=pd.read_csv(path1)
sample_submissions=pd.read_csv(path2)


# In[3]:


train


# In[4]:


test


# # Data Cleaning

# In[5]:


pd.options.display.max_rows=90
train.isna().sum()


# In[6]:


test.isna().sum()


# In[7]:


#ombine train and test data for cleaning
target=train['SalePrice']
test_ids=test['Id']

train1=train.drop(['Id','SalePrice'],axis=1)
test1=test.drop('Id',axis=1)

data1=pd.concat([train1,test1],axis=0).reset_index(drop=True)


# In[8]:


data1


# In[9]:


target


# In[10]:


#Data types
data1.select_dtypes(np.number)


# In[11]:


data2=data1.copy()


# In[12]:


# Changing Data Type Of MSSubClass Column As It Is A Categorical Column
data2['MSSubClass']=data2['MSSubClass'].astype(str)


# In[13]:


#Missing value teatment for catogorial data

#Some features have meaning to na values ex. In Alley column has na for no alley

#Impute using constant value

for column in[
    'Alley',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature'
    
    
    
    
]:
    data2[column]=data2[column].fillna('None')

#Impute using column mode
for column in[
    'MSZoning',
    'Utilities',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Electrical',
    'KitchenQual',
    'Functional',
    'SaleType'
    
    
    
]:
    data2[column]=data2[column].fillna(data2[column].mode()[0])



# In[14]:


data2.select_dtypes('object').loc[:,data2.isna().sum()>0].columns


# In[15]:


data2.select_dtypes('object').isna().sum().sum()


# In[16]:


data2.select_dtypes(np.number).isna().sum().sum()


# In[17]:


data3=data2.copy()


# In[18]:


##Missing value treatment for numeric data
#Imputing numeric missing values using Regression "(KNN imputation)"

data3.select_dtypes(np.number).isna().sum()


# In[19]:


data3.loc[data3['LotFrontage'].isna()==False,'LotFrontage']


# In[20]:


def knn_impute(df,na_target):
    df=df.copy()
    
    numeric_df=df.select_dtypes(np.number)
    non_na_cloumns=numeric_df.loc[:,numeric_df.isna().sum()==0].columns
    
    y_train=numeric_df.loc[numeric_df[na_target].isna()==False,na_target]
    X_train=numeric_df.loc[numeric_df[na_target].isna()==False,non_na_cloumns]
    X_test=numeric_df.loc[numeric_df[na_target].isna()==True,non_na_cloumns]
    
    knn=KNeighborsRegressor()
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    
    df.loc[df[na_target].isna()==True,na_target]=y_pred

    
    return df


# In[21]:


knn_impute(data3,'LotFrontage')


# In[22]:


knn_impute(data3,'LotFrontage').isna().sum()


# In[23]:


data3.columns[data3.isna().sum()>0]


# In[24]:


for column in [
    'LotFrontage',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    'BsmtFullBath',
    'BsmtHalfBath',
    'GarageYrBlt',
    'GarageCars',
    'GarageArea'
]:
    data3=knn_impute(data3,column)


# In[25]:


data3.isna().sum()


# In[26]:


data4=data3.copy()


# # Feature Transformation
#    Feature transformation is done to remove skewness in data distribution.
#    Use log transform to remove skewness from data.Skewness above 0.5 needs to be transformed to normal distribution

# In[27]:


data3.select_dtypes(np.number)


# In[28]:


scipy.stats.skew(data3['LotFrontage'])


# In[29]:


#Creating a new dataframe for skew values of all columns with numeric data
skew_df=pd.DataFrame(data4.select_dtypes(np.number).columns,columns=['Feature'])
skew_df['Skew']=skew_df['Feature'].apply(lambda feature: scipy.stats.skew(data4[feature]))
skew_df['Absolute_Skew']=skew_df['Skew'].apply(abs) #Neglecting the signs of skew value (negative or positive skew)
skew_df['Skewed']=skew_df['Absolute_Skew'].apply(lambda x:True if x>=0.5 else False)
skew_df


# In[30]:


skew_df.query("Skewed==True")['Feature'].values


# In[31]:


for column in skew_df.query("Skewed==True")['Feature'].values:
    data4[column]=np.log1p(data4[column])


# In[32]:


skew_df=pd.DataFrame(data4.select_dtypes(np.number).columns,columns=['Feature'])
skew_df['Skew']=skew_df['Feature'].apply(lambda feature: scipy.stats.skew(data4[feature]))
skew_df['Absolute_Skew']=skew_df['Skew'].apply(abs) 
skew_df['Skewed']=skew_df['Absolute_Skew'].apply(lambda x:True if x>=0.5 else False)
skew_df


# In[33]:


#Month column needs to be processed as they are cyclic
#December=12 is close to January=1
#Apply sine or cosine trnsform to add cyclic effect
data4['MoSold'].unique()


# In[34]:


#0.5236 is added to align the 0 and 12 to the point on the graph or the transform value
-np.cos(0.5236*data4['MoSold'])


# In[35]:


print(-np.cos(0.5236*12)) #cos trnsform for December
print(-np.cos(0.5236*1))  #cos trnsform for January
print(-np.cos(0.5236*6))  #cos trnsform for June


# In[36]:


data4['MoSold']=(-np.cos(0.5236*data4['MoSold']))


# In[37]:


data4['MoSold']


# In[38]:


data5=data4.copy()


# # Emcoding Catogorial Data

# In[39]:


data5=pd.get_dummies(data5)
data5


# In[40]:


data6=data5.copy()


# # Scaling

# In[41]:


scaler=StandardScaler()
scaler.fit(data6)

data6=pd.DataFrame(scaler.transform(data6), index=data6.index, columns=data6.columns)
data6


# In[42]:


data7=data6.copy()


# # Target Transformatio

# In[43]:


#Target without log trnsformation
plt.figure(figsize=(20,10))
sns.displot(target,kde=True)
plt.title("Without log transform")
plt.show()


# In[44]:


#Target with log trnsformation
plt.figure(figsize=(20,10))
sns.displot(np.log(target),kde=True)
plt.title("With log transform")
plt.show()


# In[45]:


log_target=np.log(target)
log_target


# # Split train and test data

# In[46]:


train.index.max()


# In[47]:


data6.loc[:train.index.max(),:]


# In[48]:


data6.loc[train.index.max()+1:,:]


# In[49]:


train_final=data7.loc[:train.index.max(),:].copy()
test_final=data7.loc[train.index.max()+1:,:].reset_index(drop=True).copy()


# In[50]:


train_final


# In[51]:


test_final


# # Model Selection

# In[52]:


log_target


# In[53]:


_=setup(data=pd.concat([train_final,log_target],axis=1),target='SalePrice')


# In[54]:


compare_models()


# # Baseline model

# from sklearn.linear_model import BayesianRidge,Ridge,OrthogonalMatchingPursuit
# from lightgbm import LGBMRegressor
# from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor,ExtraTreesRegressor

# In[55]:


baseline_model=GradientBoostingRegressor()
baseline_model.fit(train_final,log_target)


# In[56]:


br=BayesianRidge()
br.fit(train_final,log_target)


# In[ ]:





# # Evaluate

# In[57]:


kf=KFold(n_splits=10)
result=cross_val_score(baseline_model,train_final,log_target,scoring='neg_mean_squared_error',cv=kf)


# In[58]:


-result


# In[59]:


kf=KFold(n_splits=10)
result1=cross_val_score(br,train_final,log_target,scoring='neg_mean_squared_error',cv=kf)


# In[60]:


-result1


# In[61]:


np.exp(np.sqrt(np.mean(-result))) #Gradient Boosting


# In[62]:


np.exp(np.sqrt(np.mean(-result1))) #Baysian Ridge


# In[63]:


#GBR
plt.figure(figsize=(10,10)) 

sns.distplot(-result,kde=True)


# In[64]:


#BR
plt.figure(figsize=(10,10)) 

sns.distplot(-result1,kde=True)


# # Make submission

# In[65]:


test_ids


# In[66]:


sample_submissions


# In[67]:


submission=pd.concat([test_ids,pd.Series(final_predictions,name='SalePrice')],axis=1)
submission


# In[ ]:


submission.to_csv('./submission1.csv',index=False,header=True)


# # Bagging Enemble

# In[68]:


models={
    "br":BayesianRidge(),
    "gbr":GradientBoostingRegressor(),
    "lightgbm":LGBMRegressor(),
    "Ridge":Ridge(),
    "omp":OrthogonalMatchingPursuit()
}


# In[69]:


for name, model in models.items():
    model.fit(train_final,log_target)
    print(name+" trained")


# In[70]:


results={}
kf=KFold(n_splits=10)

for name, model in models.items():
    result=np.exp(np.sqrt(-cross_val_score(baseline_model,train_final,log_target,scoring='neg_mean_squared_error',cv=kf)))
    results[name]=result
results


# In[71]:


plt.figure(figsize=(20,10))

for name, model in models.items():
    sns.displot(results[name],kde=True,label=name)
    
plt.title("CV Error Distribution")
plt.show()
        


# In[72]:


for name, result in results.items():
    print("---------\n"+name+"\n---------\n")
    print(np.mean(result))
    print(np.std(result))


# # Combine results from all selected models

# In[73]:


final_predictions=(
    0.2*np.exp(models['br'].predict(test_final))+
    0.2*np.exp(models['gbr'].predict(test_final))+
    0.2*np.exp(models['lightgbm'].predict(test_final))+
    0.2*np.exp(models['Ridge'].predict(test_final))+
    0.2*np.exp(models['omp'].predict(test_final))
)
final_predictions


# In[76]:


bagging_submission=pd.concat([test_ids,pd.Series(final_predictions,name='SalePrice')],axis=1)


# In[77]:


bagging_submission


# In[79]:


bagging_submission.to_csv("C:\\Users\\admin\\Desktop\\GITHUB Projects\\KK\\House Price Regression Advanced Techniques\\final_sub.csv",index=False,header=True)


# In[ ]:




