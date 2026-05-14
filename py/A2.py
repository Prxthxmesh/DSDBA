#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


data = {
    'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Math': [85, 90, np.nan, 40, 500, 78, 89, 95],
    'Science': [88, 92, 85, np.nan, 91, 76, 84, 200],
    'English': [75, 80, 79, 82, 81, np.nan, 77, 78]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# In[25]:


sns.scatterplot(data=df[['Math','English']])


# In[18]:


sns.boxplot(data=df[['Math','Science','English']])


# In[40]:


df.info()


# In[41]:


print(df.dtypes)


# In[42]:


df.describe()


# In[48]:


df.isnull().sum()


# In[70]:


df['Math']=df['Math'].fillna(df['Math'].mean())
df['Science']=df['Science'].fillna(df['Science'].mean())
df['English']=df['English'].fillna(df['English'].mean())
print("\nDataset After Handling Missing Values:")
df


# In[71]:


Q1 = df['Math'].quantile(0.25)
Q3 = df['Math'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)


# In[51]:


# Detect outliers

outliers = df[(df['Math'] < lower_limit) |
              (df['Math'] > upper_limit)]

print("\nOutliers in Math:")
print(outliers)


# In[72]:


meadin=df['Math'].median()
# syntax np.where(condition , if_true,if_false)
df['Math']=np.where(((df['Math']<=lower_limit) | (df['Math']>=upper_limit)),meadin,df['Math']) 


# In[73]:


df


# In[74]:


# if u want to remove the outliers
# df = df[(df["Math"] >= lower) & (df["Math"] <= upper)]
# print(df)


# In[76]:


df["Math_Normalized"] = (df["Math"] - df["Math"].min())/(df["Math"].max() - df["Math"].min())
print(df[["Math", "Math_Normalized"]])


# In[28]:


plt.hist(df['Math'])

plt.title("Marks Distribution")

plt.show()


# In[ ]:




