#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv(r"C:\Users\prath\Desktop\SEM-6\DSDBA PRAC\Iris.csv")


# In[58]:


df


# In[37]:


print(df.head())


# In[38]:


print("\nDataset Information:")
print(df.info())


# In[6]:


print("\nShape of Dataset:")
print(df.shape)


# In[7]:


print("\nMissing Values:")
print(df.isnull().sum())


# In[8]:


print("\nStatistical Summary:")
print(df.describe())


# In[9]:


print("\nData Types:")
print(df.dtypes)


# In[12]:


print("Duplicated values")
df.duplicated().sum()


# In[46]:


# as id column is same as index so will use that 

df=df.set_index('Id')
df


# In[39]:


# to find unique names in species column to change it to numerical 
df['Species'].unique().sum()


# In[49]:


# to change the species from categorical to numerical
df['Species']=df['Species'].replace({
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2})


# In[50]:


df['Species']


# In[51]:


numerical_columns = ['SepalLengthCm', 'SepalWidthCm',
                     'PetalLengthCm', 'PetalWidthCm']

for column in numerical_columns:
    df[column] = (df[column] - df[column].min()) / \
                 (df[column].max() - df[column].min())

print("\nNormalized Dataset:")
print(df.head())


# In[62]:


dff=pd.read_csv(r"C:\Users\prath\Downloads\Titanic-Dataset.csv")


# In[63]:


dff.info()


# In[64]:


print(dff.isnull().sum())


# In[65]:


print(dff.dtypes)


# In[66]:


dff["Age"] = dff["Age"].fillna(dff["Age"].mean())


# In[67]:


print(dff.isnull().sum())


# In[75]:


dff["Age"] = dff["Age"].astype(int)


# In[72]:


dff["Sex"] = dff["Sex"].replace(['male','female'],[0,1])
dff


# In[73]:


dff


# In[5]:


plt.hist(df['SepalLengthCm'])

plt.title("Histogram of Sepal Length")

plt.show()


# In[6]:


sns.boxplot(x=df['SepalLengthCm'])

plt.title("Boxplot")

plt.show()


# In[7]:


plt.scatter(df['SepalLengthCm'], df['PetalLengthCm'])

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.show()


# In[ ]:




