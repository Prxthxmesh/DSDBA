#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


data = {
    'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Gender': ['Male', 'Female', 'Male', 'Female',
               'Male', 'Female', 'Male', 'Female'],
    'Age': [21, 22, 20, 23, 21, 22, 20, 23],
    'Income': [25000, 30000, 27000, 32000,
               29000, 31000, 26000, 33000]
}
df=pd.DataFrame(data)
df


# In[16]:


sns.scatterplot(x='Age',y='Income',data=df)


# In[14]:


plt.plot('Age','Income',data=df)


# In[10]:


sns.barplot(x='Name',y='Income',data=df)
plt.show()


# In[5]:


grouped = df.groupby('Gender')['Income']


# In[6]:


print("Mean: ")
print(grouped.mean())


# In[7]:


print("Median: ")
print(grouped.median())


# In[10]:


print("Min: ")
print(grouped.min())


# In[11]:


print("Max: ")
print(grouped.max())


# In[12]:


print("std: ")
print(grouped.std())


# In[14]:


#Create List for Each Category
male_income = df[df['Gender'] == 'Male']['Income'].tolist()

female_income = df[df['Gender'] == 'Female']['Income'].tolist()

print("\nMale Income List:")
print(male_income)

print("\nFemale Income List:")
print(female_income)


# In[51]:


import seaborn as sns


# In[18]:


#iris=sns.load_dataset('iris')


# In[52]:


dff=pd.read_csv('iris.csv')


# In[53]:


dff


# In[54]:


dff.describe()


# In[55]:


dff['Species'].unique()


# In[58]:


dff['Species']=dff['Species'].replace({'Iris-virginica' : 'virginica' , 'Iris-setosa': 'setosa','Iris-versicolor':'versicolor'})


# In[59]:


dff


# In[77]:


setosa=dff[dff['Species']=='setosa']


# In[78]:


setosa.describe()


# In[79]:


virginica=dff[dff['Species']=='virginica']


# In[80]:


virginica.describe()


# In[81]:


versicolor=dff[dff['Species']=='versicolor']


# In[82]:


versicolor.describe()


# In[24]:


sns.lineplot(x='Age',y='Income',data=df)

plt.title("Income Distribution")

plt.show()


# In[ ]:




