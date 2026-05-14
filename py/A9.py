#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


df[(df['age']>20) & (df['pclass']<3)]


# In[3]:


df = sns.load_dataset('titanic')

print("First 5 Rows:")
print(df.head())


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df['age']=df['age'].fillna(df['age'].mean())


# In[6]:


df.isnull().sum()


# In[7]:


sns.boxplot(x = 'sex', y = 'age', hue = 'survived', data = df)
plt.title("Age Distribution by Gender and Survival")
plt.show()


# In[12]:


#A Violin Plot is a graph used to show: Distribution of data Spread of data Density of data

sns.violinplot(x='sex',y='age',data=df, hue= 'survived')
#What does hue do?
#Adds subgroup categorization using colors.


# In[16]:


sns.countplot(x='sex', hue='survived', data=df)


# In[9]:


#Histogram	         Box Plot
#Shows distribution	 Shows spread and outliers
#Uses bars	         Uses box structure


# In[6]:


plt.hist(df['age'])


# In[5]:


df['age'].skew()


# In[ ]:


#Skewness: Value	                   Meaning
#          0	                       Perfectly symmetric
#          > 0	                   Positive skew (right skew)
#          < 0	                   Negative skew (left skew)


# In[ ]:


#Scatter Plot	               Line Plot
#Shows individual points	   Shows connected trend
#Used for relationships	       Used for continuous trends

#Box Plot	                    Violin Plot
#Shows quartiles and outliers	Shows distribution shape also
#Simpler	                        More detailed

