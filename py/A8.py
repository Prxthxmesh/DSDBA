#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = sns.load_dataset('titanic')

print("First 5 Rows:")
print(df.head())


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df['age']=df['age'].fillna(df['age'].mean())


# In[8]:


df.isnull().sum()


# In[9]:


# Count Plot
# Shows number of survived and non-survived passengers
# ------------------------------------------------

sns.countplot(x='survived', data=df)

plt.title("Survived vs Not Survived")

plt.show()


# In[10]:


# Count Plot based on Gender
# ------------------------------------------------

sns.countplot(x='survived', hue='sex', data=df)

plt.title("Survival Count based on Gender")

plt.show()


# In[11]:


# Count Plot based on Passenger Class
# ------------------------------------------------

sns.countplot(x='survived', hue='class', data=df)

plt.title("Survival Count based on Passenger Class")

plt.show()


# In[4]:


plt.hist(df['fare'])
plt.show()


# In[12]:


# Histogram of Fare Shows distribution of continuous data.
# Shows distribution of ticket prices
# ------------------------------------------------

plt.hist(df['fare'])

plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.title("Distribution of Fare")

plt.show()


# In[19]:


#Shows:minimum value maximum value median quartiles outliers 
#Middle line → Median
#Box → Interquartile Range (IQR)
#Dots outside → Outliers
sns.boxplot(x='class',y='age',data=df) 
plt.show()


# In[16]:


#Shows relationship between two variables.,dots represent each passenger
sns.scatterplot(x='age',y='fare',data=df)
plt.show()


# In[17]:


sns.scatterplot(x='age',y='fare',data=df,hue='survived')
plt.show()


# In[24]:


#A Violin Plot is a graph used to show: Distribution of data Spread of data Density of data
plt.figure(figsize=(10,8))
vp = sns.violinplot(x='age',y='fare',palette='rainbow',data=df)
plt.show()


# In[7]:


plt.hist(df['fare'])


# In[8]:


df['fare'].skew()


# In[ ]:


#Skewness: Value	                   Meaning
#          0	                       Perfectly symmetric
#          > 0	                   Positive skew (right skew)
#          < 0	                   Negative skew (left skew)


# In[ ]:


#Histogram	         Box Plot
#Shows distribution	 Shows spread and outliers
#Uses bars	         Uses box structure


# In[ ]:


#Scatter Plot	               Line Plot
#Shows individual points	   Shows connected trend
#Used for relationships	       Used for continuous trends


# In[ ]:


#What does hue do?
#Adds subgroup categorization using colors.


# In[ ]:


#Box Plot	                    Violin Plot
#Shows quartiles and outliers	Shows distribution shape also
#Simpler	                        More detailed

