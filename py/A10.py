#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


df = pd.read_csv("iris.csv")

print("First 5 Rows:")
print(df.head())


# In[3]:


df.info()


# In[11]:


print(df.dtypes)


# In[12]:


df.describe()


# In[13]:


# 1. Features and Datatypes
# ------------------------------------------------

print("\nFeatures in Dataset:")
print(df.columns)

print("\nDatatypes:")
print(df.dtypes)


# In[20]:


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes.flatten()

sns.histplot(df['SepalLengthCm'], ax=ax[0])
ax[0].set_title("Sepal Length")

sns.histplot(df['SepalWidthCm'], ax=ax[1])
ax[1].set_title("Sepal Width")

sns.histplot(df['PetalLengthCm'], ax=ax[2])
ax[2].set_title("Petal Length")

sns.histplot(df['PetalWidthCm'], ax=ax[3])
ax[3].set_title("Petal Width")

plt.tight_layout()

plt.show()


# In[21]:


# 3. Boxplot for Each Feature
# ------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes.flatten()

# Boxplot for Sepal Length

sns.boxplot(x=df['SepalLengthCm'], ax=ax[0])

ax[0].set_title("Sepal Length")

# Boxplot for Sepal Width

sns.boxplot(x=df['SepalWidthCm'], ax=ax[1])

ax[1].set_title("Sepal Width")

# Boxplot for Petal Length

sns.boxplot(x=df['PetalLengthCm'], ax=ax[2])

ax[2].set_title("Petal Length")

# Boxplot for Petal Width

sns.boxplot(x=df['PetalWidthCm'], ax=ax[3])

ax[3].set_title("Petal Width")

plt.tight_layout()

plt.show()


# In[ ]:


#Histogram 
#Shows frequency distribution of data.
#Used to understand:
#spread of values
#concentration of values


# In[ ]:


#Boxplot
#median
#quartiles
#spread
#outliers


# In[ ]:


#What are Outliers?
#Values far away from normal values.
#Displayed as points outside boxplot.


# In[ ]:


#Numeric	        Nominal
#Numbers	        Categories
#Example: age	Example: species


# In[ ]:


#What is subplot?
#Multiple plots inside one figure.


# In[ ]:


#Why use flatten()?
#For easy indexing of subplot axes.Converts 2D axes array into simple 1D array for easy indexing.


# In[ ]:


#Skewness: Value	                   Meaning
#          0	                       Perfectly symmetric
#          > 0	                   Positive skew (right skew)
#          < 0	                   Negative skew (left skew)

