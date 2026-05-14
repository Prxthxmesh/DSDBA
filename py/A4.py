#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().system('pip install pandas numpy seaborn matplotlib')
get_ipython().system('pip install -U scikit-learn')


# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #sklearn Libraries train_test_split Used to divide dataset into:training data testing data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r"C:\Users\prath\Desktop\SEM-6\DSDBA PRAC\BostonHousing.csv")


# In[9]:


df


# In[10]:


df.info()


# In[12]:


df.shape


# In[13]:


df.isnull().sum()


# In[15]:


print(df.dtypes)


# In[17]:


df=df.astype('float64')


# In[18]:


print(df.dtypes)


# In[5]:


# Features (Independent Variables) MEDV is used because it is the target variable in the Boston Housing dataset.
#Median Value of Owner-Occupied Homes (medv) It represents the house price.
X = df.drop('medv', axis=1) #X contains all input columns except medv.
#These columns help predict house prices.
#These are called:
#Independent Variables
#Predictor Variables
#Feature
# --- ----- 
# Target Variable (Dependent Variable)
y = df['medv']
#y stores output variable. medv = Median value of owner-occupied homes.
#This is what model predicts.
#Called:
#Dependent Variable
#Target Variable
#Output Variable


# In[ ]:


# why drop medv
#Removes target column from input features.
#Because:
#medv is output we want to predict.
#It should not be included in inputs.


# In[6]:


# Split Dataset :Splits dataset into:Training Data->Used to train model. 
#Testing Data->used to test model accuracy.
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,# 20% data-testing , # 80% data -training
    random_state=42 #Ensures same random split every time program runs. Used for reproducibility.
)


# In[8]:


model = LinearRegression()


# In[9]:


# Train Model , Trains model using training data. The model learns patterns from dataset
model.fit(X_train, y_train) 
#fit() helps model understand relationship between:~features ~house prices


# In[10]:


# Prediction
# -----------------------------------------

y_pred = model.predict(X_test) #Predicts house prices using testing data.

print("\nPredicted Values:")
print(y_pred)


# In[11]:


mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error:")
print(mse)
#Measures prediction error.
#MSE=1/n summation(y-ŷ)2
#y → actual value
#ŷ → predicted value

#Lower MSE means: Better model performance.


# In[30]:


score = model.score(X_test, y_test)

print("\nAccuracy (R2 Score):")
print(score)
#Measures accuracy of regression model.
#Range:
#0 to 1
#Closer to 1:
#better prediction


# In[31]:


result = pd.DataFrame({
    'Actual_Price': y_test,
    'Predicted_Price': y_pred
})

result.head()


# In[33]:


import matplotlib.pyplot as plt
#scatter plot
#Shows relationship between: #actual prices ,#predicted prices
#Each point represents one house.


# In[43]:


plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2) # Diagonal line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Boston House Prices')
plt.show()


# In[ ]:


#To predict house prices using Linear Regression.

The dataset contains house-related features such as:

crime rate
number of rooms
tax rate
pollution level

Target:

house price (medv)

#What is ideal regression result?

Points close to diagonal line.

