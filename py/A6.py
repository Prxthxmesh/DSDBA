#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# In[3]:


df=pd.read_csv('iris.csv')
df


# In[4]:


# Features and Target
# -----------------------------------------

X = df.iloc[:, 1:5]  #Independent Variables input features used for prediction

y = df['Species'] #Dependent Variable o/p we want to predict ,


# In[5]:


# Split Dataset:Splits dataset into:Training Data->Used to train model
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,# 20% data-testing , # 80% data -training
    random_state=42 #Ensures same random split every time program runs. Used for reproducibility.
)


# In[6]:


# Create Naive Bayes Model
# -----------------------------------------

model = GaussianNB()

# Train Model:Trains model using training data. The model learns patterns from dataset

model.fit(X_train, y_train)
#fit() helps model understand relationship between:~features ~house prices


# In[8]:


# Prediction
# -----------------------------------------

y_pred = model.predict(X_test)#Predicts house prices using testing data.

print("\nPredicted Values:")
print(y_pred)


# In[9]:


# Confusion Matrix Used to evaluate classification model.
# -----------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# In[ ]:


#Heatmap
#A graphical representation using colors.
#darker color → larger values
#lighter color → smaller values


# In[10]:


# Heatmap
# -----------------------------------------
#cm -Confusion matrix data
#annot=True -Displays numbers inside boxes.Without this:only colors appear.
#fmt='d' Formats values as integers. d=decimal integer
#cmap='Blues' Sets color theme to blue shades.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# In[11]:


accuracy = np.sum(y_pred == y_test) / len(y_test) #Measures overall correctness.

print("\nAccuracy:", accuracy)

error_rate = 1 - accuracy # Measures incorrect predictions. Error Rate=1−Accuracy

print("Error Rate:", error_rate)


# In[12]:


# TP, TN, FP, FN
# ---------------------------------------------------

# For multiclass classification,
# we consider one class vs all others.

# Taking Iris-setosa as Positive Class

TP = cm[0][0]

FP = cm[1][0] + cm[2][0]

FN = cm[0][1] + cm[0][2]

TN = np.sum(cm) - (TP + FP + FN)

print("\nTrue Positive (TP):", TP)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)
print("True Negative (TN):", TN)


# In[13]:


# Precision Measures prediction quality.
# ---------------------------------------------------

precision = TP / (TP + FP)

print("\nPrecision:", precision)

# ---------------------------------------------------
# Recall Measures ability to identify positives
#Precision -How many predicted positives are correct.
#Recall -How many actual positives are identified.
# ---------------------------------------------------

recall = TP / (TP + FN)

print("Recall:", recall)


# In[17]:


#What is Naive Bayes?
#Classification algorithm based on Bayes theorem.
#Why called Naive?
#Assumes all features are independent.
#What is GaussianNB?
#Naive Bayes model for continuous numeric data.
#What is classification?
#Predicting categories/classes.


# In[ ]:


# Naive Bayes?
#It is mainly used for:
#Classification problems
#Prediction tasks
#Examples:
#Spam email detection
#Sentiment analysis
#Disease prediction

