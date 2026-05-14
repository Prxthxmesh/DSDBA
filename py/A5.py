#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[2]:


df=pd.read_csv(r"C:\Users\prath\Desktop\SEM-6\DSDBA PRAC\Social_Network_Ads.csv")
df


# In[3]:


X = df[['Age', 'EstimatedSalary']] #Independent Variables input features used for prediction
y = df['Purchased'] #Dependent Variable o/p we want to predict ,Target variable: 1 → purchased ,0 → not purchased


# In[4]:


# Split Dataset :Splits dataset into:Training Data->Used to train model. 
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,# 20% data-testing , # 80% data -training
    random_state=42  #Ensures same random split every time program runs. Used for reproducibility.
)


# In[5]:


model = LogisticRegression()

# Train Model :Trains model using training data. The model learns patterns from dataset

model.fit(X_train, y_train)
#fit() helps model understand relationship between:~features ~house prices


# In[7]:


# Prediction
# -----------------------------------------

y_pred = model.predict(X_test) #Predicts house prices using testing data.

print("\nPredicted Values:")
print(y_pred)


# In[9]:


# Confusion Matrix Used to evaluate classification model.
# -----------------------------------------

cm = confusion_matrix(y_test, y_pred) 

print("\nConfusion Matrix:")
print(cm)


# In[11]:


# TP, TN, FP, FN
# -----------------------------------------

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]
print("\nTrue Positive:", TP)
print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)


# In[12]:


# Accuracy Measures overall correctness.
# -----------------------------------------

accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\nAccuracy:", accuracy)


# In[13]:


# Error Rate  Measures incorrect predictions.
# -----------------------------------------

error_rate = (FP + FN) / (TP + TN + FP + FN)

print("Error Rate:", error_rate)


# In[14]:


# Precision Measures prediction quality.
# -----------------------------------------

precision = TP / (TP + FP)

print("Precision:", precision)


# In[15]:


# Recall Measures ability to identify positives.
# -----------------------------------------Precision -How many predicted positives are correct.
#Recall -How many actual positives are identified.
recall = TP / (TP + FN)

print("Recall:", recall)


# In[ ]:


#Heatmap
#A graphical representation using colors.
#darker color → larger values
#lighter color → smaller values


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
#cm -Confusion matrix data
#annot=True -Displays numbers inside boxes.Without this:only colors appear.
#fmt='d' Formats values as integers. d=decimal integer
#cmap='Blues' Sets color theme to blue shades.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


#False Positive = 2
#Model predicted:
#1 (Purchased)
#But actual was:
#0 (Not Purchased)

