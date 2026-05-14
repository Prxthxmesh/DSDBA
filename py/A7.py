#!/usr/bin/env python
# coding: utf-8

# In[6]:


import nltk
import pandas as pd

# Tokenization
from nltk.tokenize import word_tokenize

# POS Tagging
from nltk import pos_tag

# Stopwords Removal
from nltk.corpus import stopwords

# Stemming
from nltk.stem import PorterStemmer

# Lemmatization
from nltk.stem import WordNetLemmatizer

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('punkt_tab')


# In[12]:


# Download Required Packages
# ------------------------------------------------

nltk.download('punkt') # used for tokenization
nltk.download('averaged_perceptron_tagger_eng') # pos tagging
nltk.download('stopwords')
nltk.download('wordnet')#lemmatization


# In[3]:


# Sample Text Document
# ------------------------------------------------

text = "Data Science is very useful and Machine Learning is part of Data Science."

print("Original Text:\n")
print(text)


# In[7]:


# 1. Tokenization
# Splits sentence into words
# ------------------------------------------------

tokens = word_tokenize(text)

print("\nTokens:")
print(tokens)


# In[13]:


# 2. POS Tagging
# Gives grammatical tags to words
# ------------------------------------------------

pos = pos_tag(tokens)

print("\nPOS Tagging:")
print(pos)


# In[14]:


# 3. Stopwords Removal
# Removes common words like is, and, of
# ------------------------------------------------

stop_words = stopwords.words('english')

filtered_words = []

for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print("\nAfter Stopword Removal:")
print(filtered_words)


# In[15]:


# 4. Stemming
# Converts word into root form
# ------------------------------------------------

ps = PorterStemmer() #playing -> play studies -> studi

stemmed_words = []

for word in filtered_words:
    stemmed_words.append(ps.stem(word))

print("\nStemmed Words:")
print(stemmed_words)


# In[16]:


# 5. Lemmatization
# Converts word into meaningful base word
# ------------------------------------------------

lemmatizer = WordNetLemmatizer() #better -> good playing -> play

lemmatized_words = []

for word in filtered_words:
    lemmatized_words.append(lemmatizer.lemmatize(word))

print("\nLemmatized Words:")
print(lemmatized_words)


# In[18]:


# 6. TF-IDF
# Calculates importance of words
# ------------------------------------------------

documents = [text] #TF-IDF requires documents in list format.

vectorizer = TfidfVectorizer() #Creates TF-IDF object.

tfidf = vectorizer.fit_transform(documents) #fit() Learns words from document.
#transform() - Converts words into TF-IDF numerical values.
# Convert into table format
df = pd.DataFrame(
    tfidf.toarray(),#Converts matrix into array.
    columns=vectorizer.get_feature_names_out() #Gets all words as column names.
)

print("\nTF-IDF Representation:")
print(df)


# In[ ]:


#NLTK is used for Natural Language Processing (NLP).
#It provides functions for:
#tokenization
#POS tagging
#stemming
#stopword removal


# In[ ]:


#Stemming	                      Lemmatization
#May produce incorrect word	      Produces meaningful word
#Faster	                           More accurate

