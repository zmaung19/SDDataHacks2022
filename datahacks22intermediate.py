#!/usr/bin/env python
# coding: utf-8

# In[83]:


import re
import string
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[84]:


from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# In[85]:


stop = stopwords.words('english')


# In[86]:


# Import CSV file

df = pd.read_csv("intermediate_trainset - intermediate_trainset.csv")
print(df.head())
text = df


# In[87]:


# Cleaning Data

text.Pitched_Business_Desc.str.replace('[^a-zA-Z0-9]', '') # removing all characters except for alphanumeric characters
text.Pitched_Business_Desc.str.replace(r'[^\w\s]+', '')
# df['Pitched_Business_Desc'] = df['text'].str.replace('[^a-zA-Z0-9]', '')
# df['Pitched_Business_Desc'] = df['text'].str.replace(r'[^\w\s]+', '')
df_success = text.dropna() #create a new dataframe with only pitches with successful deals
df_success['Pitched_Business_Desc'] = df_success['Pitched_Business_Desc'].str.lower()
df_success['Pitched_Business_Desc_without_StopWords'] = df_success['Pitched_Business_Desc'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(df_success)
print(df_success['Pitched_Business_Desc_without_StopWords'])


# In[101]:


#Exploratory Data Analysis

word_count = Counter(" ".join(df_success['Pitched_Business_Desc_without_StopWords']).split()).most_common(20) #finding the 20 most common words in successful deals
word_frequency = pd.DataFrame(word_count, columns = ['Word', 'Frequency'])
print(word_frequency)


# In[90]:


#Data Manipulation
x_train, x_test, y_train, y_test = train_test_split(df["Pitched_Business_Desc"], df["Deal_Status"])
x_train.shape, x_test.shape


# In[92]:


text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC(C=0.1))])
text_clf.fit(x_train, y_train)


# In[93]:


#Results
print(text_clf.score(x_train, y_train))
print(text_clf.score(x_test, y_test))


# In[ ]:




