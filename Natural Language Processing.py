#!/usr/bin/env python
# coding: utf-8

# In[22]:


# IMPORTING MODULES


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[23]:


# IMPORTING DATASETS


# In[3]:


dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)


# In[4]:


dataset


# In[21]:


# CLEANING THE TEXTS


# In[9]:


import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
     review = re.sub('[^a-zA-Z]' , ' ', dataset['Review'][i])
     review = review.lower()
     review = review.split()
        
     ps = PorterStemmer()
     all_stopwords = stopwords.words('english')
     all_stopwords.remove('not')
     review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
     review = ' '.join(review)
     corpus.append(review)


# In[10]:


print(corpus)


# In[20]:


# CREATING BAG OF WORDS MODEL


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values


# In[15]:


len(X[0])


# In[18]:


# SPLITTING TRAINING SET AND TEST SET


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[19]:


# NAIVE BAYES MODEL


# In[26]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[27]:


# PREDICTING TEST RESULTS


# In[28]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[32]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[ ]:




