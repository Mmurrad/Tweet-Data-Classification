#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import re
from sklearn.model_selection import train_test_split

from nltk.stem.isri import ISRIStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec


# In[3]:


import os
print(os.getcwd())


# In[4]:


file_path = '/home/student/code/Tweet/Dataset/Tweet-Arabic.xls'


# In[5]:


nltk.download('stopwords')


# In[6]:


def preprocess_text(text):
    
    arabic_stopwords = set(stopwords.words('arabic'))
    stemmer = ISRIStemmer()
    
    
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^ء-ي\s]', '', text)  

    
    tokens = word_tokenize(text)
    
    
    tokens = [stemmer.stem(word) for word in tokens if word not in arabic_stopwords]
    
    return ' '.join(tokens)


# In[9]:


def process_arabic_data(file_path):
    data = pd.read_excel(file_path)
    data.rename(columns = {data.columns[3]: 'Class'}, inplace = True)
    
    data['Cleaned_data'] = data['Text'].apply(preprocess_text)
    
    max_words = 5000
    max_len = 100

    X = data['Cleaned_data'].tolist()
    y = data['Class'].values
    
    tokenizer = Tokenizer(num_words = max_words, oov_token = "<OOV>")
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen = max_len, padding = 'post')
    
    sentences = data['Cleaned_data'].apply(lambda x: x.split()).tolist()

    word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=2, workers=4)
    
    embedding_dim = 300

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))

    for word, index in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[index] = word2vec_model.wv[word]


    return data, X_pad, y, embedding_dim, embedding_matrix, tokenizer


# In[10]:


data = process_arabic_data(file_path)


# In[ ]:




