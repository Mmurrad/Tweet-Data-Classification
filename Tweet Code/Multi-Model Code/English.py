#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import zipfile
import os
from gensim.models import KeyedVectors
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, BatchNormalization, LSTM, Bidirectional, Activation, GRU, SpatialDropout1D


# In[4]:


file_path = '/home/student/code/Tweet/Dataset/CyberThreatData.csv'


# In[5]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[6]:


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    
    return ' '.join(tokens)


# In[7]:


def process_english_data(file_path):
    data = pd.read_csv(file_path)
    
    data['cleaned_tweet'] = data['Tweet'].apply(preprocess_text)
    
    max_words = 5000
    max_len = 100
    
    X = data['cleaned_tweet'].tolist()
    y = data['Analysis'].values
    
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)
    
    tokenizer = Tokenizer(num_words = max_words, oov_token = "<OOV>")
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen = max_len, padding = 'post')
    
    word2vec_model = KeyedVectors.load_word2vec_format('/home/student/code/Tweet/word2vec/GoogleNews-vectors-negative300.bin', binary = True)
    embedding_dim = 300

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, index in tokenizer.word_index.items():
        if word in word2vec_model:
            embedding_matrix[index] = word2vec_model[word]
            
    return data, X_pad, y, embedding_dim, embedding_matrix, tokenizer


# In[8]:


data = process_english_data(file_path)


# In[9]:


data


# In[ ]:





# In[ ]:




