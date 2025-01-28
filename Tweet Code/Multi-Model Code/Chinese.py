#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from nltk.corpus import stopwords
from nltk.util import ngrams
import re
import jieba
import requests

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix

import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import jieba
import re


# In[2]:


file_path = '/home/student/code/Tweet/Dataset/C-Tweets_classified.xlsx'


# In[3]:


def preprocess_text_chinese(text):
    
    url = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"

    response = requests.get(url)
    with open('chinese_stopwords.txt', 'wb') as f:
        f.write(response.content)

    with open('chinese_stopwords.txt', 'r', encoding='utf-8') as f:
        chinese_stopwords = set(line.strip() for line in f)
    
    # Remove special characters and links
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[A-Za-z]', '', text)
    text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5]', '', text)  # Retain Chinese characters and numbers

    # Word segmentation using jieba
    tokens = jieba.cut(text)
    tokens = [word for word in tokens if word not in chinese_stopwords]  # Remove stopwords

    return ' '.join(tokens)


# In[4]:


def process_chinese_data(file_path):
    data = pd.read_excel(file_path)
    
    data['Text'] = data['Text'].apply(preprocess_text_chinese)
    
    max_words = 5000  
    max_len = 100  

    X = data['Text'].tolist()
    y = data['Class'].values
    
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')
    
    embedding_index = {}
    with open('/home/student/code/Tweet/word2vec/glove.840B.300d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                # Check if all vector elements are valid floats
                coefs = np.asarray([float(val) for val in values[1:]], dtype='float32')
                embedding_index[word] = coefs
            except ValueError:
                # Skip the line if it contains non-numeric values
                print(f"Skipping line for word: {word} due to non-numeric vector values.")
                continue

    embedding_dim = 300
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    
    for word, index in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    
    return data, X_pad, y, embedding_dim, embedding_matrix, tokenizer


# In[5]:


data = process_chinese_data(file_path)


# In[6]:


data


# In[ ]:




