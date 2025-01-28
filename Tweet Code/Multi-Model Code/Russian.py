#!/usr/bin/env python

import numpy as np
import pandas as pd
import re
import joblib
import nltk
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec, KeyedVectors
import gensim


# In[3]:


file_path = '/home/student/code/Tweet/Dataset/R-Tweets. Taras Ignatiuk.xlsx'
#w2v_model_path = '/home/student/code/Tweet/word2vec/tweets_model.w2v'


# In[3]:


nltk.download('stopwords')
nltk.download('punkt')


# In[4]:


def preprocess_text_russian(text):
    russian_stopwords = set(stopwords.words('russian'))
    
    text = re.sub(r"http\s+|www\s+|https\s", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^А-Яа-яёЁ ]+', '', text)
    text = text.lower()
    tokens = word_tokenize(text, language ='russian')
    tokens = [word for word in tokens if word not in russian_stopwords]
    return ' '.join(tokens)


# In[5]:


def process_russian_data(file_path):
    data = pd.read_excel(file_path)
    data.rename(columns = {data.columns[3]: 'Label'}, inplace = True)
    
    data['cleaned_data'] = data['Text'].apply(preprocess_text_russian)
    
    max_words = 5000
    max_len = 100
    
    NUM = 100000
    
    X = data['cleaned_data'].tolist()
    y = data['Label'].values
    
    tokenizer = Tokenizer(num_words = max_words, oov_token = "<OOV>")
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen = max_len, padding = 'post')
    
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    embedding_dim = 300

    embedding_matrix = np.zeros((len(tokenizer.word_index)+1, embedding_dim))

    return data, X_pad, y_encoded, embedding_dim, embedding_matrix, tokenizer

data = process_russian_data(file_path)

