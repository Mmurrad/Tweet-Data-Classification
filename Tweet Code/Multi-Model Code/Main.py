#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nbimporter
import pandas as pd
import numpy as np
from English import process_english_data
from Arabic import process_arabic_data
from Russian import process_russian_data
from Chinese import process_chinese_data

import pickle
import numpy as np 

import nltk
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from keras.regularizers import l2
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

from tensorflow.keras.models import Model 

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, BatchNormalization, LSTM, Bidirectional, Activation, GRU, SpatialDropout1D, Input, GlobalAveragePooling1D


file_path = {
    'arabic': '/home/student/code/Tweet/Dataset/Tweet-Arabic.xls',
    'english': '/home/student/code/Tweet/Dataset/CyberThreatData.csv',
    'chinese':  '/home/student/code/Tweet/Dataset/C-Tweets_classified.xlsx',
    'russian': '/home/student/code/Tweet/Dataset/R-Tweets. Taras Ignatiuk.xlsx'
}


os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/student/anaconda3/pkgs/cuda-nvcc-11.6.124-hbba6d2d_0/nvvm/'


max_words = 5000
max_len = 100


arabic_data, arabic_X_pad, arabic_y, arabic_embedding_dim, arabic_embedding_matrix, arabic_tokenizer = process_arabic_data(file_path['arabic'])
eng_data, eng_X_pad, eng_y, eng_embedding_dim, eng_embedding_matrix, eng_tokenizer = process_english_data(file_path['english'])
china_data, china_X_pad, china_y, china_embedding_dim, china_embedding_matrix, china_tokenizer = process_chinese_data(file_path['chinese'])
rus_data, rus_X_pad, rus_y, rus_embedding_dim, rus_embedding_matrix, rus_tokenizer = process_russian_data(file_path['russian'])

print(f"Arabic: {arabic_embedding_dim} ")
print(f"English: {eng_embedding_dim}")
print(f"Chinese: {china_embedding_dim}")
print(f"Russian: {rus_embedding_dim}")

assert arabic_embedding_dim == eng_embedding_dim == china_embedding_dim == rus_embedding_dim, "Embedding dimensions must match."


combined_X_pad = np.concatenate([arabic_X_pad, eng_X_pad, china_X_pad,
                                rus_X_pad], axis = 0)
combined_embedding_matrix = np.concatenate([arabic_embedding_matrix, eng_embedding_matrix, 
                                            china_embedding_matrix, rus_embedding_matrix], axis = 0)


combined_y = np.concatenate([arabic_y, eng_y, china_y, rus_y], axis=0)


X_train, X_test, y_train, y_test = train_test_split(combined_X_pad, combined_y, test_size = 0.2, random_state = 42)


arabic_word_index_size = len(arabic_tokenizer.word_index)
eng_word_index_size = len(eng_tokenizer.word_index)
china_word_index_size = len(china_tokenizer.word_index)
rus_word_index_size = len(rus_tokenizer.word_index)
total_vocab_size = arabic_word_index_size + eng_word_index_size + china_word_index_size + rus_word_index_size


def train_dl_model(model, model_name, X_train, y_train, X_test, y_test):
    
    checkpoint = ModelCheckpoint(f'best_{model_name}.keras', monitor='val_loss', save_best_only=True)
#     early_stop = EarlyStopping(monitor='val_loss', patience=5)
    
#     history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
#                         callbacks=[checkpoint, early_stop], verbose=1)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                        callbacks=[checkpoint], verbose=1)
    

    model.load_weights(f'best_{model_name}.keras')
    

    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Threat', 'Non-Threat', 'Neutral'], output_dict=True)
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Threat', 'Non-Threat','Neutral']))
    
    return acc, report, history


num_classes = len(np.unique(combined_y))

model_RNN = Sequential([
    Embedding(input_dim = total_vocab_size+4, 
             output_dim = arabic_embedding_dim,
             weights = [combined_embedding_matrix],
             input_length = max_len,
             trainable = True),
    Bidirectional(SimpleRNN(64, return_sequences = True, kernel_regularizer = l2(0.01))),
    Dense(32, activation = 'relu'),
    Dropout(0.4),
    Bidirectional(GRU(64, return_sequences = True, kernel_regularizer = l2(0.01))),
    Dense(32, activation = 'relu'),
    Dropout(0.4),
    Bidirectional(LSTM(32, return_sequences = False, kernel_regularizer = l2(0.01))),
    Dense(32, activation = 'relu'),
    Dropout(0.4),
    Dense(num_classes, activation = 'softmax')
])

model_RNN.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 0.005),
                 metrics = ['accuracy'])
model_RNN.build(input_shape = (None, max_len))
model_RNN.summary()


#history_RNN = model_RNN.fit(X_train, y_train, epochs = 50, batch_size = 16,
#                          validation_data = (X_test, y_test))

rnn_acc, rnn_report, rnn_history = train_dl_model(model_RNN, 'RNN', X_train, y_train, X_test, y_test)

with open('combined_rnn_history.pkl', 'wb') as f:
    pickle.dump(rnn_history.history, f)
    

with open('combined_rnn_history.pkl', 'rb') as f:
    loaded_rnn_history = pickle.load(f)


ci_range = 0.005  
epochs = np.arange(len(loaded_rnn_history['loss']))

plt.figure(figsize=(5, 3), dpi=300)

plt.plot(epochs, loaded_rnn_history['loss'], label='Training Loss', linestyle='-', marker='o', color='navy', linewidth=0.5, markersize=2)
plt.plot(epochs, loaded_rnn_history['val_loss'], label='Validation Loss', linestyle='--', marker='s', color='darkorange', linewidth=0.5, markersize=2)

plt.fill_between(epochs, np.array(loaded_rnn_history['loss']) - ci_range, 
                 np.array(loaded_rnn_history['loss']) + ci_range, 
                 color='navy', alpha=0.1, label='Training Loss Confidence Interval')
plt.fill_between(epochs, np.array(loaded_rnn_history['val_loss']) - ci_range, 
                 np.array(loaded_rnn_history['val_loss']) + ci_range, 
                 color='darkorange', alpha=0.1, label='Validation Loss Confidence Interval')

plt.ylim(0.2, 1)  

plt.xlabel('Epochs', fontsize=5, fontweight='bold')
plt.ylabel('Loss', fontsize=5, fontweight='bold')

plt.title('RNN Loss Curve', fontsize=7, fontweight='bold')
plt.xlabel('Epochs (iterations)', fontsize=5, fontweight='bold')
plt.ylabel('Loss (Cross-Entropy)', fontsize=5, fontweight='bold')


plt.xticks(fontsize=3, fontweight='bold')
plt.yticks(fontsize=3, fontweight='bold')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)


min_val_loss_epoch = loaded_rnn_history['val_loss'].index(min(loaded_rnn_history['val_loss']))
min_val_loss = min(loaded_rnn_history['val_loss'])
plt.annotate(f'Lowest Val Loss: {min_val_loss:.4f}', 
             xy=(min_val_loss_epoch, min_val_loss), xytext=(min_val_loss_epoch + 1, min_val_loss + 0.0015),
             arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=2),
             fontsize=4, fontweight='bold')

plt.annotate('', xy=(1, loaded_rnn_history['loss'][1]), xytext=(3, loaded_rnn_history['loss'][3]),
             arrowprops=dict(facecolor='blue', shrink=0.005, width=0.5, headwidth=2, linestyle='--'),
             fontsize=2)
plt.text(3, loaded_rnn_history['loss'][3] + 0.0015, 'Decreasing Slope', color='blue', fontsize=4)

plt.axvspan(5, 7, color='lightgreen', alpha=0.2, label='Potential Plateau Region')

plt.savefig('RNN_Combined_training_validation_loss.png', format='png', dpi=300)

plt.legend(loc='best', fontsize=4, fancybox=True, shadow=True, borderpad=0.8, edgecolor='black')

plt.tight_layout(pad=1.0)

plt.show()



ci_range_acc = 0.01  
epochs = np.arange(len(loaded_rnn_history['accuracy']))

plt.figure(figsize=(5, 3), dpi=300)

plt.plot(epochs, loaded_rnn_history['accuracy'], label='Training Accuracy', linestyle='-', marker='o', color='green', linewidth=0.5, markersize=2)
plt.plot(epochs, loaded_rnn_history['val_accuracy'], label='Validation Accuracy', linestyle='--', marker='s', color='blue', linewidth=0.5, markersize=2)

plt.fill_between(epochs, np.array(loaded_rnn_history['accuracy']) - ci_range_acc, 
                 np.array(loaded_rnn_history['accuracy']) + ci_range_acc, 
                 color='green', alpha=0.1, label='Training Accuracy Confidence Interval')
plt.fill_between(epochs, np.array(loaded_rnn_history['val_accuracy']) - ci_range_acc, 
                 np.array(loaded_rnn_history['val_accuracy']) + ci_range_acc, 
                 color='blue', alpha=0.1, label='Validation Accuracy Confidence Interval')

plt.ylim(0.2, 1.0) 

plt.title('RNN Accuracy Curve', fontsize=7, fontweight='bold')
plt.xlabel('Epochs', fontsize=5, fontweight='bold')
plt.ylabel('Accuracy', fontsize=5, fontweight='bold')

plt.xticks(fontsize=3, fontweight='bold')
plt.yticks(fontsize=3, fontweight='bold')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

max_val_acc_epoch = loaded_rnn_history['val_accuracy'].index(max(loaded_rnn_history['val_accuracy']))
max_val_acc = max(loaded_rnn_history['val_accuracy'])
plt.annotate(f'Highest Val Accuracy: {max_val_acc:.4f}', 
             xy=(max_val_acc_epoch, max_val_acc), xytext=(max_val_acc_epoch + 1, max_val_acc - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=2),
             fontsize=4, fontweight='bold')

plt.annotate('', xy=(2, loaded_rnn_history['accuracy'][2]), xytext=(4, loaded_rnn_history['accuracy'][4]),
             arrowprops=dict(facecolor='green', shrink=0.005, width=0.5, headwidth=2, linestyle='--'),
             fontsize=2)
plt.text(4, loaded_rnn_history['accuracy'][4] + 0.02, 'Increasing Accuracy', color='green', fontsize=4)

plt.axvspan(6, 9, color='lightblue', alpha=0.2, label='Improvement Region')

plt.savefig('RNN_Combined_training_validation_accuracy.png', format='png', dpi=300)

plt.legend(loc='best', fontsize=4, fancybox=True, shadow=True, borderpad=0.8, edgecolor='black')

plt.tight_layout(pad=1.0)  

plt.show()



num_classes = len(np.unique(combined_y))
model_lstm = Sequential([
    Embedding(input_dim = total_vocab_size+4, 
             output_dim = arabic_embedding_dim,
             weights = [combined_embedding_matrix],
             input_length = max_len,
             trainable = True),
    Bidirectional(LSTM(64, return_sequences = True, kernel_regularizer = l2(0.01))),
    Dense(32, activation = 'relu'),
    Dropout(0.4),
    Bidirectional(LSTM(64, return_sequences = True, kernel_regularizer = l2(0.01))),
    Dense(32, activation = 'relu'),
    Dropout(0.4),
    Bidirectional(LSTM(32, return_sequences = False, kernel_regularizer = l2(0.01))),
    Dropout(0.4),
    Dense(num_classes, activation = 'softmax')
])
model_lstm.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 0.0005), metrics = ['accuracy'])
model_lstm.build(input_shape= (None, max_len))
model_lstm.summary()

lstm_acc, lstm_report, lstm_history = train_dl_model(model_lstm, 'LSTM', X_train, y_train, X_test, y_test)

#history_lstm = model_lstm.fit(X_train, y_train,
                              #                             epochs = 50, batch_size = 32, validation_split = 0.2)


with open('combined_lstm_history.pkl', 'wb') as f:
    pickle.dump(lstm_history.history, f)
    

with open('combined_lstm_history.pkl', 'rb') as f:
    loaded_lstm_history = pickle.load(f)


ci_range = 0.05  
epochs = np.arange(len(loaded_lstm_history['loss']))


plt.figure(figsize=(5, 3), dpi=300)

plt.plot(epochs, loaded_lstm_history['loss'], label='Training Loss', linestyle='-', marker='o', color='navy', linewidth=0.5, markersize=2)
plt.plot(epochs, loaded_lstm_history['val_loss'], label='Validation Loss', linestyle='--', marker='s', color='darkorange', linewidth=0.5, markersize=2)

plt.fill_between(epochs, np.array(loaded_lstm_history['loss']) - ci_range, 
                 np.array(loaded_lstm_history['loss']) + ci_range, 
                 color='navy', alpha=0.2, label='Training Loss Confidence Interval')
plt.fill_between(epochs, np.array(loaded_lstm_history['val_loss']) - ci_range, 
                 np.array(loaded_lstm_history['val_loss']) + ci_range, 
                 color='darkorange', alpha=0.2, label='Validation Loss Confidence Interval')

plt.ylim(0, 5)  

plt.title('LSTM Loss Curve', fontsize=7, fontweight='bold')
plt.xlabel('Epochs (iterations)', fontsize=5, fontweight='bold')
plt.ylabel('Loss (Cross-Entropy)', fontsize=5, fontweight='bold')

plt.xticks(fontsize=3, fontweight='bold')
plt.yticks(fontsize=3, fontweight='bold')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

min_val_loss_epoch = loaded_lstm_history['val_loss'].index(min(loaded_lstm_history['val_loss']))
min_val_loss = min(loaded_lstm_history['val_loss'])
plt.annotate(f'Lowest Val Loss: {min_val_loss:.4f}', 
             xy=(min_val_loss_epoch, min_val_loss), xytext=(min_val_loss_epoch + 1, min_val_loss + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=2),
             fontsize=4, fontweight='bold')

plt.annotate('', xy=(1, loaded_lstm_history['loss'][1]), xytext=(3, loaded_lstm_history['loss'][3]),
             arrowprops=dict(facecolor='blue', shrink=0.005, width=0.5, headwidth=2, linestyle='--'),
             fontsize=3)
plt.text(3, loaded_lstm_history['loss'][3] + 0.1, 'Decreasing Slope', color='blue', fontsize=4)

plt.axvspan(5, 7, color='lightgreen', alpha=0.2, label='Potential Plateau Region')

plt.savefig('LSTM_Combined_training_validation_loss_small.png', format='png', dpi=300)

plt.legend(loc='best', fontsize=4, fancybox=True, shadow=True, borderpad=0.8, edgecolor='black')

plt.tight_layout(pad=1.0) 

plt.show()


ci_range_acc = 0.01  
epochs = np.arange(len(loaded_lstm_history['accuracy']))

plt.figure(figsize=(5, 3), dpi=300)

plt.plot(epochs, loaded_lstm_history['accuracy'], label='Training Accuracy', linestyle='-', marker='o', color='green', linewidth=0.5, markersize=2)
plt.plot(epochs, loaded_lstm_history['val_accuracy'], label='Validation Accuracy', linestyle='--', marker='s', color='blue', linewidth=0.5, markersize=2)

plt.fill_between(epochs, np.array(loaded_lstm_history['accuracy']) - ci_range_acc, np.array(loaded_lstm_history['accuracy']) + ci_range_acc, 
                 color='green', alpha=0.1, label='Training Accuracy Confidence Interval')
plt.fill_between(epochs, np.array(loaded_lstm_history['val_accuracy']) - ci_range_acc, np.array(loaded_lstm_history['val_accuracy']) + ci_range_acc, 
                 color='blue', alpha=0.1, label='Validation Accuracy Confidence Interval')

plt.ylim(0.4, 1.0) 

plt.title('LSTM Accuracy Curve', fontsize=7, fontweight='bold')
plt.xlabel('Epochs', fontsize=5, fontweight='bold')
plt.ylabel('Accuracy', fontsize=5, fontweight='bold')

plt.xticks(fontsize=3, fontweight='bold')
plt.yticks(fontsize=3, fontweight='bold')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

max_val_acc_epoch = loaded_lstm_history['val_accuracy'].index(max(loaded_lstm_history['val_accuracy']))
max_val_acc = max(loaded_lstm_history['val_accuracy'])
plt.annotate(f'Highest Val Accuracy: {max_val_acc:.4f}', 
             xy=(max_val_acc_epoch, max_val_acc), xytext=(max_val_acc_epoch + 1, max_val_acc - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=2),
             fontsize=4, fontweight='bold')

plt.annotate('', xy=(2, loaded_lstm_history['accuracy'][2]), xytext=(4, loaded_lstm_history['accuracy'][4]),
             arrowprops=dict(facecolor='green', shrink=0.005, width=0.5, headwidth=2, linestyle='--'),
             fontsize=2)
plt.text(4, loaded_lstm_history['accuracy'][4] + 0.02, 'Increasing Accuracy', color='green', fontsize=4)

plt.axvspan(6, 9, color='lightblue', alpha=0.2, label='Improvement Region')

plt.savefig('LSTM_Combined_training_validation_accuracy.png', format='png', dpi=300)

plt.legend(loc='best', fontsize=4, fancybox=True, shadow=True, borderpad=0.8, edgecolor='black')

plt.tight_layout(pad=1.0) 

plt.show()



xlm_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlm_roberta_model = TFXLMRobertaModel.from_pretrained('xlm-roberta-base')

max_len = 100
learning_rate = 0.0005
batch_size = 16
num_epochs = 50

def tokenize_roberta(data):
    return xlm_tokenizer(
        data,
        padding = True,
        truncation = True,
        max_length = max_len,
        return_tensors = 'tf'
    )

combined_text_data = pd.concat([arabic_data['Cleaned_data'], eng_data['cleaned_tweet'], china_data['Text'], rus_data['cleaned_data']], axis = 0)

tokenized_data = tokenize_roberta(combined_text_data.tolist())

X = tokenized_data['input_ids'].numpy()  
y = np.array(combined_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_layer = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
xlm_outputs = xlm_roberta_model(input_layer)
last_hidden_state = xlm_outputs.last_hidden_state

pooled_output = GlobalAveragePooling1D()(last_hidden_state)

dense_1 = Dense(128, activation='relu')(pooled_output)
dropout_1 = Dropout(0.4)(dense_1)

dense_2 = Dense(64, activation='relu')(dropout_1)
dropout_2 = Dropout(0.4)(dense_2)

dense_3 = Dense(32, activation='relu')(dropout_2)
dropout_3 = Dropout(0.4)(dense_3)

output_layer = Dense(len(np.unique(combined_y)), activation='softmax')(dropout_3)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x={'input_ids': X_train},
    y=y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=1
)

loss, accuracy = model.evaluate(x={'input_ids': X_test}, y=y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")



ci_range = 0.005 
xlm_history = history.history 

import pickle

with open('Combined_llm_result', 'wb') as f:
    pickle.dump(history.history, f)

with open('Combined_llm_result', 'rb') as f:
    xlm_history = pickle.load(f)



from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Predict on the test set
y_pred_logits = model.predict(x={'input_ids': X_test})
y_pred = np.argmax(y_pred_logits, axis=1)  # Convert logits to class predictions

# Evaluate model performance
print("Model Evaluation:")
loss, accuracy = model.evaluate(x={'input_ids': X_test}, y=y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Calculate additional metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Individual metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Save metrics to a file (if needed)
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'loss': loss
}

import pickle
with open('llm_model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("Metrics saved to 'model_metrics.pkl'")





epochs = np.arange(len(xlm_history['loss']))

plt.figure(figsize=(5, 3), dpi=300)

plt.plot(epochs, xlm_history['loss'], label='Training Loss', linestyle='-', marker='o', color='navy', linewidth=0.5, markersize=2)
plt.plot(epochs, xlm_history['val_loss'], label='Validation Loss', linestyle='--', marker='s', color='darkorange', linewidth=0.5, markersize=2)

plt.fill_between(epochs, np.array(xlm_history['loss']) - ci_range, 
                 np.array(xlm_history['loss']) + ci_range, 
                 color='navy', alpha=0.1, label='Training Loss Confidence Interval')
plt.fill_between(epochs, np.array(xlm_history['val_loss']) - ci_range, 
                 np.array(xlm_history['val_loss']) + ci_range, 
                 color='darkorange', alpha=0.1, label='Validation Loss Confidence Interval')

plt.ylim(0.1, 1)  

plt.xlabel('Epochs', fontsize=5, fontweight='bold')
plt.ylabel('Loss', fontsize=5, fontweight='bold')

plt.title('XLM-RoBERTa Loss Curve', fontsize=7, fontweight='bold')
plt.xlabel('Epochs (iterations)', fontsize=5, fontweight='bold')
plt.ylabel('Loss (Cross-Entropy)', fontsize=5, fontweight='bold')

plt.xticks(fontsize=3, fontweight='bold')
plt.yticks(fontsize=3, fontweight='bold')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

min_val_loss_epoch = xlm_history['val_loss'].index(min(xlm_history['val_loss']))
min_val_loss = min(xlm_history['val_loss'])
plt.annotate(f'Lowest Val Loss: {min_val_loss:.4f}', 
             xy=(min_val_loss_epoch, min_val_loss), xytext=(min_val_loss_epoch + 1, min_val_loss + 0.0015),
             arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=2),
             fontsize=4, fontweight='bold')

plt.annotate('', xy=(1, xlm_history['loss'][1]), xytext=(3, xlm_history['loss'][3]),
             arrowprops=dict(facecolor='blue', shrink=0.005, width=0.5, headwidth=2, linestyle='--'),
             fontsize=2)
plt.text(3, xlm_history['loss'][3] + 0.0015, 'Decreasing Slope', color='blue', fontsize=4)

plt.axvspan(5, 7, color='lightgreen', alpha=0.2, label='Potential Plateau Region')

plt.savefig('XLM_RoBERTa_training_validation_loss.png', format='png', dpi=300)

plt.legend(loc='best', fontsize=4, fancybox=True, shadow=True, borderpad=0.8, edgecolor='black')

plt.tight_layout(pad=1.0)

plt.show()



ci_range_acc = 0.01  
xlm_history = history.history
epochs = np.arange(len(xlm_history['accuracy']))

plt.figure(figsize=(5, 3), dpi=300)

plt.plot(epochs, xlm_history['accuracy'], label='Training Accuracy', linestyle='-', marker='o', color='green', linewidth=0.5, markersize=2)
plt.plot(epochs, xlm_history['val_accuracy'], label='Validation Accuracy', linestyle='--', marker='s', color='blue', linewidth=0.5, markersize=2)

plt.fill_between(epochs, np.array(xlm_history['accuracy']) - ci_range_acc, 
                 np.array(xlm_history['accuracy']) + ci_range_acc, 
                 color='green', alpha=0.1, label='Training Accuracy Confidence Interval')
plt.fill_between(epochs, np.array(xlm_history['val_accuracy']) - ci_range_acc, 
                 np.array(xlm_history['val_accuracy']) + ci_range_acc, 
                 color='blue', alpha=0.1, label='Validation Accuracy Confidence Interval')

plt.ylim(0.2, 1.0) 
plt.title('XLM-RoBERTa Accuracy Curve', fontsize=7, fontweight='bold')
plt.xlabel('Epochs', fontsize=5, fontweight='bold')
plt.ylabel('Accuracy', fontsize=5, fontweight='bold')

plt.xticks(fontsize=3, fontweight='bold')
plt.yticks(fontsize=3, fontweight='bold')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on() 
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

max_val_acc_epoch = xlm_history['val_accuracy'].index(max(xlm_history['val_accuracy']))
max_val_acc = max(xlm_history['val_accuracy'])
plt.annotate(f'Highest Val Accuracy: {max_val_acc:.4f}', 
             xy=(max_val_acc_epoch, max_val_acc), xytext=(max_val_acc_epoch + 1, max_val_acc - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=2),
             fontsize=4, fontweight='bold')

plt.annotate('', xy=(2, xlm_history['accuracy'][2]), xytext=(4, xlm_history['accuracy'][4]),
             arrowprops=dict(facecolor='green', shrink=0.005, width=0.5, headwidth=2, linestyle='--'),
             fontsize=2)
plt.text(4, xlm_history['accuracy'][4] + 0.02, 'Increasing Accuracy', color='green', fontsize=4)

plt.axvspan(6, 9, color='lightblue', alpha=0.2, label='Improvement Region')

plt.savefig('XLM_RoBERTa_training_validation_accuracy.png', format='png', dpi=300)

plt.legend(loc='best', fontsize=4, fancybox=True, shadow=True, borderpad=0.8, edgecolor='black')

plt.tight_layout(pad=1.0) 
plt.show()


