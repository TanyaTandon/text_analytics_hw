import re
import os
import yaml
import logging
import math
import pickle
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import gensim
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import keras
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

train_df = pd.read_csv("train-balanced-sarcasm.csv")
train_df.dropna()
def pre_process_data(df):
    df['comment'] = df['comment'].str.lower()
    df['token_sentence'] = df['comment'].apply(text_edit)
    df['count_token'] = df['token_sentence'].apply(len)
    df = df[df.count_token > 1]
    return(df)


#pre_process_data(train_df)
def text_edit(s):
    #tokens= str.lower(s)
    tokens = (re.sub('[^A-Za-z0-9 ]+', '', str(s)))
    tokens= word_tokenize(tokens)
    return(tokens)

train_df = pre_process_data(train_df)

x_train, x_test, y_train, y_test = train_test_split(train_df["token_sentence"], train_df["label"], test_size=0.3)
train_texts = x_train.tolist()
train_labels = y_train.tolist()

dictionary = gensim.corpora.Dictionary(texts)

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))

def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

# model hyper parameters
EMBEDDING_DIM = 100
SEQUENCE_LENGTH_PERCENTILE = 99
n_layers = 6
n_fc_layers = 2
hidden_units = 256
batch_size = 200
pretrained_embedding = False
# if we have pre-trained embeddings, specify if they are static or non-static embeddings
TRAINABLE_EMBEDDINGS = True
patience = 3
dropout_rate = 0.3
n_filters = 256
window_size = 6
dense_activation = "relu"
l2_penalty = 0.0003
epochs = 25
VALIDATION_SPLIT = 0.1

assert len(train_texts)==len(train_labels)
# compute the max sequence length
# why do we need to do that?
lengths=list(map(lambda x: len(x), train_texts))
a = np.array(lengths)
MAX_SEQUENCE_LENGTH = int(np.percentile(a, SEQUENCE_LENGTH_PERCENTILE))
# convert all texts to dictionary indices
#train_texts_indices = list(map(lambda x: texts_to_indices(x[0], dictionary), train_texts))
train_texts_indices = list(map(lambda x: texts_to_indices(x, dictionary), train_texts))
# pad or truncate the texts
x_data = pad_sequences(train_texts_indices, maxlen=int(MAX_SEQUENCE_LENGTH))
# convert the train labels to one-hot encoded vectors
train_labels = keras.utils.to_categorical(train_labels)
y_data = train_labels

model = Sequential()

# create embeddings matrix from word2vec pre-trained embeddings, if provided
if pretrained_embedding:
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_MODEL_FILE, binary=True)
    embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    for word, i in dictionary.token2id.items():
        embedding_vector = embeddings_index[word] if word in embeddings_index else None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(dictionary) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=TRAINABLE_EMBEDDINGS))
else:
    model.add(Embedding(len(dictionary) + 1,
                        EMBEDDING_DIM,
                        input_length=MAX_SEQUENCE_LENGTH))
# add drop out for the input layer, why do you think this might help?
model.add(Dropout(dropout_rate))
# add a 1 dimensional conv layer
# a rectified linear activation unit, returns input if input > 0 else 0
model.add(Conv1D(filters=n_filters,
                 kernel_size=window_size,
                 activation='relu'))
# add a max pooling layer
model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - window_size + 1))
model.add(Flatten())

# add 0 or more fully connected layers with drop out
for _ in range(n_layers):
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units,
                    activation=dense_activation,
                    kernel_regularizer=l2(l2_penalty),
                    bias_regularizer=l2(l2_penalty),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))

# add the last fully connected layer with softmax activation
model.add(Dropout(dropout_rate))
model.add(Dense(len(train_labels[0]),
                activation='softmax',
                kernel_regularizer=l2(l2_penalty),
                bias_regularizer=l2(l2_penalty),
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'))

# compile the model, provide an optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# print a summary
print(model.summary())


# train the model with early stopping
early_stopping = EarlyStopping(patience=patience)
Y = np.array(y_data)

fit = model.fit(x_data,
                Y,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=VALIDATION_SPLIT,
                verbose=1,
                callbacks=[early_stopping])

print(fit.history.keys())
val_accuracy = fit.history['acc'][-1]
print(val_accuracy)
# save the model

pickle.dump(model, open("cnn.pkl", "wb"))