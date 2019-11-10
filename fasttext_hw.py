import logging
import os
import pandas as pd
import yaml
import logging
import datetime
import pickle
import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import fasttext
import csv


from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
   
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import json


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

accuracy_list = []
precision_list = []
recall_list = []
fscore_list = []

labels = ["__label__" + str(ele) for ele in y_train]
texts = [" " + ele + "\n" for ele in x_train]

kfold = KFold(n_splits=5, random_state=12345)

for train,test in kfold.split(x_train, y_train):
    TrainCorpus = [i + j for i, j in zip([labels[i] for i in train], [texts[i] for i in train])]
    TestCorpus = [i + j for i, j in zip([labels[i] for i in test], [texts[i] for i in test])]
    TrainCorpus = ''.join(TrainCorpus)
    TestCorpus = ''.join(TestCorpus)

with open('fast.train', 'w') as f:
            f.write(TrainCorpus)
            f.close()

with open('fast.test', 'w') as f:
            f.write(TestCorpus)
            f.close()


base = "~/fastText-0.9.1/fasttext predict "


accuracy_list = []
prec_list = []
recall_list = []
fscore_list = []
wordNgrams = [1,2]
lr = [0.01, 0.1, 0.5, 1] 
dim = [100, 500, 1000, 5000]
for i in tqdm(lr, desc = "Learning Rate"):
    for j in tqdm(wordNgrams, desc = "Word Grams"):
        for k in tqdm(dim, desc = "Dimensions"):
            model = fastText.train_supervised(input="fast.train", wordNgrams=wordNgrams, lr=lr, loss='ns', dim = dim, epoch = 10, verbose = 0, neg = 25) 
            model.save_model("fastText_CV_model.bin")
        
            commandtext = base + "fastText_CV_model.bin" + " fast.test 1 > fastTextPreds.txt"
            lr_list.append(i)
            wordNgrams_list.append(j)
            dim_list.append(k)
            accuracy_list.append(res[0])
            prec_list.append(res[1])
            recall_list.append(res[2])
            fscore_list.append(res[3])
        