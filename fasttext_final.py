import logging
import os
import pandas as pd
import yaml
import logging
import datetime
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import fasttext
import csv
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import json
train_df = pd.read_csv("train-balanced-sarcasm.csv")
train_df.dropna()
def pre_process_data(df):
    df['comment'] = df['comment'].str.lower()
    df['token_sentence'] = df['comment'].apply(text_edit)
    df['count_token'] = df['token_sentence'].apply(len)
    df = df[df.count_token > 1]
    return(df)
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
labels = ["__label__" + str(ele) for ele in train_df["label"]]
texts = [" " + ele + "\n" for ele in train_df["comment"]]
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.1)
TrainCorpus = [i + j for i, j in zip(y_train, X_train)]
TestCorpus = [i + j for i, j in zip(y_test, X_test)]
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
lr = [0.01, 0.1, 1] 
dim = [100, 1000, 5000]
lr_list = []
wordNgrams_list = []
dim_list = []
for i in lr:
    for j in wordNgrams:
        for k in dim:
            model = fasttext.train_supervised(input="fast.train", wordNgrams=j, lr=i, loss='ns', dim = k, epoch = 10, verbose = 0, neg = 25)
            model.save_model("fastText_CV_model.bin")
            commandtext = base + "fastText_CV_model.bin" + " fast.test 1 > fastTextPreds.txt"
            os.system(commandtext)
            f = open("fastTextPreds.txt", 'r')
            y_pred = f.read().splitlines()
            f.close()
            perf_metrics = precision_recall_fscore_support(y_test,y_pred)
            accuracy_list.append(accuracy_score(y_test,y_pred))
            precision_list.append(perf_metrics[0][1])
            recall_list.append(perf_metrics[1][1])
            fscore_list.append(perf_metrics[2][1])
            lr_list.append(i)
            wordNgrams_list.append(j)
            dim_list.append(k)
res_df = pd.DataFrame(list(zip(lr_list, wordNgrams_list, dim_list, accuracy_list, precision_list, recall_list, fscore_list)), columns = ['lr', 'wordNgrams', 'dim', 'accuracy', 'precision', 'recall', 'fscore'])
res_df.to_csv('fasttext_results.csv')