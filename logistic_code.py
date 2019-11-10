import re
import ast
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, f1_score, recall_score, precision_score
from nltk.tokenize import sent_tokenize
import datetime

import pickle
#from gensim.test.utils import common_texts, get_tmpfile
#from gensim.models import Word2Vec

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


#import datetime
#Defining Pipeline
pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='word', token_pattern=r'[A-Za-z0-9@-]+')),
        ('model', LogisticRegression(random_state=12345, verbose = 1, solver = 'saga')),
    ])

#Defining parameters to vary
parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__max_features': (None, 5000, 10000, 50000),
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'model__C': (0.01, 1, 100) }

scoring_list = ["accuracy", "f1", "precision", "recall", "roc_auc"]

X = train_df['comment']
y = train_df['label']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

x_train = x_train.astype(str)
y_train = y_train.astype(int)
model = GridSearchCV(pipeline, parameters, cv=2,
                                n_jobs=-1, verbose=1, scoring=scoring_list, refit='f1',)

model.fit(x_train.tolist(), y_train.to_numpy())

res_df = pd.DataFrame(model.cv_results_)


res_df.to_csv("Output_logistic.csv")
pickle.dump(model, open("LogisticRegression.pkl", "wb"))