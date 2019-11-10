import pickle
from nltk.tokenize import word_tokenize
import pandas as pd
import re
from sklearn.model_selection import train_test_split


model = pickle.load( open( "SVM.pkl", "rb" ) )

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

X = train_df['comment']
y = train_df['label']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

x_test = x_test.astype(str)
y_test = y_test.astype(int)


    
g = (pd.DataFrame(x_test)).reset_index(drop= True).comment.iloc[0]
true_label = (pd.DataFrame(y_test)).reset_index(drop= True).label.iloc[0]
test_label= str(model.predict([g])[0])
doct = {}
doct['true_label'] = str(true_label)
doct['test_label'] = test_label


import json
with open('svm_output.json', 'w') as outfile:
    json.dump(doct, outfile)