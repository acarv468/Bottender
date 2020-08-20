# Import Libraries
import pickle
import logging
import pandas as pd
import numpy as np
# from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from datetime import datetime

nltk.download('stopwords')

# Load Data
intent_df = pd.read_csv('data/augmented_data/augmented_intents.csv', encoding='utf-8-sig')
print(intent_df.head())

# Preprocess data

# Remove null values
intent_df = intent_df[pd.notnull(intent_df['intent'])]

# Clean Text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    #text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    return text

intent_df['text'] = intent_df['text'].apply(clean_text)

# Train Test Split
X = intent_df.text
y = intent_df.intent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Model

# Logistic Regression
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
intent_model = logreg.fit(X_train, y_train)
y_pred = intent_model.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

# Save model
datetime_obj = datetime.now()
timestamp_str = datetime_obj.strftime('%d-%b-%Y(%H-%M-%S)')
print('Model saved as intent_classification_' + timestamp_str)
pickle.dump(intent_model, open('models/intent_models/intent_classification_' + timestamp_str, 'wb')) 