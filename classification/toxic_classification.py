# Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import pickle

# Load Data
train = pd.read_csv('data/original_data/toxic_comments.csv')
print(train.head())

# Preprocess Data

x=train.iloc[:,2:].sum()
#marking comments without any tags as "clean"
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
#count number of clean entries
print("Total comments = ",len(train))
print("Total clean comments = ",train['clean'].sum())
print("Total tags =",x.sum())

# Remove null values
print("Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print("filling NA with \"unknown\"")
train["comment_text"].fillna("unknown", inplace=True)

# Applying Binaries
train['dirty'] = train['clean'].replace({False:1, True:0})

# Cleaning Text
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
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    return text

train['comment_text'] = train['comment_text'].apply(clean_text)

# Setting X and y
X = train['comment_text']
y = train['dirty']

# Vectorizing
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Save Vectorizer File
datetime_obj = datetime.now()
timestamp_str = datetime_obj.strftime('%d-%b-%Y(%H-%M-%S)')
print('Vectorizer saved as vectorizer_' + timestamp_str)
pickle.dump(vectorizer, open('models/vectorizers/vectorizer_' + timestamp_str, 'wb')) 

# Resampling
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_sample(X_vec, y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state = 42)

# Model

# Logistic Regression
logreg = Pipeline([
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter=10000))
               ])
toxic_model = logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

# Save Model
print('Model saved as toxic_classification_model_' + timestamp_str)
pickle.dump(toxic_model, open('models/toxic_models/toxic_classification_model_' + timestamp_str, 'wb'))





