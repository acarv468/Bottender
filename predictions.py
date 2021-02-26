import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Intent Predictions
intent_model = pickle.load(open('classification/models/intent_models/intent_classification_model2020-07-29', 'rb'))

def predict_intent(text):
    x = [text]
    pred_intent = intent_model.predict(x)[0]
    prob = np.max(np.round(intent_model.predict_proba(x), 8))
    
    return pred_intent, prob

def top_intents(text):
    x = [text]
    probabilities = intent_model.predict_log_proba(x)[0]
    intents = intent_model.classes_
    log_intents = pd.DataFrame({'intent':intents, 'log_prob':probabilities})
    top_intents = log_intents.sort_values(by='log_prob', ascending=False)[:4]

    return top_intents

# Toxic Predictions
toxic_model = pickle.load(open('classification/models/toxic_models/toxic_classification_model2020-07-29', 'rb'))
vectorizer = pickle.load(open('classification/models/vectorizers/vectorizer2020-07-29', 'rb'))

def predict_toxic(text):
    x = [text]
    x_vec = vectorizer.transform(x)
    pred_sentiment = toxic_model.predict(x_vec)[0]
    prob = np.max(np.round(toxic_model.predict_proba(x_vec), 8))
    
    return pred_sentiment, prob