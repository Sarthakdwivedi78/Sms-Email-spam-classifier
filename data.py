

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


import re
import string
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    # Lowercase
    text = text.lower()
    # Basic tokenization (alphanumeric words only)
    tokens = re.findall(r'\b\w+\b', text)
    # Remove stopwords
    tokens = [i for i in tokens if i not in ENGLISH_STOP_WORDS]
    # Stemming
    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

# Load trained model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input for testing
input_sms = "Congratulations! You have won a free lottery ticket."
transformed_sms = transform_text(input_sms)
vector_input = tfidf.transform([transformed_sms])
result = model.predict(vector_input)[0]

print("Prediction:", "Spam" if result == 1 else "Ham")

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
