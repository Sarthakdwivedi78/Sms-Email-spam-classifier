import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('spam.csv', encoding='latin1')  # adjust path and encoding if needed
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing (same as in your app)
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    cleaned_tokens = [
        ps.stem(word) for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(cleaned_tokens)

df['transformed_text'] = df['text'].apply(transform_text)

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed_text'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))