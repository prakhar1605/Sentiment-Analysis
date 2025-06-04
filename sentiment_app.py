import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Simple preprocessing
def clean_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha() and w not in stopwords.words('english')]
    return " ".join(words)

# App interface
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if user_input:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        st.success(f"Result: {prediction.upper()}")
    else:
        st.warning("Please enter some text")
