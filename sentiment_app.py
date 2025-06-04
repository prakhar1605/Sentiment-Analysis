import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess user input
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
    return " ".join(tokens)

# UI
st.title("🧠 Social Media Sentiment Analyzer")
st.write("Built with Scikit-learn and Streamlit")

user_input = st.text_area("✍️ Enter a post, tweet or comment:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        clean = preprocess(user_input)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]
        st.success(f"**Predicted Sentiment: {pred.upper()}**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
