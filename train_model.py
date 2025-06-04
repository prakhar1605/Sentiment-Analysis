import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load data
data = pd.read_csv('social_media_data.csv')

# Clean text
def clean_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha() and w not in stopwords.words('english')]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# Vectorize and train
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

model = MultinomialNB()
model.fit(X, y)

# Save model
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
