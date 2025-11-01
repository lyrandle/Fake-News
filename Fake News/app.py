import streamlit as st
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Load model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# Download stopwords if not available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Cleaning function (same as training)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news title and content below to predict if it’s **Real** or **Fake**.")

title = st.text_input("News Title:")
text = st.text_area("News Text:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some news content.")
    else:
        cleaned = clean_text(text)
        vectorized = tfidf.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]

        if prediction == "REAL":
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("🚨 This news seems **FAKE**.")