# streamlit_app.py

import streamlit as st
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk

# Load nltk data
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load the dataset with caching
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/SaurabhsTuf/Hate_Speech_Twitter/refs/heads/master/twitter.csv"
    data = pd.read_csv(url)
    data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "Normal"})
    return data[["tweet", "labels"]]

data = load_data()

# Preprocessing function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Apply preprocessing to the 'tweet' column
data["tweet"] = data["tweet"].apply(clean)

# Feature extraction and model pipeline with caching
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(data["tweet"])
    y = data["labels"]
    
    # Use SVM with balanced class weights
    model = SVC(class_weight='balanced', probability=True)
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = train_model()

# Streamlit UI
st.title("Hate Speech Detection and Prevention")
st.write("Enter a tweet or message to classify and censor any detected Hate Speech or Offensive Language.")

# Input text box
user_input = st.text_area("Enter your text here")

if st.button("Predict and Censor"):
    if user_input:
        # Preprocess the input text
        cleaned_text = clean(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Prediction
        prediction = model.predict(transformed_text)
        
        if prediction[0] in ["Hate Speech", "Offensive Language"]:
            censored_text = re.sub(r'\b\w+\b', '****', user_input)
            st.write("Warning: The content contains censored language.")
            st.write("Censored Text:", censored_text)
        else:
            st.write("Prediction:", prediction[0])
            st.write("Text is classified as normal.")
    else:
        st.write("Please enter text to classify.")
