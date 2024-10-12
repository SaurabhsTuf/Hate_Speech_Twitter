# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import nltk

# Load nltk data
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load the dataset
url = "https://raw.githubusercontent.com/SaurabhsTuf/Hate_Speech_Twitter/refs/heads/master/twitter.csv"
try:
    data = pd.read_csv(url)
except Exception as e:
    st.error(f"Error reading the CSV file: {e}")

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "Normal"})
data = data[["tweet", "labels"]]

# Preprocessing function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply preprocessing to the 'tweet' column
data["tweet"] = data["tweet"].apply(clean)

# Feature extraction and model pipeline
pipeline = make_pipeline(CountVectorizer(), DecisionTreeClassifier())
X = data["tweet"]
y = data["labels"]

# Train the model
pipeline.fit(X, y)

# Streamlit UI
st.title("Hate Speech Detection and Prevention")
st.write("Enter a tweet or message to classify and censor any detected Hate Speech or Offensive Language.")

# Input text box
user_input = st.text_area("Enter your text here")

if st.button("Predict and Censor"):
    if user_input:
        # Preprocess the input text
        cleaned_text = clean(user_input)
        transformed_text = pipeline.named_steps['countvectorizer'].transform([cleaned_text])
        
        # Prediction
        prediction = pipeline.predict([cleaned_text])
        prediction_proba = pipeline.predict_proba([cleaned_text])  # Get probabilities
        
        # Determine if the prediction confidence is high enough
        confidence = np.max(prediction_proba)  # Get max probability
        threshold = 0.6  # Set confidence threshold

        if confidence >= threshold:
            # Check if the text contains hate speech or offensive language
            if prediction[0] in ["Hate Speech", "Offensive Language"]:
                # Censor detected hate speech or offensive language
                censored_text = re.sub(r'\b\w+\b', '****', user_input)
                st.write("Warning: The content contains censored language.")
                st.write("Censored Text:", censored_text)
            else:
                st.write("Prediction:", prediction[0])
                st.write("Text is classified as normal.")
        else:
            st.write("The model is not confident enough to classify this text.")
    else:
        st.write("Please enter text to classify.")
