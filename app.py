# Import required libraries
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import login

# Authenticate with Hugging Face using your token
login(token="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")  # Replace with your actual token

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("unhcr/hatespeech-detection")
model = AutoModelForSequenceClassification.from_pretrained("unhcr/hatespeech-detection")

# Streamlit interface for hate speech detection and prevention
st.title("Hate Speech Detection and Prevention")
st.write("Enter a tweet or message to classify and censor any detected hate speech or offensive language.")

# Input text box for user input
user_input = st.text_area("Enter your text here")

# Function to censor offensive or hate speech words by replacing with asterisks
def censor_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    # If hate speech or offensive language is detected, censor the text
    if predicted_class in [0, 1]:  # 0: Hate Speech, 1: Offensive Language
        # Split the text into words and replace with asterisks
        censored_text = " ".join(['*' * len(word) if word.lower() in text.lower() else word for word in text.split()])
        return censored_text, predicted_class
    else:
        return text, predicted_class

# Classify and censor text when the button is pressed
if st.button("Classify and Censor"):
    if user_input:
        # Censor and classify the input text
        censored_text, prediction = censor_text(user_input)
        
        # Mapping the output to class labels
        label_mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Normal"}
        prediction_label = label_mapping.get(prediction, "Unknown")

        # Display the result and censored text if applicable
        st.write("Prediction:", prediction_label)
        st.write("Censored Text:", censored_text)
    else:
        st.write("Please enter some text for classification.")
