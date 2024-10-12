# Install required packages


# Import required libraries
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import login # Import login for authentication

# Authenticate with Hugging Face using your token
# Get your token from https://huggingface.co/settings/tokens
login(token="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") # Replace with your actual token

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("unhcr/hatespeech-detection")
model = AutoModelForSequenceClassification.from_pretrained("unhcr/hatespeech-detection")

# Streamlit interface for hate speech detection
st.title("Hate Speech Detection and Prevention")
st.write("Enter a tweet or message to classify hate speech and offensive language.")

# Input text box for user input
user_input = st.text_area("Enter your text here")

# Classify text when button is pressed
if st.button("Classify"):
    if user_input:
        # Tokenize and prepare input for the model
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()

        # Mapping the output to class labels
        label_mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Normal"}
        prediction_label = label_mapping.get(predicted_class, "Unknown")

        # Display the result
        st.write("Prediction:", prediction_label)
    else:
        st.write("Please enter some text for classification.")
