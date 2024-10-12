# Import required libraries
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import login
import os

# Authenticate with Hugging Face using your token
login(token="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")  # Replace with your actual token

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("unhcr/hatespeech-detection")
model = AutoModelForSequenceClassification.from_pretrained("unhcr/hatespeech-detection")

# Load or create feedback data CSV
feedback_file = "feedback_data.csv"
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)
else:
    feedback_df = pd.DataFrame(columns=["text", "label"])

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
        censored_text = " ".join(['*' * len(word) if word.lower() in text.lower() else word for word in text.split()])
        return censored_text, predicted_class
    else:
        return text, predicted_class

# Check feedback data for previously corrected entries
def check_feedback(text):
    if not feedback_df.empty:
        exact_matches = feedback_df[feedback_df["text"] == text]
        if not exact_matches.empty:
            return exact_matches["label"].values[0]
    return None

# Streamlit interface
st.title("Hate Speech Detection, Prevention, and Self-Training")
st.write("Enter a tweet or message to classify and censor any detected hate speech or offensive language.")

# Input text box for user input
user_input = st.text_area("Enter your text here")

# Classify and censor text when the button is pressed
if st.button("Classify and Censor"):
    if user_input:
        # Check if there is user feedback for this input
        feedback_label = check_feedback(user_input)
        
        if feedback_label is not None:
            label_mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Normal"}
            prediction_label = label_mapping.get(feedback_label, "Unknown")
            st.write("Prediction (based on feedback):", prediction_label)
        else:
            censored_text, prediction = censor_text(user_input)
            
            # Mapping the output to class labels
            label_mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Normal"}
            prediction_label = label_mapping.get(prediction, "Unknown")
            
            # Display the result and censored text if applicable
            st.write("Prediction:", prediction_label)
            st.write("Censored Text:", censored_text)
        
            # Ask for feedback
            st.write("Is this prediction correct?")
            if st.button("Yes"):
                st.write("Thank you for your feedback!")
            if st.button("No"):
                correct_label = st.selectbox("Select the correct label:", ["Hate Speech", "Offensive Language", "Normal"])
                if st.button("Submit Correction"):
                    correct_label_idx = list(label_mapping.keys())[list(label_mapping.values()).index(correct_label)]
                    
                    # Update feedback data and save to CSV
                    feedback_df.loc[len(feedback_df)] = [user_input, correct_label_idx]
                    feedback_df.to_csv(feedback_file, index=False)
                    st.write("Thank you for your feedback! The model has updated with this new data.")

    else:
        st.write("Please enter some text for classification.")
