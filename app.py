import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unhcr/hatespeech-detection")
model = AutoModelForSequenceClassification.from_pretrained("unhcr/hatespeech-detection")

# Function to censor offensive language
def censor_text(text):
    # Tokenize the input text and get predictions
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # Get the label with the highest score
    scores = outputs.logits.softmax(dim=1)
    label = torch.argmax(scores, dim=1).item()
    
    # If hate speech or offensive, censor the detected words
    if label == 0 or label == 1:
        # Censor all words in the text (you may refine this logic as needed)
        censored_text = re.sub(r'\b\w+\b', '****', text)
        return f"Content contains hate speech or offensive language:\n{censored_text}"
    else:
        return f"Text is classified as normal:\n{text}"

# Example usage
text = "Your example text here with possible offensive language"
print(censor_text(text))
