import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.title("📰 Fake News Detection using BERT")

st.write("Enter news text below to check if it's Fake or Real")

# load model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

def predict(text):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return "Real" if predicted_class == 1 else "Fake"


user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input:
        result = predict(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text")