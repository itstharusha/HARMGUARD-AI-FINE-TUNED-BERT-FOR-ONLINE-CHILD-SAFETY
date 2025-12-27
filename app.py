import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text):
    cleaned = clean_text(text)
    encoding = tokenizer(cleaned, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    return {label: float(prob) for label, prob in zip(labels, probs)}

# Streamlit UI
st.title("🛡️ Child Online Safety Text Classifier")
st.markdown("""
Detect harmful content in online messages (e.g., chat, forums) to help protect children.
Trained on Jigsaw Toxic Comments dataset using BERT.
""")

user_input = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if user_input.strip():
        results = predict(user_input)
        st.subheader("Risk Levels")
        
        # Color-coded results
        for label, prob in results.items():
            if prob > 0.7:
                color = "🔴"
            elif prob > 0.4:
                color = "🟡"
            else:
                color = "🟢"
            st.write(f"{color} **{label.replace('_', ' ').title()}**: {prob:.3f}")
        
        if max(results.values()) > 0.5:
            st.error("⚠️ Potential harmful content detected!")
        else:
            st.success("✅ Appears safe")
    else:
        st.warning("Please enter some text")
