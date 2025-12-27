# **HARMGUARD** 🛡️  
**Fine-Tuned BERT for Online Child Safety**  

A real-time text classifier to detect harmful/toxic content in online communications, helping protect children in digital environments.

## 🚀 Live Demo
Try it now: [Streamlit App](https://harmguard-ai-fine-tuned-bert-for-online-child-safety-dbtj68xcb.streamlit.app/)

## 🎯 Purpose
This project demonstrates AI engineering skills for **child digital safety** — directly aligned with solutions from companies like Qoria (Linewize/Smoothwall).  
It fine-tunes BERT to detect multi-label toxic content (e.g., cyberbullying, threats, obscenity) in text, simulating real-world filtering in school chats, forums, or social platforms.

## ✨ Features
- **Multi-label classification**: Detects `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
- **Real-time inference**: User-friendly Streamlit interface for instant predictions.
- **High accuracy**: Fine-tuned on the Jigsaw Toxic Comment Classification dataset.
- **Production-ready**: Includes Dockerfile for containerization and easy deployment (Streamlit, Cloud Run, Vertex AI).

## 🧠 Model Architecture
Fine-tuned **BERT-base-uncased** for sequence classification.

## 🛠 Tech Stack
- Python  
- Transformers & PyTorch (Hugging Face)  
- Streamlit (web app)  
- Docker (containerization)  
- Dataset: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## 🚀 Quick Start (Run Locally)
```bash
git clone https://github.com/itstharusha/HARMGUARD-AI-FINE-TUNED-BERT-FOR-ONLINE-CHILD-SAFETY.git
cd HARMGUARD-AI-FINE-TUNED-BERT-FOR-ONLINE-CHILD-SAFETY
pip install -r requirements.txt
streamlit run app.py
