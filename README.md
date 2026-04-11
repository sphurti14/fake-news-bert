# 📰 Fake News Detection using BERT

## 📌 Project Overview
This project implements a **Transformer-based NLP model (BERT)** to classify text as **Fake or Real**.  
It demonstrates a real-world deep learning pipeline including preprocessing, tokenization, training, and prediction.

Due to high computational requirements, training is performed using **Google Colab (GPU)**, while the project structure and deployment are maintained locally.

---

## 🚀 Features

- 🧠 BERT-based text classification  
- 🔄 Data preprocessing pipeline  
- 🔤 Tokenization using HuggingFace Transformers  
- ⚡ GPU-based training (Google Colab)  
- 🔍 Prediction system for new input text  

---

## 🧠 Tech Stack

- Python  
- PyTorch  
- HuggingFace Transformers  
- Pandas / NumPy  
- Google Colab (GPU)

---

## 📊 Model Architecture

- Pretrained Model: `bert-base-uncased`
- Task: Binary Text Classification
- Input: News/Text data
- Output: Fake (0) / Real (1)

---

## 📂 Project Structure
fake-news-bert/
│
├── data/ # Dataset files
├── src/ # Source code (data loader, model, training)
├── main.py # Entry point (prediction demo)
├── requirements.txt
└── README.md


---

## ⚙️ How to Run Locally

### 1. Clone Repository
git clone https://github.com/sphurti14/fake-news-bert.git
cd fake-news-bert

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run Prediction
python main.py


---

## ☁️ Model Training (Google Colab)

Training is performed in Google Colab using GPU to handle the heavy computation required by BERT.

Steps:
- Load dataset  
- Tokenize text  
- Train BERT model  
- Evaluate loss  
- Perform predictions  

---

## 🌐 Live Demo

https://fake-news-bert-d8ff6x9hvt7gqw3p63lmhx.streamlit.app/

## 📈 Example Prediction

Input:"Breaking news: Market crashes due to global crisis"


Output:
---

## 🎯 Future Improvements

- 📊 Use full-scale fake news dataset  
- 📉 Add evaluation metrics (Accuracy, Precision, Recall, F1-score)  
- 🌐 Deploy as web application (Streamlit/Flask)  
- 🤖 Integrate OpenAI for explainable predictions  
- 🔍 Add fake news source verification  

---

## 🧑‍💻 Author

**Sphurti Patil**

---

## ⭐ Note

This project demonstrates practical implementation of **Transformer-based NLP models** and follows an industry-level workflow combining local development with cloud-based training.