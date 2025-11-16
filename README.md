# 🧠 Personality Type Prediction using MBTI and Machine Learning


This project predicts a person's **MBTI (Myers–Briggs Type Indicator)** personality type based on text input.  It uses **TF-IDF vectorization** and **Logistic Regression** to classify text into one of the 16 MBTI types.


## 🧠 Project Overview
The goal of this project is to analyze text input and classify personality characteristics. The pipeline includes:
1. **Dataset Collection**
2. **Text Preprocessing** (cleaning, tokenizing, removing stopwords)
3. **Feature Extraction using TF-IDF**
4. **Train-Test Split**
5. **Model Training using Logistic Regression**
6. **Model Evaluation (Accuracy and Confusion Matrix)**
7. **Final Personality Prediction**

## 📂 Directory Structure
├── dataset/
│ └── personality_dataset.csv
├── src/
│ └── train_model.py
│ └── preprocess.py
│ └── predict.py
├── models/
│ └── logistic_regression_model.pkl
├── README.md
└── requirements.txt
