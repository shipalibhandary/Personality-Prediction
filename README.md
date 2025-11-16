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

PersonalityPred/
│
├── dataset/
│   └── mbti_cleaned.csv
│
├── src/
│   ├── train_all_traits.py
│   ├── predict.py
│   └── preprocess.py
│
├── models/
│   ├── I_E_model.pkl
│   ├── N_S_model.pkl
│   ├── T_F_model.pkl
│   └── J_P_model.pkl
│
├── static/
│   └── (CSS/JS files if you have UI)
│
├── templates/
│   └── index.html   (Flask UI page if applicable)
│
├── README.md
└── requirements.txt


## 🔧 Technologies Used
| Category | Tools / Libraries |
|---------|-------------------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| NLP | NLTK / spaCy, Scikit-learn TF-IDF Vectorizer |
| Model | Logistic Regression |
| Evaluation | Accuracy Score, Confusion Matrix |
