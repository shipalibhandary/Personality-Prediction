import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Trait descriptions and feedback
trait_expansion = {
    'I': 'Introverted',
    'E': 'Extroverted',
    'N': 'Intuitive',
    'S': 'Sensing',
    'T': 'Thinking',
    'F': 'Feeling',
    'J': 'Judging',
    'P': 'Perceiving'
}

mbti_feedback = {
    'INTJ': "Strategic and independent thinker.",
    'INTP': "Analytical and curious problem-solver.",
    'ENTJ': "Confident leader with a goal-oriented mindset.",
    'ENTP': "Inventive and energetic, loves new ideas.",
    'INFJ': "Idealistic and empathetic, values deep meaning.",
    'INFP': "Creative dreamer guided by strong values.",
    'ENFJ': "Charismatic and warm, inspires others easily.",
    'ENFP': "Enthusiastic, imaginative, and people-focused.",
    'ISTJ': "Responsible, organized, and detail-oriented.",
    'ISFJ': "Loyal and caring, values harmony and duty.",
    'ESTJ': "Practical, structured, and takes charge.",
    'ESFJ': "Friendly, nurturing, and community-minded.",
    'ISTP': "Calm problem-solver, likes hands-on work.",
    'ISFP': "Gentle, artistic, and values personal freedom.",
    'ESTP': "Energetic risk-taker who lives in the moment.",
    'ESFP': "Playful, spontaneous, and loves social fun."
}

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dataset 
df = pd.read_csv(r'D:\PersonalityPred\mbti_cleaned.csv', encoding='latin1')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['clean_posts'])

# Train models for each trait
models = {}
traits = ['I_E', 'N_S', 'T_F', 'J_P']

for trait in traits:
    y = df[trait]
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    models[trait] = model

# Streamlit App 
st.set_page_config(page_title="Personality Prediction App", page_icon="🧠", layout="centered")

st.title("🧠 Personality Type Prediction (MBTI)")
st.markdown("Enter a text about yourself — your thoughts, behavior, or interests — and get your **MBTI personality type** instantly!")

user_text = st.text_area("✏️ Describe yourself:", placeholder="Example: I enjoy spending time alone reading and thinking about new ideas...")

if st.button("🔍 Predict Personality"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        
        clean_input = clean_text(user_text)
        input_tfidf = vectorizer.transform([clean_input])

        pred_I_E = models['I_E'].predict(input_tfidf)[0]
        pred_N_S = models['N_S'].predict(input_tfidf)[0]
        pred_T_F = models['T_F'].predict(input_tfidf)[0]
        pred_J_P = models['J_P'].predict(input_tfidf)[0]

        I_E = 'I' if pred_I_E == 0 else 'E'
        N_S = 'N' if pred_N_S == 0 else 'S'
        T_F = 'T' if pred_T_F == 0 else 'F'
        J_P = 'J' if pred_J_P == 0 else 'P'
        predicted_type = I_E + N_S + T_F + J_P

        full_form = " - ".join([trait_expansion[c] for c in predicted_type])
        feedback = mbti_feedback.get(predicted_type, "Unique personality with balanced traits.")

        st.success(f"🎯 **Predicted MBTI Type:** {predicted_type}")
        st.info(f"**Here is the Full Form\n:** {full_form}")
        st.markdown(f"💬 **Feedback\n:** {feedback}")
