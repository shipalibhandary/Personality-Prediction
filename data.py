import pandas as pd
import re

# Load dataset
df = pd.read_csv(r'D:\PersonalityPred\mbti_1.csv', encoding='latin1')

# Function to clean text
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\|\|\|', ' ', text)  
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

# Apply cleaning
df['clean_posts'] = df['posts'].apply(clean_text)

# Display cleaned text
print(df[['type', 'clean_posts']].head())

# Split MBTI types into 4 traits
df['I_E'] = df['type'].apply(lambda x: 0 if 'I' in x[0] else 1)  # 0 for Introvert, 1 for Extrovert
df['N_S'] = df['type'].apply(lambda x: 0 if 'N' in x[1] else 1)  # 0 for Intuitive, 1 for Sensing
df['T_F'] = df['type'].apply(lambda x: 0 if 'T' in x[2] else 1)  # 0 for Thinking, 1 for Feeling
df['J_P'] = df['type'].apply(lambda x: 0 if 'J' in x[3] else 1)  # 0 for Judging, 1 for Perceiving

print(df[['type', 'I_E', 'N_S', 'T_F', 'J_P']].head())

# Save cleaned DataFrame to a new CSV file for later use
df.to_csv(r'D:\PersonalityPred\mbti_cleaned.csv', index=False, encoding='latin1')
print("✅ Cleaned data saved to D:\\PersonalityPred\\mbti_cleaned.csv")
