import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r'D:\PersonalityPred\mbti_cleaned.csv', encoding='latin1')

#step 3: Extract One Personality Trait
df['IE'] = df['type'].apply(lambda x: 'I' if x[0] == 'I' else 'E')

#Step 4: Split the Data
X = df['clean_posts']
y = df['IE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 5: Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF shape:", X_train_tfidf.shape)

#Step 6: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

#Step 7: Evaluate Model Performance
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Step 8: Test the Model with a Custom Example
test_text = ["I love spending time alone reading books about psychology"]
test_vector = vectorizer.transform(test_text)
prediction = model.predict(test_vector)
print("Predicted Personality (I/E):", prediction[0])
