
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_excel('preprocessed2.xlsx')
X = df['reply']
y = df['label_encoded']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("Training model...")
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))


joblib.dump(pipeline, 'sentiment1_model.pkl')
print("Model saved as 'sentiment1_model.pkl'")


loaded_model = joblib.load('sentiment1_model.pkl')
test_prediction = loaded_model.predict(["this looks great!"])[0]
test_proba = loaded_model.predict_proba(["this looks great!"])[0]
print(f"Test prediction: {test_prediction}, probabilities: {test_proba}")