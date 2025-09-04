import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
df = pd.read_csv("sample_tweets.csv")

# Features and labels
X = df["tweet"]
y = df["sentiment"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vect, y)

# Save model and vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training complete. Files saved: sentiment_model.pkl, vectorizer.pkl")
