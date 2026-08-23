import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample Data
data = {
    "text": [
        "This product is amazing and works perfectly",
        "Worst experience ever, totally useless",
        "Great quality and fast delivery",
        "Terrible customer service, very disappointed",
        "Loved it, highly recommended!",
        "Poor quality, broke on the first day"
    ],
    "sentiment": ["positive", "negative", "positive", "negative", "positive", "negative"]
}

df = pd.DataFrame(data)

# Pipeline
model_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train & Save
model_pipeline.fit(df["text"], df["sentiment"])

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Model successfully trained and saved!")