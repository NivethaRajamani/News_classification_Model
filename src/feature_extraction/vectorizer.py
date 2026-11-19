import pandas as pd
import joblib
import regex
from sklearn.feature_extraction.text import TfidfVectorizer



cleaned_news_data = pd.read_csv("data/processed/cleaned_news_dataset.csv")
cleaned_news_data["cleaned_news"] = cleaned_news_data['cleaned_news'].fillna("")


vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)

feature_extraction = vectorizer.fit_transform(cleaned_news_data["cleaned_news"])

joblib.dump(vectorizer, "models/vectorized_features.pkl")

print("Vectorized file saved successfully")

print(feature_extraction.shape)



