import pandas as pd
import spacy
import re
import joblib

from src.data.preprocess import preprocess_batch
from src.data.preprocess import load_data
from sklearn.feature_extraction.text import TfidfVectorizer

def process_dataset() :

   print("Loading dataset")

   news_data = load_data("data/raw")

   print("Dataset Loaded")

   combined_text = news_data["news_headline"].astype(str) + " " + news_data["news_article"].astype(str)

   text = combined_text.str.lower()
   text = text.str.replace(r"https?:\S+", "", regex=True)
   text = text.str.replace(r"[^a-zA-z\s]", "", regex = True)

   news_data["cleaned_news"] = preprocess_batch(text)

   news_data.to_csv("data/processed/cleaned_news_dataset.csv", index=False)

   print("cleaned news data")


#    tfidfVectorizer = TfidfVectorizer(ngram_range = (1,2))

#    feature_extraction = tfidfVectorizer.fit_transform(preprocess_batch(text))

#    joblib.dump(tfidfVectorizer, "models/vectorizer.pkl")

#    print("File saved")


   




