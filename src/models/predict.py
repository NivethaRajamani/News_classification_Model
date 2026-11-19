import pandas as pd
import joblib
from src.data.preprocess import preprocess_text

model = joblib.load("models/news_classification_model.pkl")
vectorizer = joblib.load("models/vectorized_features.pkl")

while True :

    news = input("enter news article to classify")

    if news!= " " :

        vectorized = vectorizer.transform([preprocess_text(news)])
        print(model.predict(vectorized)[0])

    else :
        break


