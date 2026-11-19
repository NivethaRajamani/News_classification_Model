from fastapi import FastAPI
from pydantic import BaseModel
import requests
import joblib
from src.routing.store import route_article
from src.data.preprocess import preprocess_text

app = FastAPI()

model = joblib.load("models/news_classification_model.pkl")
vectorizer = joblib.load("models/vectorized_features.pkl")

class News(BaseModel) :
    text : str

@app.post("/predict")
def news_classify(news : News):

    x = vectorizer.transform([preprocess_text(news.text)])
    predicted_category = model.predict(x)[0]


    route_article(news.text, predicted_category)

    return {"category :", predicted_category }


