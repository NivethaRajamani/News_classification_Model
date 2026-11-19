import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.models.build_model import build_model
from collections import Counter
from imblearn.over_sampling import RandomOverSampler


vectorizer = joblib.load("models/vectorized_features.pkl")

news_data = pd.read_csv("data/processed/cleaned_news_dataset.csv")
news_data["cleaned_news"] = news_data["cleaned_news"].fillna("")

x = vectorizer.transform(news_data["cleaned_news"])

y = news_data["news_category"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Before balancing: ", Counter(y_train))

ros = RandomOverSampler(random_state = 4)
x_train, y_train = ros.fit_resample(x_train, y_train)

print("After balancing: ", Counter(y_train))

model = build_model()

model.fit(x_train, y_train)

joblib.dump(model, "models/news_classification_model.pkl")

print("Training completed")


