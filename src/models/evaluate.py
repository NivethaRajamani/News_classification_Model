import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

news_data = pd.read_csv("data/processed/cleaned_news_dataset.csv")

vectorizer = joblib.load("models/vectorized_features.pkl")
model = joblib.load("models/news_classification_model.pkl")

x = vectorizer.transform(news_data["cleaned_news"])
y = news_data["news_category"]

x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state=42)

prediction = model.predict(x_test)

print(classification_report(y_test, prediction))

labels = sorted(y.unique())

cm = confusion_matrix(y_test, prediction, labels = labels)

cm_table = pd.DataFrame(cm, index = labels, columns= labels)

print("\n Confusion matrix (rows = actual, columns = predicted)")

print(cm_table)






