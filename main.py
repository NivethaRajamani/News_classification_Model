from src.data.load_data import load_data
from src.data.preprocess import preprocess_text
from src.data.process_dataset import process_dataset

# df = load_data("data/raw")

# print(df.shape)
# print(df.columns)
# print(df.head(5))
# print(df["news_category"].value_counts())

# sample = "Breaking News!!! Visit https://example.com for the latest updates on technology, AI, and business developments."

# print(preprocess_text(sample))

process_dataset()

