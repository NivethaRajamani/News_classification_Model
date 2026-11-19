import pandas as pd
import spacy
import re

from src.data.load_data import load_data
import nltk


df = load_data("data/raw")

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text) :

    text = str(text).lower()
    text = re.sub(r"https?\S+", "", text)
    text = re.sub(r"[^a-zA-z\s]", "", text)

    doc = nlp(text)

    words = []

    for token in doc :
        if not token.is_stop :
            words.append(token.lemma_)

    return " ".join(words)

def preprocess_batch(text) :

    cleaned = []

    text = text.astype(str).str.lower()

    docs = nlp.pipe(text, batch_size = 200, disable = ["parser", "ner"] )

    for doc in docs :

        words = [token.lemma_ for token in doc if not token.is_stop]
        cleaned.append(" ".join(words))

    return cleaned


        


