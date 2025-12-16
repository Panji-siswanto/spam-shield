import os
import string
import joblib
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import config

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stemmer = PorterStemmer()


def clean_data(df):
    stop_set = set(stopwords.words("english"))
    corpus = []

    for text in df["text"]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation)).split()
        text = [stemmer.stem(w) for w in text if w not in stop_set]
        corpus.append(" ".join(text))

    return corpus


def load_and_merge_datasets(paths):
    dfs = []

    for path in paths:
        df = pd.read_csv(path)

        if "label_num" in df.columns:
            df["label"] = df["label_num"]
        else:
            df["label"] = df["label"].map({"ham": 0, "spam": 1})

        dfs.append(df[["text", "label"]])

    return pd.concat(dfs).dropna()


def train():
    print("Training Naive Bayes...")

    df = load_and_merge_datasets(config.DATASETS)
    corpus = clean_data(df)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(corpus)
    y = df["label"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    model = MultinomialNB()
    model.fit(Xtr, ytr)

    print(classification_report(yte, model.predict(Xte)))

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)

    print("NB training completed")
