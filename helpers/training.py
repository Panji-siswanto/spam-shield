import numpy as np
import pandas as pd
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
stemmer = PorterStemmer()


def reader(path):
    df = pd.read_csv(path)
    return df


def clean_data(data):
    data["text"] = data["text"].apply(lambda x: x.replace("\r\n", " "))

    stopwords_set = set(stopwords.words("english"))
    corpus = []

    for i in range(len(data)):
        text = data["text"].iloc[i].lower()
        text = text.translate(str.maketrans("", "", string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopwords_set]
        text = " ".join(text)
        corpus.append(text)

    return corpus


def vectorize_data(corpus, data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    mail = vectorizer.fit_transform(corpus)
    label = data.label_num.values

    mail_train, mail_test, label_train, label_test = train_test_split(
        mail, label, test_size=0.2
    )

    return vectorizer, mail_train, mail_test, label_train, label_test


def train_model(mail_train, label_train):
    model = MultinomialNB()
    model.fit(mail_train, label_train)
    return model


def evaluate_model(model, mail_test, label_test):
    accuracy = model.score(mail_test, label_test)
    return accuracy
