import pandas as pd
import string
import config
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stemmer = PorterStemmer()


def reader(path):
    return pd.read_csv(path)


def clean_data(data):
    # Ensure text column exists
    if "text" not in data.columns:
        raise ValueError("Dataset must contain a 'text' column")

    # Drop rows with missing text
    data = data.dropna(subset=["text"])

    # Force text to string (handles floats safely)
    data["text"] = data["text"].astype(str)

    # Normalize line breaks
    data["text"] = data["text"].apply(lambda x: x.replace("\r\n", " "))

    stopwords_set = set(stopwords.words("english"))
    corpus = []

    for text in data["text"]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopwords_set]
        corpus.append(" ".join(text))

    return corpus


def vectorize_data(corpus, data):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    mail = vectorizer.fit_transform(corpus)
    label = data.label_num.values

    mail_train, mail_test, label_train, label_test = train_test_split(
        mail, label, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    return vectorizer, mail_train, mail_test, label_train, label_test


def train_model(mail_train, label_train):
    model = MultinomialNB()
    model.fit(mail_train, label_train)
    return model


def evaluate_model(model, mail_test, label_test):
    accuray = model.score(mail_test, label_test)
    return accuray


def load_and_merge_datasets(paths):
    dfs = []

    for path in paths:
        df = pd.read_csv(path)

        # Ensure required columns exist
        if "text" not in df.columns:
            raise ValueError(f"{path} missing 'text' column")

        # Normalize label column
        if "label_num" in df.columns:
            df["label_num"] = df["label_num"].astype(int)
        elif "label" in df.columns:
            # common alternative
            df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
        else:
            raise ValueError(f"{path} missing label column")

        # Keep only valid labels
        df = df[df["label_num"].isin([0, 1])]

        dfs.append(df[["text", "label_num"]])

    merged_df = pd.concat(dfs, ignore_index=True)

    # Drop bad rows
    merged_df = merged_df.dropna(subset=["text", "label_num"])

    # Shuffle
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged_df


def evaluate_detailed(model, mail_test, label_test):
    preds = model.predict(mail_test)

    print("Confusion Matrix:")
    print(confusion_matrix(label_test, preds, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            label_test, preds, labels=[0, 1], target_names=["Ham", "Spam"]
        )
    )
