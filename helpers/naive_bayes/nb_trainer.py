import os
import string
import joblib
import pandas as pd
import nltk
import config

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# NLTK setup
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stemmer = PorterStemmer()


# Text preprocessing
def clean_data(data: pd.DataFrame):
    if "text" not in data.columns:
        raise ValueError("Dataset must contain a 'text' column")

    data = data.dropna(subset=["text"])
    data["text"] = data["text"].astype(str)
    data["text"] = data["text"].apply(lambda x: x.replace("\r\n", " "))

    stopwords_set = set(stopwords.words("english"))
    corpus = []

    for text in data["text"]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation)).split()
        text = [stemmer.stem(w) for w in text if w not in stopwords_set]
        corpus.append(" ".join(text))

    return corpus


# loading dataset
def load_and_merge_datasets(paths):
    dfs = []

    for path in paths:
        df = pd.read_csv(path)

        if "text" not in df.columns:
            raise ValueError(f"{path} missing 'text' column")

        if "label_num" in df.columns:
            df["label_num"] = df["label_num"].astype(int)
        elif "label" in df.columns:
            df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
        else:
            raise ValueError(f"{path} missing label column")

        df = df[df["label_num"].isin([0, 1])]
        dfs.append(df[["text", "label_num"]])

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna()
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged


# Training entry point
def train():
    print("Training Naive Bayes model...")

    # Ensure output directory exists
    os.makedirs(config.NAIVE_BAYES_DIR, exist_ok=True)

    # Load + preprocess
    df = load_and_merge_datasets(config.DATASETS)
    corpus = clean_data(df)

    # Vectorizer (restored GOOD settings)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,  # 🔥 improves NB performance
    )

    X = vectorizer.fit_transform(corpus)
    y = df["label_num"].values

    # Stratified split (CRITICAL)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # Naive Bayes with smoothing
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        labels=[0, 1],
        zero_division=0,
    )

    # Clean output
    print("\nNaive Bayes Evaluation")
    print("-" * 25)
    print(f"Accuracy : {accuracy:.4f}\n")

    print(
        f"Ham  | Precision: {precision[0]:.4f} | "
        f"Recall: {recall[0]:.4f} | F1: {f1[0]:.4f}"
    )

    print(
        f"Spam | Precision: {precision[1]:.4f} | "
        f"Recall: {recall[1]:.4f} | F1: {f1[1]:.4f}"
    )

    # Save artifacts
    joblib.dump(model, config.NB_MODEL_PATH)
    joblib.dump(vectorizer, config.NB_VECTORIZER_PATH)

    print("\nNaive Bayes training completed")


if __name__ == "__main__":
    train()
