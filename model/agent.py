from helpers.training import load_and_merge_datasets

import os
import joblib
import pandas as pd

import config
from helpers.training import (
    reader,
    clean_data,
    vectorize_data,
    train_model,
    evaluate_model,
)


class SpamAgent:
    def __init__(self):
        # Ensure model & vectorizer exist, train if needed
        SpamAgent.ensure()
        self.model = joblib.load(config.MODEL_PATH)
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)

    @staticmethod
    def ensure():
        """
        Ensure trained model and vectorizer exist.
        Train and save if not.
        """
        model_exists = os.path.exists(config.MODEL_PATH)
        vectorizer_exists = os.path.exists(config.VECTORIZER_PATH)
        if model_exists and vectorizer_exists:
            return

        df = load_and_merge_datasets(config.DATASETS)
        corpus = clean_data(df)
        vectorizer, mail_train, mail_test, label_train, label_test = vectorize_data(
            corpus, df
        )
        model = train_model(mail_train, label_train)
        accuracy = evaluate_model(model, mail_test, label_test)

        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(vectorizer, config.VECTORIZER_PATH)

        print(f"Training completed | Accuracy: {accuracy:.4f}")
        print("Model & vectorizer saved")

    @staticmethod
    def retrain():
        """
        Force retraining of the model.
        Deletes existing artifacts and retrains from scratch.
        """
        # Delete old artifacts if they exist
        if os.path.exists(config.MODEL_PATH):
            os.remove(config.MODEL_PATH)

        if os.path.exists(config.VECTORIZER_PATH):
            os.remove(config.VECTORIZER_PATH)

        print("Old model artifacts deleted. Retraining...")

        # Reuse ensure() to train again
        SpamAgent.ensure()

    def _prepare(self, text: str):
        """
        Clean and vectorize input text.
        """
        df = pd.DataFrame({"text": [text]})
        cleaned = clean_data(df)
        vector = self.vectorizer.transform(cleaned)
        return vector

    def predict(self, text: str) -> int:
        """
        Hard prediction.
        Returns:
            1 = spam
            0 = ham
        """
        vector = self._prepare(text)
        return self.model.predict(vector)[0]

    def predict_proba(self, text: str) -> dict:
        """
        Probability prediction.
        Returns:
            {
                "ham": float,
                "spam": float
            }
        """
        vector = self._prepare(text)
        proba = self.model.predict_proba(vector)[0]

        return {
            "ham": proba[0],
            "spam": proba[1],
        }

    def smart_predict(self, text: str, threshold: float = 0.75) -> dict:
        """
         prediction confidence threshold.
        Returns:
            {
                "label": "Spam" | "Ham" | "Uncertain",
                "spam_prob": float,
                "ham_prob": float
            }
        """
        proba = self.predict_proba(text)
        spam_prob = proba["spam"]
        ham_prob = proba["ham"]

        if spam_prob >= threshold:
            label = "Spam"
        elif ham_prob >= threshold:
            label = "Ham"
        else:
            label = "Uncertain"

        return {
            "label": label,
            "spam_prob": spam_prob,
            "ham_prob": ham_prob,
        }
