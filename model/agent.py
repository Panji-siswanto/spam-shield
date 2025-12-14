# model/agent.py

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
        # laod model & vectorizer
        SpamAgent.ensure()
        self.model = joblib.load(config.MODEL_PATH)
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)

    @staticmethod
    def ensure():
        # check if model & vectorizer exist if not, train the model
        model_exists = os.path.exists(config.MODEL_PATH)
        vectorizer_exists = os.path.exists(config.VECTORIZER_PATH)
        if model_exists and vectorizer_exists:
            return

        print("Trained model not found. Training now...")

        df = reader(config.DATA_PATH)
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
