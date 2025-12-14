# model/agent.py

import joblib
import pandas as pd

import config
from helpers.training import clean_data


class SpamAgent:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)

    def _prepare(self, text):
        """
        Internal helper to clean and vectorize input text
        """
        df = pd.DataFrame({"text": [text]})
        cleaned = clean_data(df)
        vector = self.vectorizer.transform(cleaned)
        return vector

    def predict(self, text):
        """
        Return hard prediction: 1 = spam, 0 = ham
        """
        vector = self._prepare(text)
        return self.model.predict(vector)[0]

    def predict_proba(self, text):
        """
        Return probability scores for ham and spam
        """
        vector = self._prepare(text)
        proba = self.model.predict_proba(vector)[0]
        return {
            "ham": proba[0],
            "spam": proba[1],
        }
