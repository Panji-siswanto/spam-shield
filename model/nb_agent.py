import joblib
import pandas as pd
import config


class NBAgent:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)

    def predict_proba(self, text: str) -> dict:
        vec = self.vectorizer.transform([text])
        proba = self.model.predict_proba(vec)[0]
        return {"ham": proba[0], "spam": proba[1]}

    def smart_predict(self, text: str, threshold=0.62) -> dict:
        proba = self.predict_proba(text)

        if proba["spam"] >= threshold:
            label = "Spam"
        elif proba["ham"] >= threshold:
            label = "Ham"
        else:
            label = "Uncertain"

        return {"label": label, **proba}
