import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import config


class BertAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.BERT_OUTPUT_DIR)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            config.BERT_OUTPUT_DIR
        ).to(self.device)

        self.model.eval()

    def predict_proba(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1)[0]
        return {"ham": probs[0].item(), "spam": probs[1].item()}

    def predict(self, text: str) -> dict:
        proba = self.predict_proba(text)
        label = "Spam" if proba["spam"] >= proba["ham"] else "Ham"
        return {"label": label, **proba}
