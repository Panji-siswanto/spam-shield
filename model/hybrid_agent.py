from model.bert_agent import BertAgent
from model.nb_agent import NBAgent


class HybridAgent:
    def __init__(self):
        self.bert = BertAgent()
        self.nb = NBAgent()

    def predict(self, text: str, threshold=0.6) -> dict:
        bert = self.bert.predict_proba(text)
        nb = self.nb.predict_proba(text)

        # base probabilities (start from BERT)
        spam = bert["spam"]
        ham = bert["ham"]

        # reinforce by nb
        if bert["spam"] > 0.6 or nb["spam"] > 0.7:
            spam = max(spam, nb["spam"], bert["spam"], 0.9)
        if bert["ham"] > 0.7 and nb["ham"] > 0.7:
            ham = max(ham, 0.9)

        if spam >= threshold:
            label = "Spam"
        elif ham >= threshold:
            label = "Ham"
        else:
            label = "Uncertain"

        return {
            "label": label,
            "spam_prob": spam,
            "ham_prob": ham,
            "bert_spam": bert["spam"],
            "nb_spam": nb["spam"],
        }
