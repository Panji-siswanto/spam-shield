from model.bert_agent import BertAgent
from model.nb_agent import NBAgent


class HybridAgent:
    def __init__(self):
        self.bert = BertAgent()
        self.nb = NBAgent()

    def predict(self, text: str, threshold=0.6) -> dict:
        # normalize text
        clean_text = text.strip()
        word_count = len(clean_text.split())

        # user birt only if short
        if word_count <= 3:
            bert = self.bert.predict_proba(clean_text)

            spam = bert["spam"]
            ham = bert["ham"]

            if spam >= threshold and spam > ham:
                label = "Spam"
            else:
                label = "Ham"

            return {
                "label": "Ham",
                "spam_prob": spam,
                "ham_prob": ham,
                "bert_spam": bert["spam"],
                "nb_spam": None,
            }

        # for longer text
        bert = self.bert.predict_proba(clean_text)
        nb = self.nb.predict_proba(clean_text)

        spam = bert["spam"]
        ham = bert["ham"]

        if bert["spam"] > 0.7 and nb["spam"] > 0.7:
            spam = max(bert["spam"], nb["spam"])

        if bert["ham"] > 0.7 and nb["ham"] > 0.7:
            ham = max(bert["ham"], nb["ham"])

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
