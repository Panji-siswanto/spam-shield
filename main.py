from helpers import training
import config
import joblib
from model.agent import SpamAgent


def debug():
    df = training.reader(config.DATA_PATH)
    corpus = training.clean_data(df)
    vectorizer, mail_train, mail_test, label_train, label_test = (
        training.vectorize_data(corpus, df)
    )
    model = training.train_model(mail_train, label_train)
    accuracy = training.evaluate_model(model, mail_test, label_test)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)

    print("Training completed")
    print(f"Accuracy: {accuracy:.4f}")
    print("Model & vectorizer saved")
    print("-" * 30)


def test_agent():
    agent = SpamAgent()

    sample_ham = "Please review the attached document and let me know your feedback."
    sample_spam = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."

    samples = [
        sample_ham,
        sample_spam,
    ]
    for sample in samples:
        proba = agent.predict_proba(sample)
        pred = agent.predict(sample)
        print(f"Input: {sample}")
        print(f"Confidence : Spam {proba['spam']:.2%} | Ham {proba['ham']:.2%}")
        print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")
        print("-" * 30)


if __name__ == "__main__":
    debug()
    test_agent()
