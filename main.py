from model.agent import SpamAgent


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


def user_input():
    agent = SpamAgent()
    while True:
        user_text = input("Enter email text (or 'exit' to quit): ")
        if user_text.lower() == "exit":
            break

        proba = agent.predict_proba(user_text)
        pred = agent.predict(user_text)

        print(f"Input: {user_text}")
        print(f"Confidence : Spam {proba['spam']:.2%} | Ham {proba['ham']:.2%}")
        print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")
        print("-" * 30)


if __name__ == "__main__":
    user_input()
