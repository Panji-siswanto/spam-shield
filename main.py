from model.agent import SpamAgent


def test_agent():
    agent = SpamAgent()

    sample_ham = "hello."
    sample_spam = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."

    samples = [
        sample_ham,
        sample_spam,
    ]
    for sample in samples:
        result = agent.smart_predict(sample)

        print("=" * 60)
        print(f"Input      : {sample}")
        print(f"Prediction : {result['label']}")
        print(
            f"Confidence : Spam {result['spam_prob']:.2%} | "
            f"Ham {result['ham_prob']:.2%}"
        )


def user_input():
    agent = SpamAgent()
    while True:
        user_text = input("Enter email text (or 'exit' to quit): ")
        if user_text.lower() == "exit":
            break
        result = agent.smart_predict(user_text, threshold=0.75)

        print(f"Input: {user_text}")
        print(f"\nPrediction : {result['label']}")
        print(
            f"Confidence : Spam {result['spam_prob']:.2%} | "
            f"Ham {result['ham_prob']:.2%}"
        )
        print("-" * 60)


if __name__ == "__main__":
    SpamAgent.retrain()
    test_agent()
    # user_input()
