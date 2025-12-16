from model.hybrid_agent import HybridAgent


def main():
    agent = HybridAgent()

    samples = [
        "hello, how are you?",
        "Are you free later tonight?",
        "Congratulations! You've won $1000, click now!",
        "URGENT! Your account has been suspended. Verify now.",
    ]

    for text in samples:
        result = agent.predict(text)

        print("=" * 60)
        print(f'Input      : "{text}"')
        print(f"Prediction : {result['label']}")
        print(
            f"Confidence : Spam {result['spam_prob']:.2%} | "
            f"Ham {result['ham_prob']:.2%}"
        )
        print(
            f"(BERT spam: {result['bert_spam']:.2%}, NB spam: {result['nb_spam']:.2%})"
        )


if __name__ == "__main__":
    main()
