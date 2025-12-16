from model.bert_agent import BertAgent


def main():
    agent = BertAgent()

    samples = [
        "hello, how are you?",
        "Are you free later tonight?",
        "Congratulations! You've won $1000, click now!",
    ]

    for text in samples:
        result = agent.predict(text)

        print("=" * 60)
        print(f'Input      : "{text}"')
        print(f"Prediction : {result['label']}")
        print(f"Confidence : Spam {result['spam']:.2%} | Ham {result['ham']:.2%}")


if __name__ == "__main__":
    main()
