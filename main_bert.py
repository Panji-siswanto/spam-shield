from model.bert_agent import BertAgent


def main():
    agent = BertAgent()
    result = agent.predict("hello my name is panji, how are you?")
    print(result["label"])


if __name__ == "__main__":
    main()
