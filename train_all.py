from helpers.naive_bayes.nb_trainer import train as train_nb
from helpers.bert.bert_trainer import train as train_bert


def main():
    # print("=== Training Naive Bayes ===")
    # train_nb()

    print("\n=== Training DistilBERT ===")
    train_bert()

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
