import os
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import config
from helpers.bert.dataset import SpamDataset


def load_and_merge_datasets(paths):
    dfs = []

    for path in paths:
        df = pd.read_csv(path)

        if "text" not in df.columns:
            raise ValueError(f"{path} missing 'text' column")

        if "label_num" in df.columns:
            df["label"] = df["label_num"].astype(int)
        elif "label" in df.columns:
            df["label"] = df["label"].map({"ham": 0, "spam": 1})
        else:
            raise ValueError(f"{path} missing label column")

        df = df[df["label"].isin([0, 1])]
        dfs.append(df[["text", "label"]])

    merged = pd.concat(dfs, ignore_index=True).dropna()
    return merged.sample(frac=1, random_state=42).reset_index(drop=True)


def train():
    print("Loading datasets...")
    df = load_and_merge_datasets(config.DATASETS)

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels=2,
    )

    split = int(0.8 * len(df))
    train_df, val_df = df[:split], df[split:]

    train_dataset = SpamDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        config.MAX_SEQ_LEN,
    )

    val_dataset = SpamDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        config.MAX_SEQ_LEN,
    )

    args = TrainingArguments(
        output_dir=config.BERT_OUTPUT_DIR,
        num_train_epochs=config.BERT_EPOCHS,
        per_device_train_batch_size=config.BERT_BATCH_SIZE,
        learning_rate=config.BERT_LR,
        logging_dir=os.path.join(config.BERT_OUTPUT_DIR, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.BERT_OUTPUT_DIR)
    tokenizer.save_pretrained(config.BERT_OUTPUT_DIR)

    print("DistilBERT training completed")
