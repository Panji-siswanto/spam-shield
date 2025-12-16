import os
import pandas as pd
import torch

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

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna()
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged


def train_bert():
    print("Loading datasets...")
    df = load_and_merge_datasets(config.DATASETS)

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME)

    model = DistilBertForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels=2,
    )

    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]

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

    training_args = TrainingArguments(
        output_dir=config.BERT_OUTPUT_DIR,
        num_train_epochs=config.BERT_EPOCHS,
        per_device_train_batch_size=config.BERT_BATCH_SIZE,
        per_device_eval_batch_size=config.BERT_BATCH_SIZE,
        learning_rate=config.BERT_LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(config.BERT_OUTPUT_DIR, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("Starting DistilBERT training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(config.BERT_OUTPUT_DIR)
    tokenizer.save_pretrained(config.BERT_OUTPUT_DIR)

    print("DistilBERT training completed")


if __name__ == "__main__":
    train_bert()
