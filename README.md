spam-shield/
├── data/                         # Datasets
│   ├── email_origin.csv
│   ├── spam_ham_dataset.csv
│   ├── spam_sms.csv
│   └── train.csv
│
├── helpers/                      # Training & utilities
│   ├── bert/                     # BERT-related helpers
│   │   ├── dataset.py            # PyTorch dataset for BERT
│   │   ├── tokenizer.py          # DistilBERT tokenizer
│   │   └── training.py           # BERT fine-tuning logic
│   │
│   └── naive_bayes/              # Classical ML helpers
│       ├── training.py           # TF-IDF + Naive Bayes training
│       └── evaluation.py         # Evaluation metrics & reports
│
├── model/                        # Inference-only model interfaces
│   ├── agent.py                  # SpamAgent (TF-IDF + Naive Bayes)
│   └── bert_agent.py             # BertSpamAgent (DistilBERT)
│
├── output/                       # Model artifacts (gitignored)
│   ├── naive_bayes/
│   │   ├── spam_model.pkl
│   │   └── vectorizer.pkl
│   │
│   └── bert/                     # Fine-tuned BERT model files
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer.json
│
├── .gitignore
├── .python-version
├── config.py                     # Global configuration
├── main.py                       # Entry point / demo
├── pyproject.toml
├── README.md
├── spam-shield.code-workspace
└── uv.lock


Datasets:
    spam_ham_dataset.csv // https://www.kaggle.com/datasets/venky73/spam-mails-dataset?resource=download
    spam_sms.csv // https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset
    email_origin //https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-enron-2006