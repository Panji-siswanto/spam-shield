# Spam Shield 🍆💦
Spam Shield is a spam detection system built using a hybrid machine learning approach.
It combines classical text classification with modern transformer-based models to detect spam
in chat messages and email-like text.

The system supports **conversation-level context**, allowing multiple chat bubbles to be analyzed
together instead of treating each message independently.

Features

- Naive Bayes (TF-IDF) spam classifier
- DistilBERT-based deep learning classifier
- Hybrid model combining NB + BERT
- Conversation-context prediction
- REST API built with FastAPI
- Batch and single-message prediction
- Locally trained models (no external inference)

---

Models
1. Naive Bayes (TF-IDF)
- Uses unigrams and bigrams
- Effective for keyword-based spam detection
- Fast and lightweight

2. DistilBERT
- Fine-tuned on spam/ham datasets
- Captures semantic meaning
- More conservative on ambiguous messages

3. Hybrid Model
- Combines NB and BERT probabilities
- Uses OR-based escalation for contextual input
- Reduces false negatives in chat-based spam

---

##  Conversation Context
Instead of predicting a single message, Spam Shield can analyze
multiple chat messages as one context:

```json
{
  "messages": [
    "Hi",
    "Are you busy right now?",
    "You have been selected for a free gift card",
    "Act now before it expires"
  ]
}



Project Structure
spam-shield/
│
├── api/                    # FastAPI layer
│   ├── __init__.py
│   ├── main.py             # FastAPI app entry
│   ├── nb_routes.py        # Naive Bayes endpoints
│   ├── bert_routes.py      # BERT endpoints
│   ├── hybrid_routes.py    # Hybrid endpoints
│   └── schemas.py          # Request schemas
│
├── data/                   # Datasets (CSV)
│   ├── email_text.csv
│   ├── spam_ham_dataset.csv
│   ├── spam_sms.csv
│   └── train.csv
│
├── helpers/                # Training utilities (offline)
│   ├── bert/
│   │   ├── bert_trainer.py
│   │   └── dataset.py
│   └── naive_bayes/
│       ├── nb_trainer.py
│       └── evaluation.py
│
├── model/                  # Inference-only agents
│   ├── __init__.py
│   ├── nb_agent.py
│   ├── bert_agent.py
│   └── hybrid_agent.py
│
├── utils/                  # Shared utilities
│   ├── __init__.py
│   └── context.py          # Conversation context builder
│
├── output/                 # Trained models (gitignored)
│   ├── bert/
│   └── naive_bayes/
│
├── config.py               # Global configuration
├── train_all.py            # Train all models
├── main_nb.py              # Local NB testing
├── main_bert.py            # Local BERT testing
├── main_hybrid.py          # Local Hybrid testing
│
├── README.md               # Project documentation
├── pyproject.toml
├── uv.lock
├── .gitignore
└── .python-version

data/ → Datasets
helpers/ -> training & preprocessing utils
model/ -> inference for agents
utils/ → Shared utilities (context builder)
output/ -> generated training models (gitignored)
config.py -> stores All paths and hyperparameters


to run models:
naive bayes 
 "uv run python main_nb.py" /"python main_nb.py"
DistilBERT 
 "uv run python main_bert.py" /"python main_bert.py"
Hybrid Model 
 "uv run python main_hybrid.py" / "python main_hybrid.py"

on pull, the output folders will be empty and to initiate models training, run:
 "uv run python train_all.py"


APIs:
to initate:
 "uv run uvicorn api.api:app --reload"
NB:
POST http://127.0.0.1:8000/predict/nb
BERT:
POST http://127.0.0.1:8000/predict/bert
Hybrid:
POST http://127.0.0.1:8000/predict/hybrid






Datasets:
    spam_ham_dataset.csv // https://www.kaggle.com/datasets/venky73/spam-mails-dataset?resource=download
    spam_sms.csv // https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset
    email_text.csv //https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-enron-2006