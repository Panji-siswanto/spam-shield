## Details Of Features
- Naive Bayes (TF-IDF) spam classifier
- DistilBERT-based deep learning classifier
- Hybrid model combining NB + BERT
- Conversation-context prediction
- REST API built with FastAPI
- Batch and single-message prediction
- Locally trained models (no external inference)


## Details of Agent Models
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



## Project Structure
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

donwload and setup 'uv' for pyhton first before run, or convert the commands given to standard python instruction.

to run:
fresh after pull, train the agents first,
run on terminal :
"uv run python train_all.py"
config for training can be found in config.py as '#BERT hyperparameters'
and the train result is stored as .pkl file at 'output/'

after training is done, initiate API.
run on terminal : 
"uv run uvicorn api.main:app --reload"
press CTRL + C to shut it down

by this point agents is set and APIs are ready to use.
move to spam-shield-app ("https://github.com/Panji-siswanto/spam-shield-app")
pull the repositories and run the programme, for details of it can be found it the files @README.md


to test each models:
naive bayes 
 "uv run python main_nb.py" /"python main_nb.py"
DistilBERT 
 "uv run python main_bert.py" /"python main_bert.py"
Hybrid Model 
 "uv run python main_hybrid.py" / "python main_hybrid.py"


APIs:
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