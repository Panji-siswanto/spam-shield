spam-shield/
│
├── api/                          # FastAPI backend
│   ├── main.py                   # API launcher
│   ├── schemas.py                # Request/response schemas
│   ├── nb_routes.py              # Naive Bayes endpoints
│   ├── bert_routes.py            # BERT endpoints
│   └── hybrid_routes.py          # Hybrid endpoints
│
├── data/                         # Datasets (CSV)
│   ├── email_text.csv
│   ├── spam_ham_dataset.csv
│   ├── spam_sms.csv
│   └── train.csv
│
├── helpers/                      # Training utilities
│   ├── bert/
│   │   ├── bert_trainer.py       # DistilBERT fine-tuning
│   │   ├── dataset.py            # PyTorch Dataset
│   │
│   └── naive_bayes/
│       ├── nb_trainer.py         # TF-IDF + MultinomialNB training
│       └── evaluation.py         # Precision / Recall / F1 / Accuracy
│
├── model/                        # Inference-only agents
│   ├── nb_agent.py               # Naive Bayes agent
│   ├── bert_agent.py             # DistilBERT agent
│   └── hybrid_agent.py           # Hybrid agent
│
├── output/                       # Model artifacts (gitignored)
│   ├── naive_bayes/   
│   │
│   └── bert/
│
├── main_nb.py                    # Run NB prediction locally
├── main_bert.py                  # Run BERT prediction locally
├── main_hybrid.py                # Run Hybrid prediction locally
├── train_all.py                  # Train all models
│
└──config.py                     # Global configuration

helpers/ -> training & preprocessing
model/ -> inference only
output/ -> generated models (not pushed to Git)
config,py -> stores All paths and hyperparameters


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