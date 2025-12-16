spam-shield/
├── data/                         # CSV datasets
│   ├── email_text.csv
│   ├── spam_ham_dataset.csv
│   ├── spam_sms.csv
│   └── train.csv
│
├── helpers/                      # Training & preprocessing logic
│   ├── bert/
│   │   ├── bert_trainer.py       # DistilBERT fine-tuning
│   │   ├── dataset.py            # PyTorch Dataset for BERT
│   │   └── prediction.py         # (Optional) BERT utilities
│   │
│   └── naive_bayes/
│       └── nb_trainer.py         # TF-IDF + Naive Bayes training
│
├── model/                        # Inference-only agents
│   ├── nb_agent.py               # Naive Bayes inference
│   ├── bert_agent.py             # DistilBERT inference
│   └── hybrid_agent.py           # Hybrid (NB + BERT)
│
├── output/                       # Trained models (gitignored)
│   ├── naive_bayes/
│   │   ├── spam_model.pkl
│   │   └── vectorizer.pkl
│   └── bert/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer.json
│
├── train_all.py                  # Train all models
├── main_nb.py                    # Run Naive Bayes demo
├── main_bert.py                  # Run DistilBERT demo
├── main_hybrid.py                # Run Hybrid demo
│
├── config.py                     # Global configuration
├── .gitignore
├── pyproject

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






Datasets:
    spam_ham_dataset.csv // https://www.kaggle.com/datasets/venky73/spam-mails-dataset?resource=download
    spam_sms.csv // https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset
    email_text.csv //https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-enron-2006