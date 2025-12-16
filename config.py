import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# Data
# ======================
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASETS = [
    os.path.join(DATA_DIR, "spam_ham_dataset.csv"),
    os.path.join(DATA_DIR, "spam_sms.csv"),
    os.path.join(DATA_DIR, "train.csv"),
    os.path.join(DATA_DIR, "email_text.csv"),
]

# ======================
# Output base
# ======================
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# Naive Bayes (Classical)
# ======================
NAIVE_BAYES_DIR = os.path.join(OUTPUT_DIR, "naive_bayes")
os.makedirs(NAIVE_BAYES_DIR, exist_ok=True)

NB_MODEL_PATH = os.path.join(NAIVE_BAYES_DIR, "spam_model.pkl")
NB_VECTORIZER_PATH = os.path.join(NAIVE_BAYES_DIR, "vectorizer.pkl")

# (Optional backward compatibility)
MODEL_PATH = NB_MODEL_PATH
VECTORIZER_PATH = NB_VECTORIZER_PATH

# ======================
# BERT
# ======================
BERT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "bert")
os.makedirs(BERT_OUTPUT_DIR, exist_ok=True)

BERT_MODEL_NAME = "distilbert-base-uncased"

# ======================
# Training settings
# ======================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# BERT hyperparameters
BERT_EPOCHS = 1
BERT_BATCH_SIZE = 16
MAX_SEQ_LEN = 64
BERT_LR = 2e-5


def ensure_dirs():
    os.makedirs(NAIVE_BAYES_DIR, exist_ok=True)
    os.makedirs(BERT_OUTPUT_DIR, exist_ok=True)
