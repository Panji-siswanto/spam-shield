# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset paths (MULTIPLE)
DATASETS = [
    os.path.join(DATA_DIR, "spam_ham_dataset.csv"),
    os.path.join(DATA_DIR, "spam_sms.csv"),
    os.path.join(DATA_DIR, "train.csv"),
]

MODEL_PATH = os.path.join(OUTPUT_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "vectorizer.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42
