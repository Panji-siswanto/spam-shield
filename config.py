# config.py
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# File Paths
DATA_PATH = os.path.join(DATA_DIR, "spam_ham_dataset.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "vectorizer.pkl")

# Model Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
