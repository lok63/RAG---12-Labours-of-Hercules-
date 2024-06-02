import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # take environment variables from .env.

OPENAI_KEY = os.getenv('OPENAI_KEY')

ROOT_DIR = Path(__file__).parent.parent.parent.parent
SERVICES_DIR = ROOT_DIR / 'services'
RAG_DIR = SERVICES_DIR / 'rag'
DATA_DIR = RAG_DIR / 'src/data'

HERCULES_VALIDATION_SET = DATA_DIR / 'hercules-validation.csv'
HERCULES_TRAINING_SET = DATA_DIR / 'hercules.txt'

if __name__ == '__main__':
    print(DATA_DIR)
