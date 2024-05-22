import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

OPENAI_KEY = os.getenv('OPENAI_KEY')

if __name__ == '__main__':
    print(OPENAI_KEY)