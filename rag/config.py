import os
from dotenv import load_dotenv

# تحديد المسار الصحيح للـ .env في root المشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INDEX_DIR = os.path.join(DATA_DIR, "index")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
TOP_K = int(os.getenv("TOP_K", "4"))

# Debug (احذفه بعد ما تتأكد)
print("MODEL_NAME:", MODEL_NAME)
