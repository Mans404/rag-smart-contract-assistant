from langchain_openai import OpenAIEmbeddings
from .config import OPENAI_API_KEY, OPENAI_BASE_URL

def get_embeddings():
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
