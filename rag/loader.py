from langchain_community.document_loaders import PyPDFLoader
from typing import List

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
