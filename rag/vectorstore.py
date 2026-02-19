from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings

def create_vectorstore(chunks):

    embeddings = get_embeddings()

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore