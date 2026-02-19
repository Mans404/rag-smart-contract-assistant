from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL

def llm_chunk_text(text: str):

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Divide the following document into logical semantic chunks.
        Keep each chunk meaningful and coherent.

        Return chunks separated by: <CHUNK>

        Text:
        {text}
        """
    )

    chain = prompt | llm

    result = chain.invoke({"text": text})
    chunks = result.content.split("<CHUNK>")

    return [c.strip() for c in chunks if c.strip()]
