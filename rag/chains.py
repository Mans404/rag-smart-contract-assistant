from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .config import OPENAI_API_KEY, MODEL_NAME, TOP_K, OPENAI_BASE_URL

def get_llm():

    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.5
    )

def build_qa_chain(vectorstore):

    llm = get_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the provided context.

        Context:
        {context}

        Question:
        {question}
        """
    )

    def qa_fn(question: str):
        try:
            docs = retriever.invoke(question)
            context = "\n\n".join([d.page_content for d in docs])

            chain = prompt | llm
            return chain.invoke({"context": context, "question": question}).content
        except Exception as e:
            return f"Error processing question: {str(e)}"

    return qa_fn


def summarize_text(full_text: str):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        Summarize this document clearly:

        {text}
        """
    )

    chain = prompt | llm
    return chain.invoke({"text": full_text}).content
