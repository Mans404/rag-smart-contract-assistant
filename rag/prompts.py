from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt used for document (contract) question answering with history
CANDIDATE_PROMPT  = ChatPromptTemplate.from_messages([
    ("system", "You are a Smart Contract assistant. Use ONLY the provided contract context to answer. "
     "If the answer is not in the context, say you don't know and suggest what to ask instead."),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}\n\nContract Context:\n{context}")
])

# (Optional) summarization prompt used by the simple screen chain
SCREEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Smart Contract summarizer. Use the provided document evidence to produce a concise summary."),
    ("human",
     "Document:\n{evidence}\n\nReturn a short plain-text summary and 3 key points.")
])