from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings as ChromaSettings


def get_embeddings(embedding_model: str, openai_api_key: str):
    return OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)


def get_vectorstore(persist_directory: str, collection_name: str, embedding_model: str, openai_api_key: str):
    embeddings = get_embeddings(embedding_model, openai_api_key)
    chroma_settings = ChromaSettings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=persist_directory,
    )
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=chroma_settings,
        persist_directory=persist_directory,
    )
    return vs


def make_retriever(vectorstore: Chroma, k: int = 5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_qa_chain(retriever, model_name: str, temperature: float, openai_api_key: str):
    llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=openai_api_key)

    prompt_template = """You are a helpful assistant. Use the provided context to answer the question succinctly.
If the answer is not contained in the context, say "I don't know."

Context:
{context}
Question: {question}
Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa