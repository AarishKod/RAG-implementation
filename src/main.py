"""
By Aarish Kodnaney
"""

import os
from typing import List, Tuple, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent


# loading environment variables
load_dotenv()
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING: str | None = os.getenv("LANGSMITH_TRACING")

# turn pdf into a list of document objects
FILE_PATH: str = "./data/porsche_2024_annual_report_english.pdf"
loader: PyPDFLoader = PyPDFLoader(file_path=FILE_PATH)
docs: List[Document] = loader.load()

# initializing components
claude: BaseChatModel = init_chat_model("claude-sonnet-4-5-20250929")
embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
vector_store: Chroma = Chroma(
    collection_name="porsche-2024-annual-report",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# initializing text splitter and creating list of chunks with type Document
# (each document obj is now a much smaller chunk)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits: List[Document] = text_splitter.split_documents(docs)

# embedding and storing documents
document_ids: List[str] = vector_store.add_documents(documents=all_splits)

# Rag Agent Implementation

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> Tuple[str, List[Document]]:
    """
    Retrieve info to feed to llm to answer query

    args:
        query:
            type: str
            the query being prompted to the llm

    returns:

    """
    retrieved_documents: List[Document] = vector_store.similarity_search(query, k=2) # k=2 means returns at most 2 chunks
    serialized: str = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_documents
    )

    return serialized, retrieved_documents


tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post."
    "Use the tool to help answer user queries."
)

# creating agent
agent: Any = create_agent(model=claude, tools=tools, system_prompt=prompt)

query = (
    "What is Porsche's outlook for the future?\n\n"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]}, stream_mode="values"
):
    event["messages"][-1].pretty_print()

