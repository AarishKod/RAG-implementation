"""
By Aarish Kodnaney
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


print(document_ids)