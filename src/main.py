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
from langchain.agents.middleware import dynamic_prompt, ModelRequest


# loading environment variables
load_dotenv()
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING: str | None = os.getenv("LANGSMITH_TRACING")
PERSIST_DIRECTORY: str = "./chroma_langchain_db"
FILE_PATH: str = "./data/porsche_2024_annual_report_english.pdf"

# turn pdf into a list of document objects
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


if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    print("loading existing vector store")
    print(f"Loaded {vector_store._collection.count()} documents from existing store")
else:
    print("creating new vector store. this will take a bit of time. go grab a snack")
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

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """
    inject context into state messages
    """
    last_query: str = request.state["messages"][-1].text
    retrieved_documents: List[Document] = vector_store.similarity_search(last_query)

    docs_content: str = "\n\n".join(doc.page_content for doc in retrieved_documents)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )
    return system_message

tools: list = []

# creating agent
agent: Any = create_agent(model=claude, tools=tools, middleware=[prompt_with_context])

query = (
    "what are they saying about the Porsche 911"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]}, stream_mode="values"
):
    event["messages"][-1].pretty_print()

