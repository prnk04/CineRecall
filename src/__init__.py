# src/__init__.py

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


@st.cache_resource
def get_vector_store(version=None):
    """
    Get vector store with consistent configuration

    Args:
        version: ChromaDB version (e.g., 'v0'). If None, uses env var.
    """
    embedding_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-mpnet-base-v2")
    collection_name = os.getenv("CHROMA_MOVIES_COLLECTION", "plotseek_movies")
    chroma_version = version or os.getenv("CHROMA_DB_VERSION", "v0")
    chroma_base_dir = os.getenv("CHROMA_DIR", "data/chroma")

    # Construct full collection name with version
    full_collection_name = f"{collection_name}_{chroma_version}"

    # Construct full path with version
    persist_path = f"{chroma_base_dir}/{chroma_version}"

    print(f"[VectorStore] Collection: {full_collection_name}")
    print(f"[VectorStore] Path: {persist_path}")
    print(f"[VectorStore] Model: {embedding_model}")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=full_collection_name,
        embedding_function=embeddings,
        persist_directory=persist_path,
    )
