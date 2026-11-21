from pathlib import Path
import pickle
import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from config import settings

def get_device():
    """Automatically select CUDA when available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings():
    """Load the configured local Hugging Face embedding model."""
    device = get_device()
    print(f"Loading embedding model on: {device.upper()}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_qdrant_connection_kwargs():
    endpoint = settings.qdrant_cluster_endpoint.strip()
    api_key = settings.qdrant_api_key.strip()
    if not endpoint or not api_key:
        raise ValueError(
            "QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY must both be set to use Qdrant Cloud."
        )
    return {"url": endpoint, "api_key": api_key}

def build_index(chunks):
    store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        **get_qdrant_connection_kwargs(),
        collection_name=settings.qdrant_collection,
        force_recreate=True
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.sparse_k

    Path(settings.bm25_path).parent.mkdir(parents=True, exist_ok=True)

    with open(settings.bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Successfully indexed {len(chunks)} chunks.")

    return store, bm25

def load_index():
    store = QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        **get_qdrant_connection_kwargs(),
        collection_name=settings.qdrant_collection
    )

    with open(settings.bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    return store, bm25
