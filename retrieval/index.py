from pathlib import Path
import pickle
import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from config import settings
from langchain_core.embeddings import Embeddings
import logging

def get_device():
    """Automatically select CUDA when available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings():
    """Return an embedding model. If HuggingFaceEmbeddings cannot be imported, fall back to a dummy
    implementation that satisfies LangChain's Embeddings interface with 384‑dim vectors.
    """
    device = get_device()
    print(f"Loading embedding model on: {device.upper()}")
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        logging.warning("HuggingFaceEmbeddings import failed (%s). Using dummy embeddings.", e)
        class DummyEmbeddings(Embeddings):
            """Dummy embeddings returning zero vectors of dimension 384 for compatibility."""
            def embed_documents(self, texts):
                import numpy as np
                return np.zeros((len(texts), 384)).tolist()
            def embed_query(self, text):
                import numpy as np
                return np.zeros(384).tolist()
        return DummyEmbeddings()

def build_index(chunks):
    Path(settings.qdrant_path).parent.mkdir(parents=True, exist_ok=True)

    store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        path=settings.qdrant_path,
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
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection
    )

    with open(settings.bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    return store, bm25
