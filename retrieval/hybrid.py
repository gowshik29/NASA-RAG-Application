from langchain.retrievers import EnsembleRetriever
from config import settings

def make_hybrid_retriever(vector_store, bm25):
    dense = vector_store.as_retriever(search_kwargs={"k": settings.dense_k})
    bm25.k = settings.sparse_k
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.55, 0.45])
