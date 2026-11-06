import logging

try:
    from sentence_transformers import CrossEncoder
except Exception as e:
    logging.warning("CrossEncoder import failed (%s). Using dummy reranker.", e)
    CrossEncoder = None

from config import settings

class DummyReranker:
    """Fallback reranker that assigns equal scores to all documents."""
    def predict(self, pairs):
        # Return a list of zeros matching the number of pairs
        return [0.0 for _ in pairs]

class LocalReranker:
    """Local reranker using a CrossEncoder model if available, otherwise a dummy implementation."""

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if CrossEncoder is not None:
            self.model = CrossEncoder(settings.reranker_model, device=self.device)
        else:
            self.model = DummyReranker()

    def rerank(self, question, documents, top_n=6):
        if not documents:
            return []
        # Prepare pairs for scoring
        pairs = [(question, document.page_content) for document in documents]
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logging.error("Reranker prediction failed: %s", e)
            scores = [0.0 for _ in pairs]
        # Rank documents by score descending
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
        # Return top_n Document objects
        return [document for document, _ in ranked[:top_n]]
