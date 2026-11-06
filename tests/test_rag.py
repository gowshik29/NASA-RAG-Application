from langchain_core.documents import Document
from fastapi.testclient import TestClient

import api
from api import AskRequest
from rag import NASAResearchRAG
from retrieval.rerank import LocalReranker
from config import settings


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, question):
        return self.documents if question else []


class FakeReranker:
    def rerank(self, question, documents, top_n):
        if not question:
            return []
        return documents[:top_n]


class FakeChain:
    def invoke(self, values):
        return f"Answer for: {values['question']}"


def make_rag(documents):
    rag = NASAResearchRAG.__new__(NASAResearchRAG)
    rag.retriever = FakeRetriever(documents)
    rag.reranker = FakeReranker()
    rag.chain = FakeChain()
    return rag


def test_retrieve_and_ask_return_ranked_evidence_and_citations():
    document = Document(
        page_content="Solar wind speed was measured at 400 km/s.",
        metadata={"source": "paper.pdf", "page": 2, "doi": "10.1234/example"},
    )
    rag = make_rag([document])

    assert rag.retrieve("What was the speed?") == [document]
    result = rag.ask("What was the speed?")

    assert result["answer"] == "Answer for: What was the speed?"
    assert result["sources"] == [{
        "filename": "paper.pdf",
        "page": 2,
        "doi": "10.1234/example",
        "snippet": "Solar wind speed was measured at 400 km/s.",
        "score": None,
    }]


def test_ask_returns_no_evidence_without_calling_the_llm():
    rag = make_rag([])

    assert rag.ask("Question with no matches") == {
        "answer": "I could not find relevant evidence in the indexed NASA research papers.",
        "sources": [],
    }


def test_retrieve_merges_tuple_metadata():
    document = Document(page_content="Evidence", metadata={})
    rag = make_rag([(document, {"source": "paper.pdf", "page": 3})])

    assert rag.retrieve("question")[0].metadata == {"source": "paper.pdf", "page": 3}


def test_api_uses_default_top_k():
    assert AskRequest(question="What is in the papers?").top_k == 6


def test_reranker_uses_configured_model(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device

    monkeypatch.setattr("retrieval.rerank.CrossEncoder", FakeCrossEncoder)

    model = LocalReranker().model
    assert model.model_name == settings.reranker_model
    assert model.device in {"cpu", "cuda"}


def test_api_routes_use_lifespan_rag(monkeypatch):
    class FakeRAG:
        def ask(self, question, top_k):
            return {"answer": question, "top_k": top_k}

    monkeypatch.setattr(api, "NASAResearchRAG", FakeRAG)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok", "rag_loaded": True}
        response = client.post("/ask", json={"question": "What is in the papers?"})

    assert response.json() == {"answer": "What is in the papers?", "top_k": 6}


def test_api_reports_initialization_failure(monkeypatch):
    def fail_initialization():
        raise RuntimeError("index is unavailable")

    monkeypatch.setattr(api, "NASAResearchRAG", fail_initialization)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 503
        assert client.post(
            "/ask",
            json={"question": "What is in the papers?"},
        ).status_code == 503
