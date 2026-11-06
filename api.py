import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from rag import NASAResearchRAG

logger = logging.getLogger(__name__)
rag = None
rag_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.rag = NASAResearchRAG()
    except Exception:
        app.state.rag = None
        logger.exception("Failed to initialize the NASA research assistant.")
    yield


app = FastAPI(title="NASA Space Physics RAG API", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=20)

@app.get("/health")
def health(request: Request):
    if request.app.state.rag is None:
        raise HTTPException(503, "RAG unavailable. Check the API logs for initialization details.")
    return {"status": "ok", "rag_loaded": True}

@app.post("/ask")
def ask(req: AskRequest, request: Request):
    rag = request.app.state.rag
    if rag is None:
        raise HTTPException(503, "RAG unavailable. Check the API logs for initialization details.")
    return rag.ask(req.question, req.top_k)
