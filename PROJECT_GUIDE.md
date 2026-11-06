# Project Guide – NASA ORBIT (Scientific Research Intelligence Platform)

## 1. What is this project?
NASA ORBIT is a **local, open‑source research assistant** built to help you explore and understand NASA scientific papers (PDFs).  It indexes the PDFs, lets you ask natural‑language questions, and returns answers that are **grounded in the source documents**.  All processing runs on your own machine – no external paid APIs are used.

## 2. High‑level architecture
```
+-------------------+      +-------------------+      +-------------------+
|   PDF Upload UI   | ---> |   Ingestion       | ---> |   Vector Store    |
| (Streamlit page) |      | (pdf_loader,      |      | (Qdrant)          |
+-------------------+      |  chunking, OCR)   |      +-------------------+
                              |                     |
                              v                     v
                     +-------------------+   +-------------------+
                     |  Retrieval Layer   |   |  Reranking Layer |
                     | (Hybrid BM25 +   |   | (Cross‑Encoder)   |
                     |  Vector Store)    |   +-------------------+
                     +-------------------+            |
                              |                     v
                              v            +-------------------+
                     +-------------------+|  Generation Layer |
                     |  Generation Layer  | (ChatOllama)       |
                     | (ChatOllama)      +-------------------+
                     +-------------------+            |
                              |                     v
                              v            +-------------------+
                     +-------------------+|  Evaluation Layer |
                     |  Evaluation Layer  | (DeepEval)        |
                     | (DeepEvalEngine)   +-------------------+
                     +-------------------+
                              |
                              v
                     +-------------------+      +-------------------+
                     |   Output UI       | ---> |   API (FastAPI)   |
                     | (Streamlit)      |      +-------------------+
                     +-------------------+
```

### Main components
| Component | Purpose | Key files |
|-----------|---------|-----------|
| **PDF ingestion** | Validates PDFs, extracts text, OCR for images, splits into chunks. | `ingestion/pdf_loader.py`, `ingestion/chunking.py` |
| **Vector store** | Stores dense embeddings for fast similarity search. | `retrieval/index.py` (Qdrant) |
| **Sparse BM25** | Classic keyword search to complement vectors. | `retrieval/index.py` (BM25) |
| **Hybrid retriever** | Merges dense and sparse results. | `retrieval/hybrid.py` |
| **Reranker** | Ranks the retrieved chunks more accurately using a cross‑encoder. | `retrieval/rerank.py` |
| **LLM generation** | Generates the final answer using a local Ollama model. | `rag.py` (`NASAResearchRAG`) |
| **Evaluation** | Runs a suite of DeepEval metrics (faithfulness, relevance, etc.). | `evaluation/__init__.py` (`DeepEvalEngine`) |
| **UI** | Interactive Streamlit web app for uploading PDFs, asking questions, and viewing scores. | `app.py` |
| **API** | FastAPI endpoints for programmatic access (question, ingest, evaluate). | `api.py` |
| **Tests** | Pytest suite ensuring each piece works. | `tests/` |

## 3. How to get it running (step‑by‑step)
1. **Clone / download the repository** – you already have it under `c:/Users/AIGow/Downloads/NASA‑Space‑Physics‑Research‑Intelligence`.
2. **Create a Python virtual environment** (optional but recommended):
   ```bash
   cd "c:/Users/AIGow/Downloads/NASA-Space-Physics-Research-Intelligence"
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies** – they are pinned for Windows compatibility:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start Ollama locally** (if not already running).  Pull a suitable model, e.g.:
   ```bash
   ollama pull qwen2.5:3b   # or any other open‑source model you prefer
   ```
5. **Run the Streamlit UI** (the easiest way to explore the system):
   ```bash
   streamlit run app.py
   ```
   - Upload PDF files via the “Upload PDFs” sidebar.
   - Ask a question in the main textbox.
   - The answer, supporting context, and a DeepEval score report appear.
6. **Or start the FastAPI server** for programmatic use:
   ```bash
   uvicorn api:app --reload
   ```
   - Endpoints: `/ingest`, `/ask`, `/evaluate`.
7. **Run the test suite** to verify everything works:
   ```bash
   pytest -q
   ```

## 4. Data flow – what happens when you ask a question?
1. **Upload** → PDFs are validated, text is extracted (with OCR if needed), and split into chunks.
2. **Indexing** → Dense embeddings (via `sentence‑transformers`) are stored in Qdrant; a BM25 index is built on the raw text.
3. **Retrieval** → The query is sent to both Qdrant (vector similarity) *and* BM25 (keyword). Results are merged.
4. **Reranking** → A lightweight cross‑encoder reorders the top‑k chunks.
5. **Generation** → The selected context is fed to `ChatOllama` which produces the final answer.
6. **Evaluation** → `DeepEvalEngine` runs all enabled metrics on `answer`, `context`, and `question`. A markdown report is produced.
7. **Display** → Streamlit shows the answer, highlighted citations, and the evaluation report.

## 5. Extending the project
- **Add new metrics** – implement a class inheriting from `deepeval.metrics.base_metric.BaseMetric` and register it in `DeepEvalEngine`.
- **Swap the LLM** – change the model name in `settings.ollama_model` (found in `config.py`). The wrapper auto‑detects CUDA/CPU.
- **Custom agents** – create LangGraph agents in a new `agents/` package and plug them into the RAG pipeline.
- **Persist structured extraction** – extend `ingestion/pdf_loader.py` to pull tables/equations using `docling` and store them in SQLite.

## 6. Quick reference of important files
- `config.py` – central settings (model name, Ollama URL, device detection). 
- `rag.py` – `NASAResearchRAG` class orchestrates retrieval → rerank → generation.
- `evaluation/__init__.py` – `DeepEvalEngine` wrapper.
- `app.py` – Streamlit UI entry point.
- `api.py` – FastAPI endpoints.
- `tests/` – pytest suite (including `test_deepeval.py`).

---
### TL;DR
1. **Upload PDFs →** they get indexed in Qdrant + BM25.  
2. **Ask a question →** hybrid retrieval → rerank → LLM answer.  
3. **DeepEval evaluates** the answer and gives you a clear score report.  
4. All of this runs **locally on Windows** with free, open‑source tools (Ollama, LangChain, LangGraph, DeepEval, Qdrant).

Enjoy exploring NASA research with your own private, GPU‑accelerated assistant!   

---
*File created on 2026‑11‑01.*
