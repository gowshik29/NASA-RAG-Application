"""
NASA Space Physics Research Intelligence — Main App
====================================================
Streamlit entry-point. Upload PDFs, ask questions, view answers and
evaluation scores. The DeepEval Metrics page is at pages/2_DeepEval_Metrics.py
"""

import streamlit as st
from rag import NASAResearchRAG

# Activate LangSmith + OpenTelemetry as early as possible
try:
    from observability import configure_observability
    _obs = configure_observability()
except Exception:
    _obs = {}

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA Space Physics Research Intelligence",
    page_icon="🛰️",
    layout="wide",
)

# ── Light custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #111827 100%); }
.nasa-header {
    background: linear-gradient(135deg,#1a2744 0%,#0f1e3c 60%,#162032 100%);
    border: 1px solid rgba(99,179,237,.2);
    border-radius: 16px;
    padding: 22px 32px;
    margin-bottom: 22px;
    display: flex; align-items: center; gap: 16px;
}
.nasa-header h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.nasa-header p { color: #90cdf4; font-size: 0.9rem; margin: 4px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nasa-header">
  <span style="font-size:2.8rem">🛰️</span>
  <div>
    <h1>NASA Space Physics Research Intelligence</h1>
    <p>Local scientific RAG · LangChain · Qdrant · Ollama · DeepEval</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ System")
    st.write("**LLM:** Qwen via Ollama")
    st.write("**Vector DB:** Qdrant")
    st.write("**Retrieval:** Dense + BM25 + Cross-encoder reranking")
    st.info("Run `python ingest.py` before querying.", icon="ℹ️")
    st.markdown("---")
    st.markdown("### 🔗 Navigation")
    st.page_link("app.py", label="🏠 Research Q&A")
    st.page_link("pages/2_DeepEval_Metrics.py", label="🧪 DeepEval Metrics")
    st.page_link("pages/3_LangSmith_Traces.py", label="🔍 LangSmith Traces")

    # LangSmith status badge
    if _obs.get("langsmith_hosted"):
        st.success(f"🟢 LangSmith active · `{_obs.get('langsmith_project')}`", icon="✅")
    else:
        st.warning("🔴 LangSmith off · add key to `.env`", icon="⚠️")
    st.markdown("---")
    st.markdown("### ⚗️ Evaluation")
    run_eval = st.checkbox("Auto-evaluate answers with DeepEval", value=False)
    if run_eval:
        st.caption(
            "When enabled, each answer is scored with Faithfulness, Answer Relevancy "
            "and Hallucination metrics. Full evaluation is on the 🧪 page."
        )


@st.cache_resource
def get_rag():
    return NASAResearchRAG()

# ── Chat ──────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about the research papers…")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reranking evidence…"):
            try:
                result = get_rag().ask(question)
                st.markdown(result["answer"])

                with st.expander(f"📄 Retrieved evidence ({len(result['sources'])})"):
                    for s in result["sources"]:
                        st.markdown(f"**{s['filename']} · page {s['page']}**")
                        if s["doi"]:
                            st.caption(f"DOI: {s['doi']}")
                        st.write(s["snippet"])
                        st.caption(f"Reranker score: {s['score']}")

                # ── Optional quick DeepEval badge ─────────────────────────
                if run_eval:
                    with st.spinner("⏳ Running quick DeepEval (3 metrics)…"):
                        try:
                            from evaluation import DeepEvalEngine
                            ctx_list = [s["snippet"] for s in result["sources"]]
                            engine = DeepEvalEngine(
                                enabled_metrics=["faithfulness", "answer_relevancy", "hallucination"]
                            )
                            scores = engine.evaluate(
                                question=question,
                                answer=result["answer"],
                                context=ctx_list,
                            )
                            overall = scores.get("overall", {}).get("score")
                            faith   = scores.get("faithfulness", {}).get("score")
                            relev   = scores.get("answer_relevancy", {}).get("score")
                            halluc  = scores.get("hallucination", {}).get("score")

                            def _fmt(v):
                                return f"{v:.2f}" if isinstance(v, float) else "N/A"

                            st.markdown(f"""
<div style="background:rgba(15,26,50,0.8);border:1px solid rgba(99,179,237,.25);
border-radius:12px;padding:14px 18px;margin-top:12px;">
  <span style="color:#90cdf4;font-size:.8rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.08em">⚗️ Quick DeepEval</span><br>
  <span style="color:#68d391;font-weight:700">Faithfulness&nbsp;&nbsp;{_fmt(faith)}</span>&nbsp;&nbsp;
  <span style="color:#63b3ed;font-weight:700">Relevancy&nbsp;&nbsp;{_fmt(relev)}</span>&nbsp;&nbsp;
  <span style="color:#f6ad55;font-weight:700">Hallucination&nbsp;&nbsp;{_fmt(halluc)}</span>&nbsp;&nbsp;
  <span style="color:#a78bfa;font-weight:700">Overall&nbsp;&nbsp;{_fmt(overall)}</span>
  <br><span style="color:#718096;font-size:.75rem">→ Full 9-metric analysis on the <a href="/DeepEval_Metrics" style="color:#63b3ed">🧪 DeepEval Metrics</a> page</span>
</div>
""", unsafe_allow_html=True)
                        except Exception as ee:
                            st.warning(f"DeepEval not available: {ee}")

            except Exception as exc:
                st.error(str(exc))
                st.info("Check Ollama, Qdrant index and local model configuration.")
