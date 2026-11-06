"""
DeepEval Metrics — NASA ORBIT
===============================
A dedicated Streamlit page for running and visualising all DeepEval
evaluation metrics against a NASA research RAG answer.
"""

import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepEval Metrics · NASA ORBIT",
    page_icon="🧪",
    layout="wide",
)

# ── custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ----------  Google fonts  ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ----------  Page background  ---------- */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #111827 100%);
    color: #e2e8f0;
}

/* ----------  Main header  ---------- */
.deepeval-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2a4a 50%, #162032 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
}
.deepeval-header h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.deepeval-header p { color: #90cdf4; font-size: 0.95rem; margin: 4px 0 0; }

/* ----------  Metric card  ---------- */
.metric-card {
    background: linear-gradient(135deg, rgba(30,58,95,0.6) 0%, rgba(15,42,74,0.8) 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 14px;
    padding: 20px 22px;
    margin: 0 0 16px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 14px 0 0 14px;
}
.metric-card.pass::before  { background: linear-gradient(180deg, #48bb78, #38a169); }
.metric-card.fail::before  { background: linear-gradient(180deg, #fc8181, #e53e3e); }
.metric-card.warn::before  { background: linear-gradient(180deg, #f6ad55, #dd6b20); }
.metric-card.pending::before { background: linear-gradient(180deg, #76e4f7, #0bc5ea); }

.metric-card h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.0rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 6px;
}
.metric-score-big {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    line-height: 1;
}
.score-pass  { color: #68d391; }
.score-fail  { color: #fc8181; }
.score-warn  { color: #f6ad55; }
.score-na    { color: #718096; }
.metric-desc { font-size: 0.82rem; color: #a0aec0; margin: 8px 0 4px; line-height: 1.45; }
.metric-reason {
    font-size: 0.8rem;
    color: #90cdf4;
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 8px;
    line-height: 1.45;
    border-left: 3px solid rgba(99,179,237,0.4);
}

/* ----------  Overall score banner  ---------- */
.overall-banner {
    background: linear-gradient(135deg, #1a365d 0%, #2a4a7f 50%, #1e3a6a 100%);
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 16px;
    padding: 24px 32px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.overall-banner .label { font-size: 0.85rem; color: #90cdf4; text-transform: uppercase; letter-spacing: 0.1em; }
.overall-banner .score { font-size: 4rem; font-weight: 700; font-family: 'Outfit', sans-serif; }

/* ----------  Section titles  ---------- */
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #90cdf4;
    border-bottom: 1px solid rgba(99,179,237,0.2);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* ----------  Input area  ---------- */
.stTextArea textarea {
    background: rgba(15,26,50,0.9) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,179,237,0.7) !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
}

/* ----------  Buttons  ---------- */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(49,130,206,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(49,130,206,0.5) !important;
}

/* ----------  Tabs  ---------- */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,22,41,0.8) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #90cdf4 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(49,130,206,0.3) !important;
    color: #63b3ed !important;
}

/* ----------  Expanders  ---------- */
.streamlit-expanderHeader {
    background: rgba(15,26,50,0.7) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #90cdf4 !important;
}

/* ----------  DataFrame  ---------- */
.dataframe { background: rgba(15,22,41,0.9) !important; }
</style>
""", unsafe_allow_html=True)

# ── imports (guarded so page loads even if packages missing) ────────────────
try:
    from evaluation import DeepEvalEngine
    ENGINE_OK = True
except Exception as _e:
    ENGINE_OK = False
    ENGINE_ERR = str(_e)

# ── session-state defaults ───────────────────────────────────────────────────
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ── HELPERS ──────────────────────────────────────────────────────────────────

def score_class(name: str, score: float | None) -> str:
    """Return CSS class for a metric score."""
    if score is None:
        return "pending"
    lower_is_better = name in DeepEvalEngine.LOWER_IS_BETTER
    threshold = DeepEvalEngine.PASS_THRESHOLDS.get(name, 0.7)
    if lower_is_better:
        return "pass" if score <= threshold else "fail"
    return "pass" if score >= threshold else ("warn" if score >= threshold * 0.7 else "fail")

def score_color_class(name: str, score: float | None) -> str:
    sc = score_class(name, score)
    return {"pass": "score-pass", "fail": "score-fail", "warn": "score-warn"}.get(sc, "score-na")

def render_metric_card(name: str, data: dict, engine: "DeepEvalEngine"):
    label   = DeepEvalEngine.METRIC_LABELS.get(name, name.replace("_", " ").title())
    desc    = DeepEvalEngine.METRIC_DESCRIPTIONS.get(name, "")
    score   = data.get("score")
    reason  = data.get("reason") or ""
    sc      = score_class(name, score)
    cc      = score_color_class(name, score)
    score_txt = f"{score:.3f}" if isinstance(score, float) else "N/A"

    lower_note = " ↓ lower is better" if name in DeepEvalEngine.LOWER_IS_BETTER else ""
    st.markdown(f"""
<div class="metric-card {sc}">
  <h3>🔬 {label}</h3>
  <span class="metric-score-big {cc}">{score_txt}</span><span style="color:#718096;font-size:0.8rem">{lower_note}</span>
  <p class="metric-desc">{desc}</p>
  {"<div class='metric-reason'>💬 " + reason[:400] + ("…" if len(reason) > 400 else "") + "</div>" if reason else ""}
</div>
""", unsafe_allow_html=True)

def render_overall_banner(score: float | None):
    if score is None:
        color = "#718096"
        label = "No Score"
        emoji = "❓"
    elif score >= 0.75:
        color = "#68d391"
        label = "Excellent"
        emoji = "🌟"
    elif score >= 0.55:
        color = "#f6ad55"
        label = "Acceptable"
        emoji = "⚠️"
    else:
        color = "#fc8181"
        label = "Poor"
        emoji = "❌"
    score_txt = f"{score:.3f}" if score is not None else "—"
    st.markdown(f"""
<div class="overall-banner">
  <div class="label">Overall DeepEval Score</div>
  <div class="score" style="color:{color}">{emoji} {score_txt}</div>
  <div style="color:#a0aec0;font-size:0.9rem;margin-top:4px">{label}</div>
</div>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="deepeval-header">
  <span style="font-size:3rem">🧪</span>
  <div>
    <h1>DeepEval Metrics Dashboard</h1>
    <p>Evaluate RAG answer quality with 9 LLM-graded metrics · Powered by deepeval 4.x + local Ollama</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not ENGINE_OK:
    st.error(f"⚠️ Could not load evaluation engine: {ENGINE_ERR}")
    st.info("Make sure `deepeval` is installed and `config.py` settings are correct.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_name = st.text_input(
        "Ollama Eval Model",
        value="qwen2.5:3b",
        help="Local Ollama model used by DeepEval to judge answers",
    )
    st.markdown("---")
    st.markdown("### 📋 Metrics to Run")
    metric_toggles = {}
    for m_key, m_label in DeepEvalEngine.METRIC_LABELS.items():
        metric_toggles[m_key] = st.checkbox(m_label, value=True, key=f"chk_{m_key}")
    selected_metrics = [k for k, v in metric_toggles.items() if v]
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "DeepEval v4.x uses an LLM judge (local Ollama) to score RAG outputs "
        "across faithfulness, relevancy, contextual metrics, hallucination, "
        "summarisation quality, bias and toxicity."
    )

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📝 Input for Evaluation</div>', unsafe_allow_html=True)

tab_manual, tab_golden = st.tabs(["✏️ Manual Input", "📚 Golden Dataset"])

with tab_manual:
    col_q, col_a = st.columns(2)
    with col_q:
        question = st.text_area(
            "Question",
            height=120,
            placeholder="What electron outflow jet velocity was reported for the 3 Aug 2017 event?",
            key="q_manual",
        )
    with col_a:
        answer = st.text_area(
            "RAG Answer",
            height=120,
            placeholder="The electron outflow jet velocity was 1,000–1,500 km/s...",
            key="a_manual",
        )
    context_raw = st.text_area(
        "Retrieved Context (one chunk per line — or paste the full context block)",
        height=140,
        placeholder="[Source: poh_2019.pdf, p.3]\nThe electron outflow jets reached speeds of 1,000–1,500 km/s during the August 2017 event.\n\n[Source: vines_2019.pdf, p.7]\nOff-equatorial EMIC waves were observed by MMS spacecraft...",
        key="ctx_manual",
    )
    expected_output = st.text_area(
        "Expected / Reference Answer (optional — improves Contextual Recall scoring)",
        height=80,
        placeholder="The expected answer or a ground-truth summary...",
        key="exp_manual",
    )
    run_manual = st.button("🚀 Run DeepEval Evaluation", key="run_manual")

with tab_golden:
    # Load golden dataset
    golden_path = Path("evaluation/golden_dataset.json")
    if golden_path.exists():
        with open(golden_path) as f:
            golden_data = json.load(f)
        selected_golden = st.selectbox(
            "Choose a test case",
            options=golden_data,
            format_func=lambda x: f"[{x['id']}] {x['question'][:80]}...",
        )
        st.info(f"**Question:** {selected_golden['question']}")
        golden_answer = st.text_area(
            "Provide RAG Answer for this question",
            height=120,
            placeholder="Paste or type the RAG-generated answer here...",
            key="a_golden",
        )
        golden_context = st.text_area(
            "Retrieved Context",
            height=120,
            placeholder="Paste the retrieved context here...",
            key="ctx_golden",
        )
        run_golden = st.button("🚀 Run DeepEval on Golden Case", key="run_golden")
    else:
        st.warning("Golden dataset not found at `evaluation/golden_dataset.json`.")
        run_golden = False

# ── RUN EVALUATION ────────────────────────────────────────────────────────────

def _run(q: str, a: str, ctx_text: str, exp: str = ""):
    if not q.strip() or not a.strip():
        st.error("Question and Answer are both required.")
        return
    if not selected_metrics:
        st.error("Select at least one metric in the sidebar.")
        return

    ctx_list = [c.strip() for c in ctx_text.split("\n\n") if c.strip()] if ctx_text.strip() else []

    with st.spinner("⏳ Running DeepEval — this may take 1–3 minutes per metric with a local LLM…"):
        t0 = time.perf_counter()
        engine = DeepEvalEngine(enabled_metrics=selected_metrics, model_name=model_name)
        results = engine.evaluate(
            question=q,
            answer=a,
            context=ctx_list,
            expected_output=exp or None,
            enabled_metrics=selected_metrics,
        )
        elapsed = time.perf_counter() - t0

    st.session_state.last_results = {
        "question": q,
        "answer": a,
        "context": ctx_list,
        "results": results,
        "elapsed": elapsed,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "metrics": selected_metrics,
    }
    st.session_state.eval_history.append(st.session_state.last_results)

if run_manual:
    _run(question, answer, context_raw, expected_output)

if "run_golden" in dir() and run_golden:
    _run(
        selected_golden["question"],
        golden_answer,
        golden_context,
    )

# ── RESULTS ──────────────────────────────────────────────────────────────────
if st.session_state.last_results:
    res = st.session_state.last_results
    results = res["results"]
    overall = results.get("overall", {}).get("score")

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Evaluation Results</div>', unsafe_allow_html=True)
    st.caption(f"⏱️ Evaluation completed in **{res['elapsed']:.1f}s** · {res['timestamp']}")

    # --- Overall banner ---
    render_overall_banner(overall)

    # --- Individual metric cards (3 per row) ---
    metric_names = [m for m in results.keys() if m != "overall"]
    cols_per_row = 3
    for i in range(0, len(metric_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, m in enumerate(metric_names[i:i + cols_per_row]):
            with cols[j]:
                render_metric_card(m, results[m], None)

    # --- DataFrame summary ---
    st.markdown('<div class="section-title">📋 Score Summary Table</div>', unsafe_allow_html=True)
    rows = []
    for m in metric_names:
        d = results[m]
        score = d.get("score")
        lower = m in DeepEvalEngine.LOWER_IS_BETTER
        threshold = DeepEvalEngine.PASS_THRESHOLDS.get(m, 0.7)
        if score is None:
            status = "⚪ N/A"
        elif lower:
            status = "✅ Pass" if score <= threshold else "❌ Fail"
        else:
            status = "✅ Pass" if score >= threshold else "❌ Fail"
        rows.append({
            "Metric": DeepEvalEngine.METRIC_LABELS.get(m, m),
            "Score": f"{score:.4f}" if isinstance(score, float) else "N/A",
            "Threshold": f"{'≤' if lower else '≥'} {threshold}",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Bar chart ---
    chart_data = {
        DeepEvalEngine.METRIC_LABELS.get(m, m): results[m].get("score") or 0.0
        for m in metric_names
        if isinstance(results[m].get("score"), float)
    }
    if chart_data:
        st.markdown('<div class="section-title">📈 Score Distribution</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({"Score": chart_data})
        st.bar_chart(chart_df, use_container_width=True, height=280, color="#3182ce")

    # --- Raw JSON expander ---
    with st.expander("🔍 Raw JSON Results", expanded=False):
        st.json(results)

    # --- Download ---
    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json.dumps({
            "question": res["question"],
            "answer": res["answer"],
            "timestamp": res["timestamp"],
            "results": {
                k: {"score": v.get("score"), "reason": v.get("reason"), "success": v.get("success")}
                for k, v in results.items()
            },
        }, indent=2),
        file_name=f"deepeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

# ── HISTORY TABLE ─────────────────────────────────────────────────────────────
if len(st.session_state.eval_history) > 1:
    st.markdown("---")
    st.markdown('<div class="section-title">🕓 Evaluation History (this session)</div>', unsafe_allow_html=True)
    hist_rows = []
    for entry in reversed(st.session_state.eval_history):
        overall_score = entry["results"].get("overall", {}).get("score")
        hist_rows.append({
            "Time": entry["timestamp"],
            "Question": entry["question"][:70] + ("…" if len(entry["question"]) > 70 else ""),
            "Overall": f"{overall_score:.4f}" if isinstance(overall_score, float) else "N/A",
            "Metrics": len(entry["metrics"]),
            "Duration (s)": f"{entry['elapsed']:.1f}",
        })
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History"):
        st.session_state.eval_history = []
        st.session_state.last_results = None
        st.rerun()

# ── DOCS / REFERENCE ──────────────────────────────────────────────────────────
with st.expander("📖 Metric Reference Guide", expanded=False):
    st.markdown("""
| Metric | What it measures | Ideal |
|--------|-----------------|-------|
| **Faithfulness** | Answer is factually consistent with retrieved context | ≥ 0.7 |
| **Answer Relevancy** | Answer directly addresses the question | ≥ 0.7 |
| **Contextual Precision** | Retrieved chunks are ranked most-relevant-first | ≥ 0.7 |
| **Contextual Recall** | Retrieved context covers the expected answer | ≥ 0.7 |
| **Contextual Relevancy** | Overall context relevance to the query | ≥ 0.7 |
| **Hallucination** | Content not supported by context (lower = fewer hallucinations) | ≤ 0.5 |
| **Summarization** | Answer covers key facts without adding noise | ≥ 0.7 |
| **Bias** | Political / gender / racial bias (lower = less biased) | ≤ 0.3 |
| **Toxicity** | Harmful or offensive language (lower = safer) | ≤ 0.3 |

> **Note:** All metrics are LLM-graded by the local Ollama model you specified in the sidebar.
> Results depend on the quality and capabilities of that model. For best results use `qwen2.5:3b`
> or larger.
    """)
