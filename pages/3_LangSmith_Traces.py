"""
LangSmith Traces — NASA ORBIT
================================
View live LangSmith tracing status, local OpenTelemetry span logs
stored in SQLite, and quick links to the LangSmith dashboard.
"""

import json
from datetime import datetime
from observability import configure_observability
from config import settings
from catalog.database import list_traces, init_db

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangSmith Traces · NASA ORBIT",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #111827 100%);
    color: #e2e8f0;
}
.ls-header {
    background: linear-gradient(135deg, #1a2d1a 0%, #0f2a0f 50%, #162016 100%);
    border: 1px solid rgba(72,187,120,0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.ls-header h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #68d391, #63b3ed, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.ls-header p { color: #9ae6b4; font-size: 0.95rem; margin: 4px 0 0; }

/* Status pill */
.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.pill-green { background: rgba(72,187,120,0.2); color: #68d391; border: 1px solid rgba(72,187,120,0.4); }
.pill-red   { background: rgba(252,129,129,0.2); color: #fc8181; border: 1px solid rgba(252,129,129,0.4); }
.pill-blue  { background: rgba(99,179,237,0.2); color: #63b3ed; border: 1px solid rgba(99,179,237,0.4); }

/* Info cards */
.info-card {
    background: linear-gradient(135deg, rgba(26,45,26,0.6), rgba(15,42,15,0.8));
    border: 1px solid rgba(72,187,120,0.15);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.info-card h3 { color: #9ae6b4; font-family: 'Outfit', sans-serif; margin: 0 0 8px; font-size: 0.95rem; }
.info-card p  { color: #e2e8f0; margin: 0; font-size: 1.5rem; font-weight: 700; }
.info-card small { color: #718096; font-size: 0.8rem; }

/* Trace row */
.trace-row {
    background: rgba(15,22,41,0.7);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.trace-name  { color: #63b3ed; font-weight: 600; }
.trace-dur   { color: #f6ad55; }
.trace-err   { color: #fc8181; }
.trace-ok    { color: #68d391; }
.trace-ts    { color: #718096; font-size: 0.78rem; }

.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem; font-weight: 600;
    color: #9ae6b4;
    border-bottom: 1px solid rgba(72,187,120,0.2);
    padding-bottom: 8px; margin: 24px 0 16px;
}

.stButton > button {
    background: linear-gradient(135deg, #276749 0%, #2f855a 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(47,133,90,0.3) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.external-link-card {
    background: linear-gradient(135deg, rgba(30,58,95,0.6), rgba(15,42,74,0.8));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ls-header">
  <span style="font-size:3rem">🔍</span>
  <div>
    <h1>LangSmith Traces Dashboard</h1>
    <p>Monitor LangChain pipeline traces, latency, and errors — hosted + local OpenTelemetry</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load observability config ─────────────────────────────────────────────────
try:
    from observability import configure_observability
    from config import settings
    obs = configure_observability()
    langsmith_active = obs.get("langsmith_hosted", False)
    project          = obs.get("langsmith_project") or "nasa-orbit"
    api_key          = settings.langsmith_api_key or ""
    endpoint         = settings.langsmith_endpoint or "https://api.smith.langchain.com"
    send_content     = obs.get("send_document_content", False)
    CONFIG_OK = True
except Exception as e:
    langsmith_active = False
    project          = "nasa-orbit"
    api_key          = ""
    endpoint         = "https://api.smith.langchain.com"
    send_content     = False
    CONFIG_OK = False
    st.error(f"Could not load config: {e}")

# ── Load local SQLite traces ──────────────────────────────────────────────────
try:
    from catalog.database import list_traces, init_db
    init_db()
    local_traces = list_traces(limit=200)
    TRACES_OK = True
except Exception as e:
    local_traces = []
    TRACES_OK = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"""
**LangSmith Status:**  
{'<span class="pill pill-green">🟢 Active</span>' if langsmith_active else '<span class="pill pill-red">🔴 Inactive</span>'}
""", unsafe_allow_html=True)
    st.markdown(f"**Project:** `{project}`")
    st.markdown(f"**Endpoint:** `{endpoint}`")
    st.markdown(f"**Send content:** `{send_content}`")
    st.markdown("---")
    if langsmith_active:
        st.markdown(f"[🌐 Open LangSmith Dashboard]({endpoint.replace('api.', 'app.').replace('/api','')}/projects/{project})")
    st.markdown("---")
    st.markdown("### 🔗 Navigation")
    st.page_link("app.py", label="🏠 Research Q&A")
    st.page_link("pages/2_DeepEval_Metrics.py", label="🧪 DeepEval Metrics")
    st.page_link("pages/3_LangSmith_Traces.py", label="🔍 LangSmith Traces")

# ── Status Overview ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📡 Connection Status</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    status_text = "Active" if langsmith_active else "Inactive"
    status_color = "#68d391" if langsmith_active else "#fc8181"
    st.markdown(f"""
<div class="info-card">
  <h3>LangSmith Hosted</h3>
  <p style="color:{status_color}">{status_text}</p>
  <small>{'Traces sent to LangSmith cloud' if langsmith_active else 'Add key to .env to activate'}</small>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
<div class="info-card">
  <h3>Project</h3>
  <p style="font-size:1.1rem">{project}</p>
  <small>LangSmith project name</small>
</div>
""", unsafe_allow_html=True)

with col3:
    total_local = len(local_traces)
    st.markdown(f"""
<div class="info-card">
  <h3>Local Traces</h3>
  <p>{total_local}</p>
  <small>Spans stored in SQLite</small>
</div>
""", unsafe_allow_html=True)

with col4:
    errors = sum(1 for t in local_traces if t.get("error"))
    err_color = "#fc8181" if errors > 0 else "#68d391"
    st.markdown(f"""
<div class="info-card">
  <h3>Errors</h3>
  <p style="color:{err_color}">{errors}</p>
  <small>Spans with recorded errors</small>
</div>
""", unsafe_allow_html=True)

# ── Setup Guide (if not active) ───────────────────────────────────────────────
if not langsmith_active:
    st.markdown('<div class="section-title">🚀 Activate LangSmith</div>', unsafe_allow_html=True)
    st.markdown("""
> **LangSmith is not connected.** Follow these steps to enable it:

**1.** Go to [smith.langchain.com](https://smith.langchain.com) → **Settings → API Keys** → Create key

**2.** Open your [`.env`](file:///c:/Users/AIGow/Downloads/NASA-Space-Physics-Research-Intelligence/NASA-Space-Physics-Research-Intelligence/.env) file and fill in:
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__xxxxxxxxxxxxxxxx   ← paste your key here
LANGSMITH_PROJECT=nasa-orbit
```

**3.** Restart the Streamlit app:
```bash
.venv\\Scripts\\streamlit.exe run app.py
```

**4.** Ask a question on the main page — the trace will appear in LangSmith within seconds.
    """)

# ── LangSmith Dashboard Link ──────────────────────────────────────────────────
if langsmith_active:
    st.markdown('<div class="section-title">🌐 LangSmith Cloud Dashboard</div>', unsafe_allow_html=True)
    dashboard_url = f"https://smith.langchain.com/projects/{project}"
    st.markdown(f"""
<div class="external-link-card">
  <span style="font-size:2.5rem">🔭</span>
  <h3 style="color:#63b3ed;font-family:'Outfit',sans-serif;margin:8px 0">Your LangSmith Project: <code>{project}</code></h3>
  <p style="color:#90cdf4;margin:0 0 16px">View waterfall traces, token usage, latency breakdown, and prompt history</p>
  <a href="{dashboard_url}" target="_blank" style="
    background:linear-gradient(135deg,#276749,#2f855a);
    color:white;padding:10px 28px;border-radius:10px;
    text-decoration:none;font-weight:600;font-size:0.95rem;
    box-shadow:0 4px 12px rgba(47,133,90,0.4);
  ">🌐 Open Dashboard →</a>
</div>
""", unsafe_allow_html=True)
    st.info("💡 Tip: Each question you ask on the main page creates a new trace. You can filter by project, date, and latency on the dashboard.")

# ── Local Trace Log ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗃️ Local OpenTelemetry Span Log (SQLite)</div>', unsafe_allow_html=True)

if not TRACES_OK:
    st.warning("Could not load local traces from SQLite. The database may not be initialised yet.")
elif not local_traces:
    st.info("No local traces yet. Ask a question on the main page to generate spans.")
else:
    # Filters
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        search = st.text_input("🔎 Filter by span name", placeholder="retrieval, rerank, generation…")
    with col_f2:
        show_errors_only = st.checkbox("Show errors only")
    with col_f3:
        limit = st.selectbox("Show last N", [25, 50, 100, 200], index=0)

    filtered = local_traces[:limit]
    if search:
        filtered = [t for t in filtered if search.lower() in (t.get("name") or "").lower()]
    if show_errors_only:
        filtered = [t for t in filtered if t.get("error")]

    st.caption(f"Showing {len(filtered)} spans (most recent first)")

    # Render as styled cards
    for t in filtered:
        name       = t.get("name") or "unknown"
        dur        = t.get("duration_ms")
        err        = t.get("error")
        ts         = t.get("started_at") or ""
        attrs_json = t.get("attributes_json") or "{}"

        dur_str  = f"{dur:.1f} ms" if isinstance(dur, (int, float)) else "—"
        dur_color = "#fc8181" if isinstance(dur, float) and dur > 5000 else "#f6ad55" if isinstance(dur, float) and dur > 1000 else "#68d391"
        err_html  = f'<span class="trace-err">⚠️ {err[:120]}</span>' if err else '<span class="trace-ok">✅ OK</span>'

        try:
            attrs = json.loads(attrs_json)
            attrs_display = json.dumps(attrs, indent=2)[:300]
        except Exception:
            attrs_display = attrs_json[:300]

        with st.expander(f"🔹 {name}  ·  {dur_str}  ·  {ts[:19]}", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"**Span:** `{name}`")
                st.markdown(f"**Duration:** <span style='color:{dur_color}'>{dur_str}</span>", unsafe_allow_html=True)
                st.markdown(f"**Status:** {err_html}", unsafe_allow_html=True)
                st.markdown(f"**Timestamp:** `{ts}`")
            with col_d2:
                if attrs_display.strip() not in ("{}", ""):
                    st.code(attrs_display, language="json")
                else:
                    st.caption("No attributes recorded.")

    # ── DataFrame table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Trace Summary Table</div>', unsafe_allow_html=True)
    rows = []
    for t in filtered:
        rows.append({
            "Span":       t.get("name", ""),
            "Duration (ms)": round(t.get("duration_ms") or 0, 1),
            "Status":     "❌ Error" if t.get("error") else "✅ OK",
            "Timestamp":  (t.get("started_at") or "")[:19],
            "Error":      (t.get("error") or "")[:80],
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Latency chart
        st.markdown('<div class="section-title">📈 Span Latency Chart</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Duration (ms)": [r["Duration (ms)"] for r in rows],
            "Span": [r["Span"] for r in rows],
        }).set_index("Span")
        st.bar_chart(chart_df, use_container_width=True, height=240, color="#68d391")

        # Download
        st.download_button(
            "⬇️ Download Traces (JSON)",
            data=json.dumps(filtered, indent=2, default=str),
            file_name=f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

# ── Refresh ────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔄 Refresh Traces"):
    st.rerun()
