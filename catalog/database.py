from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings

_LOCK = threading.Lock()
_INIT = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_name TEXT,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_size INTEGER,
    page_count INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    uploaded_at TEXT,
    updated_at TEXT,
    duplicate_of TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE TABLE IF NOT EXISTS paper_intelligence (
    document_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    extracted_at TEXT,
    model TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT,
    filename TEXT,
    page INTEGER,
    section TEXT,
    content TEXT,
    content_hash TEXT,
    qdrant_id TEXT,
    start_index INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE TABLE IF NOT EXISTS tables (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    filename TEXT,
    page INTEGER,
    table_number TEXT,
    title TEXT,
    headers_json TEXT,
    rows_json TEXT,
    units TEXT,
    footnotes TEXT,
    markdown TEXT
);
CREATE TABLE IF NOT EXISTS equations (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    filename TEXT,
    page INTEGER,
    equation_number TEXT,
    raw_text TEXT,
    latex TEXT,
    variables_json TEXT,
    context TEXT,
    ocr_flag INTEGER
);
CREATE TABLE IF NOT EXISTS figures (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    filename TEXT,
    page INTEGER,
    figure_number TEXT,
    caption TEXT,
    explanation TEXT,
    image_path TEXT
);
CREATE TABLE IF NOT EXISTS semantic_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    query_hash TEXT,
    embedding_json TEXT,
    response_json TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT,
    finished_at TEXT,
    status TEXT,
    results_json TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    started_at TEXT,
    duration_ms REAL,
    attributes_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS workflow_states (
    id TEXT PRIMARY KEY,
    question TEXT,
    state_json TEXT,
    updated_at TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    global _INIT
    with _LOCK:
        if _INIT:
            return
        conn = get_connection()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
        _INIT = True


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def upsert_document(doc: dict) -> None:
    init_db()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO documents (id, filename, original_name, path, content_hash, file_size,
                page_count, status, error, uploaded_at, updated_at, duplicate_of)
            VALUES (:id, :filename, :original_name, :path, :content_hash, :file_size,
                :page_count, :status, :error, :uploaded_at, :updated_at, :duplicate_of)
            ON CONFLICT(id) DO UPDATE SET
                filename=excluded.filename, path=excluded.path, file_size=excluded.file_size,
                page_count=excluded.page_count, status=excluded.status, error=excluded.error,
                updated_at=excluded.updated_at, duplicate_of=excluded.duplicate_of
            """,
            doc,
        )
        conn.commit()
    finally:
        conn.close()


def update_document_status(doc_id: str, status: str, error: str | None = None, **extra: Any) -> None:
    init_db()
    fields = {"status": status, "error": error, "updated_at": utcnow(), "id": doc_id}
    assignments = ["status=:status", "error=:error", "updated_at=:updated_at"]
    for key, value in extra.items():
        fields[key] = value
        assignments.append(f"{key}=:{key}")
    conn = get_connection()
    try:
        conn.execute(f"UPDATE documents SET {', '.join(assignments)} WHERE id=:id", fields)
        conn.commit()
    finally:
        conn.close()


def get_document(doc_id: str) -> dict | None:
    init_db()
    conn = get_connection()
    try:
        return row_to_dict(conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone())
    finally:
        conn.close()


def get_document_by_hash(content_hash: str) -> dict | None:
    init_db()
    conn = get_connection()
    try:
        return row_to_dict(
            conn.execute("SELECT * FROM documents WHERE content_hash=?", (content_hash,)).fetchone()
        )
    finally:
        conn.close()


def list_documents() -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(doc_id: str) -> None:
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM paper_intelligence WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM tables WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM equations WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM figures WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def save_chunks(rows: list[dict]) -> None:
    init_db()
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO chunks (chunk_id, document_id, filename, page, section, content,
                content_hash, qdrant_id, start_index)
            VALUES (:chunk_id, :document_id, :filename, :page, :section, :content,
                :content_hash, :qdrant_id, :start_index)
            ON CONFLICT(chunk_id) DO UPDATE SET content=excluded.content, qdrant_id=excluded.qdrant_id
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def load_all_chunks() -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        return [row_to_dict(r) for r in conn.execute("SELECT * FROM chunks ORDER BY filename, page").fetchall()]
    finally:
        conn.close()


def document_full_text(filename: str) -> str:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE filename=? ORDER BY page, start_index", (filename,)
        ).fetchall()
        return "\n".join(r["content"] for r in rows)
    finally:
        conn.close()


def save_json_row(table: str, row: dict) -> None:
    init_db()
    conn = get_connection()
    try:
        keys = ",".join(row.keys())
        placeholders = ",".join(f":{k}" for k in row)
        updates = ",".join(f"{k}=excluded.{k}" for k in row if k != "id")
        pk = "id" if "id" in row else "document_id"
        conn.execute(
            f"INSERT INTO {table} ({keys}) VALUES ({placeholders}) ON CONFLICT({pk}) DO UPDATE SET {updates}",
            row,
        )
        conn.commit()
    finally:
        conn.close()


def list_rows(table: str, document_id: str | None = None) -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        if document_id:
            rows = conn.execute(f"SELECT * FROM {table} WHERE document_id=?", (document_id,)).fetchall()
        else:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def save_paper_intelligence(document_id: str, payload: dict, model: str) -> None:
    save_json_row(
        "paper_intelligence",
        {
            "document_id": document_id,
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "extracted_at": utcnow(),
            "model": model,
        },
    )


def get_paper_intelligence(document_id: str | None = None) -> list[dict]:
    rows = list_rows("paper_intelligence", document_id)
    out = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        payload["document_id"] = row["document_id"]
        payload["extracted_at"] = row["extracted_at"]
        payload["model"] = row["model"]
        out.append(payload)
    return out


def cache_get_all() -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        return [row_to_dict(r) for r in conn.execute("SELECT * FROM semantic_cache").fetchall()]
    finally:
        conn.close()


def cache_put(query: str, query_hash: str, embedding: list[float], response: dict) -> None:
    init_db()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO semantic_cache (query, query_hash, embedding_json, response_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (query, query_hash, json.dumps(embedding), json.dumps(response), utcnow()),
        )
        conn.commit()
    finally:
        conn.close()


def save_eval_run(run: dict) -> None:
    save_json_row("eval_runs", run)


def list_eval_runs() -> list[dict]:
    return list_rows("eval_runs")


def log_trace(name: str, duration_ms: float, attributes: Any, error: str | None) -> None:
    init_db()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO traces (name, started_at, duration_ms, attributes_json, error) VALUES (?, ?, ?, ?, ?)",
            (name, utcnow(), duration_ms, json.dumps(attributes, default=str), error),
        )
        conn.commit()
    finally:
        conn.close()


def list_traces(limit: int = 100) -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def save_workflow(state_id: str, question: str, state: dict) -> None:
    save_json_row(
        "workflow_states",
        {
            "id": state_id,
            "question": question,
            "state_json": json.dumps(state, default=str),
            "updated_at": utcnow(),
        },
    )
