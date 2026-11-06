from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from ingestion.hashing import text_sha256

SECTION_RE = re.compile(
    r"(?im)^(abstract|introduction|methods?|methodology|data|observations?|"
    r"results?|discussion|conclusions?|references|acknowledgments?)\b"
)


def infer_section(text: str, fallback: str = "body") -> str:
    match = SECTION_RE.search(text[:800])
    return match.group(1).lower() if match else fallback


def split_documents(documents, chunk_size: int | None = None, chunk_overlap: int | None = None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    counters: dict[tuple, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        start = chunk.metadata.get("start_index", 0)
        key = (source, page)
        counters[key] = counters.get(key, 0) + 1
        local_i = counters[key]
        content_hash = text_sha256(chunk.page_content)
        chunk.metadata["section"] = infer_section(chunk.page_content, chunk.metadata.get("section", "body"))
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["chunk_id"] = f"{source}:{page}:{start}:{content_hash[:12]}"
        chunk.metadata["local_index"] = local_i
    return chunks
