from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

NAMESPACE = uuid.UUID("8c2e4a1b-6d9f-4c33-9a70-7b1d0e4f91aa")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_id(*parts: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(parts)))
