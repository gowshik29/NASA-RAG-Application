from __future__ import annotations

import re
import logging
from pathlib import Path

import fitz
from langchain_core.documents import Document

from config import settings
from ingestion.hashing import file_sha256

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
MAX_PAGES_GUARD = 400
logger = logging.getLogger(__name__)

STATUSES = [
    "Uploaded",
    "Validating",
    "Extracting",
    "Analyzing",
    "Chunking",
    "Embedding",
    "Indexing",
    "Completed",
]


class PDFValidationError(ValueError):
    pass


def validate_pdf(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise PDFValidationError("File does not exist.")
    if path.suffix.lower() != ".pdf":
        raise PDFValidationError("Only PDF files are accepted.")
    size = path.stat().st_size
    if size < 64:
        raise PDFValidationError("File is too small to be a valid PDF.")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size > max_bytes:
        raise PDFValidationError(f"File exceeds {settings.max_upload_mb} MB limit.")
    try:
        pdf = fitz.open(path)
    except Exception as exc:
        raise PDFValidationError(f"PyMuPDF could not open the file: {exc}") from exc
    if pdf.is_encrypted:
        raise PDFValidationError("Encrypted PDFs are not supported.")
    if pdf.page_count < 1:
        raise PDFValidationError("PDF has no pages.")
    if pdf.page_count > MAX_PAGES_GUARD:
        raise PDFValidationError(f"PDF has {pdf.page_count} pages; maximum is {MAX_PAGES_GUARD}.")
    header = path.read_bytes()[:8]
    if b"%PDF" not in header:
        raise PDFValidationError("Missing %PDF header.")
    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "file_size": size,
        "page_count": pdf.page_count,
        "content_hash": file_sha256(path),
    }


def _ocr_page(page) -> str:
    if not settings.ocr_enabled:
        return ""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning(
            "OCR is enabled but pytesseract or Pillow is unavailable; "
            "install project dependencies to OCR scanned PDF pages."
        )
        return ""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(image) or ""
    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "OCR is enabled but the Tesseract executable was not found; "
            "install Tesseract OCR and make it available on PATH."
        )
        return ""
    except Exception:
        logger.exception("OCR failed while processing a PDF page.")
        return ""


def extract_page_text(page) -> tuple[str, bool]:
    text = page.get_text("text", sort=True).strip()
    ocr_used = False
    if len(text) < 40:
        ocr = _ocr_page(page).strip()
        if len(ocr) > len(text):
            text = ocr
            ocr_used = True
    return text, ocr_used


def load_pdf(path) -> list[Document]:
    path = Path(path)
    info = validate_pdf(path)
    pdf = fitz.open(path)
    first_pages = "\n".join(p.get_text("text") for p in pdf[: min(3, len(pdf))])
    match = DOI_RE.search(first_pages)
    doi = match.group(0).rstrip(".,)") if match else ""
    docs: list[Document] = []
    ocr_pages = 0
    for page_no, page in enumerate(pdf, start=1):
        text, ocr_used = extract_page_text(page)
        if ocr_used:
            ocr_pages += 1
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "title": path.stem,
                    "page": page_no,
                    "doi": doi,
                    "section": "page_text",
                    "chunk_type": "scientific_text",
                    "ocr_used": ocr_used,
                    "page_count": info["page_count"],
                    "content_hash": info["content_hash"],
                },
            )
        )
    if docs:
        docs[0].metadata["ocr_pages"] = ocr_pages
    return docs


def load_directory(directory) -> list[Document]:
    docs: list[Document] = []
    for pdf in sorted(Path(directory).glob("*.pdf")):
        docs.extend(load_pdf(pdf))
    return docs


def preview_text(path: str | Path, max_chars: int = 2500) -> str:
    docs = load_pdf(path)
    text = "\n\n".join(d.page_content for d in docs[:2])
    return text[:max_chars]
