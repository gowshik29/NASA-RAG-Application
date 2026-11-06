import argparse
from ingestion.pdf_loader import load_directory
from ingestion.chunking import split_documents
from retrieval.index import build_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/raw")
    args = parser.parse_args()
    pages = load_directory(args.input)
    if not pages:
        raise SystemExit("No extractable PDF text found. Scanned PDFs need OCR.")
    chunks = split_documents(pages)
    build_index(chunks)
    print(f"Indexed {len(pages)} pages into {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
