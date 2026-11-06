from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval.index import load_index
from retrieval.hybrid import make_hybrid_retriever
from retrieval.rerank import LocalReranker
from config import settings


SYSTEM_PROMPT = """
You are NASA ORBIT, a scientific research assistant specializing in
NASA space physics research papers.

Answer questions using only the supplied research-paper context.

Rules:
1. Use evidence from the retrieved documents.
2. Never invent scientific findings, numerical values, or citations.
3. Cite the source filename and page number when available.
4. If the context does not contain the answer, clearly say so.
5. Distinguish reported findings from your own interpretation.
6. Preserve scientific terminology, equations, and units.
7. When comparing papers, identify which paper supports each claim.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    (
        "human",
        """
        Research question:
        {question}

        Retrieved scientific evidence:
        {context}

        Provide a clear, evidence-grounded answer with source citations.
        """,
    ),
])


class NASAResearchRAG:
    def __init__(self):
        self.store, self.bm25 = load_index()
        self.retriever = make_hybrid_retriever(self.store, self.bm25)
        self.reranker = LocalReranker()
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self.chain = PROMPT | self.llm | StrOutputParser()

    def retrieve(self, question: str, top_k: int = 6):
        candidates = self.retriever.invoke(question)
        documents = []

        for item in candidates:
            if isinstance(item, tuple):
                document = item[0]
                if len(item) > 1 and isinstance(item[1], dict):
                    if not getattr(document, "metadata", None):
                        document.metadata = {}
                    document.metadata.update(item[1])
            else:
                document = item

            if not getattr(document, "metadata", None):
                document.metadata = {}
            if hasattr(document, "page_content"):
                documents.append(document)

        if not documents:
            return []

        ranked_documents = self.reranker.rerank(
            question,
            documents,
            top_n=top_k,
        )

        normalized = []
        for item in ranked_documents:
            if isinstance(item, tuple):
                document = item[0]
                if len(item) > 1 and isinstance(item[1], dict):
                    if not getattr(document, "metadata", None):
                        document.metadata = {}
                    document.metadata.update(item[1])
            else:
                document = item

            if not getattr(document, "metadata", None):
                document.metadata = {}
            if hasattr(document, "page_content"):
                normalized.append(document)

        return normalized

    def ask(self, question: str, top_k: int = 6):
        ranked_documents = self.retrieve(question, top_k=top_k)

        if not ranked_documents:
            return {
                "answer": (
                    "I could not find relevant evidence in the "
                    "indexed NASA research papers."
                ),
                "sources": [],
            }

        context_parts = []
        sources = []

        for document in ranked_documents:
            metadata = document.metadata or {}
            filename = metadata.get("filename", metadata.get("source", "Unknown document"))
            page = metadata.get("page", metadata.get("page_number", "Unknown"))
            doi = metadata.get("doi", "Not available")

            context_parts.append(
                f"""
                Source: {filename}
                Page: {page}
                DOI: {doi}

                Evidence:
                {document.page_content}
                """
            )
            sources.append({
                "filename": filename,
                "page": page,
                "doi": doi,
                "snippet": document.page_content[:500],
                "score": metadata.get("rerank_score"),
            })

        answer = self.chain.invoke({
            "question": question,
            "context": "\n\n".join(context_parts),
        })

        return {"answer": answer, "sources": sources}
