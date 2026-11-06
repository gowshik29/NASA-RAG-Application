from mcp.server.fastmcp import FastMCP
from rag import NASAResearchRAG

mcp = FastMCP("nasa-space-physics-research")
rag = None

def get_rag():
    global rag
    if rag is None:
        rag = NASAResearchRAG()
    return rag

@mcp.tool()
def ask_research_papers(question: str) -> str:
    # Ask the indexed corpus and return cited answer text.
    return get_rag().ask(question)["answer"]

@mcp.tool()
def search_research_evidence(question: str) -> list[dict]:
    # Return ranked evidence passages and their page metadata.
    return [{
        "filename": d.metadata.get("source"),
        "page": d.metadata.get("page"),
        "doi": d.metadata.get("doi"),
        "text": d.page_content[:1500],
        "reranker_score": d.metadata.get("rerank_score"),
    } for d in get_rag().retrieve(question)]

if __name__ == "__main__":
    mcp.run()
