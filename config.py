from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_eval_model: str = "qwen2.5:3b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_model_upgrade: str = "BAAI/bge-m3"
    use_bge_m3: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    qdrant_api_key: str = ""
    qdrant_cluster_endpoint: str = ""
    qdrant_collection: str = "nasa_space_physics"
    bm25_path: str = "./storage/bm25.pkl"
    sqlite_path: str = "./storage/orbit.sqlite"
    library_dir: str = "./data/library"
    raw_dir: str = "./data/raw"
    figures_dir: str = "./data/figures"
    chunk_size: int = 850
    chunk_overlap: int = 140
    dense_k: int = 20
    sparse_k: int = 20
    rerank_top_n: int = 6
    dense_weight: float = 0.55
    sparse_weight: float = 0.45
    enable_query_rewrite: bool = True
    enable_multi_query: bool = True
    enable_query_decompose: bool = True
    enable_context_compression: bool = True
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.92
    max_upload_mb: int = 80
    ocr_enabled: bool = True
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "nasa-orbit"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_send_content: bool = False
    otel_enabled: bool = True
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def active_embedding_model(self) -> str:
        return self.embedding_model_upgrade if self.use_bge_m3 else self.embedding_model

    def ensure_dirs(self) -> None:
        paths = [
            Path(self.bm25_path).parent,
            Path(self.sqlite_path).parent,
            self.library_dir,
            self.raw_dir,
            self.figures_dir,
        ]
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
