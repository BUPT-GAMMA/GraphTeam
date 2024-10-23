from dataclasses import dataclass, field

@dataclass(frozen=True)
class RAGConfig:
    r"""
    Defines the configuration parameters for RAG queries.
    """
    temperature: float = 0.3  # Model temperature
    similarity_top_k: int = 3  # Number of top-k results to return
    llm_model: str = "gpt-4o-mini"  # The model used for RAG queries
    persist_dir: str = "./storageGPT"  # Directory to persist the RAG index
