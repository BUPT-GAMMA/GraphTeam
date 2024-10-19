from dataclasses import dataclass, field

@dataclass(frozen=True)
class MemoryConfig:
    r"""
    Defines the configuration parameters for RAG queries.
    """
    temperature: float = 0.3  # Model temperature
    similarity_top_k: int = 1  # Number of top-k results to return
    llm_model: str = "gpt-4o-mini"  # The model used for RAG queries
    persist_dir: str = "./storageGPT_memory"  # Directory to persist the RAG index
    threshold: float = 0.85  # Threshold for filtering the results