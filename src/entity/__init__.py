from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class QdrantConfig:
    url: str
    port: int
    collection_name: str

@dataclass(frozen=True)
class ParseConfig:
    file_folder: Path
    clean_metadata: bool

@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int
    chunk_overlap: int

@dataclass(frozen=True)
class EmbedConfig:
    embedding_model_ckpt: str
    vector_dim: int


