import argparse
from pathlib import Path

from src.components.chunk import Chunk
from src.components.embed import Embed
from src.components.parse import Parse
from src.config.configuration import ConfigurationManager
from src.logging import logger


def run_ingestion(data_folder: Path = None, collection: str = None) -> list:
    configuration_manager = ConfigurationManager()
    qdrant_config = configuration_manager.get_qdrant_config()
    parse_config  = configuration_manager.get_parse_config()
    chunk_config  = configuration_manager.get_chunk_config()
    embed_config  = configuration_manager.get_embed_config()

    resolved_folder     = data_folder or parse_config.file_folder
    resolved_collection = collection  or qdrant_config.collection_name

    parser = Parse(resolved_folder, parse_config.clean_metadata)
    docs   = parser.parse_docs()

    chunker = Chunk(chunk_size=chunk_config.chunk_size,
                    chunk_overlap=chunk_config.chunk_overlap)
    chunks  = chunker.chunk_docs(docs)

    embedder = Embed(
        model_ckpt=embed_config.embedding_model_ckpt,
        vector_dim=embed_config.vector_dim,
        qdrant_url=qdrant_config.url,
        qdrant_port=qdrant_config.port,
        collection_name=resolved_collection,
    )
    embedder.embed_chunks(chunks)

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into Qdrant.")
    parser.add_argument(
        "--data-folder", "-d",
        type=Path,
        default=None,
        help="Folder with PDF files to ingest (default: parse_config.file_folder from config.yaml)",
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=None,
        help="Qdrant collection name (default: collection_name from qdrant_config.yaml)",
    )
    args = parser.parse_args()

    chunks = run_ingestion(data_folder=args.data_folder, collection=args.collection)
    logger.info(f"Ingestion complete — {len(chunks)} chunks stored.")
