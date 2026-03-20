import argparse

from src.components.generate import Generate
from src.components.retrieve import Retrieve
from src.config.configuration import ConfigurationManager


def run_query(
    user_query: str,
    top_k: int = None,
    threshold: float = None,
    collection: str = None,
) -> str:
    configuration_manager = ConfigurationManager()
    qdrant_config   = configuration_manager.get_qdrant_config()
    embed_config    = configuration_manager.get_embed_config()
    retrieve_config = configuration_manager.get_retrieve_config()
    generate_config = configuration_manager.get_generate_config()

    retriever = Retrieve(
        qdrant_url=qdrant_config.url,
        qdrant_port=qdrant_config.port,
        model_ckpt=embed_config.embedding_model_ckpt,
        collection_name=collection or qdrant_config.collection_name,
        top_k=top_k or retrieve_config.top_k,
        threshold_score=threshold or retrieve_config.threshold_score,
    )
    context = retriever.retrieve(user_query)

    if not context:
        return ""

    generator = Generate(
        model_ckpt=generate_config.model_ckpt,
        system_prompt=generate_config.system_prompt,
        temperature=generate_config.temperature,
        max_new_tokens=generate_config.max_new_tokens,
    )
    return generator.generate(query=user_query, context=context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query ingested documents via RAG.")
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Question to ask the documents.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: retrieve_config.top_k from config.yaml)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Minimum similarity score to keep a chunk (default: retrieve_config.threshold_score from config.yaml)",
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=None,
        help="Qdrant collection to query (default: collection_name from qdrant_config.yaml)",
    )
    args = parser.parse_args()

    response = run_query(
        user_query=args.query,
        top_k=args.top_k,
        threshold=args.threshold,
        collection=args.collection,
    )
    print(response if response else "No relevant passages found.")
