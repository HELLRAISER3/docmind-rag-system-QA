from qdrant_client.models import Distance, VectorParams, PointStruct
from src.components.shared import get_qdrant_client, get_embedding_model


class Embed:
    def __init__(self, model_ckpt, vector_dim, qdrant_url, qdrant_port, collection_name):
        self.collection_name = collection_name
        self.client = get_qdrant_client(qdrant_url, qdrant_port)  
        self.model  = get_embedding_model(model_ckpt)              

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )

    def embed_chunks(self, chunks):
        texts   = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, show_progress_bar=True)

        points = [
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                }
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        return points