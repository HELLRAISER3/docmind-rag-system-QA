from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

class Embed:
    def __init__(self, 
                 model_ckpt:str, 
                 vector_dim:int, 
                 qdrant_url:str, 
                 qdrant_port:int, 
                 qdrant_collection_name:str):
        self.model_ckpt = model_ckpt
        self.vector_dim = vector_dim
        self.qdrant_url = qdrant_url
        self.qdrant_port = qdrant_port
        self.qdrant_collection_name = qdrant_collection_name

        self.client = QdrantClient(qdrant_url, port=qdrant_port)
        self.model = SentenceTransformer(model_ckpt)

        vectors_args = VectorParams(size=vector_dim,
                                   distance=Distance.COSINE)
        
        self.client.recreate_collection(
            collection_name=qdrant_collection_name,
            vectors_config=vectors_args
        )

    def embed_docs(self, docs):
        vectors = self.model.encode(docs)

        points = [
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "text":        chunk["text"],
                    "source":      chunk["metadata"]["source"],
                    "page":        chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                }
            )
            for idx, (chunk, vector) in enumerate(zip(docs, vectors))
        ]

        self.client.upsert(collection_name=self.qdrant_collection_name,
                           points=points)
        


