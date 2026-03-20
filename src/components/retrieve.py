from src.components.shared import get_qdrant_client, get_embedding_model
from src.logging import logger


class Retrieve:
    def __init__(self,
                 qdrant_url: str,
                 qdrant_port: int, 
                 model_ckpt: str,
                 collection_name: str, 
                 top_k: int,
                 threshold_score: float):
        self.collection_name = collection_name
        self.threshold_score = threshold_score
        self.top_k = top_k

        self.client = get_qdrant_client(qdrant_url, qdrant_port)
        self.model = get_embedding_model(model_ckpt)

    def retrieve(self, query: str):
        query_vector = self.model.encode(query).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.top_k
        )

        if not results.points or results.points[0].score < self.threshold_score:
            logger.info("No relevant information!")
            return []

        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "page": r.payload["page"],
                "score": r.score,
            }
            for r in results.points
        ]