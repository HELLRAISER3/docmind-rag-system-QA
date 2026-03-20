from src.constants import *
from src.utils.common import read_yaml
from src.entity import QdrantConfig, ChunkConfig, ParseConfig, EmbedConfig
from src.logging import logger

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        qdrant_config_filepath = QDRANT_CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.qdrant_config = read_yaml(qdrant_config_filepath)


    def get_qdrant_config(self) -> QdrantConfig:
        config = self.qdrant_config
        
        qdrant_config = QdrantConfig(
            url=config["url"],
            port=config["port"],
            collection_name=config["collection_name"],
        )
        return qdrant_config
    
    def get_parse_config(self) -> ParseConfig:
        config = self.config["parse_config"]

        parse_config = ParseConfig(
            file_folder=Path(config["file_folder"]),
            clean_metadata=config["clean_metadata"],
        )

        return parse_config
    
    def get_chunk_config(self) -> ChunkConfig:
        config = self.config["chunk_config"]

        chunk_config = ChunkConfig(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )

        return chunk_config

    def get_embed_config(self) -> EmbedConfig:
        config = self.config["embed_config"]

        embed_config = EmbedConfig(
            embedding_model_ckpt=config["embedding_model_ckpt"],
            vector_dim=config["vector_dim"]
        )

        return embed_config