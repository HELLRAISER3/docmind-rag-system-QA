from src.components.parse import Parse
from src.components.chunk import Chunk
from src.components.embed import Embed
from src.config.configuration import ConfigurationManager

configuration_manager = ConfigurationManager()
qdrant_config = configuration_manager.get_qdrant_config()
parse_config = configuration_manager.get_parse_config()
chunk_config = configuration_manager.get_chunk_config()
embed_config = configuration_manager.get_embed_config()

parser = Parse(parse_config.file_folder, 
               parse_config.clean_metadata)
docs = parser.parse_docs()

chunker = Chunk(chunk_size=chunk_config.chunk_size, 
                chunk_overlap=chunk_config.chunk_overlap)
chunks = chunker.chunk_docs(docs)

embeder = Embed(model_ckpt=embed_config.embedding_model_ckpt,
                vector_dim=embed_config.vector_dim,
                qdrant_url=qdrant_config.url,
                qdrant_port=qdrant_config.port,
                collection_name=qdrant_config.collection_name)
embeder.embed_chunks(chunks)

