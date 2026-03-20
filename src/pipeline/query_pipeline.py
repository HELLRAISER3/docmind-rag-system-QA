from src.components.retrieve import Retrieve
from src.components.generate import Generate

from src.config.configuration import ConfigurationManager


if __name__ == "__main__":
    
    configuration_manager = ConfigurationManager()
    qdrant_config = configuration_manager.get_qdrant_config()
    embed_config = configuration_manager.get_embed_config()
    retrieve_config = configuration_manager.get_retrieve_config()
    generate_config = configuration_manager.get_generate_config()

    user_query = """What was the BLEU score the Transformer could achieved  on the 
    English-to-German and English-to-French newstest2014 test?
    """

    retriver = Retrieve(qdrant_url=qdrant_config.url,
                        qdrant_port=qdrant_config.port,
                        model_ckpt=embed_config.embedding_model_ckpt,
                        collection_name=qdrant_config.collection_name,
                        top_k=retrieve_config.top_k,
                        threshold_score=retrieve_config.threshold_score)

    context = retriver.retrieve(user_query)

    generator = Generate(model_ckpt=generate_config.model_ckpt,
                        system_prompt=generate_config.system_prompt,
                        temperature=generate_config.temperature,
                        max_new_tokens=generate_config.max_new_tokens)

    response = generator.generate(query=user_query,
                    context = context)

    print(response)
