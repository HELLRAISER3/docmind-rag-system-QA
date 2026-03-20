from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from src.logging import logger
import gc
import torch

# Singletone for qdrant_client and embedding model

_qdrant_client = None
_embedding_model = None
_LM_model = None
_LM_tokenizer = None

def get_qdrant_client(url: str, port: int) -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Initializing Qdrant client...")
        _qdrant_client = QdrantClient(url, port=port)
    return _qdrant_client

def get_embedding_model(model_ckpt: str) -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model {model_ckpt}...")
        _embedding_model = SentenceTransformer(model_ckpt)
    return _embedding_model

def get_LM_model(model_ckpt: str) -> AutoModelForCausalLM:
    global _LM_model
    if _LM_model is None:
        gc.collect()
        if torch.cuda.is_available():
            logger.info(f"Loading LM model: {model_ckpt}...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            _LM_model = AutoModelForCausalLM.from_pretrained(
                model_ckpt,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            logger.info(f"Loading LM model (float32, CPU): {model_ckpt}...")
            _LM_model = AutoModelForCausalLM.from_pretrained(
                model_ckpt,
                dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
    return _LM_model

def get_LM_tokenizer(model_ckpt: str) -> AutoTokenizer:
    global _LM_tokenizer
    if _LM_tokenizer is None:
        logger.info(f"Loading LM tokenizer: {model_ckpt}...")
        _LM_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        if _LM_tokenizer.pad_token is None:
            _LM_tokenizer.pad_token = _LM_tokenizer.eos_token
            
    return _LM_tokenizer