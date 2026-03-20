from src.components.shared import get_LM_model, get_LM_tokenizer
import torch
from src.logging import logger

class Generate:
    def __init__(self, model_ckpt:str, 
                 system_prompt:str, 
                 temperature:float,
                 max_new_tokens:int):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = get_LM_model(model_ckpt)
        self.tokenizer = get_LM_tokenizer(model_ckpt)
        

    # generate.py

    def generate(self, query: str, context: list):
        logger.info("Building prompt...")
        
        context_str = "\n\n".join([
            f"[{c['source']}, page {c['page']}]\n{c['text']}"
            for c in context
        ])

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        logger.info("Applying chat template...")
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        logger.info(f"Input tokens: {tokenized['input_ids'].shape[-1]}")
        logger.info("Running model.generate()...")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        logger.info("Decoding response...")
        input_length = tokenized["input_ids"].shape[-1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        logger.info("Done.")
        return response