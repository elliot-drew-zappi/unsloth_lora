from unsloth import FastLanguageModel
import torch

from unslothlora.prompt import formatting_prompts_func
from unslothlora.config import *

class UnslothModel:
    def __init__(self, model_name, max_seq_length, dtype, load_in_4bit):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

    def load_model(self, for_inference=False):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        self.model = model
        self.tokenizer = tokenizer
        if for_inference:
            FastLanguageModel.for_inference(self.model)
        return True

    def unload_model(self, model):
        del model
        torch.cuda.empty_cache()
        return True

    def inference(
        inputs, prompt_template = alpaca_prompt, max_new_tokens = 512
        ):
        # format the prompt
        prompts = []
        for inp in inputs:
            prompt = formatting_prompts_func(inp)
            prompts.append(prompt)
        
        # generate the response
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        tokenized_prompts = self.tokenizer(
            prompts,
            return_tensors = "pt"
        ).to(self.model.device)
        
        response = self.model.generate(**tokenized_prompts, max_new_tokens = max_new_tokens, use_cache = True)

        return response
