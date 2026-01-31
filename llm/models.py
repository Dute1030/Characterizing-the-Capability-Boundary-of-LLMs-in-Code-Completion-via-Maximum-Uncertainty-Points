import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import AutoTokenizer

def init_qwen3_600m(model_path="Qwen/Qwen3-0.6B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwen3_2b(model_path="Qwen/Qwen3-1.7B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwen3_4b(model_path="Qwen/Qwen3-4B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwen3_8b(model_path="Qwen/Qwen3-8B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_codegpt_small_py(model_path="microsoft/CodeGPT-small-py", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

MODEL_FACTORY = {
    "qwen3-0.6b": init_qwen3_600m,
    "qwen3-1.7b": init_qwen3_2b,
    "qwen3-4b": init_qwen3_4b,
    "qwen3-8b": init_qwen3_8b,
    "codegpt-small-py": init_codegpt_small_py,
}