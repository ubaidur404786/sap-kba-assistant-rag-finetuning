"""Load a Hugging Face causal LM and generate answers - shared by every phase.

One place for: model loading (bf16 or 4-bit, optionally with a PEFT adapter), chat prompt
formatting, and batched greedy/sampled generation with token counts.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# The tiny Hugging Face models compared in this project (all Apache-2.0, all <= 1.1B params).
TINY_MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def load_model(model_id, quant_4bit=False, adapter_dir=None):
    """Return (model, tokenizer, load_seconds). Everything lives on the single GPU."""
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"                 # left padding: all prompts end at the same position
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None
    if quant_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0},
                                                 quantization_config=quant)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    torch.cuda.synchronize()
    return model, tok, time.perf_counter() - t0


def chat_prompt(tok, user, system=None):
    """Format one user message with the model's own chat template.
    enable_thinking=False turns off Qwen3's reasoning mode; other templates ignore it."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.inference_mode()
def generate(model, tok, prompts, max_new_tokens=200, batch_size=8, temperature=0.0, seed=42):
    """Generate for a list of formatted prompts. Greedy if temperature == 0 (deterministic).
    Returns a list of dicts: text, input_tokens, output_tokens."""
    torch.manual_seed(seed)
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            kwargs.update(do_sample=False)
        out = model.generate(**batch, **kwargs)
        new = out[:, batch["input_ids"].shape[1]:]
        for row_in, row in zip(batch["attention_mask"], new):
            n_out = int((row != tok.pad_token_id).sum())
            results.append({"text": tok.decode(row, skip_special_tokens=True).strip(),
                            "input_tokens": int(row_in.sum()), "output_tokens": n_out})
    return results
