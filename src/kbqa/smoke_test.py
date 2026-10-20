"""Phase 1 - GPU smoke test: can we load a model in 4-bit and train LoRA for a few steps?

Run:  uv run python -m kbqa.smoke_test [model_id]

Uses a dummy sentence, NOT the confidential KBA, so the output is safe to share.
It checks the whole QLoRA tool chain (torch CUDA -> bitsandbytes -> transformers -> peft)
and prints load time, peak VRAM and peak process RAM.
"""
import sys
import time

import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main(model_id="Qwen/Qwen3-0.6B"):
    MODEL_ID = model_id
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | bf16 supported: {torch.cuda.is_bf16_supported()}")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()

    # 4-bit NF4 quantization = the "Q" in QLoRA.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0})
    print(f"Loaded {MODEL_ID} in {time.perf_counter() - t0:.1f} s")
    print(f"attention_bias in config: {getattr(model.config, 'attention_bias', 'n/a')}")

    # LoRA adapters on all attention and MLP projections.
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Dummy training text (illustrative only).
    batch = tok(["A database index can speed up queries that filter on a column."] * 2,
                return_tensors="pt").to(0)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    for step in range(5):
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(f"step {step} loss {loss.item():.4f}")

    torch.cuda.synchronize()
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"Process RAM (RSS) now: {psutil.Process().memory_info().rss / 2**30:.2f} GiB")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main(*sys.argv[1:2])
