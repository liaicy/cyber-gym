#!/usr/bin/env python3
"""Merge a LoRA adapter into its base and save a plain full model, so vLLM can
serve it WITHOUT --enable-lora (punica JIT dies on this cluster: no usable nvcc).

2026-11-01 (27B): load with the class named in config.json `architectures`
(Qwen3_5ForConditionalGeneration -- the adapter keys are
`base_model.model.model.language_model.layers...`, i.e. the multimodal wrapper,
NOT AutoModelForCausalLM's text-only class; a class mismatch makes PEFT drop
every key silently and "merge" nothing). Verified by diffing a target weight
before/after merge and by PEFT's load_result. Aux files (preprocessor configs,
chat template, vocab) are copied so vLLM sees the same model_type as the base.
"""
import argparse, json, os, shutil, sys

import torch
import transformers
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

arch = json.load(open(os.path.join(a.base, "config.json")))["architectures"][0]
cls = getattr(transformers, arch)
print(f"loading base {a.base} as {arch}", flush=True)
base = cls.from_pretrained(a.base, dtype=torch.bfloat16, low_cpu_mem_usage=True)

# pick a probe weight that the adapter targets
from safetensors import safe_open
with safe_open(os.path.join(a.adapter, "adapter_model.safetensors"), "pt") as f:
    keys = list(f.keys())
probe_key = next(k for k in keys if k.endswith("lora_B.weight"))
probe_mod = probe_key[len("base_model.model."):-len(".lora_B.weight")]  # e.g. model.language_model.layers.0.mlp.down_proj
probe_before = base.get_submodule(probe_mod).weight.detach().clone()
print(f"adapter tensors={len(keys)} probe={probe_mod}", flush=True)

print(f"applying adapter {a.adapter}", flush=True)
model = PeftModel.from_pretrained(base, a.adapter)
n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
print(f"lora params attached={n_lora} (expect {len(keys)})", flush=True)
if n_lora != len(keys):
    print("MERGE_FAIL: attached lora param count != adapter tensor count", flush=True); sys.exit(1)
# make sure the loaded B matrices are not all zero (fresh init would be zero -> silent no-op)
b_norm = sum(p.float().norm().item() for n, p in model.named_parameters() if "lora_B" in n)
print(f"sum ||lora_B||={b_norm:.4f}", flush=True)
if b_norm == 0.0:
    print("MERGE_FAIL: lora_B all zero -> adapter weights were not loaded", flush=True); sys.exit(1)

model = model.merge_and_unload()
probe_after = model.get_submodule(probe_mod).weight.detach()
delta = (probe_after.float() - probe_before.float()).abs().max().item()
print(f"probe max|delta|={delta:.3e}", flush=True)
if delta == 0.0:
    print("MERGE_FAIL: probe weight unchanged after merge", flush=True); sys.exit(1)

model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
for fn in ("chat_template.jinja", "generation_config.json", "merges.txt", "vocab.json",
           "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
           "video_preprocessor_config.json"):
    src = os.path.join(a.base, fn)
    if os.path.exists(src) and not os.path.exists(os.path.join(a.out, fn)):
        shutil.copy(src, a.out)
print(f"MERGE_OK merged -> {a.out}", flush=True)
