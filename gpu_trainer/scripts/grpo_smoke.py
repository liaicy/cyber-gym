"""GRPO smoke test: does the full loop close (generate -> real CyberGym tool
calls via the bridge -> real docker verify() reward -> advantage -> one LoRA
step), not whether it trains anything useful. 1-2 steps, small K, two known-
reachable tasks (arvo:3938 near-trivial, arvo:8896 has real within-task
variance so advantage isn't degenerate).

environment_factory points at CyberGymGRPOEnvClient, a thin HTTP client --
the real CyberGymEnv/docker/verify all live on the cloud machine behind a
reverse SSH tunnel (see traj/grpo_bridge_server.py there).

Usage (GPU node, venv active):
  python grpo_smoke.py --bridge-port 18666
"""
import argparse

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo_env_client import CyberGymGRPOEnvClient  # same dir, copied alongside this script

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--bridge-port", type=int, default=18666)
ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=16)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)

# No "prompt" column -- TRL defaults to an empty user-turn placeholder and
# reset()'s returned task description gets appended to it. task_id is the
# only column; GRPOConfig.num_generations controls how many rollouts get
# sampled per row automatically, no manual repetition needed.
ds = Dataset.from_dict({"task_id": ["arvo:3938", "arvo:8896"]})

cfg = GRPOConfig(
    output_dir=args.out,
    max_steps=2,
    per_device_train_batch_size=2,
    num_generations=4,
    generation_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    beta=0.1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    use_vllm=True,
    vllm_mode="colocate",
    max_tool_calling_iterations=10,
    temperature=0.8,
    seed=0,
)
peft_cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")

trainer = GRPOTrainer(
    model=args.model,
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
    peft_config=peft_cfg,
    environment_factory=lambda: CyberGymGRPOEnvClient(
        base_url=f"http://localhost:{args.bridge_port}"
    ),
)
trainer.train()
print("SMOKE_TEST_DONE")
