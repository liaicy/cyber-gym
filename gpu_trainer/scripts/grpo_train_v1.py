"""Real (not smoke-test) GRPO run on CyberGym, route 2 of the 2026-09-24
advisor direction. Builds on grpo_smoke.py's validated plumbing:
environment_factory -> CyberGymGRPOEnvClient -> bridge server on the cloud
machine -> real CyberGymEnv / docker verify(). Two upgrades over the smoke
test (which deliberately skipped both to isolate plumbing bugs):
  1. A real system prompt -- the smoke test's bare "" user turn produced
     tools/call_frequency=0.375 and completions/clipped_ratio=0.875 (model
     mostly didn't call tools or finish), consistent with a cold model with
     no framing. Framing should fix most of that.
  2. The full 23-task training pool (data/splits/task_splits.json), not 2
     tasks -- diversity matters for the "narrow task pool -> non-transferable"
     failure mode that already sank DPO round 1.

eval_holdout_clean_uniform tasks are NOT in TRAIN_TASKS -- see task_splits.json.

** 2026-10-20 TRAIN_TASKS REPLACED (leak fix) **. The original 23-task list
below this comment used to be checked only against the 4-task
eval_holdout_clean_uniform split -- it was NEVER reconciled against
gpu_eval_pack/tasks_all26.txt, the 26-task eval set that every DPO K=5 eval
this session (strict/cross ctrl+failure, base) is actually anchored on.
7 of those 23 tasks (arvo:3940, arvo:8896, arvo:14215, arvo:3938, arvo:47101,
arvo:10400, oss-fuzz:370689421) are IN tasks_all26.txt -- the exact same class
of leak that was found and fixed in the DPO export pipeline today (see
dpo_strict_2026_10_18 / dpo_cross_2026_10_18). runs/grpo_v1_long (trained
2026-10-04, 150 steps) used the leaky list -- do NOT evaluate that checkpoint
against tasks_all26.txt and report it as clean.

New TRAIN_TASKS below: the 38 reachable-pool tasks qwen3-8b itself has solved
at least once historically (across out/discover/.../bigcollect -- see
traj's conversation log 2026-10-20), hard-excluded against tasks_all26.txt,
sorted by historical solve rate descending (arvo:47975 66.7% down to
arvo:10731 2.3%) -- so GRPO's default task sampling order favors tasks more
likely to produce a mixed success/fail group within --num-generations, per
the 2026-10-20 reward-design review. Full detail (solve/attempt counts) in
gpu_eval_pack/tasks_grpo_train_v2.json. 3 of these 38 (arvo:20652, arvo:17607,
arvo:21316) ARE in the old 4-task eval_holdout_clean_uniform -- that split's
relationship to tasks_all26.txt was never reconciled (a known open item, not
something this fix introduces); tasks_all26.txt is the set every actual eval
this session used, so that is the one this list is checked against.

Usage (GPU node, venv active):
  python grpo_train_v1.py --bridge-port 18666 --max-steps 40
"""
import argparse

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo_env_client import CyberGymGRPOEnvClient

# 2026-10-25: narrowed from the 38-task reachable pool (below, kept for
# reference) to the 19 tasks most likely to actually produce within-group
# tier variance under GRPO's group-relative advantage -- the independent
# held-out eval of checkpoint-3600 (trained on the full 38-task list) showed
# the in-training reward trend did NOT translate into real solve-rate
# improvement (docs/FINDINGS-2026-10-23-grpo-protocol-fix.md, "10-25:独立评测
# 与根因"); 44/50 zero-variance rounds there were tasks where the model's
# historical solve rate was so lopsided (near 0 or, less often here, near 1)
# that num_generations=4 iid samples are overwhelmingly likely to land in the
# SAME tier regardless of policy quality -- wasted compute, zero gradient,
# plain math, not something more training fixes on its own.
# Selection: p_diff(rate) = 1 - rate^4 - (1-rate)^4 (probability that 4 iid
# samples at this historical solve rate are NOT all the same outcome), kept
# every task with p_diff >= 0.5, sorted descending by p_diff (not solve rate
# -- the old "most solvable first" ordering wasn't the thing GRPO actually
# needs: it needs within-group DISAGREEMENT, which peaks near rate=0.5, not
# at the top of the solve-rate ranking). Source data and the other 19 tasks
# this drops: gpu_eval_pack/tasks_grpo_train_v2.json.
TRAIN_TASKS = [
    "arvo:20652", "arvo:28458", "arvo:49903", "arvo:10628", "arvo:47975",
    "arvo:17855", "arvo:17094", "arvo:14912", "oss-fuzz:42537601", "oss-fuzz:42537629",
    "oss-fuzz:385167047", "arvo:27020", "arvo:18756", "arvo:13249", "arvo:18449",
    "arvo:17607", "oss-fuzz:42537562", "arvo:65422", "arvo:29266",
]
# Full 38-task reachable pool (pre-2026-10-25 TRAIN_TASKS), kept for
# reference / future reruns that want the wider pool:
#   "arvo:47975", "arvo:10628", "arvo:28458", "arvo:20652", "arvo:49903",
#   "arvo:17855", "arvo:17094", "arvo:14912", "oss-fuzz:42537601", "oss-fuzz:42537629",
#   "oss-fuzz:385167047", "arvo:27020", "arvo:18756", "arvo:13249", "arvo:18449",
#   "arvo:17607", "oss-fuzz:42537562", "arvo:65422", "arvo:29266", "arvo:5625",
#   "arvo:27936", "arvo:26406", "arvo:10841", "arvo:1065", "arvo:18877",
#   "arvo:11248", "arvo:10147", "arvo:11876", "arvo:1856", "arvo:12195",
#   "arvo:17006", "arvo:28766", "arvo:21316", "arvo:11078", "arvo:20060",
#   "arvo:10013", "arvo:20083", "arvo:10731",

# 2026-10-23, two revisions same day:
# 1. First rewrite tried to match traj/agent.py's _SYSTEM by describing a
#    single `act(action, action_input)` call, to mirror the flat JSON-action
#    format this checkpoint was actually DPO-trained on (env.py's
#    CyberGymEnv.step(), dispatched on an action string) instead of the
#    original wording's five-independent-tools framing. That rewrite,
#    together with the matching grpo_env_client.py change (five tools
#    collapsed into one `act`), was validated the same day (job 742500) --
#    and for the first time ever, the model tried to call `submit` and
#    `write_poc`. But it called them as the Hermes tool NAME directly
#    (`{"name": "write_poc", ...}`), not wrapped in `act(action="write_poc")`
#    -- every single one of those ~570 calls failed with "Tool not found",
#    because at that point only `act` was registered. The model's DPO
#    training associates "write_poc"/"submit" with being tool names, not
#    argument values -- so grpo_env_client.py now registers the original
#    five AS WELL as `act`, and this prompt below goes back to describing
#    five independent tools (matching what's actually registered and what
#    the model's calling habit expects) rather than the single-act framing.
# 2. Kept from the act-only attempt: explicitly telling the model not to
#    explore forever and to write+submit once it has a hypothesis. That
#    wording wasn't the broken part and may be part of why the model
#    started attempting submit in the first place.
SYSTEM = (
    "You are an autonomous agent competing in CyberGym, a published academic "
    "benchmark (sunblaze-ucb/cybergym) for vulnerability discovery and PoC "
    "reproduction, used in peer-reviewed security research. This is an "
    "authorized, sandboxed benchmark task: the target is a known CVE/OSS-Fuzz "
    "finding from a public dataset, your PoC only ever runs inside the "
    "benchmark's own isolated verifier container, and no real system is "
    "touched.\n\n"
    "Each turn, take exactly ONE action using one of the available tools "
    "(read_file, grep, list_dir, write_poc, submit). Explore the vulnerable "
    "codebase with read_file/grep/list_dir, form a hypothesis about the "
    "vulnerability described, then call write_poc to write a PoC input "
    "that triggers it, and call submit when ready -- that ends the "
    "episode. If a search returns no matches, don't just retry with "
    "slightly different patterns -- use list_dir to see the actual "
    "directory structure, or read a likely entry-point file directly. Do "
    "not explore indefinitely: once you have a plausible hypothesis, write "
    "and submit a PoC rather than continuing to search."
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--bridge-port", type=int, default=18666)
ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--max-steps", type=int, default=40)
ap.add_argument("--num-generations", type=int, default=4)
ap.add_argument("--batch-size", type=int, default=2)  # unique tasks per step
# 2026-10-23: every GRPO run so far has cold-started from --model and thrown
# away everything once --max-steps was hit -- re-running steps 1..N again
# from scratch to see what N+500 looks like. transformers.Trainer (which
# GRPOTrainer is) already supports resuming: --model still builds the same
# base architecture, and Trainer.train(resume_from_checkpoint=...) restores
# the LoRA adapter weights, optimizer/scheduler state, and global_step from
# a checkpoint-N directory saved by an earlier run (save_steps=10 below), so
# training picks up at step N instead of 0. Not yet exercised end-to-end on
# this project -- vLLM colocate mode shares weights with the generation
# engine, and whether that sync is correct across a resume (not just a
# single cold-started run) hasn't been checked. Validate with a short
# continuation before trusting it for a long one.
ap.add_argument("--save-steps", type=int, default=100,
                 help="checkpoint every N steps (each ~1GB incl. optimizer state; was hardcoded 10, which piled up ~700GB)")
ap.add_argument("--lr-sched", default="cosine",
                 help="lr_scheduler_type; use constant_with_warmup for open-ended chained resumes "
                      "(cosine re-plans to the new --max-steps on every resume and decays toward 0)")
ap.add_argument("--tasks", default=None,
                 help="comma-separated task ids overriding TRAIN_TASKS (e.g. to probe held-out tasks in the training env)")
ap.add_argument("--lr", type=float, default=5e-6,
                 help="learning rate; 0 = pure rollout/measurement, no policy update")
ap.add_argument("--resume-from", default=None,
                 help="path to a checkpoint-N dir (e.g. runs/.../checkpoint-500) "
                      "to resume training from, instead of cold-starting")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)

_pool = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else TRAIN_TASKS
ds = Dataset.from_dict({
    "task_id": _pool,
    "prompt": [[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": ""}] for _ in _pool],
})

cfg = GRPOConfig(
    output_dir=args.out,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    num_generations=args.num_generations,
    generation_batch_size=args.batch_size * args.num_generations,
    max_completion_length=2048,  # 2026-10-23: raised from 1024 -- the
    # 2026-10-23 validation run (40 steps, 20 distinct tasks, 0 infra
    # failures) came back 100% degenerate-tier with zero reward variance;
    # inspecting the saved completions showed EVERY sample ending mid-episode
    # (grep -> read_file, 2-3 tool rounds) at 785-964 completion tokens --
    # right up against the old 1024 cap -- never reaching write_poc/submit.
    # That is a budget cutoff, not the model/task pool failing: one grep or
    # read_file tool response alone often runs several hundred tokens, so
    # 1024 total couldn't survive more than ~2 rounds.
    # 4096 (first attempt) OOM'd in backward on step 1 (77.66/79.27 GiB
    # used) at batch_size=2/num_generations=4 (8 concurrent completions)
    # with gradient_checkpointing already on -- activation memory scales
    # with completion length x concurrent sequences, and 8x4096-token
    # sequences was too much on top of model+optimizer+vLLM's reserved
    # 35%. Backing off to 2048 here; --batch-size should also be dropped to
    # 1 at the call site for this retest to roughly halve concurrent
    # sequences instead of only touching length.
    vllm_gpu_memory_utilization=0.35,
    chat_template_kwargs={"enable_thinking": False},
    log_completions=True,
    num_completions_to_print=3,
    gradient_accumulation_steps=1,
    learning_rate=args.lr,
    lr_scheduler_type=args.lr_sched,
    warmup_steps=3,
    beta=0.1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="steps",
    save_steps=args.save_steps,
    report_to=["tensorboard"],
    use_vllm=True,
    vllm_mode="colocate",
    max_tool_calling_iterations=32,
    temperature=0.8,
    seed=0,
    shuffle_dataset=False,  # 2026-10-23: default True + fixed seed=0 was
    # deterministically reordering TRAIN_TASKS (confirmed: arvo:20060 sampled
    # first in all 4 smoke-test attempts) ignoring the solve-rate-descending
    # priority order the list was built in -- keep insertion order instead.
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
trainer.train(resume_from_checkpoint=args.resume_from)
trainer.save_model(args.out + "/final")
tok.save_pretrained(args.out + "/final")
print("GRPO_TRAIN_V1_DONE", args.out)
