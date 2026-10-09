"""Run one instrumented GRPO training job from a YAML config.

    python -m rlstudy.train --config configs/baseline.yaml --out results/E1/baseline/rep0
    python -m rlstudy.train --config configs/baseline.yaml --out ... --set grpo.num_generations=16

``--set`` overrides use dotted keys and YAML values. The fully resolved config is
what lands in run.json, so an override is never lost.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "experiment": "adhoc",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype": "bfloat16",
    "seed": 0,
    "data": {"split": "train", "limit": None},
    "lora": {"enabled": False, "r": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"},
    "grpo": {
        "prompts_per_step": 8,        # B: distinct questions per optimizer step
        "num_generations": 8,         # G: samples per question
        "max_completion_length": 256,
        "temperature": 1.0,
        "learning_rate": 1.0e-6,
        "max_steps": 50,
        "beta": 0.0,                  # no reference model, no KL term (TRL default)
        "gradient_checkpointing": False,
    },
    "rollout": {
        "backend": "vllm",            # vllm | hf
        "gpu_memory_utilization": 0.3,
        "sleep_mode": False,          # must stay off for sync_every > 1 (NOTES.md)
        "sync_every": 1,
    },
    "timing": {"enabled": True},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_set(config: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    node = config
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def load_config(path: Path | None, sets: list[str]) -> dict:
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        config = deep_merge(config, yaml.safe_load(Path(path).read_text()) or {})
    for s in sets:
        apply_set(config, s)
    return config


def build_grpo_config(config: dict, out_dir: Path):
    import torch
    from trl import GRPOConfig

    g, r = config["grpo"], config["rollout"]
    completions_per_step = g["prompts_per_step"] * g["num_generations"]
    on_cuda = torch.cuda.is_available()
    bf16 = config["dtype"] == "bfloat16" and (on_cuda or torch.backends.mps.is_available())
    return GRPOConfig(
        output_dir=str(out_dir / "hf_trainer"),
        seed=config["seed"],
        max_steps=g["max_steps"],
        learning_rate=g["learning_rate"],
        # TRL counts completions, not prompts. With steps_per_generation=1 the
        # generation batch equals this batch: one generation round per step.
        per_device_train_batch_size=completions_per_step,
        gradient_accumulation_steps=1,
        steps_per_generation=1,
        num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"],
        temperature=g["temperature"],
        beta=g["beta"],
        bf16=bf16,
        gradient_checkpointing=g["gradient_checkpointing"],
        use_vllm=r["backend"] == "vllm",
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=r["gpu_memory_utilization"],
        vllm_enable_sleep_mode=r["sleep_mode"],
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        log_completions=False,
        disable_tqdm=True,
        use_cpu=not on_cuda and not torch.backends.mps.is_available(),
    )


def run(config: dict, out_dir: Path) -> Path:
    import torch

    from rlstudy.data import as_hf_dataset
    from rlstudy.instrumented import ProfiledGRPOTrainer
    from rlstudy.reward import correctness_reward, format_reward
    from rlstudy.timing import StepTimer, device_sync_fn
    from rlstudy.trace import TraceWriter

    out_dir = Path(out_dir)
    args = build_grpo_config(config, out_dir)
    trace = TraceWriter(out_dir, config, extra={"experiment": config["experiment"]})

    peft_config = None
    if config["lora"]["enabled"]:
        from peft import LoraConfig

        lc = config["lora"]
        peft_config = LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], task_type="CAUSAL_LM",
        )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    timer = StepTimer(sync=device_sync_fn(device), enabled=config["timing"]["enabled"])
    dataset = as_hf_dataset(config["data"]["split"], limit=config["data"]["limit"])

    t0 = time.perf_counter()
    trainer = ProfiledGRPOTrainer(
        model=config["model"],
        reward_funcs=[correctness_reward, format_reward],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        timer=timer,
        trace=trace,
        sync_every=config["rollout"]["sync_every"],
    )
    setup_s = time.perf_counter() - t0
    status = "failed"
    try:
        t1 = time.perf_counter()
        trainer.train()
        status = "completed"
    finally:
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        trace.finish(
            status=status,
            setup_s=setup_s,
            train_s=time.perf_counter() - t1 if status == "completed" else None,
            device=device,
            cuda_max_memory_allocated=peak,
        )
    return out_dir


def main() -> None:
    if sys.platform == "darwin":
        # transformers' threaded weight loader segfaults in this process on macOS/MPS
        # (seen with transformers 5.17.0, torch 2.13.0). Local smoke runs only.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run(load_config(args.config, args.set), args.out)


if __name__ == "__main__":
    main()
