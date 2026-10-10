"""Held-out GSM8K evaluation: score a model on the frozen 200-problem eval split.

    python -m rlstudy.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --out results/E6_heldout/base
    python -m rlstudy.evaluate --model /path/to/trained --out results/E6_heldout/trained_seed0

The eval split comes from GSM8K's *test* split, so the model never trained on it
(data/gsm8k_subset/manifest.json). Decoding is greedy (temperature 0, one answer per
problem), with the same system prompt as training and a 512-token budget, twice the
training cap, so a longer answer is not marked wrong just for being long.

Three metrics per model, all written with every individual answer to eval.json:

- strict accuracy: the number after the last ``####`` is correct (the training reward)
- format rate: the answer ends with a well-formed ``#### <number>`` line
- lenient accuracy: the last number anywhere in the answer is correct, ignoring format

Strict minus lenient separates "learned the answer format" from "got the math right".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rlstudy.data import load_split
from rlstudy.reward import has_valid_format, is_correct, is_correct_lenient
from rlstudy.trace import environment, git_info

MAX_TOKENS = 512


def _generate_vllm(model: str, prompts: list[list[dict]]) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, gpu_memory_utilization=0.5, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outputs = llm.chat(prompts, params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def _generate_hf(model: str, prompts: list[list[dict]], batch: int = 16) -> list[str]:
    """Fallback for machines without vLLM (local tests)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model)
    texts = []
    for i in range(0, len(prompts), batch):
        chunk = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return texts


def score(rows: list[dict], texts: list[str]) -> dict:
    items = []
    for row, text in zip(rows, texts, strict=True):
        items.append({
            "id": row["id"],
            "gold": row["answer"],
            "strict": is_correct(text, row["answer"]),
            "lenient": is_correct_lenient(text, row["answer"]),
            "format_ok": has_valid_format(text),
            "completion": text,
        })
    n = len(items)
    return {
        "n": n,
        "strict_accuracy": sum(i["strict"] for i in items) / n,
        "lenient_accuracy": sum(i["lenient"] for i in items) / n,
        "format_rate": sum(i["format_ok"] for i in items) / n,
        "items": items,
    }


def evaluate(model: str, out_dir: Path, backend: str = "vllm", limit: int | None = None, meta: dict | None = None) -> dict:
    rows = load_split("eval", limit=limit)
    prompts = [r["prompt"] for r in rows]
    t0 = time.perf_counter()
    texts = (_generate_vllm if backend == "vllm" else _generate_hf)(model, prompts)
    result = score(rows, texts)
    result.update({
        "model": model,
        "backend": backend,
        "decoding": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "samples_per_problem": 1},
        "eval_split": "data/gsm8k_subset eval (from GSM8K test)",
        "generate_s": time.perf_counter() - t0,
        "git": git_info(Path(__file__).resolve().parent.parent),
        "env": environment(),
        **(meta or {}),
    })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    r = evaluate(args.model, args.out, args.backend, args.limit, {"label": args.label})
    print(f"{args.label or args.model}: strict {r['strict_accuracy']:.3f}  lenient {r['lenient_accuracy']:.3f}  "
          f"format {r['format_rate']:.3f}  (n={r['n']})")


if __name__ == "__main__":
    main()
