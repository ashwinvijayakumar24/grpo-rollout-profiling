"""GSM8K subset: a fixed, seeded slice of the dataset, frozen to JSONL.

The subset is built once (``python -m rlstudy.data build``) and committed under
``data/gsm8k_subset/`` together with a manifest recording the dataset revision,
seed, and the original row indices. Every experiment then reads the same frozen
files, so runs never depend on network access or on the Hub's current revision.
That matters on PACE, where compute nodes have no internet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from rlstudy.reward import gold_answer

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "gsm8k_subset"

SYSTEM_PROMPT = (
    "Solve the math problem. Think step by step, then give the final answer on the "
    "last line in the form '#### <number>', with no units."
)


def make_prompt(question: str) -> list[dict]:
    """Conversational prompt in TRL's format: a list of chat messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def to_record(row: dict, split: str, index: int) -> dict:
    return {
        "id": f"{split}-{index}",
        "prompt": make_prompt(row["question"]),
        # Gold number as a plain string; the reward parses it with normalize_number.
        "answer": str(gold_answer(row["answer"])),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out_dir: Path, n_train: int, n_eval: int, seed: int) -> dict:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET_ID).sha
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, revision=revision)

    rng = random.Random(seed)
    train_idx = sorted(rng.sample(range(len(ds["train"])), n_train))
    # Held-out problems come from GSM8K's test split, so they never overlap training.
    eval_idx = sorted(rng.sample(range(len(ds["test"])), n_eval))

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, src, idx in (("train", "train", train_idx), ("eval", "test", eval_idx)):
        path = out_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for i in idx:
                f.write(json.dumps(to_record(ds[src][i], src, i)) + "\n")
        files[split] = {"path": path.name, "rows": len(idx), "sha256": _sha256(path)}

    manifest = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": revision,
        "seed": seed,
        "source_splits": {"train": "train", "eval": "test"},
        "source_sizes": {"train": len(ds["train"]), "test": len(ds["test"])},
        "train_indices": train_idx,
        "eval_indices": eval_idx,
        "system_prompt": SYSTEM_PROMPT,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(split: str, data_dir: Path = DEFAULT_DIR, limit: int | None = None) -> list[dict]:
    """Read a frozen split, verifying it still matches the manifest checksum."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    path = data_dir / manifest["files"][split]["path"]
    if _sha256(path) != manifest["files"][split]["sha256"]:
        raise RuntimeError(f"{path} does not match its manifest checksum; rebuild the subset")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows[:limit] if limit is not None else rows


def as_hf_dataset(split: str, data_dir: Path = DEFAULT_DIR, limit: int | None = None):
    from datasets import Dataset

    return Dataset.from_list(load_split(split, data_dir, limit))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download GSM8K and freeze the subset")
    b.add_argument("--out", type=Path, default=DEFAULT_DIR)
    b.add_argument("--n-train", type=int, default=1000)
    b.add_argument("--n-eval", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.cmd == "build":
        m = build(args.out, args.n_train, args.n_eval, args.seed)
        print(f"wrote {args.out} at revision {m['revision']}: "
              f"{m['files']['train']['rows']} train, {m['files']['eval']['rows']} eval")


if __name__ == "__main__":
    main()
