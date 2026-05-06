"""
Natural Questions loader using the cleaner `nq_open` split on HuggingFace.
nq_open has just questions + short answers — no HTML, no huge downloads.
"""

import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, NQ_DEBUG_SIZE, NQ_MIDTERM_SIZE, NQ_FULL_SIZE

NQ_CACHE = DATA_DIR / "nq"


def load_nq(split: str = "debug") -> List[Dict]:
    """
    split: "debug"   → NQ_DEBUG_SIZE  (100)
           "midterm" → NQ_MIDTERM_SIZE (500)
           "full"    → NQ_FULL_SIZE   (3610)
    """
    size_map = {"debug": NQ_DEBUG_SIZE, "midterm": NQ_MIDTERM_SIZE, "full": NQ_FULL_SIZE}
    if split not in size_map:
        raise ValueError(f"split must be one of {list(size_map)}, got '{split}'")
    n = size_map[split]

    cache_file = NQ_CACHE / f"nq_{split}.jsonl"
    if cache_file.exists():
        print(f"[NQ] Loading cached {split} split ({n} examples)")
        with open(cache_file) as f:
            return [json.loads(l) for l in f]

    # nq_open is tiny (~40MB) — just questions + short answers, no HTML
    print("[NQ] Downloading nq_open from HuggingFace…")
    NQ_CACHE.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "google-research-datasets/nq_open",
        split="validation",
        cache_dir=str(NQ_CACHE / "hf_cache"),
    )

    examples = []
    for i, ex in enumerate(dataset):
        answers = ex.get("answer", [])
        gold = answers[0] if answers else ""
        if not gold:
            continue
        examples.append({
            "id":          str(i),
            "question":    ex["question"],
            "gold_answer": gold,
        })
        if len(examples) >= NQ_FULL_SIZE:
            break

    NQ_CACHE.mkdir(parents=True, exist_ok=True)
    for name, k in size_map.items():
        out = NQ_CACHE / f"nq_{name}.jsonl"
        with open(out, "w") as f:
            for item in examples[:k]:
                f.write(json.dumps(item) + "\n")
        print(f"[NQ] Saved {name} ({min(k, len(examples))} examples) → {out}")

    return examples[:n]


if __name__ == "__main__":
    data = load_nq("debug")
    print(f"\nLoaded {len(data)} NQ examples")
    for ex in data[:3]:
        print(f"  Q: {ex['question']}")
        print(f"  A: {ex['gold_answer']}\n")
