"""ASQA loader — multi-hop long-form QA dataset."""

import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, ASQA_DEBUG_SIZE, ASQA_FULL_SIZE

ASQA_CACHE = DATA_DIR / "asqa"


def load_asqa(split: str = "debug") -> List[Dict]:
    """
    split: "debug" → ASQA_DEBUG_SIZE, "full" → ASQA_FULL_SIZE
    Each item: {id, question, gold_answers (list), annotations}
    """
    size_map = {"debug": ASQA_DEBUG_SIZE, "full": ASQA_FULL_SIZE}
    if split not in size_map:
        raise ValueError(f"split must be one of {list(size_map)}")
    n = size_map[split]

    cache_file = ASQA_CACHE / f"asqa_{split}.jsonl"
    if cache_file.exists():
        print(f"[ASQA] Loading cached {split} split from {cache_file}")
        with open(cache_file) as f:
            return [json.loads(l) for l in f]

    print("[ASQA] Downloading ASQA…")
    ASQA_CACHE.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "din0s/asqa",
        split="dev",
        cache_dir=str(ASQA_CACHE / "hf_cache"),
    )

    examples = []
    for ex in dataset:
        # Collect all gold short answers
        gold_answers = list({
            ann["short_answers"][0]
            for ann in ex.get("annotations", [])
            if ann.get("short_answers")
        })
        examples.append({
            "id":           ex.get("sample_id", str(len(examples))),
            "question":     ex["ambiguous_question"],
            "gold_answers": gold_answers,
        })
        if len(examples) >= ASQA_FULL_SIZE:
            break

    for name, k in size_map.items():
        out = ASQA_CACHE / f"asqa_{name}.jsonl"
        with open(out, "w") as f:
            for item in examples[:k]:
                f.write(json.dumps(item) + "\n")
        print(f"[ASQA] Saved {name} split ({k} examples) → {out}")

    return examples[:n]


if __name__ == "__main__":
    data = load_asqa("debug")
    print(f"Loaded {len(data)} ASQA examples")
    for ex in data[:2]:
        print(ex)
