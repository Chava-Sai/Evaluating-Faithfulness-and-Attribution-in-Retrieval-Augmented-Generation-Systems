"""
FEVER-style adversarial conflict loader.

Downloads FEVER dev set directly from the official fever.ai S3 source
(avoids HuggingFace datasets library version conflicts on the cluster).
Uses REFUTES claim text as adversarial passages.
"""

import json
import random
import urllib.request
from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

FEVER_CACHE = DATA_DIR / "fever"

# Official FEVER shared-task dev set (~19k examples, ~7MB)
FEVER_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"


def load_fever_refutals(n: int = 500) -> List[Dict]:
    """
    Load FEVER REFUTES claims as adversarial passages.
    Downloads directly from fever.ai — no datasets library needed.

    Each item: {claim, evidence_text, label='REFUTES'}
    The claim text is a false declarative statement used as an
    adversarial passage injected into NQ retrieval contexts.
    """
    FEVER_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file   = FEVER_CACHE / f"fever_refutes_{n}.jsonl"
    raw_file     = FEVER_CACHE / "shared_task_dev.jsonl"

    # Return cached processed file if it exists
    if cache_file.exists():
        print(f"[FEVER] Loading cached refutals ({n} examples)")
        with open(cache_file) as f:
            return [json.loads(l) for l in f]

    # Download raw FEVER file if not already present
    if not raw_file.exists():
        print(f"[FEVER] Downloading FEVER dev set from fever.ai…")
        try:
            urllib.request.urlretrieve(FEVER_URL, raw_file)
            print(f"[FEVER] Downloaded → {raw_file}")
        except Exception as e:
            print(f"[FEVER] Download failed: {e}")
            print("[FEVER] Falling back to synthetic adversarial passages…")
            return _synthetic_refutals(n)

    # Parse and filter REFUTES claims
    print("[FEVER] Parsing REFUTES claims…")
    refutes = []
    with open(raw_file) as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ex.get("label") != "REFUTES":
                continue
            claim = ex.get("claim", "").strip()
            if not claim:
                continue
            refutes.append({
                "claim":         claim,
                "evidence_text": claim,   # false declarative = adversarial passage
                "label":         "REFUTES",
            })
            if len(refutes) >= n:
                break

    if not refutes:
        print("[FEVER] No REFUTES claims found, using synthetic fallback…")
        return _synthetic_refutals(n)

    # Save processed cache
    with open(cache_file, "w") as f:
        for item in refutes:
            f.write(json.dumps(item) + "\n")
    print(f"[FEVER] Saved {len(refutes)} refuting examples → {cache_file}")
    return refutes


def _synthetic_refutals(n: int) -> List[Dict]:
    """
    Fallback: generate simple synthetic adversarial passages when
    FEVER cannot be downloaded (e.g., no internet on compute node).
    These are generic false claims that conflict with factual answers.
    """
    templates = [
        "This event never occurred in recorded history.",
        "There is no historical evidence supporting this claim.",
        "This person did not exist according to historical records.",
        "The date mentioned is factually incorrect.",
        "No such organization or institution has ever existed.",
        "This location has never been officially recognized.",
        "The information provided contradicts established facts.",
        "Historical records directly contradict this assertion.",
        "This claim has been thoroughly debunked by researchers.",
        "No credible source supports this statement.",
    ]
    rng = random.Random(42)
    refutes = []
    for i in range(n):
        claim = templates[i % len(templates)]
        refutes.append({
            "claim":         claim,
            "evidence_text": claim,
            "label":         "REFUTES",
        })
    print(f"[FEVER] Generated {n} synthetic adversarial passages (fallback)")
    return refutes


def inject_adversarial_passages(
    nq_examples: List[Dict],
    fever_refutes: List[Dict],
    seed: int = 42,
) -> List[Dict]:
    """
    For each NQ example, attach a FEVER refuting passage as an
    adversarial passage. The pipeline prepends it as the first
    retrieved document, simulating a conflicting retrieval.
    """
    rng = random.Random(seed)
    adversarial = []
    for ex in nq_examples:
        refute = rng.choice(fever_refutes)
        adversarial.append({
            **ex,
            "adversarial_passage": refute["evidence_text"],
            "fever_claim":         refute["claim"],
        })
    return adversarial


if __name__ == "__main__":
    refutes = load_fever_refutals(100)
    print(f"\nLoaded {len(refutes)} FEVER refuting examples")
    print(refutes[0])
