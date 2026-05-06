"""
sample_for_human_eval.py
─────────────────────────
Samples 50 diverse examples from E1/E3/E5 results for human annotation.
Saves two files:
  human_eval_sample.json  — full data for Claude to annotate
  human_eval_sheet.csv    — clean sheet for team member to fill in

Usage:
    python sample_for_human_eval.py
"""

import json
import csv
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

OUT_DIR = RESULTS_DIR.parent / "human_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

def load_latest(prefix):
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.jsonl"))
    if not files:
        return []
    records = []
    with open(files[-1]) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                continue
    return records

# Load from three conditions for diversity
pools = {
    "e1_bm25":       load_latest("e1_bm25_midterm"),
    "e3_grounded":   load_latest("e3_grounded_midterm"),
    "e5_adversarial":load_latest("e5_adversarial_midterm"),
}

# Sample ~17 from each pool (total 51, trim to 50)
sampled = []
for condition, records in pools.items():
    n = 17 if condition != "e5_adversarial" else 16
    chosen = random.sample(records, min(n, len(records)))
    for r in chosen:
        sampled.append({
            "condition": condition,
            "id":         r.get("id", ""),
            "question":   r["question"],
            "passages":   [p["text"] for p in r.get("passages", [])[:3]],  # top-3 only
            "answer":     r["answer"],
            "nli_faith":  round(r["metrics"].get("faithfulness", 0), 3),
        })

random.shuffle(sampled)
sampled = sampled[:50]

# Add annotation ID
for i, ex in enumerate(sampled):
    ex["ann_id"] = i + 1

# Save full JSON for Claude annotation
out_json = OUT_DIR / "human_eval_sample.json"
with open(out_json, "w") as f:
    json.dump(sampled, f, indent=2)
print(f"Saved {len(sampled)} examples → {out_json}")

# Save clean CSV for team member
out_csv = OUT_DIR / "human_eval_sheet.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ann_id", "question", "answer_preview",
                     "annotator_label (1=Faithful 0=Not)", "notes"])
    for ex in sampled:
        writer.writerow([
            ex["ann_id"],
            ex["question"],
            ex["answer"][:120].replace("\n", " "),
            "",   # annotator fills this in
            "",
        ])
print(f"Saved annotation sheet → {out_csv}")
print("\nNext steps:")
print("  1. rsync human_eval_sample.json back to Mac for Claude annotation")
print("  2. Share human_eval_sheet.csv with one team member to annotate independently")
