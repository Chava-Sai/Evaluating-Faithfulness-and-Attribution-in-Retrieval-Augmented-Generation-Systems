"""
compute_kappa.py
────────────────
Load two annotator CSV files and compute:
  - Raw agreement (%)
  - Cohen's κ (inter-annotator agreement)
  - Per-condition breakdown
  - Comparison of human labels vs. NLI-based faithfulness score

Usage:
    python compute_kappa.py \
        --ann1 ../human_eval_annotator1.csv \
        --ann2 ../human_eval_annotator2.csv \
        --sample ../human_eval_sample.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Cohen's κ ─────────────────────────────────────────────────────────────────

def cohen_kappa(labels1, labels2):
    """Compute Cohen's kappa for two lists of binary labels."""
    assert len(labels1) == len(labels2), "Label lists must have equal length."
    n = len(labels1)
    if n == 0:
        return 0.0

    agree = sum(a == b for a, b in zip(labels1, labels2))
    po = agree / n

    # Marginals
    p1_pos = sum(1 for x in labels1 if x == 1) / n
    p2_pos = sum(1 for x in labels2 if x == 1) / n
    p1_neg = 1 - p1_pos
    p2_neg = 1 - p2_pos

    pe = p1_pos * p2_pos + p1_neg * p2_neg

    if pe == 1.0:
        return 1.0  # degenerate case: both raters use only one label
    return (po - pe) / (1 - pe)


def interpret_kappa(k):
    if k < 0:       return "Poor (< 0)"
    elif k < 0.20:  return "Slight"
    elif k < 0.40:  return "Fair"
    elif k < 0.60:  return "Moderate"
    elif k < 0.80:  return "Substantial"
    else:            return "Almost Perfect"


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_annotations(csv_path):
    """Return dict  ann_id (int) → label (int)."""
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The label column header may vary slightly; search for the right one
            label_col = None
            for col in row:
                if "label" in col.lower() or "annotator" in col.lower():
                    label_col = col
                    break
            if label_col is None:
                raise ValueError(f"Cannot find label column in {csv_path}")
            label_str = row[label_col].strip()
            if label_str == "":
                continue
            try:
                labels[int(row["ann_id"])] = int(label_str)
            except (ValueError, KeyError):
                continue
    return labels


# ── NLI faithfulness comparison ───────────────────────────────────────────────

def load_sample_nli(json_path):
    """Return dict ann_id → nli_faith score (float)."""
    with open(json_path) as f:
        data = json.load(f)
    return {ex["ann_id"]: ex.get("nli_faith", 0.0) for ex in data}


def nli_to_binary(nli_scores, threshold=0.5):
    """Binarise NLI faithfulness at threshold."""
    return {ann_id: 1 if score >= threshold else 0
            for ann_id, score in nli_scores.items()}


def condition_lookup(json_path):
    """Return dict ann_id → condition string."""
    with open(json_path) as f:
        data = json.load(f)
    return {ex["ann_id"]: ex.get("condition", "unknown") for ex in data}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann1",   default="../human_eval_annotator1.csv",
                        help="CSV with Annotator-1 labels")
    parser.add_argument("--ann2",   default="../human_eval_annotator2.csv",
                        help="CSV with Annotator-2 labels")
    parser.add_argument("--sample", default="../human_eval_sample.json",
                        help="JSON with NLI faithfulness scores")
    parser.add_argument("--tau",    type=float, default=0.5,
                        help="Threshold to binarise NLI score (default 0.5)")
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    ann1 = load_annotations(args.ann1)
    ann2 = load_annotations(args.ann2)
    nli  = load_sample_nli(args.sample)
    nli_bin = nli_to_binary(nli, args.tau)
    conds = condition_lookup(args.sample)

    # Align to common ann_ids
    ids = sorted(set(ann1) & set(ann2))
    if not ids:
        print("ERROR: no common ann_ids found between the two annotation files.")
        sys.exit(1)

    l1 = [ann1[i] for i in ids]
    l2 = [ann2[i] for i in ids]
    ln = [nli_bin.get(i, 0) for i in ids]

    n = len(ids)

    # ── Overall stats ────────────────────────────────────────────────────────
    kappa_12 = cohen_kappa(l1, l2)
    agree_12 = sum(a == b for a, b in zip(l1, l2)) / n * 100

    kappa_1n  = cohen_kappa(l1, ln)
    kappa_2n  = cohen_kappa(l2, ln)

    # Gold = majority vote (both annotators agree) or annotator-1 when they disagree
    gold = [a if a == b else a for a, b in zip(l1, l2)]
    kappa_gn  = cohen_kappa(gold, [nli_bin.get(i, 0) for i in ids])

    print("=" * 60)
    print("  Human Evaluation — Inter-Annotator Agreement")
    print("=" * 60)
    print(f"  N examples              : {n}")
    print(f"  Annotator-1 faithful    : {sum(l1)}/{n}  ({sum(l1)/n*100:.1f}%)")
    print(f"  Annotator-2 faithful    : {sum(l2)}/{n}  ({sum(l2)/n*100:.1f}%)")
    print(f"  NLI faithful (τ={args.tau}): {sum(ln)}/{n}  ({sum(ln)/n*100:.1f}%)")
    print()
    print(f"  Raw agreement (A1 vs A2): {agree_12:.1f}%")
    print(f"  Cohen's κ  (A1 vs A2)  : {kappa_12:.3f}  [{interpret_kappa(kappa_12)}]")
    print()
    print(f"  κ  (A1 vs NLI)         : {kappa_1n:.3f}  [{interpret_kappa(kappa_1n)}]")
    print(f"  κ  (A2 vs NLI)         : {kappa_2n:.3f}  [{interpret_kappa(kappa_2n)}]")
    print(f"  κ  (Gold vs NLI)       : {kappa_gn:.3f}  [{interpret_kappa(kappa_gn)}]")

    # ── Per-condition breakdown ───────────────────────────────────────────────
    print()
    print("  Per-condition breakdown")
    print("  " + "-" * 50)
    cond_data = defaultdict(list)
    for i, g, n_lbl in zip(ids, gold, [nli_bin.get(j, 0) for j in ids]):
        cond_data[conds.get(i, "unknown")].append((g, n_lbl))

    for cond, pairs in sorted(cond_data.items()):
        c_gold = [p[0] for p in pairs]
        c_nli  = [p[1] for p in pairs]
        nc     = len(pairs)
        k      = cohen_kappa(c_gold, c_nli)
        faith_rate = sum(c_gold) / nc * 100
        print(f"  {cond:25s}  N={nc}  Faithful={faith_rate:5.1f}%  κ(gold,NLI)={k:.3f}")

    # ── Disagreement analysis ─────────────────────────────────────────────────
    disagree_ids = [ids[i] for i in range(n) if l1[i] != l2[i]]
    print()
    print(f"  Disagreements (A1 ≠ A2): {len(disagree_ids)} / {n}")
    if disagree_ids:
        print(f"  Disagreement ann_ids    : {disagree_ids}")

    print()
    print("  LaTeX snippet for paper:")
    print("  ─────────────────────────────────────────────────")
    print(f"  \\kappa & {kappa_12:.2f} (human IAA) &"
          f" {kappa_gn:.2f} (NLI vs gold) \\\\")
    print()


if __name__ == "__main__":
    main()
