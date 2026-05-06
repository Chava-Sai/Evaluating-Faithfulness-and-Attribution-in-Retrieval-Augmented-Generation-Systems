"""
analyze_results.py
──────────────────
Reads all JSONL result files and prints a formatted summary table.
Also generates a LaTeX table snippet ready to paste into the final report.

Usage:
    python analyze_results.py                # all results
    python analyze_results.py --filter e3    # only e3_* files
    python analyze_results.py --latex        # also print LaTeX table
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

METRICS = [
    ("faithfulness",             "Faith"),
    ("attribution_precision",    "AttrP"),
    ("retrieval_relevance",      "Rel(exact)"),
    ("soft_retrieval_relevance", "Rel(soft)"),
    ("rfg",                      "RFG(exact)"),
    ("soft_rfg",                 "RFG(soft)"),
]


def load_results(filter_prefix: str = "") -> dict:
    """
    Returns {run_name: {metric: avg_value}}.
    Each JSONL filename is like  e3_plain_full_1714000000.jsonl
    We strip the trailing timestamp to get a clean run name.
    """
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if filter_prefix:
        files = [f for f in files if f.stem.startswith(filter_prefix)]
    if not files:
        print(f"No result files found in {RESULTS_DIR}")
        return {}

    # Group by run_name (strip timestamp suffix)
    groups: dict[str, list] = defaultdict(list)
    for f in files:
        parts = f.stem.rsplit("_", 1)
        run_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else f.stem
        with open(f) as fh:
            for line in fh:
                try:
                    groups[run_name].append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    summary = {}
    for run_name, records in groups.items():
        if not records:
            continue
        avgs = {}
        for key, _ in METRICS:
            vals = [r["metrics"].get(key, 0.0) for r in records if "metrics" in r]
            # Handle missing soft metrics from older runs (before the soft Rel update)
            avgs[key] = sum(vals) / len(vals) if vals else float("nan")
        avgs["n"] = len(records)
        summary[run_name] = avgs
    return summary


def print_table(summary: dict):
    header_cols = ["Run", "N"] + [label for _, label in METRICS]
    col_widths  = [max(len(h), 30) for h in header_cols]
    col_widths[0] = 35
    col_widths[1] = 5

    def fmt_val(v):
        if isinstance(v, float):
            return f"{v:+.3f}" if "rfg" in str(v) else f"{v:.3f}"
        return str(v)

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)
    header = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header_cols)) + " |"
    print(header)
    print(sep)

    for run_name in sorted(summary):
        d = summary[run_name]
        row_vals = [run_name, str(d["n"])]
        for key, _ in METRICS:
            v = d[key]
            if isinstance(v, float):
                if "rfg" in key:
                    row_vals.append(f"{v:+.3f}")
                else:
                    row_vals.append(f"{v:.3f}")
            else:
                row_vals.append(str(v))
        row = "| " + " | ".join(str(row_vals[i]).ljust(col_widths[i]) for i in range(len(header_cols))) + " |"
        print(row)

    print(sep)


def print_latex(summary: dict):
    """Print a LaTeX tabular snippet for the final report."""
    print("\n% ─── LaTeX table snippet ───────────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering\small\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{@{} l c c c c c c @{}}")
    print(r"\toprule")
    print(r"\textbf{Run} & \textbf{N} & \textbf{Faith} & \textbf{AttrP} "
          r"& \textbf{Rel\textsubscript{ex}} & \textbf{Rel\textsubscript{soft}} "
          r"& \textbf{RFG\textsubscript{soft}} \\")
    print(r"\midrule")

    for run_name in sorted(summary):
        d = summary[run_name]
        faith = d.get("faithfulness", float("nan"))
        attrp = d.get("attribution_precision", float("nan"))
        rel_e = d.get("retrieval_relevance", float("nan"))
        rel_s = d.get("soft_retrieval_relevance", float("nan"))
        rfg_s = d.get("soft_rfg", float("nan"))
        n     = d["n"]
        # Escape underscores for LaTeX
        label = run_name.replace("_", r"\_")
        print(f"{label} & {n} & {faith:.3f} & {attrp:.3f} & "
              f"{rel_e:.3f} & {rel_s:.3f} & ${rfg_s:+.3f}$ \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Final results. $|\text{RFG}_\text{soft}|\downarrow$ is better.}")
    print(r"\label{tab:final_results}")
    print(r"\end{table}")
    print("% ────────────────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter",  default="",    help="Only show runs whose name starts with this")
    parser.add_argument("--latex",   action="store_true", help="Also print LaTeX table snippet")
    args = parser.parse_args()

    summary = load_results(args.filter)
    if not summary:
        return

    print(f"\n{'='*80}")
    print(f"  Results from: {RESULTS_DIR}")
    print(f"  Runs found:   {len(summary)}")
    print(f"{'='*80}\n")
    print_table(summary)

    if args.latex:
        print_latex(summary)


if __name__ == "__main__":
    main()
