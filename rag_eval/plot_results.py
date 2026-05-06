"""
plot_results.py
───────────────
Generates all figures for the final report from saved JSONL result files.
Saves PNGs to  <RESULTS_DIR>/../figures/

Usage (on SCC after all experiments complete):
    python plot_results.py

Figures produced:
    fig1_e3_prompt.png      — E3 bar chart: prompt style vs Faithfulness
    fig2_e4_topk.png        — E4 line plot: k vs Faithfulness and |RFG|
    fig3_e5_adversarial.png — E5 bar chart: normal vs adversarial
    fig4_tau_sensitivity.png— τ sensitivity across key conditions
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # no display needed on cluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

FIGURES_DIR = RESULTS_DIR.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
C_BLUE   = "#4878CF"
C_GREEN  = "#6ACC65"
C_ORANGE = "#D65F5F"
C_PURPLE = "#B47CC7"
C_GREY   = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── helpers ─────────────────────────────────────────────────────────────────

def load_run(prefix: str):
    """Load all JSONL records whose filename starts with prefix."""
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.jsonl"))
    records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def avg(records, key):
    vals = [r["metrics"].get(key, 0.0) for r in records if "metrics" in r]
    return sum(vals) / len(vals) if vals else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — E3 Prompt Ablation
# ════════════════════════════════════════════════════════════════════════════

def fig1_e3():
    styles = {
        "Plain":    load_run("e3_plain_midterm"),
        "Grounded": load_run("e3_grounded_midterm"),
        "CoT":      load_run("e3_cot_midterm"),
    }

    labels   = list(styles.keys())
    faith    = [avg(styles[s], "faithfulness")           for s in labels]
    soft_rel = [avg(styles[s], "soft_retrieval_relevance") for s in labels]
    rfg_abs  = [abs(avg(styles[s], "soft_rfg"))           for s in labels]

    x  = np.arange(len(labels))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))

    bars_f = ax.bar(x - w, faith,    w, label="Faithfulness", color=C_BLUE,   alpha=0.88)
    bars_r = ax.bar(x,     soft_rel, w, label="Rel$_\\mathrm{soft}$",   color=C_GREEN,  alpha=0.88)
    bars_g = ax.bar(x + w, rfg_abs,  w, label="|RFG$_\\mathrm{soft}$|", color=C_ORANGE, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("E3: Effect of Prompt Style on Faithfulness")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 0.18)
    ax.axhline(0, color="black", linewidth=0.6)

    # annotate bars
    for bar in list(bars_f) + list(bars_r) + list(bars_g):
        h = bar.get_height()
        if h > 0.002:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    out = FIGURES_DIR / "fig1_e3_prompt.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — E4 Top-k Ablation
# ════════════════════════════════════════════════════════════════════════════

def fig2_e4():
    ks = [1, 3, 5, 10]
    faith   = []
    rfg_abs = []
    soft_rel = []

    for k in ks:
        recs = load_run(f"e4_topk{k}_midterm")
        faith.append(avg(recs,   "faithfulness"))
        rfg_abs.append(abs(avg(recs, "soft_rfg")))
        soft_rel.append(avg(recs, "soft_retrieval_relevance"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, faith,    "o-",  color=C_BLUE,   linewidth=2, markersize=7,
            label="Faithfulness")
    ax.plot(ks, rfg_abs,  "s--", color=C_ORANGE, linewidth=2, markersize=7,
            label="|RFG$_\\mathrm{soft}$|")
    ax.plot(ks, soft_rel, "^:",  color=C_GREEN,  linewidth=2, markersize=7,
            label="Rel$_\\mathrm{soft}$")

    ax.set_xlabel("Number of retrieved passages ($k$)")
    ax.set_ylabel("Score")
    ax.set_title("E4: Effect of Top-$k$ on Faithfulness and RFG")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 0.20)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_e4_topk.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — E5 Adversarial
# ════════════════════════════════════════════════════════════════════════════

def fig3_e5():
    normal = load_run("e5_normal_midterm")
    adv    = load_run("e5_adversarial_midterm")

    conditions = ["Normal RAG", "Adversarial\n(FEVER)"]
    faith   = [avg(normal, "faithfulness"),             avg(adv, "faithfulness")]
    rel     = [avg(normal, "soft_retrieval_relevance"), avg(adv, "soft_retrieval_relevance")]
    rfg_abs = [abs(avg(normal, "soft_rfg")),             abs(avg(adv, "soft_rfg"))]

    x = np.arange(len(conditions))
    w = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.bar(x - w, faith,    w, label="Faithfulness", color=C_BLUE,   alpha=0.88)
    ax.bar(x,     rel,      w, label="Rel$_\\mathrm{soft}$",   color=C_GREEN,  alpha=0.88)
    ax.bar(x + w, rfg_abs,  w, label="|RFG$_\\mathrm{soft}$|", color=C_ORANGE, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Score")
    ax.set_title("E5: Normal vs.\ Adversarial Retrieval")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 0.28)

    # highlight the adversarial faithfulness jump
    ax.annotate("Faithfulness\ndoubles",
                xy=(1 - w, avg(adv, "faithfulness")),
                xytext=(0.55, 0.23),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=8.5, color="black")

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_e5_adversarial.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — τ Sensitivity (AUC-style, no re-running needed)
# Recomputes faithfulness at different τ from saved NLI scores in details.
# ════════════════════════════════════════════════════════════════════════════

def recompute_faith_at_tau(records, tau):
    """Recompute faithfulness from saved per-claim NLI scores at a new tau."""
    totals, supported = 0, 0
    for r in records:
        details = r.get("details", {}).get("faithfulness", {})
        scores  = details.get("scores", [])   # scores[i][j] = P(entail | d_j, c_i)
        if not scores:
            continue
        for row in scores:
            totals += 1
            if any(s > tau for s in row):
                supported += 1
    return supported / totals if totals > 0 else 0.0


def fig4_tau_sensitivity():
    taus = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    conditions = {
        "BM25 + LLaMA":       load_run("e1_bm25_midterm"),
        "Contriever + LLaMA": load_run("e1_contriever_midterm"),
        "Grounded Prompt":    load_run("e3_grounded_midterm"),
        "Adversarial":        load_run("e5_adversarial_midterm"),
    }
    colors = [C_BLUE, C_GREEN, C_PURPLE, C_ORANGE]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    for (label, recs), color in zip(conditions.items(), colors):
        if not recs:
            print(f"  [WARNING] No records for {label}, skipping")
            continue
        faiths = [recompute_faith_at_tau(recs, t) for t in taus]
        ax.plot(taus, faiths, "o-", color=color, linewidth=2,
                markersize=6, label=label)

    ax.set_xlabel("Entailment threshold $\\tau$")
    ax.set_ylabel("Faithfulness")
    ax.set_title("Faithfulness vs.\ NLI Threshold $\\tau$ (Key Conditions)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlim(0.15, 0.95)
    ax.set_ylim(0, 0.45)
    ax.axvline(0.5, color="grey", linewidth=1, linestyle="--", alpha=0.6,
               label="$\\tau{=}0.5$ (used)")

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_tau_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Saving figures to {FIGURES_DIR}\n")
    print("Generating Figure 1 — E3 Prompt Ablation…")
    fig1_e3()
    print("Generating Figure 2 — E4 Top-k Ablation…")
    fig2_e4()
    print("Generating Figure 3 — E5 Adversarial…")
    fig3_e5()
    print("Generating Figure 4 — τ Sensitivity…")
    fig4_tau_sensitivity()
    print("\nAll figures saved.")
