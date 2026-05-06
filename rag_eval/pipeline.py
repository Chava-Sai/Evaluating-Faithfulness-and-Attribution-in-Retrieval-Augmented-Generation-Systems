"""
Main RAG evaluation pipeline.
Runs: retrieval → generation → claim extraction → NLI scoring → metrics → save.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, TOP_K_DEFAULT, PROMPT_PLAIN
from metrics import compute_all_metrics


def format_passages(passages: List[Dict]) -> str:
    return "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))


def run_pipeline(
    examples: List[Dict],
    retriever,
    generator,
    top_k: int = TOP_K_DEFAULT,
    prompt_template: str = PROMPT_PLAIN,
    run_name: str = "run",
    no_rag: bool = False,
    adversarial: bool = False,
) -> List[Dict]:
    """
    Core loop: for each example, retrieve → generate → score → collect.

    adversarial=True: prepend the adversarial_passage as the first retrieved doc,
    replacing one normal passage (simulates FEVER conflict experiment).
    """
    results = []
    total = len(examples)
    running_faith = 0.0
    t_start = time.time()

    for i, ex in enumerate(examples):
        print(f"[{i+1}/{total}] {ex['question'][:60]}…", flush=True)

        # ── Retrieval ────────────────────────────────────────────────
        if no_rag:
            passages = []
        else:
            passages = retriever.retrieve(ex["question"], top_k=top_k)

        if adversarial and "adversarial_passage" in ex:
            adv = {"id": "adversarial", "text": ex["adversarial_passage"], "score": 999}
            passages = [adv] + passages[: max(0, top_k - 1)]

        # ── Generation ──────────────────────────────────────────────
        if no_rag:
            answer = generator.generate_no_rag(ex["question"])
        else:
            answer = generator.generate(ex["question"], passages, prompt_template)

        # ── Metrics ─────────────────────────────────────────────────
        gold = ex.get("gold_answer") or (ex.get("gold_answers") or [""])[0]
        passage_texts = [p["text"] for p in passages]
        metric_scores = compute_all_metrics(answer, passage_texts, gold)

        # ── Live progress ────────────────────────────────────────────
        running_faith += metric_scores["faithfulness"]
        elapsed = time.time() - t_start
        avg_sec = elapsed / (i + 1)
        eta_min = avg_sec * (total - i - 1) / 60
        print(
            f"    ✓ Faith={metric_scores['faithfulness']:.3f} | "
            f"SoftRel={metric_scores['soft_retrieval_relevance']:.3f} | "
            f"Avg={running_faith/(i+1):.3f} | "
            f"~{eta_min:.1f} min left | "
            f"ans: {answer[:60].strip()}",
            flush=True,
        )

        record = {
            "id":            ex.get("id", str(i)),
            "question":      ex["question"],
            "gold_answer":   gold,
            "passages":      [{"id": p["id"], "text": p["text"][:300]} for p in passages],
            "answer":        answer,
            "metrics":       {k: v for k, v in metric_scores.items() if k != "details"},
            "details":       metric_scores["details"],
            "retriever":     getattr(retriever, "name", "no_rag") if not no_rag else "no_rag",
            "generator":     generator.name,
            "top_k":         top_k,
            "prompt_style":  "no_rag" if no_rag else "adversarial" if adversarial else run_name,
        }
        results.append(record)

    # ── Save ─────────────────────────────────────────────────────────
    out_file = RESULTS_DIR / f"{run_name}_{int(time.time())}.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[Pipeline] Saved {len(results)} results → {out_file}")

    # Print summary
    avg = lambda key: sum(r["metrics"][key] for r in results) / len(results)
    print(f"  Faithfulness:               {avg('faithfulness'):.3f}")
    print(f"  Attribution Precision:      {avg('attribution_precision'):.3f}")
    print(f"  Retrieval Relevance (exact):{avg('retrieval_relevance'):.3f}")
    print(f"  Retrieval Relevance (soft): {avg('soft_retrieval_relevance'):.3f}")
    print(f"  RFG (exact):                {avg('rfg'):.3f}")
    print(f"  RFG (soft):                 {avg('soft_rfg'):.3f}")

    return results
