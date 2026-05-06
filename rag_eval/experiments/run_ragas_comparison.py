"""Compare our metrics vs RAGAS Faithfulness on the same outputs."""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR


def main():
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness as ragas_faithfulness

    # Load one result file to compare (pick a full run)
    result_files = sorted(RESULTS_DIR.glob("e1_bm25_*.jsonl"))
    if not result_files:
        print("No E1 BM25 results found. Run run_e1_retriever.py first.")
        return

    results = []
    with open(result_files[-1]) as f:
        for line in f:
            results.append(json.loads(line))

    # RAGAS expects: question, answer, contexts, ground_truth
    ragas_data = {
        "question":     [r["question"]  for r in results],
        "answer":       [r["answer"]    for r in results],
        "contexts":     [[p["text"] for p in r["passages"]] for r in results],
        "ground_truth": [r["gold_answer"] for r in results],
    }
    dataset = Dataset.from_dict(ragas_data)

    print("Running RAGAS faithfulness evaluation…")
    ragas_scores = evaluate(dataset, metrics=[ragas_faithfulness])
    ragas_faith  = ragas_scores["faithfulness"]

    our_faith = [r["metrics"]["faithfulness"] for r in results]
    avg_ours  = sum(our_faith) / len(our_faith)

    print(f"\n{'Metric':<30} {'Score':>8}")
    print("-" * 40)
    print(f"{'Our Faithfulness':<30} {avg_ours:>8.3f}")
    print(f"{'RAGAS Faithfulness':<30} {ragas_faith:>8.3f}")
    print(f"{'Difference':<30} {avg_ours - ragas_faith:>8.3f}")

    # Save comparison
    out = RESULTS_DIR / "ragas_comparison.json"
    with open(out, "w") as f:
        json.dump({
            "our_faithfulness_avg": avg_ours,
            "ragas_faithfulness":   ragas_faith,
        }, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
