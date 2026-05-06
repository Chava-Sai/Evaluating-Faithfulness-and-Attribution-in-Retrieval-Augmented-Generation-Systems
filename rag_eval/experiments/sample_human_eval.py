"""
Sample 100 QA pairs from results for human evaluation.
Creates a CSV that all 4 team members fill in with binary faithfulness judgments.
"""

import json
import csv
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR


def main(seed: int = 42, n: int = 100):
    # Load all result files
    all_results = []
    for f in RESULTS_DIR.glob("*.jsonl"):
        with open(f) as fp:
            for line in fp:
                all_results.append(json.loads(line))

    if len(all_results) < n:
        print(f"Only {len(all_results)} results available, sampling all.")
        n = len(all_results)

    random.seed(seed)
    sample = random.sample(all_results, n)

    out_file = RESULTS_DIR / "human_eval_100.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "question", "gold_answer",
            "passage_1", "passage_2", "passage_3",
            "generated_answer",
            "auto_faithfulness", "auto_attr_precision", "auto_rfg",
            # Human annotators fill these columns (0 = not faithful, 1 = faithful)
            "human_sai", "human_jihyeon", "human_samyu", "human_shrishty",
        ])
        for r in sample:
            passages = [p["text"][:200] for p in r.get("passages", [])]
            while len(passages) < 3:
                passages.append("")
            writer.writerow([
                r["id"], r["question"], r["gold_answer"],
                passages[0], passages[1], passages[2],
                r["answer"],
                round(r["metrics"]["faithfulness"], 3),
                round(r["metrics"]["attribution_precision"], 3),
                round(r["metrics"]["rfg"], 3),
                "", "", "", "",  # human columns blank
            ])

    print(f"Human eval sheet saved → {out_file}")
    print("Each team member fills in their column: 1 = answer is faithful to passages, 0 = not.")


if __name__ == "__main__":
    main()
