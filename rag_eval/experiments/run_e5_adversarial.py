"""E5: Adversarial conflict experiment using FEVER-refuting passages."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nq_loader import load_nq
from data.fever_loader import load_fever_refutals, inject_adversarial_passages
from generators.hf_generator import HFGenerator
from retrievers.contriever_retriever import ContrieverRetriever
from pipeline import run_pipeline
from config import LLAMA_MODEL, TOP_K_DEFAULT


def main(split: str):
    examples      = load_nq(split)
    fever_refutes = load_fever_refutals(n=len(examples) * 2)
    adv_examples  = inject_adversarial_passages(examples, fever_refutes)

    retriever = ContrieverRetriever()
    generator = HFGenerator(LLAMA_MODEL)

    # Normal RAG on the same examples (control)
    print(f"\n=== E5 Control: Normal RAG ({split}) ===")
    run_pipeline(
        examples,
        retriever=retriever,
        generator=generator,
        top_k=TOP_K_DEFAULT,
        run_name=f"e5_normal_{split}",
    )

    # Adversarial: first retrieved passage replaced with FEVER refutation
    print(f"\n=== E5 Adversarial: FEVER conflict passages ({split}) ===")
    run_pipeline(
        adv_examples,
        retriever=retriever,
        generator=generator,
        top_k=TOP_K_DEFAULT,
        adversarial=True,
        run_name=f"e5_adversarial_{split}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="full", choices=["debug", "midterm", "full"])
    args = parser.parse_args()
    main(args.split)
