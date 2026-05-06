"""E4: Top-k ablation — k in {1, 3, 5, 10}."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nq_loader import load_nq
from generators.hf_generator import HFGenerator
from retrievers.contriever_retriever import ContrieverRetriever
from pipeline import run_pipeline
from config import LLAMA_MODEL, TOP_K_ABLATION


def main(split: str):
    examples  = load_nq(split)
    retriever = ContrieverRetriever()
    generator = HFGenerator(LLAMA_MODEL)

    for k in TOP_K_ABLATION:
        print(f"\n=== E4: top_k={k} ({split}) ===")
        run_pipeline(
            examples,
            retriever=retriever,
            generator=generator,
            top_k=k,
            run_name=f"e4_topk{k}_{split}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="full", choices=["debug", "midterm", "full"])
    args = parser.parse_args()
    main(args.split)
