"""E1: Retriever comparison — BM25 vs DPR vs Contriever, fixed LLaMA-3.1-8B."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nq_loader import load_nq
from generators.hf_generator import HFGenerator
from retrievers.bm25_retriever import BM25Retriever
from retrievers.contriever_retriever import ContrieverRetriever
from pipeline import run_pipeline
from config import LLAMA_MODEL, TOP_K_DEFAULT


def main(split: str):
    examples  = load_nq(split)
    generator = HFGenerator(LLAMA_MODEL)

    # DPR added for final report once FAISS index is built
    retrievers = [
        ("bm25",       BM25Retriever()),
        ("contriever", ContrieverRetriever()),
    ]

    for name, retriever in retrievers:
        print(f"\n=== E1: {name.upper()} + LLaMA ({split}, top-{TOP_K_DEFAULT}) ===")
        run_pipeline(
            examples,
            retriever=retriever,
            generator=generator,
            top_k=TOP_K_DEFAULT,
            run_name=f"e1_{name}_{split}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="midterm", choices=["debug", "midterm", "full"])
    args = parser.parse_args()
    main(args.split)
