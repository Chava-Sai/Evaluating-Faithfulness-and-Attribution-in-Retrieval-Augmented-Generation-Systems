"""E2: Generator comparison — LLaMA vs Mistral, fixed Contriever."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nq_loader import load_nq
from generators.hf_generator import HFGenerator
from retrievers.contriever_retriever import ContrieverRetriever
from pipeline import run_pipeline
from config import LLAMA_MODEL, MISTRAL_MODEL, TOP_K_DEFAULT


def main(split: str):
    examples  = load_nq(split)
    retriever = ContrieverRetriever()

    generators = [
        ("llama",   HFGenerator(LLAMA_MODEL)),
        ("mistral", HFGenerator(MISTRAL_MODEL)),
    ]

    for name, generator in generators:
        print(f"\n=== E2: Contriever + {name.upper()} ({split}) ===")
        run_pipeline(
            examples,
            retriever=retriever,
            generator=generator,
            top_k=TOP_K_DEFAULT,
            run_name=f"e2_{name}_{split}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="midterm", choices=["debug", "midterm", "full"])
    args = parser.parse_args()
    main(args.split)
