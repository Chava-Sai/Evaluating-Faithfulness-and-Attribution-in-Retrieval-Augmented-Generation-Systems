"""ASQA experiment: multi-hop long-form QA evaluation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.asqa_loader import load_asqa
from generators.hf_generator import HFGenerator
from retrievers.contriever_retriever import ContrieverRetriever
from pipeline import run_pipeline
from config import LLAMA_MODEL, TOP_K_DEFAULT


def main():
    examples = load_asqa("full")
    print(f"Loaded {len(examples)} ASQA examples\n")

    retriever = ContrieverRetriever()
    generator = HFGenerator(LLAMA_MODEL)

    print("=== ASQA: Contriever + LLaMA ===")
    run_pipeline(
        examples,
        retriever=retriever,
        generator=generator,
        top_k=TOP_K_DEFAULT,
        run_name="asqa_contriever_llama",
    )


if __name__ == "__main__":
    main()
