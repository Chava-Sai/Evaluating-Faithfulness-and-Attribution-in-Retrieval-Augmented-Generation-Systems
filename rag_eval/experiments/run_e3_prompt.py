"""E3: Prompt style comparison — plain vs grounded vs chain-of-thought."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nq_loader import load_nq
from generators.hf_generator import HFGenerator
from retrievers.contriever_retriever import ContrieverRetriever
from pipeline import run_pipeline
from config import LLAMA_MODEL, PROMPT_STYLES, TOP_K_DEFAULT


def main(split: str):
    examples  = load_nq(split)
    retriever = ContrieverRetriever()
    generator = HFGenerator(LLAMA_MODEL)

    for style_name, template in PROMPT_STYLES.items():
        print(f"\n=== E3: Prompt style = {style_name} ({split}) ===")
        run_pipeline(
            examples,
            retriever=retriever,
            generator=generator,
            top_k=TOP_K_DEFAULT,
            prompt_template=template,
            run_name=f"e3_{style_name}_{split}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="full", choices=["debug", "midterm", "full"])
    args = parser.parse_args()
    main(args.split)
