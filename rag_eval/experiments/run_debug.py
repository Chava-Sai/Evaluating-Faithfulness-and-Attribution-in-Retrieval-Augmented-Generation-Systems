"""
Debug run: 100 NQ examples, BM25 + LLaMA, full pipeline end-to-end.
Run this first on the cluster to verify everything works before the big jobs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nq_loader import load_nq
from retrievers.bm25_retriever import BM25Retriever
from generators.hf_generator import HFGenerator
from pipeline import run_pipeline
from config import LLAMA_MODEL


def main():
    print("=== DEBUG RUN: BM25 + LLaMA on NQ (100 examples) ===\n")

    # 1. Load data
    examples = load_nq("debug")
    print(f"Loaded {len(examples)} examples\n")

    # 2. Sanity check metrics standalone
    from metrics import compute_all_metrics
    test_answer  = "Paris is the capital of France."
    test_passage = [{"text": "Paris, the capital of France, is located in northern France."}]
    test_gold    = "Paris"
    scores = compute_all_metrics(test_answer, [p["text"] for p in test_passage], test_gold)
    print(f"[Metrics sanity check]")
    print(f"  Faithfulness:  {scores['faithfulness']:.3f}  (expected ~1.0)")
    print(f"  AttrP:         {scores['attribution_precision']:.3f}  (expected ~1.0)")
    print(f"  Rel:           {scores['retrieval_relevance']:.3f}  (expected 1.0)")
    print(f"  RFG:           {scores['rfg']:.3f}  (expected ~0.0)\n")

    # 3. No-RAG baseline
    print("=== No-RAG Baseline ===")
    generator = HFGenerator(LLAMA_MODEL)
    run_pipeline(
        examples[:20],  # tiny subset first
        retriever=None,
        generator=generator,
        no_rag=True,
        run_name="debug_no_rag",
    )

    # 4. Vanilla RAG: BM25 + LLaMA
    print("\n=== Vanilla RAG: BM25 + LLaMA ===")
    retriever = BM25Retriever()
    run_pipeline(
        examples[:20],
        retriever=retriever,
        generator=generator,
        top_k=5,
        run_name="debug_vanilla_rag",
    )

    print("\n=== Debug run complete. Check results/ in the shared folder. ===")


if __name__ == "__main__":
    main()
