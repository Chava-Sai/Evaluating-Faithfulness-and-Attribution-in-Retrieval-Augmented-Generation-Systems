"""
BM25 retriever using rank_bm25 (pure Python — no Java/Pyserini needed).
Builds an in-memory BM25 index over the Wikipedia DPR passages corpus.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR, DATA_DIR

PASSAGES_FILE = DATA_DIR / "passages" / "psgs_w100.tsv"
BM25_INDEX_PATH = INDEX_DIR / "bm25" / "bm25_index.pkl"
BM25_META_PATH  = INDEX_DIR / "bm25" / "passages_meta.pkl"


def _load_passages(max_passages: int = None):
    """Load Wikipedia DPR 100-word passages from TSV."""
    passages = []
    with open(PASSAGES_FILE) as f:
        next(f)  # skip header: id \t text \t title
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            passages.append({
                "id":    parts[0],
                "text":  parts[1],
                "title": parts[2] if len(parts) > 2 else "",
            })
            if max_passages and len(passages) >= max_passages:
                break
    return passages


class BM25Retriever:
    """
    Wraps rank_bm25 BM25Okapi over Wikipedia DPR passages.
    Index is built once and cached to disk as a pickle.
    """

    def __init__(self, max_passages: int = 1_000_000):
        from rank_bm25 import BM25Okapi

        self.name = "bm25"

        if BM25_INDEX_PATH.exists() and BM25_META_PATH.exists():
            print("[BM25] Loading cached index…")
            with open(BM25_INDEX_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(BM25_META_PATH, "rb") as f:
                self.passages = pickle.load(f)
            print(f"[BM25] Index loaded ({len(self.passages):,} passages)")
            return

        print(f"[BM25] Building index over {max_passages:,} passages…")
        self.passages = _load_passages(max_passages)
        tokenized = [p["text"].lower().split() for p in self.passages]

        print("[BM25] Fitting BM25Okapi…")
        self.bm25 = BM25Okapi(tokenized)

        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25, f, protocol=4)
        with open(BM25_META_PATH, "wb") as f:
            pickle.dump(self.passages, f, protocol=4)
        print(f"[BM25] Index saved → {BM25_INDEX_PATH}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        import numpy as np
        top_ids = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id":    self.passages[i]["id"],
                "text":  self.passages[i]["text"],
                "title": self.passages[i]["title"],
                "score": float(scores[i]),
            }
            for i in top_ids
        ]


if __name__ == "__main__":
    r = BM25Retriever(max_passages=100_000)
    results = r.retrieve("Who wrote Hamlet?", top_k=3)
    for p in results:
        print(f"{p['score']:.2f}  {p['text'][:100]}")
