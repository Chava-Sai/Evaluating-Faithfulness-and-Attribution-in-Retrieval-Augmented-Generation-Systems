"""Contriever dense retriever using FAISS index."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR, CACHE_DIR


class ContrieverRetriever:
    """
    Encodes queries with Contriever and searches a FAISS index built
    from Wikipedia DPR passages.  On first run, builds and saves the
    index (slow); subsequent runs load from disk.
    """

    MODEL_NAME = "facebook/contriever"
    INDEX_PATH = INDEX_DIR / "contriever"

    def __init__(self, passages_file: str = None):
        import faiss
        from transformers import AutoTokenizer, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.name = "contriever"

        print(f"[Contriever] Loading model on {device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, cache_dir=str(CACHE_DIR / "huggingface")
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME, cache_dir=str(CACHE_DIR / "huggingface")
        ).to(device).eval()

        index_file = self.INDEX_PATH / "index.faiss"
        meta_file  = self.INDEX_PATH / "passages.jsonl"

        if index_file.exists() and meta_file.exists():
            print(f"[Contriever] Loading FAISS index from {index_file}")
            self.index = faiss.read_index(str(index_file))
            import json
            with open(meta_file) as f:
                self.passages = [json.loads(l) for l in f]
        else:
            if passages_file is None:
                raise FileNotFoundError(
                    "No FAISS index found. Run build_contriever_index.py first, "
                    "or provide passages_file= to build on-the-fly."
                )
            self._build_index(passages_file)

    def _encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                # Mean pooling
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embs.append(emb)
        return np.vstack(all_embs).astype("float32")

    def _build_index(self, passages_file: str):
        import faiss, json
        print(f"[Contriever] Building FAISS index from {passages_file}…")
        self.INDEX_PATH.mkdir(parents=True, exist_ok=True)

        with open(passages_file) as f:
            self.passages = [json.loads(l) for l in f]

        texts = [p["text"] for p in self.passages]
        embs  = self._encode(texts)

        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embs)
        self.index.add(embs)

        faiss.write_index(self.index, str(self.INDEX_PATH / "index.faiss"))
        with open(self.INDEX_PATH / "passages.jsonl", "w") as f:
            for p in self.passages:
                f.write(json.dumps(p) + "\n")
        print(f"[Contriever] Index built: {len(self.passages)} passages")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        import faiss
        q_emb = self._encode([query])
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            p = self.passages[idx]
            results.append({
                "id":    p.get("id", str(idx)),
                "text":  p["text"],
                "score": float(score),
            })
        return results
