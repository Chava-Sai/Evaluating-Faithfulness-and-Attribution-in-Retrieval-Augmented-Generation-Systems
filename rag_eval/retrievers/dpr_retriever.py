"""DPR retriever using Pyserini's prebuilt dense index."""

from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR, CACHE_DIR


class DPRRetriever:
    """
    Uses Pyserini's FaissSearcher with the prebuilt DPR Wikipedia index.
    Falls back to local index if already downloaded.
    """

    INDEX_NAME = "wikipedia-dpr-100w.dpr-single-nq"  # Pyserini prebuilt

    def __init__(self):
        from pyserini.search.faiss import FaissSearcher
        from pyserini.search.faiss import DprQueryEncoder

        index_path = INDEX_DIR / "dpr"
        self.name = "dpr"

        encoder = DprQueryEncoder(
            "facebook/dpr-question_encoder-single-nq-base",
            cache_dir=str(CACHE_DIR / "huggingface"),
        )
        if index_path.exists():
            print(f"[DPR] Loading local index from {index_path}")
            self.searcher = FaissSearcher(str(index_path), encoder)
        else:
            print(f"[DPR] Downloading prebuilt index '{self.INDEX_NAME}'…")
            self.searcher = FaissSearcher.from_prebuilt_index(self.INDEX_NAME, encoder)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        hits = self.searcher.search(query, k=top_k)
        passages = []
        for hit in hits:
            doc = hit.lucene_document
            text = doc.get("contents") or doc.get("raw") or ""
            passages.append({
                "id":    hit.docid,
                "text":  text.strip(),
                "score": hit.score,
            })
        return passages
