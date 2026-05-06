"""
Build the Contriever FAISS index over Wikipedia DPR passages.

This requires GPU and takes ~4-6 hours on a single A100 for all 21M passages.
Run via: qsub jobs/build_contriever_index.sh

The script:
  1. Downloads the Wikipedia DPR passage corpus (~5GB jsonl)
  2. Encodes all passages with Contriever in batches
  3. Builds and saves a FAISS IndexFlatIP
"""

import json
import numpy as np
import torch
import faiss
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR, CACHE_DIR

CONTRIEVER_INDEX_PATH = INDEX_DIR / "contriever"
PASSAGES_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
PASSAGES_FILE = INDEX_DIR / "passages" / "psgs_w100.tsv"
BATCH_SIZE = 512


def download_passages():
    """Download the standard DPR Wikipedia 100-word passages corpus."""
    PASSAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PASSAGES_FILE.exists():
        print(f"[Passages] Already downloaded: {PASSAGES_FILE}")
        return

    import urllib.request, gzip, shutil
    gz_file = PASSAGES_FILE.with_suffix(".tsv.gz")
    print(f"[Passages] Downloading Wikipedia DPR passages (~5 GB)…")
    urllib.request.urlretrieve(PASSAGES_URL, gz_file)
    print("[Passages] Decompressing…")
    with gzip.open(gz_file, "rb") as f_in, open(PASSAGES_FILE, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_file.unlink()
    print(f"[Passages] Saved → {PASSAGES_FILE}")


def load_passages(max_passages: int = None):
    """Load passages from TSV: id \t text \t title"""
    passages = []
    with open(PASSAGES_FILE) as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            passages.append({"id": parts[0], "text": parts[1], "title": parts[2] if len(parts) > 2 else ""})
            if max_passages and i + 1 >= max_passages:
                break
    return passages


def encode_passages(passages, model, tokenizer, device):
    texts = [p["text"] for p in passages]
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embs.append(emb)
    return np.vstack(all_embs).astype("float32")


def build_index(max_passages: int = None):
    from transformers import AutoTokenizer, AutoModel

    CONTRIEVER_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index_file = CONTRIEVER_INDEX_PATH / "index.faiss"
    meta_file  = CONTRIEVER_INDEX_PATH / "passages.jsonl"

    if index_file.exists():
        print(f"[Contriever] Index already exists at {index_file}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Contriever] Building index on {device}")

    # Load model
    model_name = "facebook/contriever"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE_DIR / "huggingface"))
    model      = AutoModel.from_pretrained(model_name, cache_dir=str(CACHE_DIR / "huggingface")).to(device).eval()

    # Load passages
    print("[Contriever] Loading passages…")
    passages = load_passages(max_passages)
    print(f"[Contriever] Loaded {len(passages):,} passages")

    # Encode in chunks to avoid OOM; save intermediate embeddings
    chunk_size = 500_000
    all_embs = []
    for start in range(0, len(passages), chunk_size):
        chunk = passages[start : start + chunk_size]
        print(f"[Contriever] Encoding passages {start:,}–{start+len(chunk):,}…")
        embs = encode_passages(chunk, model, tokenizer, device)
        all_embs.append(embs)

    all_embs = np.vstack(all_embs)
    faiss.normalize_L2(all_embs)

    # Build FAISS index
    print("[Contriever] Building FAISS IndexFlatIP…")
    index = faiss.IndexFlatIP(all_embs.shape[1])
    index.add(all_embs)
    faiss.write_index(index, str(index_file))

    # Save passage metadata
    with open(meta_file, "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")

    print(f"[Contriever] Done! Index: {index_file}  Passages: {meta_file}")
    print(f"             {len(passages):,} passages, dim={all_embs.shape[1]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-passages", type=int, default=None,
                        help="Limit passages (e.g. 1000000 for 1M subset). Default: all 21M")
    args = parser.parse_args()

    download_passages()
    build_index(max_passages=args.max_passages)
