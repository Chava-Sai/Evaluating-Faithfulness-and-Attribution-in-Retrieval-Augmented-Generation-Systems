"""
Download all datasets and retrieval corpus needed for the project.
Run ONCE on the cluster before any experiments.

What this downloads (to the shared project folder):
  1. NQ open         ~40 MB    → datasets/nq/
  2. ASQA            ~10 MB    → datasets/asqa/
  3. FEVER           ~60 MB    → datasets/fever/
  4. Wikipedia DPR passages  ~5 GB compressed / ~22 GB uncompressed
                              → datasets/passages/psgs_w100.tsv

BM25 index is built on-the-fly by bm25_retriever.py (first run only).
Contriever FAISS index is built by build_contriever_index.py (needs GPU).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, INDEX_DIR, NQ_FULL_SIZE, ASQA_FULL_SIZE

PASSAGES_DIR = DATA_DIR / "passages"
PASSAGES_TSV = PASSAGES_DIR / "psgs_w100.tsv"
PASSAGES_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"


# ── 1. NQ ────────────────────────────────────────────────────────────────────
def download_nq():
    from nq_loader import load_nq
    print("\n" + "="*50)
    print("1/4  Natural Questions (nq_open)")
    print("="*50)
    data = load_nq("full")
    print(f"     Done — {len(data)} examples.\n")


# ── 2. ASQA ──────────────────────────────────────────────────────────────────
def download_asqa():
    from asqa_loader import load_asqa
    print("\n" + "="*50)
    print("2/4  ASQA")
    print("="*50)
    data = load_asqa("full")
    print(f"     Done — {len(data)} examples.\n")


# ── 3. FEVER ─────────────────────────────────────────────────────────────────
def download_fever():
    from fever_loader import load_fever_refutals
    print("\n" + "="*50)
    print("3/4  FEVER (adversarial refutals)")
    print("="*50)
    data = load_fever_refutals(n=5000)
    print(f"     Done — {len(data)} examples.\n")


# ── 4. Wikipedia DPR passages ────────────────────────────────────────────────
def download_passages():
    import gzip, shutil, urllib.request
    print("\n" + "="*50)
    print("4/4  Wikipedia DPR passages (psgs_w100)")
    print("     Compressed: ~5 GB  |  Uncompressed: ~22 GB")
    print("     Used by BM25, Contriever, and DPR retrievers")
    print("="*50)

    if PASSAGES_TSV.exists():
        size_gb = PASSAGES_TSV.stat().st_size / 1e9
        print(f"     Already downloaded ({size_gb:.1f} GB). Skipping.\n")
        return

    PASSAGES_DIR.mkdir(parents=True, exist_ok=True)
    gz_file = PASSAGES_DIR / "psgs_w100.tsv.gz"

    if not gz_file.exists():
        print(f"     Downloading from {PASSAGES_URL} …")
        print("     This will take 5–20 minutes on the cluster network.")

        def _progress(count, block_size, total_size):
            pct = count * block_size * 100 / total_size
            if count % 500 == 0:
                print(f"     {pct:.1f}%", flush=True)

        urllib.request.urlretrieve(PASSAGES_URL, gz_file, reporthook=_progress)
        print(f"     Download complete → {gz_file}")
    else:
        print(f"     .gz file already present, decompressing…")

    print("     Decompressing (~22 GB)…")
    with gzip.open(gz_file, "rb") as f_in, open(PASSAGES_TSV, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024*1024*64)  # 64 MB chunks
    gz_file.unlink()  # remove compressed file to save space

    size_gb = PASSAGES_TSV.stat().st_size / 1e9
    print(f"     Done — {size_gb:.1f} GB → {PASSAGES_TSV}\n")

    # Count passages
    with open(PASSAGES_TSV) as f:
        n = sum(1 for _ in f) - 1  # subtract header
    print(f"     Total passages: {n:,}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAG Project — Download All Datasets & Corpus")
    print(f"Data dir:  {DATA_DIR}")
    print(f"Index dir: {INDEX_DIR}")
    print("="*50)

    download_nq()
    download_asqa()
    download_fever()
    download_passages()

    print("\n" + "="*50)
    print("All downloads complete!")
    print("Next steps:")
    print("  1. Get GPU session")
    print("  2. Run: python data/build_contriever_index.py --max-passages 1000000")
    print("  3. Run: python experiments/run_debug.py")
    print("="*50)
