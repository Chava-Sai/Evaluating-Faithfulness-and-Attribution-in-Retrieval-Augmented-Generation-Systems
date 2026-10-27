#!/bin/bash
# Run once on the cluster to set up the Python environment.
# Usage: bash setup_env.sh

set -e

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_DIR="${RAG_WORK_DIR:-$CODE_DIR/../workspace}"

echo "=== Creating shared directories ==="
mkdir -p "$SHARED_DIR/datasets/nq"
mkdir -p "$SHARED_DIR/datasets/asqa"
mkdir -p "$SHARED_DIR/datasets/fever"
mkdir -p "$SHARED_DIR/indexes/bm25"
mkdir -p "$SHARED_DIR/indexes/dpr"
mkdir -p "$SHARED_DIR/indexes/contriever"
mkdir -p "$SHARED_DIR/model_cache/huggingface"
mkdir -p "$SHARED_DIR/results"

echo "=== Loading modules ==="
if command -v module >/dev/null 2>&1; then
    module load python3/3.10.12
    module load cuda/11.8
    module load java/17
fi   # Pyserini (BM25) requires Java 11+

echo "=== Creating virtual environment ==="
cd "$CODE_DIR"
python3 -m venv venv
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install "numpy<2"   # faiss-gpu requires numpy 1.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets sentence-transformers
pip install faiss-gpu
pip install rank_bm25
pip install bitsandbytes accelerate
pip install openai
pip install rouge-score ragas
pip install pandas tqdm

echo "=== Setup complete ==="
echo "To activate: source $CODE_DIR/venv/bin/activate"
