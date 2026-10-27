#!/bin/bash -l
#$ -N rag_debug
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 8
#$ -j y
#$ -o logs/

# Quick debug run: 100 NQ examples, BM25 + LLaMA, verify full pipeline works.

if command -v module >/dev/null 2>&1; then
    module load python3/3.10.12
    module load cuda/11.8
    module load java/17
fi

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE/venv/bin/activate"
cd "$CODE"

python experiments/run_debug.py

echo "Debug run complete."
