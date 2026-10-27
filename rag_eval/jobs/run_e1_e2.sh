#!/bin/bash -l
#$ -N rag_e1_e2
#$ -l h_rt=12:00:00
#$ -l gpus=2
#$ -l gpu_c=7.0
#$ -pe omp 16
#$ -j y
#$ -o logs/

# Experiments E1 (retriever comparison) and E2 (generator comparison)
# on NQ midterm subset (500 examples).

if command -v module >/dev/null 2>&1; then
    module load python3/3.10.12
    module load cuda/11.8
    module load java/17
fi

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE/venv/bin/activate"
cd "$CODE"

echo "=== Running E1: Retriever Comparison ==="
python experiments/run_e1_retriever.py --split midterm

echo "=== Running E2: Generator Comparison ==="
python experiments/run_e2_generator.py --split midterm

echo "E1+E2 complete."
