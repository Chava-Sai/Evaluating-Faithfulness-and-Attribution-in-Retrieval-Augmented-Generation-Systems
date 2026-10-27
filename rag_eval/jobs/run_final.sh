#!/bin/bash -l
#$ -N rag_final
#$ -l h_rt=24:00:00
#$ -l gpus=2
#$ -l gpu_c=7.0
#$ -pe omp 16
#$ -j y
#$ -o logs/

# Full experiment suite for the final report.
# Runs E3, E4, E5, ASQA, adversarial, and human eval prep.

if command -v module >/dev/null 2>&1; then
    module load python3/3.10.12
    module load cuda/11.8
    module load java/17
fi

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE/venv/bin/activate"
cd "$CODE"

echo "=== E3: Prompt Style ==="
python experiments/run_e3_prompt.py --split full

echo "=== E4: Top-k Ablation ==="
python experiments/run_e4_topk.py --split full

echo "=== E5: Adversarial Conflict ==="
python experiments/run_e5_adversarial.py --split full

echo "=== ASQA ==="
python experiments/run_asqa.py

echo "=== RAGAS Comparison ==="
python experiments/run_ragas_comparison.py

echo "=== Human Eval Prep (sample 100 QA pairs) ==="
python experiments/sample_human_eval.py

echo "All final experiments complete."
