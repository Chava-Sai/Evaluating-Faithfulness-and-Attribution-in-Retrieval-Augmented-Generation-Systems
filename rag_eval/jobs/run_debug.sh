#!/bin/bash -l
#$ -N rag_debug
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 8
#$ -j y
#$ -o /projectnb/cs505am/students/saichava/logs/debug_$JOB_ID.log

# Quick debug run: 100 NQ examples, BM25 + LLaMA, verify full pipeline works.

module load python3/3.10.12
module load cuda/11.8
module load java/17

CODE="/projectnb/cs505am/students/saichava/rag_eval"
source "$CODE/venv/bin/activate"
cd "$CODE"

python experiments/run_debug.py

echo "Debug run complete."
