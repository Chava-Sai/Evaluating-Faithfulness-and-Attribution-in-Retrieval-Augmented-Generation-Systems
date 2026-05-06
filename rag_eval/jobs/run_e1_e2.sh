#!/bin/bash -l
#$ -N rag_e1_e2
#$ -l h_rt=12:00:00
#$ -l gpus=2
#$ -l gpu_c=7.0
#$ -pe omp 16
#$ -j y
#$ -o /projectnb/cs505am/students/saichava/logs/e1_e2_$JOB_ID.log

# Experiments E1 (retriever comparison) and E2 (generator comparison)
# on NQ midterm subset (500 examples).

module load python3/3.10.12
module load cuda/11.8
module load java/17

CODE="/projectnb/cs505am/students/saichava/rag_eval"
source "$CODE/venv/bin/activate"
cd "$CODE"

echo "=== Running E1: Retriever Comparison ==="
python experiments/run_e1_retriever.py --split midterm

echo "=== Running E2: Generator Comparison ==="
python experiments/run_e2_generator.py --split midterm

echo "E1+E2 complete."
