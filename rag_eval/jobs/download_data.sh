#!/bin/bash -l
#$ -N rag_download
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/cs505am/students/saichava/logs/download_$JOB_ID.log

# Downloads NQ, ASQA, FEVER (fast) and BM25 + DPR Wikipedia indexes (slow, ~55 GB).
# No GPU needed. Run this FIRST before any experiment jobs.

module load python3/3.10.12
module load java/17

CODE="/projectnb/cs505am/students/saichava/rag_eval"
source "$CODE/venv/bin/activate"
cd "$CODE/data"

python download_all.py

echo "=== Dataset download complete ==="
