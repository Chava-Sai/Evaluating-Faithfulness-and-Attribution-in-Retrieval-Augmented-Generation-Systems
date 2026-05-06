#!/bin/bash -l
#$ -N rag_contriever_index
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 8
#$ -l mem_per_core=16G
#$ -j y
#$ -o /projectnb/cs505am/students/saichava/logs/contriever_index_$JOB_ID.log

# Builds the Contriever FAISS index over Wikipedia DPR passages.
# Requires GPU. Run AFTER download_data.sh completes.
# Takes ~4-6 hours for all 21M passages on a single A100.

module load python3/3.10.12
module load cuda/11.8

CODE="/projectnb/cs505am/students/saichava/rag_eval"
source "$CODE/venv/bin/activate"
cd "$CODE/data"

# For midterm: build over 2M passages (fast, ~30 min) to have something working
# For final: remove --max-passages to encode all 21M
python build_contriever_index.py --max-passages 2000000

echo "=== Contriever index build complete ==="
