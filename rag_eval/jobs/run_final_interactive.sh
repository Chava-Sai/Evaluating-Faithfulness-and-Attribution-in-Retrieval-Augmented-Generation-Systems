#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_final_interactive.sh
# Run this in the VS Code Server terminal on the SCC GPU node.
# Runs E3, E4, E5, ASQA, and RAGAS comparison in sequence.
# Estimated time on H200: ~7–9 hours total (500-example "midterm" split).
#
# Usage:
#   cd /projectnb/cs505am/students/saichava/rag_eval
#   bash jobs/run_final_interactive.sh 2>&1 | tee logs/final_run_$(date +%s).log
# ─────────────────────────────────────────────────────────────────────────────

set -e  # stop on first error

CODE="/projectnb/cs505am/students/saichava/rag_eval"
cd "$CODE"

# ── Activate environment ─────────────────────────────────────────────────────
module load miniconda/23.11.0 academic-ml/spring-2024 2>/dev/null || true
conda activate spring-2024-pyt 2>/dev/null || source activate spring-2024-pyt 2>/dev/null || true

# ── Sanity check GPU ─────────────────────────────────────────────────────────
echo "========================================================"
echo " GPU check"
echo "========================================================"
python -c "
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f'  GPU : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  sm_{major}{minor}  → 4-bit OK: {major >= 8}')
else:
    print('  WARNING: No CUDA GPU detected!')
"

SPLIT="midterm"   # 500 examples — safe for 12-hour session
# Change to "full" (3610 examples) only if you have 24+ hours

echo ""
echo "========================================================"
echo " Starting final experiments  (split=$SPLIT)"
echo " Start time: $(date)"
echo "========================================================"

# ── E1: Retriever comparison (BM25 vs Contriever) ───────────────────────────
echo ""
echo "=== [1/6] E1: Retriever Comparison (BM25 vs Contriever) ==="
python experiments/run_e1_retriever.py --split "$SPLIT"

# ── E2: Generator comparison (LLaMA vs Mistral) ─────────────────────────────
echo ""
echo "=== [2/6] E2: Generator Comparison (LLaMA vs Mistral) ==="
python experiments/run_e2_generator.py --split "$SPLIT"

# ── E3: Prompt ablation (plain / grounded / CoT) ────────────────────────────
echo ""
echo "=== [3/6] E3: Prompt Style Ablation ==="
python experiments/run_e3_prompt.py --split "$SPLIT"

# ── E4: Top-k ablation (k = 1, 3, 5, 10) ───────────────────────────────────
echo ""
echo "=== [4/6] E4: Top-k Ablation ==="
python experiments/run_e4_topk.py --split "$SPLIT"

# ── E5: Adversarial retrieval (FEVER conflict passages) ─────────────────────
echo ""
echo "=== [5/6] E5: Adversarial Conflict ==="
python experiments/run_e5_adversarial.py --split "$SPLIT"

# ── ASQA: Multi-answer faithfulness ─────────────────────────────────────────
echo ""
echo "=== [6/6] ASQA: Multi-Answer Evaluation ==="
python experiments/run_asqa.py

# ── RAGAS comparison (runs on existing E1 output, fast) ─────────────────────
echo ""
echo "=== [+] RAGAS comparison ==="
python experiments/run_ragas_comparison.py || echo "RAGAS skipped (not installed or no E1 results yet)"

# ── Final summary table ──────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " All experiments complete.  $(date)"
echo " Generating results summary..."
echo "========================================================"
python analyze_results.py --latex

echo ""
echo "Done. Results saved to:"
python -c "from config import RESULTS_DIR; print(f'  {RESULTS_DIR}')"
