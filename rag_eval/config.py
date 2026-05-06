import os
from pathlib import Path

# ── Cluster paths ────────────────────────────────────────────────────────────
CODE_DIR    = Path("/projectnb/cs505am/students/saichava/rag_eval")
SHARED_DIR  = Path("/projectnb/cs505am/projects/Faithfulness and Attribution in RAG")

DATA_DIR    = SHARED_DIR / "datasets"
INDEX_DIR   = SHARED_DIR / "indexes"
CACHE_DIR   = SHARED_DIR / "model_cache"
RESULTS_DIR = SHARED_DIR / "results"

# Create dirs if they don't exist (safe to call on every run)
for d in [DATA_DIR, INDEX_DIR, CACHE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Tell HuggingFace to cache into shared folder (saves re-downloading per user)
os.environ["HF_HOME"]            = str(CACHE_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface" / "transformers")

# ── Dataset subsets ──────────────────────────────────────────────────────────
NQ_DEBUG_SIZE     = 100
NQ_MIDTERM_SIZE   = 500
NQ_FULL_SIZE      = 3610   # full NQ test set

ASQA_DEBUG_SIZE   = 50
ASQA_FULL_SIZE    = 500

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_DEFAULT     = 5
TOP_K_ABLATION    = [1, 3, 5, 10]

# ── NLI scoring ─────────────────────────────────────────────────────────────
NLI_MODEL         = "cross-encoder/nli-deberta-v3-base"
NLI_THRESHOLD     = 0.5    # τ from the proposal

# ── Generator models ─────────────────────────────────────────────────────────
LLAMA_MODEL       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MISTRAL_MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"

# ── Prompt templates (E3 ablation) ───────────────────────────────────────────
PROMPT_PLAIN = """Answer the question based on the passages below.

Passages:
{passages}

Question: {question}
Answer:"""

PROMPT_GROUNDED = """Answer the question using ONLY information from the passages below. \
Do not use prior knowledge. Cite evidence directly.

Passages:
{passages}

Question: {question}
Answer:"""

PROMPT_COT = """You are given passages and a question. Think step by step, \
referencing the passages, then give a final answer.

Passages:
{passages}

Question: {question}
Let's think step by step:"""

PROMPT_STYLES = {
    "plain":    PROMPT_PLAIN,
    "grounded": PROMPT_GROUNDED,
    "cot":      PROMPT_COT,
}
