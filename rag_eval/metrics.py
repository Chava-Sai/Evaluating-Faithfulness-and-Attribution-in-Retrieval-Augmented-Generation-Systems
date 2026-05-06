"""
Core metrics from the proposal:
  Faithfulness   (Eq. 1)
  AttrP          (Eq. 2)
  RFG            (Eq. 3)
"""

import re
from typing import List, Tuple
import torch
from sentence_transformers import CrossEncoder
from config import NLI_MODEL, NLI_THRESHOLD, CACHE_DIR
import os

os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR / "sentence_transformers")

_nli_model: CrossEncoder = None


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _nli_model = CrossEncoder(NLI_MODEL, device=device)
    return _nli_model


def extract_claims(answer: str) -> List[str]:
    """Split answer into sentence-level claims."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def nli_support_scores(claims: List[str], passages: List[str]) -> List[List[float]]:
    """
    Returns scores[i][j] = P(entailment | passage_j, claim_i).
    Uses the label order of nli-deberta-v3-base: contradiction, entailment, neutral.
    """
    model = _get_nli_model()
    if not claims or not passages:
        return [[0.0] * len(passages) for _ in claims]

    pairs = [(p, c) for c in claims for p in passages]
    raw = model.predict(pairs, apply_softmax=True)

    # label index 1 = entailment for deberta-v3-base
    entail_idx = 1
    scores = []
    for i, _ in enumerate(claims):
        row = []
        for j, _ in enumerate(passages):
            row.append(float(raw[i * len(passages) + j][entail_idx]))
        scores.append(row)
    return scores


def faithfulness(answer: str, passages: List[str], threshold: float = NLI_THRESHOLD) -> Tuple[float, dict]:
    """Eq. 1: fraction of claims supported by at least one passage."""
    claims = extract_claims(answer)
    if not claims:
        return 0.0, {"claims": [], "supported": [], "scores": []}

    scores = nli_support_scores(claims, passages)
    supported = [any(s > threshold for s in row) for row in scores]

    score = sum(supported) / len(claims)
    detail = {"claims": claims, "supported": supported, "scores": scores}
    return score, detail


def attribution_precision(answer: str, passages: List[str], threshold: float = NLI_THRESHOLD) -> Tuple[float, dict]:
    """Eq. 2: fraction of claims where max NLI score > threshold."""
    claims = extract_claims(answer)
    if not claims:
        return 0.0, {"claims": [], "max_scores": []}

    scores = nli_support_scores(claims, passages)
    max_scores = [max(row) if row else 0.0 for row in scores]
    supported = [s > threshold for s in max_scores]

    score = sum(supported) / len(claims)
    detail = {"claims": claims, "max_scores": max_scores, "supported": supported}
    return score, detail


def retrieval_relevance(passages: List[str], gold_answer: str) -> float:
    """
    Rel(D, q): fraction of passages containing the gold answer string
    (exact-match substring check, as specified in the proposal).
    """
    if not passages or not gold_answer:
        return 0.0
    gold = gold_answer.strip().lower()
    relevant = sum(1 for p in passages if gold in p.lower())
    return relevant / len(passages)


def soft_retrieval_relevance(passages: List[str], gold_answer: str,
                              threshold: float = NLI_THRESHOLD) -> float:
    """
    Soft Rel: fraction of passages that NLI-entail the gold answer.
    Addresses brittleness of exact-match Rel (paraphrases, synonyms).
    Each (passage, gold_answer) pair is scored; passage counts as relevant
    if entailment probability > threshold.
    """
    if not passages or not gold_answer:
        return 0.0
    model = _get_nli_model()
    pairs = [(p, gold_answer) for p in passages]
    raw   = model.predict(pairs, apply_softmax=True)
    entail_idx = 1  # deberta-v3-base: [contradiction, entailment, neutral]
    relevant = sum(1 for r in raw if float(r[entail_idx]) > threshold)
    return relevant / len(passages)


def rfg(answer: str, passages: List[str], gold_answer: str,
        threshold: float = NLI_THRESHOLD) -> Tuple[float, dict]:
    """
    Eq. 3: RFG = Rel(D, q) - Faithfulness(a, D)
    Positive RFG → good docs retrieved but generator ignored them.
    Negative RFG → generator added information not in retrieved docs.
    """
    faith_score, faith_detail = faithfulness(answer, passages, threshold)
    rel_score = retrieval_relevance(passages, gold_answer)
    gap = rel_score - faith_score
    detail = {
        "retrieval_relevance": rel_score,
        "faithfulness": faith_score,
        "rfg": gap,
        "faithfulness_detail": faith_detail,
    }
    return gap, detail


def compute_all_metrics(answer: str, passages: List[str], gold_answer: str,
                        threshold: float = NLI_THRESHOLD) -> dict:
    """
    Convenience: compute Faithfulness, AttrP, exact Rel, soft NLI Rel, and RFG.
    RFG uses the soft Rel so it is less affected by exact-string brittleness.
    """
    faith_score, faith_detail = faithfulness(answer, passages, threshold)
    attr_score, attr_detail   = attribution_precision(answer, passages, threshold)
    rel_score                 = retrieval_relevance(passages, gold_answer)
    soft_rel_score            = soft_retrieval_relevance(passages, gold_answer, threshold)
    gap_exact                 = rel_score - faith_score        # classic RFG
    gap_soft                  = soft_rel_score - faith_score   # soft RFG (final report)

    return {
        "faithfulness":           faith_score,
        "attribution_precision":  attr_score,
        "retrieval_relevance":    rel_score,        # exact-match (kept for comparison)
        "soft_retrieval_relevance": soft_rel_score, # NLI-based (final report metric)
        "rfg":                    gap_exact,
        "soft_rfg":               gap_soft,
        "details": {
            "faithfulness": faith_detail,
            "attribution":  attr_detail,
        },
    }
