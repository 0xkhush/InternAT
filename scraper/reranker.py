"""Stage 2 ML Layer: Card-Context Pre-Filter & Cross-Encoder Query Re-Ranker."""

from __future__ import annotations

import math
import re
from typing import Any

from scraper.config import Settings
from scraper.schemas import DOMCandidate, JobSearchRequirement

HIGH_EXPERIENCE_RE = re.compile(
    r"\b([3-9]|1\d)\s*(?:[-–to]+\s*\d+\s*|\+\s*)(?:yrs?|years?)(?:\s+of)?(?:\s+exp|\s+experience)?\b",
    re.IGNORECASE,
)
BATCH_YEARS_RE = re.compile(r"\b(202[0-9])\b")


def passes_card_context_prefilter(candidate: DOMCandidate, req: JobSearchRequirement) -> bool:
    """Fast rule/regex pre-filter on card context before running Cross-Encoder or fetching detail pages."""
    combined = f"{candidate.text} {candidate.context}".lower()

    # 1. If user is searching for an internship or fresher role, reject cards explicitly requiring 3+ years experience
    is_fresher_or_intern = req.job_type == "internship" or any(
        b in ("2026", "2027", "2028", "2029") for b in req.target_batches
    )
    if is_fresher_or_intern and HIGH_EXPERIENCE_RE.search(combined):
        return False

    # 2. If user specified target batches (e.g., ["2028"]) and the card explicitly lists batch years,
    # reject if it only lists older batches (<= 2025) and does not mention the target batch or adjacent fresher batch.
    if req.target_batches:
        mentioned_years = set(BATCH_YEARS_RE.findall(combined))
        if mentioned_years:
            target_set = set(req.target_batches)
            # Allow if target batch is mentioned, or if card mentions recent student batches (e.g. 2027 when looking for 2028)
            allowed_years = set(target_set)
            for tb in target_set:
                if tb.isdigit():
                    allowed_years.add(str(int(tb) - 1))
                    allowed_years.add(str(int(tb) + 1))
            if mentioned_years.isdisjoint(allowed_years) and all(int(y) <= 2025 for y in mentioned_years):
                return False

    return True


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


class CrossEncoderReranker:
    """Stage 2 Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) query relevance re-ranker."""

    _shared_model: Any = None
    _shared_model_name: str | None = None

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model = self._get_model(self.settings.cross_encoder_model)

    @classmethod
    def _get_model(cls, model_name: str) -> Any:
        if cls._shared_model is None or cls._shared_model_name != model_name:
            from sentence_transformers import CrossEncoder

            cls._shared_model = CrossEncoder(model_name)
            cls._shared_model_name = model_name
        return cls._shared_model

    def rerank(
        self,
        candidates: list[DOMCandidate],
        req: JobSearchRequirement,
        top_k: int = 25,
        apply_prefilter: bool = True,
    ) -> list[DOMCandidate]:
        """Score surviving candidates against `req` and return the top-K sorted highest-first."""
        if not candidates:
            return []

        pool = [
            c for c in candidates if (not apply_prefilter or passes_card_context_prefilter(c, req))
        ]
        if not pool:
            return []

        query_str = req.to_search_summary()
        pairs = [
            (query_str, f"{c.text} — {c.context} ({c.url_path})")
            for c in pool
        ]

        raw_scores = self._model.predict(pairs, show_progress_bar=False)
        scored_candidates: list[DOMCandidate] = []
        for cand, raw_score in zip(pool, raw_scores):
            norm_score = _sigmoid(float(raw_score))
            # Boost slightly when exact role or batch keywords appear in the card text/context
            combined_lower = f"{cand.text} {cand.context}".lower()
            bonus = 0.0
            for batch in req.target_batches:
                if batch.lower() in combined_lower:
                    bonus += 0.05
            for loc in req.locations:
                if loc.lower() in combined_lower:
                    bonus += 0.03
            final_score = min(0.9999, round(norm_score + bonus, 4))
            scored_candidates.append(cand.model_copy(update={"relevance_score": final_score}))

        scored_candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return scored_candidates[:top_k]

