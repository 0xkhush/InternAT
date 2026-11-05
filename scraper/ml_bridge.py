"""JSON Bridge allowing the Go Scraper/Agent to invoke the Python Bi-Encoder + Cross-Encoder ML Pipeline."""

from __future__ import annotations

import json
import sys
from typing import Any

from scraper.ml_classifier import BiEncoderClassifier
from scraper.reranker import CrossEncoderReranker, passes_card_context_prefilter
from scraper.schemas import DOMCandidate, JobSearchRequirement


def run_ml_bridge_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Process a batch of DOMCandidates from Go through Stage 1 Bi-Encoder and Stage 2 Cross-Encoder."""
    req_data = payload.get("requirement") or {"raw_query": "software engineering internship india"}
    req = JobSearchRequirement(**req_data)
    top_k = int(payload.get("top_k", 25))
    raw_cands = payload.get("candidates") or []

    candidates = [DOMCandidate(**c) for c in raw_cands]
    if not candidates:
        return {
            "job_listings": [],
            "pagination_links": [],
            "noise_count": 0,
            "total_candidates": 0,
        }

    clf = BiEncoderClassifier()
    classified, _ = clf.classify_candidates(candidates)

    jobs_pool: list[DOMCandidate] = []
    pagination_pool: list[DOMCandidate] = []
    noise_count = 0

    for c in classified:
        if c.predicted_label == "job_listing":
            if passes_card_context_prefilter(c, req):
                jobs_pool.append(c)
            else:
                noise_count += 1
        elif c.predicted_label == "pagination_or_filter":
            pagination_pool.append(c)
        else:
            noise_count += 1

    reranker = CrossEncoderReranker()
    ranked_jobs = reranker.rerank(jobs_pool, req, top_k=top_k, apply_prefilter=False)
    ranked_pagination = reranker.rerank(pagination_pool, req, top_k=10, apply_prefilter=False)

    return {
        "job_listings": [c.model_dump() for c in ranked_jobs],
        "pagination_links": [c.model_dump() for c in ranked_pagination],
        "noise_count": noise_count,
        "total_candidates": len(candidates),
    }


def main() -> None:
    """Read JSON request from stdin and write JSON response to stdout for the Go caller."""
    raw_input = sys.stdin.read()
    if not raw_input.strip():
        print(json.dumps({"error": "empty input"}))
        return
    payload = json.loads(raw_input)
    result = run_ml_bridge_request(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

