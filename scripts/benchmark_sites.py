"""Live Portal Benchmark & Vector Space Visualization across Target Job Sites."""

from __future__ import annotations

import asyncio
from typing import Any

from scraper.config import PROJECT_ROOT, resolve_portal_for_url
from scraper.extractor import extract_dom_candidates
from scraper.fetcher import HybridFetcher
from scraper.ml_classifier import SEED_PROTOTYPES, BiEncoderClassifier, candidate_from_feature_string
from scraper.schemas import DOMCandidate
from scripts.visualize_vector_space import generate_vector_space_plots

DEFAULT_BENCHMARK_URLS = [
    "https://thejobcompany.co.in/",
    "https://careersat.tech/jobs",
    "https://internshala.com/internships/",
    "https://hasjob.co/",
]


async def run_benchmark(urls: list[str] | None = None) -> dict[str, Any]:
    """Scrape live target pages, classify candidates into 3 clusters, and generate vector space plots."""
    target_urls = urls or DEFAULT_BENCHMARK_URLS
    fetcher = HybridFetcher(timeout=15.0)
    clf = BiEncoderClassifier()

    all_classified: list[DOMCandidate] = []
    per_site_stats: list[dict[str, Any]] = []

    try:
        for url in target_urls:
            portal = resolve_portal_for_url(url)
            portal_id = portal.id if portal else "external"
            rmode = portal.render if portal else "auto"
            try:
                html = await fetcher.fetch(url, render_mode=rmode)
            except Exception:
                html = ""

            if not html:
                continue

            cands = extract_dom_candidates(html, url, portal_id)
            if not cands:
                continue

            classified, _ = clf.classify_candidates(cands)
            all_classified.extend(classified)

            jobs_cnt = sum(1 for c in classified if c.predicted_label == "job_listing")
            page_cnt = sum(1 for c in classified if c.predicted_label == "pagination_or_filter")
            noise_cnt = sum(1 for c in classified if c.predicted_label == "noise")
            per_site_stats.append(
                {
                    "portal": portal_id,
                    "url": url,
                    "raw_candidates": len(classified),
                    "job_listing": jobs_cnt,
                    "pagination_or_filter": page_cnt,
                    "noise_dropped": noise_cnt,
                }
            )
    finally:
        await fetcher.close()

    # Include seed prototypes if live network returned few candidates so the plot is always rich
    if len(all_classified) < 15:
        for lbl, exs in SEED_PROTOTYPES.items():
            for ex in exs:
                all_classified.append(candidate_from_feature_string(ex, label=lbl))

    classified_all, proj_embs = clf.classify_candidates(all_classified)
    html_out, png_out = generate_vector_space_plots(
        classified_all,
        proj_embs,
        clf.get_centroids(),
        PROJECT_ROOT / "plots" / "vector_space.html",
        PROJECT_ROOT / "plots" / "vector_space.png",
    )

    print("\n" + "=" * 78)
    print(f"{'PORTAL':<16} | {'RAW LINKS':<10} | {'JOBS KEPT':<10} | {'PAGINATION':<11} | {'NOISE DROPPED':<13}")
    print("-" * 78)
    for row in per_site_stats:
        print(
            f"{row['portal']:<16} | {row['raw_candidates']:<10} | {row['job_listing']:<10} | "
            f"{row['pagination_or_filter']:<11} | {row['noise_dropped']:<13}"
        )
    print("=" * 78)
    print(f"Vector space plots saved to:\n  - {html_out}\n  - {png_out}")

    return {
        "per_site_stats": per_site_stats,
        "total_visualized": len(classified_all),
        "html_plot": str(html_out),
        "png_plot": str(png_out),
    }


if __name__ == "__main__":
    asyncio.run(run_benchmark())

