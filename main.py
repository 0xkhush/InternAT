"""Interactive Terminal CLI Chat Agent for ML-Powered Job & Internship Scraping."""

from __future__ import annotations

import argparse
import asyncio
import json

from scraper.cli_agent import IntakeAgent
from scraper.config import Settings
from scraper.discovery import build_16_site_search_queries
from scraper.pipeline import JobScraperPipeline


async def run_interactive_cli(
    initial_query: str | None = None,
    skip_clarification: bool = False,
    top_k: int = 20,
    reset_seen: bool = False,
) -> None:
    settings = Settings()
    if reset_seen and settings.seen_jobs_path.exists():
        settings.seen_jobs_path.unlink()
        print("[SeenJobsStore] Cleared data/seen_jobs.json for a fresh crawl.")
    agent = IntakeAgent(settings)

    print("=" * 72)
    print("  InterAT 2.0 — AI-Powered Job & Internship Agent (GLM-5.3-flash)")
    print("  Stage 1: Bi-Encoder (all-MiniLM-L6-v2 + Contrastive Metric Projection)")
    print("  Stage 2: Cross-Encoder Re-Ranker (ms-marco-MiniLM-L-6-v2)")
    print("=" * 72)

    if not initial_query:
        initial_query = input("\nYou > ").strip()
        if not initial_query:
            print("No query provided. Exiting.")
            return

    req = agent.parse_initial_query(initial_query)
    if not skip_clarification:
        question = agent.get_missing_questions(req)
        if question:
            print(f"\nAgent > {question}")
            reply = input("\nYou > ").strip()
            req = agent.apply_user_clarification(req, reply)

    queries = build_16_site_search_queries(req)
    print(f"\nAgent > Locked in search profile: {req.to_search_summary()}")
    print(f"Agent > Generated {len(queries)} targeted 16-portal search queries:")
    for idx, q in enumerate(queries, 1):
        print(f"  [{idx}] {q}")

    pipeline = JobScraperPipeline(settings)
    print(f"\nAgent > Running hybrid discovery, Bi-Encoder classification & Cross-Encoder re-ranking (top_k={top_k})...")
    result = await pipeline.run(req, max_detail_pages=top_k)

    stats = result["removed_noise_stats"]
    total_in_file = 0
    if settings.output_jobs_path.exists():
        try:
            total_in_file = len(json.loads(settings.output_jobs_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    print("\n" + "-" * 72)
    print(
        f"Noise Filter Stats: {stats['total_raw_candidates']} raw links -> "
        f"{stats['job_candidates_found']} unseen jobs kept | "
        f"{stats['pagination_links_found']} pagination/filters | "
        f"{stats['noise_dropped']} noise dropped"
    )
    print(
        f"Saved {stats['final_clean_jobs_saved']} NEW deduplicated jobs this run "
        f"(Total jobs in {settings.output_jobs_path.name}: {total_in_file})"
    )
    print("-" * 72)

    display_jobs = list(result["job_listings"][:top_k])
    if len(display_jobs) < top_k and settings.output_jobs_path.exists():
        try:
            saved_all = json.loads(settings.output_jobs_path.read_text(encoding="utf-8"))
            seen_ids = {j.get("job_id") for j in display_jobs}
            target_batches = set(req.target_batches)
            for j in reversed(saved_all):
                if len(display_jobs) >= top_k:
                    break
                if j.get("job_id") in seen_ids:
                    continue
                j_batches = {b for b in j.get("batch_eligible", []) if str(b).isdigit()}
                if target_batches and j_batches and j_batches.isdisjoint(target_batches):
                    continue
                display_jobs.append(j)
                seen_ids.add(j.get("job_id"))
        except Exception:
            pass

    for idx, job in enumerate(display_jobs[:top_k], 1):
        print(
            f"{idx}. [{job['company']}] {job['title']} — {', '.join(job['location'])} "
            f"({job['stipend_or_salary']}) | Batch: {', '.join(job['batch_eligible'])}\n"
            f"   Apply: {job['apply_url']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="InterAT 2.0 Conversational Job Scraper CLI")
    parser.add_argument("--query", type=str, default=None, help="Initial job search query")
    parser.add_argument("--yes", action="store_true", help="Skip interactive clarification prompt")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K job detail pages to display/keep (default: 20)")
    parser.add_argument("--reset-seen", action="store_true", help="Clear data/seen_jobs.json before running")
    args = parser.parse_args()
    asyncio.run(
        run_interactive_cli(
            initial_query=args.query,
            skip_clarification=args.yes,
            top_k=args.top_k,
            reset_seen=args.reset_seen,
        )
    )


if __name__ == "__main__":
    main()

