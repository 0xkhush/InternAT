"""End-to-End Orchestration Pipeline connecting Stages 1 -> 4."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from scraper.config import Settings, resolve_portal_for_url
from scraper.dedup import SeenJobsStore
from scraper.detail_extractor import RawJobDetail, extract_job_detail_from_html
from scraper.discovery import discover_candidate_urls
from scraper.extractor import extract_dom_candidates
from scraper.fetcher import HybridFetcher
from scraper.llm_extractor import StructuredJobExtractor
from scraper.ml_classifier import BiEncoderClassifier
from scraper.reranker import CrossEncoderReranker
from scraper.schemas import DOMCandidate, JobRecord, JobSearchRequirement

logger = logging.getLogger(__name__)


class JobScraperPipeline:
    """Orchestrates 16-portal discovery, Two-Stage local ML filtering, and 4-layer detail extraction."""

    def __init__(self, settings: Settings | None = None, use_llm: bool = True) -> None:
        self.settings = settings or Settings()
        self.seen_store = SeenJobsStore(self.settings.seen_jobs_path)
        self.fetcher = HybridFetcher(timeout=18.0)
        self.bi_encoder = BiEncoderClassifier(self.settings)
        self.reranker = CrossEncoderReranker(self.settings)
        self.job_extractor = StructuredJobExtractor(self.settings, use_llm=use_llm)

    async def run(
        self,
        req: JobSearchRequirement,
        max_listing_pages: int = 20,
        max_detail_pages: int = 20,
        enable_ddg: bool = True,
    ) -> dict[str, Any]:
        """Run the full scraping pipeline for a user's JobSearchRequirement."""
        discovered = discover_candidate_urls(req, self.seen_store, enable_ddg=enable_ddg)
        listing_targets = discovered[:max_listing_pages]

        all_job_candidates: list[DOMCandidate] = []
        all_pagination_candidates: list[DOMCandidate] = []
        total_raw_candidates = 0
        total_noise_dropped = 0

        try:
            # Stage 2: Fetch listing pages & classify DOM candidates with Stage 1 Bi-Encoder
            sem = asyncio.Semaphore(4)

            async def _fetch_and_extract(url: str, portal_id: str) -> list[DOMCandidate]:
                async with sem:
                    try:
                        portal = resolve_portal_for_url(url)
                        rmode = portal.render if portal else "auto"
                        html = await self.fetcher.fetch(url, render_mode=rmode)
                        if not html:
                            return []
                        return extract_dom_candidates(html, url, portal_id)
                    except Exception as exc:
                        logger.debug("Listing fetch failed for %s: %s", url, exc)
                        return []

            pages_cands = await asyncio.gather(
                *[_fetch_and_extract(u, pid) for u, pid in listing_targets]
            )
            flat_cands = [c for page_list in pages_cands for c in page_list]
            total_raw_candidates = len(flat_cands)

            if flat_cands:
                classified, _ = self.bi_encoder.classify_candidates(flat_cands)
                for c in classified:
                    if c.predicted_label == "job_listing":
                        if not self.seen_store.is_url_seen(c.url):
                            all_job_candidates.append(c)
                    elif c.predicted_label == "pagination_or_filter":
                        all_pagination_candidates.append(c)
                    else:
                        total_noise_dropped += 1

            # Stage 2b: Cross-Encoder Re-Ranking
            top_pagination = self.reranker.rerank(
                all_pagination_candidates, req, top_k=10
            )

            # If fewer than max_detail_pages unseen job candidates were found on Page 1,
            # follow the top 4 pagination/filter links (1-hop expansion) to discover deeper jobs!
            if len(all_job_candidates) < max_detail_pages and top_pagination:
                hop_urls = [(p.url, p.source_portal) for p in top_pagination[:4]]
                hop_pages = await asyncio.gather(*[_fetch_and_extract(u, pid) for u, pid in hop_urls])
                hop_flat = [c for page_list in hop_pages for c in page_list]
                total_raw_candidates += len(hop_flat)
                if hop_flat:
                    hop_classified, _ = self.bi_encoder.classify_candidates(hop_flat)
                    seen_cand_urls = {c.url for c in all_job_candidates}
                    for c in hop_classified:
                        if c.predicted_label == "job_listing":
                            if not self.seen_store.is_url_seen(c.url) and c.url not in seen_cand_urls:
                                all_job_candidates.append(c)
                                seen_cand_urls.add(c.url)
                        elif c.predicted_label == "noise":
                            total_noise_dropped += 1

            # Over-fetch detail candidates (2x max_detail_pages) so strict post-detail batch filtering
            # still yields up to max_detail_pages (20) matching jobs!
            candidate_fetch_limit = max(max_detail_pages * 2, 35)
            top_job_candidates = self.reranker.rerank(
                all_job_candidates, req, top_k=candidate_fetch_limit
            )

            # Stage 3: Fetch surviving Job Detail pages & run 4-Layer Data Cleaning
            async def _fetch_detail(cand: DOMCandidate) -> RawJobDetail | None:
                if self.seen_store.is_url_seen(cand.url):
                    return None
                async with sem:
                    try:
                        portal = resolve_portal_for_url(cand.url)
                        rmode = portal.render if portal else "auto"
                        html = await self.fetcher.fetch(cand.url, render_mode=rmode)
                        if not html:
                            return None
                        return extract_job_detail_from_html(
                            html,
                            source_url=cand.url,
                            source_portal=cand.source_portal,
                            relevance_score=cand.relevance_score,
                        )
                    except Exception as exc:
                        logger.debug("Detail fetch failed for %s: %s", cand.url, exc)
                        return None

            raw_details = await asyncio.gather(*[_fetch_detail(c) for c in top_job_candidates])
            valid_details = [d for d in raw_details if d is not None]

            # Stage 4: Batched GLM-5.3-flash structured extraction + strict batch filter + deduplication
            extracted_records = self.job_extractor.extract_batch(valid_details, req)
            final_records: list[JobRecord] = []
            target_batch_set = set(req.target_batches)
            for rec in extracted_records:
                if len(final_records) >= max_detail_pages:
                    break
                # If the user asked for specific batches (e.g. 2028) and the job explicitly lists batch years,
                # drop jobs that only allow other batches (e.g. 2026/2027 only)
                rec_years = {b for b in rec.batch_eligible if b.isdigit()}
                if target_batch_set and rec_years and rec_years.isdisjoint(target_batch_set):
                    continue
                if self.seen_store.is_job_seen(rec.company, rec.title, rec.source_portal, rec.apply_url):
                    continue
                self.seen_store.mark_seen(rec)
                final_records.append(rec)

            self.seen_store.save()
            self._save_jobs_json(final_records)

            return {
                "requirement": req.model_dump(),
                "job_listings": [r.model_dump() for r in final_records],
                "navigation_links": [p.model_dump() for p in top_pagination],
                "removed_noise_stats": {
                    "total_raw_candidates": total_raw_candidates,
                    "job_candidates_found": len(all_job_candidates),
                    "pagination_links_found": len(all_pagination_candidates),
                    "noise_dropped": total_noise_dropped,
                    "final_clean_jobs_saved": len(final_records),
                },
            }
        finally:
            await self.fetcher.close()

    def _save_jobs_json(self, new_records: list[JobRecord]) -> None:
        out_path = self.settings.output_jobs_path
        existing: list[dict[str, Any]] = []
        if out_path.exists():
            try:
                loaded = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
            except Exception:
                existing = []

        existing_ids = {item.get("job_id") for item in existing if isinstance(item, dict)}
        for rec in new_records:
            if rec.job_id not in existing_ids:
                existing.append(rec.model_dump())
                existing_ids.add(rec.job_id)

        out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

