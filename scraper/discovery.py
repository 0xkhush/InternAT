"""Stage 1: 16-Portal Targeted DuckDuckGo Search + Direct Portal URL Discovery."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus

from scraper.config import TARGET_PORTALS, resolve_portal_for_url
from scraper.dedup import SeenJobsStore, normalize_url
from scraper.schemas import JobSearchRequirement

logger = logging.getLogger(__name__)


def build_16_site_search_queries(req: JobSearchRequirement) -> list[str]:
    """Generate 16 clean per-portal DuckDuckGo search queries (no quotes on batch/location so DDG's native site: filter triggers)."""
    role_term = req.role_keywords[0] if req.role_keywords else "software engineering"
    type_term = "internship" if req.job_type == "internship" else "jobs"
    batch_term = req.target_batches[0] if req.target_batches else "fresher"
    loc_terms = [c for c in req.locations if c.lower() != "india"][:2]
    loc_str = f"({' OR '.join(loc_terms)})" if loc_terms else "(India)"

    queries: list[str] = []
    for portal in TARGET_PORTALS:
        q = f"site:{portal.domain} ({role_term} OR SDE) {type_term} {batch_term} {loc_str}".strip()
        queries.append(q)

    return queries


def build_direct_portal_urls(req: JobSearchRequirement) -> list[tuple[str, str]]:
    """Build direct entry and parameterized search URLs across all 16 target portals."""
    role_slug = (req.role_keywords[0] if req.role_keywords else "software-engineering").lower().replace(" ", "-")
    kw_encoded = quote_plus(
        f"{req.role_keywords[0] if req.role_keywords else 'Software Engineer'} "
        f"{'Intern' if req.job_type == 'internship' else ''} "
        f"{req.target_batches[0] if req.target_batches else ''}".strip()
    )
    loc_encoded = quote_plus(req.locations[0] if req.locations else "India")

    urls: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(url: str, portal_id: str) -> None:
        norm = normalize_url(url)
        if norm and norm not in seen:
            seen.add(norm)
            urls.append((url, portal_id))

    # Specialized deep search/category templates where supported
    if req.target_batches:
        batch = req.target_batches[0]
        _add(f"https://thejobcompany.co.in/job-category/batch/{batch}/", "thejobcompany")
    _add(f"https://www.linkedin.com/jobs/search/?keywords={kw_encoded}&location={loc_encoded}", "linkedin")
    _add(f"https://internshala.com/internships/{role_slug}-internship/", "internshala")
    _add(f"https://in.indeed.com/jobs?q={kw_encoded}&l={loc_encoded}", "indeed")

    # Include all 16 configured entry URLs
    for portal in TARGET_PORTALS:
        _add(portal.entry_url, portal.id)

    return urls


def discover_candidate_urls(
    req: JobSearchRequirement,
    seen_store: SeenJobsStore,
    max_ddg_results_per_query: int = 8,
    enable_ddg: bool = True,
) -> list[tuple[str, str]]:
    """Run 16-portal DuckDuckGo discovery + direct portal templates, filtering out already-seen detail URLs."""
    discovered: list[tuple[str, str]] = []
    seen_in_batch: set[str] = set()

    def _register(url: str, default_portal: str = "external") -> None:
        norm = normalize_url(url)
        if not norm or norm in seen_in_batch:
            return
        if seen_store.is_url_seen(norm):
            return
        portal = resolve_portal_for_url(norm)
        portal_id = portal.id if portal else default_portal
        seen_in_batch.add(norm)
        discovered.append((norm, portal_id))

    if enable_ddg:
        queries = build_16_site_search_queries(req)
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                for q in queries:
                    try:
                        results = ddgs.text(q, region="in-en", max_results=max_ddg_results_per_query)
                        for r in results or []:
                            href = r.get("href") or r.get("url") or ""
                            if href:
                                _register(href)
                    except Exception as exc:
                        logger.debug("DDG query failed for %s: %s", q, exc)
        except ImportError:
            # Built-in httpx DuckDuckGo HTML fallback when `ddgs` package is not installed
            import httpx
            from bs4 import BeautifulSoup
            from urllib.parse import parse_qs, unquote, urlparse

            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            for q in queries:
                try:
                    resp = httpx.get(
                        f"https://html.duckduckgo.com/html/?q={quote_plus(q)}",
                        headers=headers,
                        timeout=8.0,
                    )
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, "lxml")
                        for a in soup.select("a.result__a"):
                            raw_href = a.get("href") or ""
                            if "uddg=" in raw_href:
                                qs = parse_qs(urlparse(raw_href).query)
                                if qs.get("uddg"):
                                    raw_href = unquote(qs["uddg"][0])
                            if raw_href.startswith("http"):
                                _register(raw_href)
                except Exception as exc:
                    logger.debug("DDG HTML fallback failed for %s: %s", q, exc)
        except Exception as exc:
            logger.debug("DDGS search unavailable (%s), relying on direct portal URLs", exc)

    for url, portal_id in build_direct_portal_urls(req):
        _register(url, portal_id)

    return discovered

