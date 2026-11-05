"""Stage 3: Clean Job Detail Page Extraction (`application/ld+json` JobPosting + `trafilatura` Markdown)."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from scraper.cleaner import clean_company_name, clean_text_block

EXTERNAL_APPLY_TEXT_RE = re.compile(
    r"^\s*apply\s*$|(?:\b(?:apply\s*now|apply\s*here|apply\s*online|apply\s*on\s*company|registration\s*link|application\s*link|click\s*here\s*to\s*apply)\b)",
    re.IGNORECASE,
)

BLOCKED_APPLY_DOMAINS = (
    "whatsapp.com",
    "wa.me",
    "t.me",
    "telegram.me",
    "telegram.org",
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "linkedin.com/company",
)


class RawJobDetail(BaseModel):
    """Cleaned intermediate representation of a single job detail page prior to Stage 4 structuring."""

    source_url: str
    source_portal: str
    title: str = ""
    company: str = ""
    json_ld_data: dict[str, Any] = Field(default_factory=dict)
    cleaned_markdown: str = ""
    outbound_apply_url: str = ""
    relevance_score: float = 0.0


def _find_job_posting_in_jsonld(obj: Any) -> dict[str, Any] | None:
    """Recursively search a parsed JSON-LD object/array/`@graph` for `@type == 'JobPosting'`."""
    if isinstance(obj, dict):
        t = obj.get("@type")
        if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
            return obj
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for item in obj["@graph"]:
                found = _find_job_posting_in_jsonld(item)
                if found:
                    return found
        for v in obj.values():
            if isinstance(v, (dict, list)):
                found = _find_job_posting_in_jsonld(v)
                if found:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_job_posting_in_jsonld(item)
            if found:
                return found
    return None


def _extract_jsonld_job_posting(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw_txt = (script.string or script.get_text() or "").strip()
        if not raw_txt:
            continue
        # Remove trailing commas before } or ] and strip non-whitespace control chars
        sanitized = re.sub(r",\s*([}\]])", r"\1", raw_txt)
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
        try:
            parsed = json.loads(sanitized, strict=False)
        except json.JSONDecodeError:
            continue
        found = _find_job_posting_in_jsonld(parsed)
        if found:
            return found
    return {}


def _extract_outbound_apply_url(soup: BeautifulSoup, source_url: str) -> str:
    """Resolve external application link on aggregator pages (or fall back to `source_url`)."""
    source_host = (urlparse(source_url).hostname or "").lower()
    candidates: list[tuple[int, str]] = []

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        abs_url = urljoin(source_url, href)
        host = (urlparse(abs_url).hostname or "").lower()
        if not host:
            continue
        if any(blocked in abs_url.lower() for blocked in BLOCKED_APPLY_DOMAINS):
            continue

        anchor_txt = " ".join(a.get_text(" ", strip=True).split())
        score = 0
        if EXTERNAL_APPLY_TEXT_RE.search(anchor_txt):
            score += 5
        if host != source_host:
            score += 3
        elif "apply" not in abs_url.lower():
            # Same-domain links that aren't explicit apply endpoints (e.g. related blog posts) are not apply links
            continue
        if any(
            ats in host
            for ats in (
                "myworkdayjobs.com",
                "greenhouse.io",
                "lever.co",
                "smartrecruiters.com",
                "ashbyhq.com",
                "icims.com",
                "eightfold.ai",
                "unstop.com",
                "internshala.com",
            )
        ):
            score += 4

        if score >= 5:
            candidates.append((score, abs_url))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return source_url


def extract_job_detail_from_html(
    html: str,
    source_url: str,
    source_portal: str,
    relevance_score: float = 0.0,
) -> RawJobDetail:
    """Extract structured JSON-LD JobPosting + cleaned main article text from a detail page."""
    if not html or not html.strip():
        return RawJobDetail(source_url=source_url, source_portal=source_portal, outbound_apply_url=source_url)

    soup = BeautifulSoup(html, "lxml")
    json_ld = _extract_jsonld_job_posting(soup)
    outbound_apply = _extract_outbound_apply_url(soup, source_url)

    title = str(json_ld.get("title") or "").strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = " ".join(h1.get_text(" ", strip=True).split())
        elif soup.title:
            title = " ".join(soup.title.get_text(" ", strip=True).split()).split("|")[0].strip()

    company = ""
    hiring_org = json_ld.get("hiringOrganization")
    if isinstance(hiring_org, dict):
        company = clean_company_name(str(hiring_org.get("name") or ""))
    elif isinstance(hiring_org, str):
        company = clean_company_name(hiring_org)

    # Layer 2: Main content extraction via trafilatura + article/main fallback
    trafilatura_text = ""
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_links=False, include_tables=True)
        if extracted:
            trafilatura_text = extracted
    except Exception:
        pass

    main_container = soup.find("article") or soup.find("main") or soup.body
    bs4_text = ""
    if main_container:
        for junk in main_container.find_all(["nav", "footer", "header", "script", "style", "noscript"]):
            junk.decompose()
        bs4_text = main_container.get_text("\n", strip=True)

    json_ld_desc = clean_text_block(str(json_ld.get("description") or ""))
    combined_body = "\n".join(filter(None, [json_ld_desc, trafilatura_text, bs4_text]))
    cleaned_markdown = clean_text_block(combined_body)[:4000]

    return RawJobDetail(
        source_url=source_url,
        source_portal=source_portal,
        title=title,
        company=company,
        json_ld_data=json_ld,
        cleaned_markdown=cleaned_markdown,
        outbound_apply_url=outbound_apply,
        relevance_score=relevance_score,
    )

