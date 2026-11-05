"""Persistent URL and job-fingerprint deduplication store across runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from scraper.config import TARGET_PORTALS
from scraper.schemas import JobRecord

TRACKING_PARAM_PREFIXES = ("utm_", "fbclid", "gclid", "ref", "trk", "source", "mc_", "_ga", "igshid")


def normalize_url(url: str) -> str:
    """Normalize a URL by stripping tracking query parameters, fragments, and trailing slashes."""
    if not url or not url.strip():
        return ""
    raw = url.strip()
    try:
        parsed = urlparse(raw)
    except ValueError:
        return raw

    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        # Keep www only if needed, or normalize consistently: wait - preserve original host except lowercase
        netloc = parsed.netloc.lower()

    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    clean_query_items = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=False):
        k_lower = k.lower()
        if any(k_lower == p or k_lower.startswith(p) for p in TRACKING_PARAM_PREFIXES):
            continue
        clean_query_items.append((k, v))
    clean_query_items.sort()
    query_str = urlencode(clean_query_items)

    return urlunparse((scheme, netloc, path, "", query_str, ""))


def _build_listing_entry_urls() -> set[str]:
    """Return normalized entry URLs of all 16 portals so listing roots are never blocked as seen detail pages."""
    urls: set[str] = set()
    for p in TARGET_PORTALS:
        urls.add(normalize_url(p.entry_url))
        urls.add(normalize_url(f"https://{p.domain}/"))
        urls.add(normalize_url(f"https://www.{p.domain}/"))
    return urls


PORTAL_LISTING_ROOTS: set[str] = _build_listing_entry_urls()


def _fingerprint(company: str, title: str, source_portal: str) -> str:
    norm_c = re.sub(r"[^a-z0-9]+", " ", company.lower()).strip()
    norm_t = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    norm_p = source_portal.lower().strip()
    raw = f"{norm_c}|{norm_t}|{norm_p}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class SeenJobsStore:
    """Persistent store preventing already-scraped jobs from a site from being scraped or listed again."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.seen_urls: set[str] = set()
        self.seen_fingerprints: set[str] = set()
        self.seen_job_ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.seen_urls = set(data.get("seen_urls", []))
            self.seen_fingerprints = set(data.get("seen_fingerprints", []))
            self.seen_job_ids = set(data.get("seen_job_ids", []))
        except (OSError, json.JSONDecodeError):
            pass

    def is_listing_root(self, url: str) -> bool:
        """Check whether a URL is a top-level portal listing entry page."""
        norm = normalize_url(url)
        return norm in PORTAL_LISTING_ROOTS

    def is_url_seen(self, url: str) -> bool:
        """Return True if a job detail URL was already scraped in an earlier run."""
        norm = normalize_url(url)
        if not norm or norm in PORTAL_LISTING_ROOTS:
            return False
        return norm in self.seen_urls

    def is_job_seen(self, company: str, title: str, source_portal: str, apply_url: str = "") -> bool:
        """Return True if a job from a portal has already been scraped/listed."""
        fp = _fingerprint(company, title, source_portal)
        if fp in self.seen_fingerprints:
            return True
        if apply_url and self.is_url_seen(apply_url):
            return True
        return False

    def mark_url_seen(self, url: str) -> None:
        """Record a detail URL as seen (listing root URLs are ignored so they remain crawlable)."""
        norm = normalize_url(url)
        if norm and norm not in PORTAL_LISTING_ROOTS:
            self.seen_urls.add(norm)

    def mark_seen(self, record: JobRecord) -> None:
        """Mark a full JobRecord (its source_url, apply_url, job_id, and portal fingerprint) as seen."""
        self.mark_url_seen(record.source_url)
        if record.apply_url:
            self.mark_url_seen(record.apply_url)
        self.seen_fingerprints.add(_fingerprint(record.company, record.title, record.source_portal))
        if record.job_id:
            self.seen_job_ids.add(record.job_id)

    def save(self) -> None:
        """Persist seen URLs and fingerprints to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seen_urls": sorted(self.seen_urls),
            "seen_fingerprints": sorted(self.seen_fingerprints),
            "seen_job_ids": sorted(self.seen_job_ids),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

