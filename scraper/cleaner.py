"""4-Layer Data Sanitizer, Promo/Boilerplate Stripper, and Field Normalizer."""

from __future__ import annotations

import html
import re
from typing import Optional

from bs4 import BeautifulSoup

from scraper.schemas import JobRecord, compute_job_id

PROMO_LINE_RE = re.compile(
    r"(join\s+(?:our\s+)?(?:whatsapp|telegram)|whatsapp\s+(?:group|channel)|telegram\s+(?:group|channel)|"
    r"t\.me/|whatsapp\.com/channel|follow\s+us\s+on\s+(?:instagram|linkedin|twitter|telegram|whatsapp)|"
    r"click\s+here\s+to\s+join|subscribe\s+to\s+our\s+newsletter|practice\s+\d+\+?\s+interview\s+questions|"
    r"free\s+interview\s+prep|tailor\s+your\s+resume|also\s+apply\s*:|share\s+this\s+job\s+with\s+friends)",
    re.IGNORECASE,
)

CITY_SYNONYMS: dict[str, str] = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "new delhi": "Delhi NCR",
    "delhi": "Delhi NCR",
    "delhi ncr": "Delhi NCR",
    "ncr": "Delhi NCR",
    "noida": "Noida",
    "hyderabad": "Hyderabad",
    "pune": "Pune",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
    "ahmedabad": "Ahmedabad",
    "remote": "Remote",
    "work from home": "Remote",
    "wfh": "Remote",
    "pan india": "India",
    "india": "India",
}

COMPANY_NOISE_SUFFIX_RE = re.compile(
    r"\s*(?:hiring\s+off\s+campus|off\s+campus\s+drive|hiring\s+freshers|hiring|recruitment\s*\d*|careers?)\s*$",
    re.IGNORECASE,
)

INVALID_GENERIC_TITLES = {
    "",
    "home",
    "jobs",
    "internships",
    "login",
    "sign in",
    "sign up",
    "register",
    "privacy policy",
    "terms of service",
    "contact us",
    "about us",
}


def clean_text_block(text: str) -> str:
    """Strip HTML tags, entities, zero-width chars, and WhatsApp/Telegram/promo boilerplate lines."""
    if not text:
        return ""

    # Unescape HTML entities first
    raw = html.unescape(text)
    raw = raw.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")

    # If HTML tags exist, convert block tags to newlines and strip tags
    if "<" in raw and ">" in raw:
        soup = BeautifulSoup(raw, "lxml")
        for s in soup.find_all(["script", "style", "noscript", "svg"]):
            s.decompose()
        raw = soup.get_text("\n", strip=True)

    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()
    for line in raw.splitlines():
        line_clean = re.sub(r"\s+", " ", line).strip()
        if not line_clean:
            continue
        if PROMO_LINE_RE.search(line_clean):
            continue
        norm_key = line_clean.lower()
        if norm_key in seen_lines:
            continue
        seen_lines.add(norm_key)
        cleaned_lines.append(line_clean)

    return "\n".join(cleaned_lines).strip()


def clean_company_name(company: str) -> str:
    """Normalize company name by stripping 'Hiring Off Campus', 'Recruitment', etc."""
    if not company:
        return "Not Disclosed"
    cleaned = html.unescape(company).strip()
    cleaned = COMPANY_NOISE_SUFFIX_RE.sub("", cleaned).strip(" -–|•:")
    return cleaned if cleaned else "Not Disclosed"


def normalize_locations(locations: list[str] | str) -> list[str]:
    """Normalize city names (e.g., Bangalore -> Bengaluru, Gurgaon -> Gurugram) and deduplicate."""
    if isinstance(locations, str):
        raw_items = re.split(r"[,/|•;]+", locations)
    else:
        raw_items = []
        for item in locations:
            raw_items.extend(re.split(r"[,/|•;]+", str(item)))

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        token = item.strip()
        if not token:
            continue
        key = token.lower()
        mapped = CITY_SYNONYMS.get(key, token.title() if token.islower() else token)
        if mapped.lower() not in seen:
            seen.add(mapped.lower())
            normalized.append(mapped)

    return normalized if normalized else ["India"]


def normalize_job_record(record: JobRecord) -> Optional[JobRecord]:
    """Layer 4 deterministic validator and field normalizer for JobRecord."""
    clean_title = re.sub(r"\s+", " ", html.unescape(record.title or "")).strip(" -–|")
    if clean_title.lower() in INVALID_GENERIC_TITLES or len(clean_title) < 3:
        return None
    if PROMO_LINE_RE.search(clean_title):
        return None

    clean_comp = clean_company_name(record.company)
    clean_desc = clean_text_block(record.job_description)
    if not clean_desc:
        clean_desc = f"{clean_title} at {clean_comp}"

    norm_locs = normalize_locations(record.location)
    batches = [b.strip() for b in record.batch_eligible if b and b.strip()] or ["Not Specified"]
    stipend = (record.stipend_or_salary or "").strip() or "Not Disclosed"
    exp = (record.experience_required or "").strip() or "0 years (Fresher / Student)"
    apply_url = (record.apply_url or record.source_url or "").strip()

    updated = record.model_copy(
        update={
            "title": clean_title,
            "company": clean_comp,
            "location": norm_locs,
            "batch_eligible": batches,
            "stipend_or_salary": stipend,
            "experience_required": exp,
            "job_description": clean_desc,
            "apply_url": apply_url,
            "job_id": compute_job_id(clean_comp, clean_title, apply_url),
        }
    )
    return updated

