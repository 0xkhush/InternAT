"""Outer-first structural DOM candidate extractor and page-level URL deduplicator."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, Tag

from scraper.config import resolve_portal_for_url
from scraper.dedup import normalize_url
from scraper.schemas import DOMCandidate

RegionType = Literal["article", "main", "aside", "nav", "header", "footer", "body"]

REGION_PRIORITY: dict[str, int] = {
    "article": 6,
    "main": 5,
    "body": 4,
    "aside": 3,
    "nav": 2,
    "header": 1,
    "footer": 0,
}

STRIP_TAGS = ("script", "style", "noscript", "svg", "template", "canvas", "iframe")
CARD_CONTAINER_TAGS = ("article", "li", "tr", "section", "div")


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u200b", "").replace("\ufeff", "")).strip()


def resolve_region(tag: Tag) -> RegionType:
    """Resolve the semantic DOM region of a tag walking outermost-to-innermost ancestors.

    - Outer chrome tags (`footer`, `header`, `aside`, `nav`) take precedence over anything nested inside them
      when they are outside `<main>`.
    - Once inside `<main>` or `<article>`, a nested `<footer>` that contains job listing links (`/jobs/` or `/internships/`)
      is treated as `"main"` content rather than site chrome (e.g., `interndoor.com` renders its SSR job directory
      inside `<main><footer><details><ul><li><a href="/jobs/...">`).
    - Once an `<article>` card boundary is entered, inner `<header>` or `<footer>` sub-elements inside the card
      remain `"article"`.
    """
    ancestors = [p.name.lower() for p in tag.parents if isinstance(p, Tag) and p.name]
    if tag.name:
        ancestors.insert(0, tag.name.lower())

    href_val = (tag.get("href") or "").lower() if tag.name == "a" else ""
    is_job_href = any(p in href_val for p in ("/jobs/", "/job/", "/internships/", "/internship/", "job_details.php"))

    # Walk from outermost (root/body) to innermost (the tag itself)
    in_main = False
    for name in reversed(ancestors):
        if name == "article":
            return "article"
        if name == "main":
            in_main = True
            continue
        if name in ("footer", "header", "aside", "nav"):
            if in_main and name == "footer" and is_job_href:
                continue
            return name  # type: ignore[return-value]

    return "main" if in_main else "body"


def _build_tag_path(tag: Tag, max_depth: int = 5) -> str:
    names = [p.name.lower() for p in tag.parents if isinstance(p, Tag) and p.name and p.name.lower() not in ("html", "[document]")]
    names.reverse()
    names.append(tag.name.lower())
    if len(names) > max_depth:
        names = names[-max_depth:]
    return " > ".join(names)


def _find_card_container(anchor: Tag) -> Tag:
    """Find the enclosing card or list-item container around an anchor element."""
    for parent in anchor.parents:
        if not isinstance(parent, Tag) or not parent.name:
            continue
        name = parent.name.lower()
        if name in ("article", "li", "tr"):
            return parent
        if name in ("main", "body", "footer", "header", "nav", "aside"):
            break
    # Fallback: look for immediate parent/grandparent div with bounded text
    curr: Tag = anchor
    for parent in anchor.parents:
        if not isinstance(parent, Tag) or not parent.name:
            break
        if parent.name.lower() in ("main", "body", "footer", "header", "nav", "aside"):
            break
        txt = _clean_whitespace(parent.get_text(" ", strip=True))
        if 20 <= len(txt) <= 600:
            curr = parent
            # Stop if parent has multiple sibling cards of the same tag
            if parent.parent and isinstance(parent.parent, Tag):
                same_tag_siblings = [
                    sib for sib in parent.parent.find_all(parent.name, recursive=False)
                ]
                if len(same_tag_siblings) >= 2:
                    return parent
    return curr


def _compute_sibling_count(container: Tag) -> int:
    parent = container.parent
    if not isinstance(parent, Tag):
        return 1
    siblings = [
        sib
        for sib in parent.find_all(container.name, recursive=False)
        if isinstance(sib, Tag)
    ]
    return max(1, len(siblings))


def _normalize_url_path(full_url: str) -> str:
    try:
        parsed = urlparse(full_url)
    except ValueError:
        return full_url
    path = parsed.path or "/"
    if parsed.query:
        return f"{path}?{parsed.query}"
    return path


def extract_dom_candidates(
    html: str,
    base_url: str,
    source_portal: str = "unknown",
) -> list[DOMCandidate]:
    """Parse HTML, strip non-visible noise tags, extract DOMCandidates, and deduplicate per page."""
    if not html or not html.strip():
        return []

    if source_portal == "unknown":
        matched = resolve_portal_for_url(base_url)
        if matched:
            source_portal = matched.id

    soup = BeautifulSoup(html, "lxml")

    # Remove comments and non-content elements
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag_name in STRIP_TAGS:
        for t in soup.find_all(tag_name):
            t.decompose()
    for hidden in soup.select("[hidden], [aria-hidden='true']"):
        hidden.decompose()

    dedup_by_url: dict[str, DOMCandidate] = {}

    for a in soup.find_all("a", href=True):
        raw_href = (a.get("href") or "").strip()
        if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        abs_url = urljoin(base_url, raw_href)
        norm_url = normalize_url(abs_url)
        if not norm_url:
            continue

        anchor_text = _clean_whitespace(a.get_text(" ", strip=True))
        if not anchor_text:
            anchor_text = _clean_whitespace(a.get("aria-label") or a.get("title") or "")
        if not anchor_text:
            continue

        region = resolve_region(a)
        container = _find_card_container(a)
        sibling_count = _compute_sibling_count(container)
        raw_context = _clean_whitespace(container.get_text(" ", strip=True))

        # Remove duplicate anchor text prefix from context if identical
        if raw_context == anchor_text:
            context = anchor_text
        else:
            context = raw_context[:400]

        url_path = _normalize_url_path(norm_url)
        tag_path = _build_tag_path(a)

        candidate = DOMCandidate(
            url=norm_url,
            url_path=url_path,
            text=anchor_text[:200],
            context=context,
            region=region,
            tag_path=tag_path,
            sibling_count=sibling_count,
            source_portal=source_portal,
        )

        existing = dedup_by_url.get(norm_url)
        if existing is None:
            dedup_by_url[norm_url] = candidate
        else:
            # Keep the candidate in the highest-priority region; break ties by longest context
            curr_prio = (REGION_PRIORITY.get(candidate.region, 0), len(candidate.context), len(candidate.text))
            exist_prio = (REGION_PRIORITY.get(existing.region, 0), len(existing.context), len(existing.text))
            if curr_prio > exist_prio:
                dedup_by_url[norm_url] = candidate

    return list(dedup_by_url.values())

