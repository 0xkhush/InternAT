"""Hybrid async HTTP (`httpx`) + Headless Playwright Chromium fetcher engine."""

from __future__ import annotations

import logging
from typing import Literal

import httpx
from bs4 import BeautifulSoup

from scraper.config import resolve_portal_for_url

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


async def _handle_playwright_route(route) -> None:
    """Async route handler that aborts heavy media/font requests and continues document/script requests."""
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


def needs_js_rendering(html: str) -> bool:
    """Detect whether fetched HTML is an empty SPA shell or requires client-side JS rendering."""
    if not html or len(html.strip()) < 250:
        return True

    soup = BeautifulSoup(html, "lxml")
    for s in soup.find_all(["script", "style", "noscript", "template"]):
        s.decompose()

    body = soup.body
    if body is None:
        return True

    visible_text = " ".join(body.get_text(" ", strip=True).split())
    if len(visible_text) < 200:
        return True

    # Check common SPA root containers (#root, #__next, #app)
    for root_id in ("root", "__next", "app"):
        root_el = soup.find(id=root_id)
        if root_el is not None:
            root_text = " ".join(root_el.get_text(" ", strip=True).split())
            if len(root_text) < 80 and len(visible_text) < 350:
                return True

    return False


class HybridFetcher:
    """Async fetcher supporting `render="never"`, `render="always"`, and `render="auto"`."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self._http_client: httpx.AsyncClient | None = None
        self._playwright = None
        self._browser = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._http_client

    async def _fetch_http(self, url: str) -> str:
        client = await self._get_http_client()
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text

    async def _fetch_playwright(self, url: str) -> str:
        from playwright.async_api import async_playwright

        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)

        context = await self._browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"])
        page = await context.new_page()
        try:
            await page.route("**/*", _handle_playwright_route)
            await page.goto(url, wait_until="domcontentloaded", timeout=int(self.timeout * 1000))
            await page.wait_for_timeout(1200)
            return await page.content()
        finally:
            await context.close()

    async def fetch(
        self,
        url: str,
        render_mode: Literal["auto", "always", "never"] | None = None,
    ) -> str:
        """Fetch page HTML using the portal's configured render mode (`never`, `always`, `auto`)."""
        if render_mode is None:
            portal = resolve_portal_for_url(url)
            render_mode = portal.render if portal else "auto"

        if render_mode == "never":
            return await self._fetch_http(url)

        # Fast-path for InternDoor (`interndoor.com`):
        # 1. Its homepage loads 1,500+ live jobs via `/data/jobs.json` into `<ol id="joblist">` on the client.
        # 2. Many jobs in `/data/jobs.json` have `pageSlug: null` and point directly to Workday/Avature URLs.
        # We cache `/data/jobs.json` in memory so both the listing page and any direct ATS detail URLs from
        # InternDoor resolve immediately with full structured job descriptions in <200ms!
        if "interndoor.com" in url and "/jobs/" not in url:
            try:
                client = await self._get_http_client()
                resp = await client.get("https://interndoor.com/data/jobs.json")
                if resp.status_code == 200:
                    data = resp.json()
                    jobs = data.get("jobs", [])
                    self._interndoor_cache = {}
                    cards_html: list[str] = []
                    for j in jobs:
                        emp_type = str(j.get("employmentType") or "")
                        title = str(j.get("title") or "")
                        if "intern" not in emp_type.lower() and "intern" not in title.lower():
                            continue
                        if not (j.get("category") == "software" or j.get("isTech")):
                            continue
                        company = str(j.get("company") or "")
                        loc = str(j.get("location") or "India")
                        slug = j.get("pageSlug")
                        apply_link = str(j.get("applyUrl") or j.get("url") or "")
                        href = f"https://interndoor.com/jobs/{slug}" if slug else apply_link
                        if not href:
                            continue
                        exp = str(j.get("experience") or "")
                        acad = str(j.get("academic") or "")
                        stipend = str(j.get("stipend") or "")
                        skills = ", ".join(j.get("skills") or [])
                        summary = str(j.get("summary") or "")
                        desc = str(j.get("description") or summary)
                        self._interndoor_cache[href] = (
                            f"<html><head><title>{title} at {company}</title></head>"
                            f"<body><article><h1>{company} - {title}</h1>"
                            f"<p>Location: {loc}</p><p>Experience: {exp}</p><p>Eligibility: {acad}</p>"
                            f"<p>Stipend: {stipend}</p><p>Skills: {skills}</p>"
                            f"<div>{desc}</div>"
                            f'<a href="{apply_link}">Apply Now</a></article></body></html>'
                        )
                        cards_html.append(
                            f'<article class="interndoor-card">'
                            f'<h2><a href="{href}">{company} - {title}</a></h2>'
                            f'<p>{company} | {loc} | Internship | {exp} | {acad} | {stipend} | {skills} | {summary}</p>'
                            f'</article>'
                        )
                    if cards_html:
                        return f"<html><body><main>{''.join(cards_html)}</main></body></html>"
            except Exception as exc:
                logger.debug("InternDoor JSON feed synthesis failed (%s), falling back to HTML", exc)

        if hasattr(self, "_interndoor_cache") and url in self._interndoor_cache:
            return self._interndoor_cache[url]

        if render_mode == "always":
            try:
                return await self._fetch_playwright(url)
            except Exception as exc:
                logger.debug("Playwright fetch unavailable for %s (%s), falling back to httpx", url, exc)
                return await self._fetch_http(url)

        # render_mode == "auto"
        http_html = ""
        try:
            http_html = await self._fetch_http(url)
            if not needs_js_rendering(http_html):
                return http_html
        except Exception as exc:
            logger.debug("HTTP auto-fetch failed for %s (%s), escalating to Playwright", url, exc)

        try:
            return await self._fetch_playwright(url)
        except ImportError:
            logger.debug("Playwright not installed; using HTTP response for %s", url)
            return http_html
        except Exception as exc:
            logger.debug("Playwright fallback failed for %s (%s)", url, exc)
            return http_html

    async def close(self) -> None:
        """Close underlying HTTP client and Playwright browser."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

