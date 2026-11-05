# ML-Powered Job & Internship Web Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, end-to-end Conversational CLI AI Agent (`glm-5.3-flash`) and Two-Stage Local ML Job Scraper (`all-MiniLM-L6-v2` Bi-Encoder + `ms-marco-MiniLM-L-6-v2` Cross-Encoder) across 16 Indian job/internship portals with persistent deduplication, 4-layer data cleaning, and 2D vector space visualization.

**Architecture:** A 5-stage pipeline starting with a conversational CLI intake agent (`glm-5.3-flash`) that clarifies user requirements, followed by 16-portal DuckDuckGo + direct URL discovery gated by a persistent `SeenJobsStore`, an async hybrid HTTP/Playwright fetcher + outer-first DOM candidate extractor, a Two-Stage local ML filter (Bi-Encoder 3-class classifier + Cross-Encoder re-ranker), and a 4-layer detail cleaner (`JSON-LD` + `trafilatura` + boilerplate sanitizer + batched `glm-5.3-flash` extraction into validated `jobs.json`).

**Tech Stack:** Python 3.10+, `httpx`, `playwright`, `beautifulsoup4`, `lxml`, `trafilatura`, `ddgs` (`duckduckgo-search`), `sentence-transformers`, `torch`, `numpy`, `scikit-learn`, `pydantic`, `openai` (for `glm-5.3-flash`), `matplotlib`, `plotly`.

**Spec:** `docs/superpowers/specs/2026-11-05-ml-job-scraper-design.md`

## Global Constraints

- Primary LLM model name must default to `glm-5.3-flash` via OpenAI-compatible SDK (`GLM_API_KEY`, `GLM_BASE_URL` defaulting to `https://open.bigmodel.cn/api/paas/v4/`), with deterministic fallback when no API key is configured.
- Stage 1 Bi-Encoder model must be `sentence-transformers/all-MiniLM-L6-v2` (384-d L2-normalized embeddings) classifying into `"job_listing"`, `"pagination_or_filter"`, or `"noise"`.
- Stage 2 Cross-Encoder model must be `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- All 16 portals (`thejobcompany`, `careersat`, `hiredoor`, `freshersworld`, `offcampus`, `talentbattle`, `internshala`, `unstop`, `wellfound`, `instahyre`, `cutshort`, `hasjob`, `indeed`, `linkedin`, `cuvette`, `prosple`) must be registered in `TARGET_PORTALS` with their exact `id`, `domain`, `entry_url`, and `render` mode (`"auto"`, `"always"`, `"never"`).
- Outer-first DOM region resolution: `<nav>` inside `<footer>` or `<aside>` must resolve to `region="footer"` or `region="aside"`, never `"nav"`.
- Bi-Encoder input string format must be: `"[REGION] {region} | [SIBLINGS] {sibling_count} | [TEXT] {text} | [CONTEXT] {context} | [URL] {url_path}"`.
- Missing job fields must normalize to `"Not Disclosed"` or `"Not Specified"`—never hallucinated.

## Review Focus

1. **Nested `<nav aria-label="Footer">` inside `<footer>`:** Must resolve to `region="footer"` so duplicate mobile/desktop footer sitemaps are ranked lower than `<article>`/`<main>` during page deduplication and classified as `noise`.
2. **Same job URL appearing in both navbar/footer and `<article>` card on the same page:** Page-level deduplication must retain the `<article>`/`<main>` candidate with the richest `context`, not whichever link appeared first in DOM order.
3. **Malformed or `@graph`-nested `application/ld+json` blocks on job detail pages:** Detail extractor must unwrap `@graph` lists, tolerate trailing commas/control chars, and strip HTML tags inside `JobPosting.description`.
4. **Aggregator posts containing WhatsApp/Telegram promos and external apply links:** Cleaner must strip `"Join our WhatsApp/Telegram"` lines from `job_description` and resolve the outbound external career URL as `apply_url` while keeping the portal page as `source_url`.
5. **Re-scraping listing pages vs. detail pages across runs:** `SeenJobsStore` must block already-seen job detail URLs and `(company, title, source_portal)` fingerprints while still allowing listing/category entry URLs (like `https://careersat.tech/jobs`) to be fetched on subsequent runs.

---

### Task 1: Project Scaffolding, 16-Portal Config, Core Schemas & Persistent Deduplication Store

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `scraper/__init__.py`
- Create: `scraper/config.py`
- Create: `scraper/schemas.py`
- Create: `scraper/dedup.py`
- Test: `tests/test_config_and_dedup.py`

**Interfaces:**
- Consumes: None
- Produces:
  - `scraper.config.PortalConfig` (Pydantic model / dataclass with `id`, `name`, `domain`, `entry_url`, `render`)
  - `scraper.config.TARGET_PORTALS: list[PortalConfig]` (all 16 portals)
  - `scraper.config.PORTAL_BY_DOMAIN: dict[str, PortalConfig]`
  - `scraper.config.Settings` (`glm_api_key`, `glm_base_url`, `glm_model="glm-5.3-flash"`, `bi_encoder_model="sentence-transformers/all-MiniLM-L6-v2"`, `cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"`, `seen_jobs_path`, `output_jobs_path`)
  - `scraper.schemas.JobSearchRequirement` (`raw_query`, `role_keywords`, `job_type`, `target_batches`, `experience_level`, `locations`, `work_mode`, `min_stipend_or_salary`, `skills`, `to_search_summary() -> str`)
  - `scraper.schemas.DOMCandidate` (`url`, `url_path`, `text`, `context`, `region`, `tag_path`, `sibling_count`, `source_portal`, `predicted_label`, `confidence`, `relevance_score`, `to_feature_string() -> str`)
  - `scraper.schemas.JobRecord` (all 18 required fields + `compute_job_id(company: str, title: str, apply_url: str) -> str`)
  - `scraper.dedup.normalize_url(url: str) -> str`
  - `scraper.dedup.SeenJobsStore` (`is_url_seen(url: str) -> bool`, `is_job_seen(company: str, title: str, source_portal: str, apply_url: str = "") -> bool`, `mark_seen(record: JobRecord) -> None`, `mark_url_seen(url: str) -> None`, `save() -> None`)

- [ ] **Step 1: Write the failing test in `tests/test_config_and_dedup.py`**

```python
from pathlib import Path
from scraper.config import TARGET_PORTALS, Settings
from scraper.schemas import DOMCandidate, JobRecord, JobSearchRequirement
from scraper.dedup import normalize_url, SeenJobsStore


def test_16_target_portals_and_glm_settings():
    assert len(TARGET_PORTALS) == 16
    ids = {p.id for p in TARGET_PORTALS}
    assert {"thejobcompany", "careersat", "internshala", "unstop", "linkedin", "cuvette", "prosple"}.issubset(ids)
    settings = Settings()
    assert settings.glm_model == "glm-5.3-flash"


def test_dom_candidate_feature_string_format():
    cand = DOMCandidate(
        url="https://careersat.tech/jobs/honeywell-intern-2027",
        url_path="/jobs/honeywell-intern-2027",
        text="Honeywell Internship 2027",
        context="Honeywell Bangalore Internship Batch 2027",
        region="article",
        tag_path="main > article > a",
        sibling_count=20,
        source_portal="careersat",
    )
    assert cand.to_feature_string() == (
        "[REGION] article | [SIBLINGS] 20 | [TEXT] Honeywell Internship 2027 | "
        "[CONTEXT] Honeywell Bangalore Internship Batch 2027 | [URL] /jobs/honeywell-intern-2027"
    )


def test_seen_jobs_store_deduplicates_urls_and_fingerprints(tmp_path: Path):
    store_file = tmp_path / "seen_jobs.json"
    store = SeenJobsStore(store_file)

    raw_url = "https://careersat.tech/jobs/rubrik-swe-intern-2028/?utm_source=ddg&ref=home"
    clean_url = normalize_url(raw_url)
    assert clean_url == "https://careersat.tech/jobs/rubrik-swe-intern-2028"
    assert not store.is_url_seen(raw_url)

    # Listing entry URLs from TARGET_PORTALS must never be blocked as seen detail URLs
    store.mark_url_seen("https://careersat.tech/jobs")
    assert not store.is_url_seen("https://careersat.tech/jobs")

    record = JobRecord(
        title="Software Engineering Intern",
        company="Rubrik",
        job_type="Internship",
        batch_eligible=["2028"],
        experience_required="0 years",
        location=["Bengaluru"],
        work_mode="Hybrid",
        stipend_or_salary="₹1,00,000 / month",
        skills_required=["Python", "C++"],
        eligibility_criteria="B.Tech CS 2028 batch",
        job_description="Build distributed cloud storage systems.",
        apply_url="https://rubrik.com/careers/123",
        source_url=raw_url,
        source_portal="careersat",
    )
    store.mark_seen(record)
    store.save()

    reloaded = SeenJobsStore(store_file)
    assert reloaded.is_url_seen("https://careersat.tech/jobs/rubrik-swe-intern-2028")
    assert reloaded.is_job_seen("rubrik", "Software Engineering Intern", "careersat")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_and_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scraper'`

- [ ] **Step 3: Create `pyproject.toml`, `.env.example`, `scraper/config.py`, `scraper/schemas.py`, and `scraper/dedup.py`**

Implement `pyproject.toml` with dependencies (`httpx`, `playwright`, `beautifulsoup4`, `lxml`, `trafilatura`, `ddgs`, `sentence-transformers`, `torch`, `numpy`, `scikit-learn`, `pydantic`, `python-dotenv`, `openai`, `matplotlib`, `plotly`, `pytest`, `pytest-asyncio`), `scraper/config.py` with the 16 `TARGET_PORTALS` and `Settings`, `scraper/schemas.py` with `JobSearchRequirement`, `DOMCandidate`, and `JobRecord` (auto-computing `job_id` and `scraped_at`), and `scraper/dedup.py` with tracking-param stripping (`utm_*`, `ref`, `fbclid`, `gclid`, `trk`) and `SeenJobsStore` (ignoring portal `entry_url` roots so listing pages are always re-crawlable).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_and_dedup.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git init && git add pyproject.toml .env.example scraper/ tests/test_config_and_dedup.py docs/
git commit -m "feat: add 16-portal config, core schemas, and SeenJobsStore deduplication"
```

---

### Task 2: Outer-First Structural DOM Candidate Extractor & Hybrid Fetcher

**Files:**
- Create: `scraper/extractor.py`
- Create: `scraper/fetcher.py`
- Test: `tests/test_extractor_and_fetcher.py`

**Interfaces:**
- Consumes: `DOMCandidate` from `scraper.schemas`, `TARGET_PORTALS` from `scraper.config`, `normalize_url` from `scraper.dedup`
- Produces:
  - `scraper.extractor.resolve_region(tag: Tag) -> str`
  - `scraper.extractor.extract_dom_candidates(html: str, base_url: str, source_portal: str = "unknown") -> list[DOMCandidate]`
  - `scraper.fetcher.needs_js_rendering(html: str) -> bool`
  - `scraper.fetcher.HybridFetcher` (`async fetch(url: str, render_mode: str = "auto") -> str`, `async close() -> None`)

- [ ] **Step 1: Write the failing test in `tests/test_extractor_and_fetcher.py`**

```python
from scraper.extractor import extract_dom_candidates
from scraper.fetcher import needs_js_rendering


SAMPLE_CAREERSAT_HTML = """
<html>
<body>
  <header><nav><a href="/jobs/honeywell-aerospace-internship-2027">Featured Job</a></nav></header>
  <main>
    <div class="feed">
      <article class="card">
        <h2><a href="/jobs/honeywell-aerospace-internship-2027?utm_source=feed">Honeywell Aerospace Internship 2027 – Software Intern</a></h2>
        <p>Honeywell Aerospace • Bangalore, Hyderabad • 1d ago • Internship • 0–1 yrs • Batch 2026–27</p>
      </article>
      <article class="card">
        <h2><a href="/jobs/rubrik-software-intern-2028">Rubrik Software Engineering Intern 2028</a></h2>
        <p>Rubrik • Bengaluru • Internship • Batch 2028 • ₹1,00,000/month</p>
      </article>
      <div class="promo">
        <a href="https://whatsapp.com/channel/0029VaUJgMW2">Join channel</a>
        <p>WhatsApp channel 52,000+ followers Verified fresher jobs</p>
      </div>
    </div>
    <div class="pagination">
      <a href="/jobs?page=2">Next Page</a>
    </div>
  </main>
  <footer>
    <nav aria-label="Footer">
      <a href="/privacy">Privacy Policy</a>
      <a href="/jobs/honeywell-aerospace-internship-2027">Honeywell Footer Link</a>
    </nav>
  </footer>
</body>
</html>
"""


def test_outer_first_region_and_deduplication():
    candidates = extract_dom_candidates(SAMPLE_CAREERSAT_HTML, "https://careersat.tech/jobs", "careersat")
    by_path = {c.url_path: c for c in candidates}

    # 1. Nested <nav> inside <footer> MUST be tagged region="footer", not "nav"
    assert by_path["/privacy"].region == "footer"

    # 2. Duplicate URL in <header><nav>, <footer><nav>, and <main><article> must keep <article> with richest context
    honeywell = by_path["/jobs/honeywell-aerospace-internship-2027"]
    assert honeywell.region == "article"
    assert "Batch 2026–27" in honeywell.context
    assert honeywell.sibling_count >= 2


def test_needs_js_rendering_detects_empty_spa_shell():
    empty_react = "<html><body><div id='root'></div><script src='app.js'></script></body></html>"
    rich_ssr = SAMPLE_CAREERSAT_HTML
    assert needs_js_rendering(empty_react) is True
    assert needs_js_rendering(rich_ssr) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractor_and_fetcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scraper.extractor'`

- [ ] **Step 3: Implement `scraper/extractor.py` and `scraper/fetcher.py`**

In `scraper/extractor.py`:
- Strip `<script>`, `<style>`, `<noscript>`, `<svg>`, `<template>`, `<canvas>`, `<iframe>`, `[hidden]`, `[aria-hidden="true"]`.
- Resolve `region` by checking ancestors from outermost to innermost with priority rule: if any ancestor is `footer`, return `"footer"`; else if `header`, return `"header"`; else if `aside`, return `"aside"`; else if `nav`, return `"nav"`; else if `article`, return `"article"`; else if `main`, return `"main"`; else `"body"`.
- Compute `sibling_count` from the card/container element's parent (number of sibling elements sharing the same tag name).
- Deduplicate candidates on the same page by normalized URL using region priority `{"article": 6, "main": 5, "body": 4, "aside": 3, "nav": 2, "header": 1, "footer": 0}` and longest `context`.

In `scraper/fetcher.py`:
- Implement `needs_js_rendering(html: str) -> bool` checking for empty `#root`, `#__next`, `#app` or visible text < 200 chars.
- Implement `HybridFetcher` supporting `render_mode` in `{"never", "always", "auto"}` using `httpx.AsyncClient` and lazy-initialized Playwright Chromium (blocking image/media/font resources).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractor_and_fetcher.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scraper/extractor.py scraper/fetcher.py tests/test_extractor_and_fetcher.py
git commit -m "feat: add outer-first DOM candidate extractor and hybrid HTTP/Playwright fetcher"
```

---

### Task 3: Stage 1 Bi-Encoder Vector Space Classifier & Stage 2 Cross-Encoder Re-Ranker

**Files:**
- Create: `scraper/ml_classifier.py`
- Create: `scraper/reranker.py`
- Create: `scripts/train_classifier.py`
- Test: `tests/test_ml_pipeline.py`

**Interfaces:**
- Consumes: `DOMCandidate`, `JobSearchRequirement` from `scraper.schemas`, `Settings` from `scraper.config`
- Produces:
  - `scraper.ml_classifier.CLASS_LABELS = ["job_listing", "pagination_or_filter", "noise"]`
  - `scraper.ml_classifier.SEED_PROTOTYPES: dict[str, list[str]]` (20+ rich prototype strings per class)
  - `scraper.ml_classifier.BiEncoderClassifier` (`encode(texts: list[str]) -> np.ndarray`, `classify_candidates(candidates: list[DOMCandidate]) -> tuple[list[DOMCandidate], np.ndarray]`, `fit_classifier_head(texts: list[str], labels: list[str], save_path: Path) -> dict`, `get_centroids() -> dict[str, np.ndarray]`)
  - `scraper.reranker.passes_card_context_prefilter(candidate: DOMCandidate, req: JobSearchRequirement) -> bool`
  - `scraper.reranker.CrossEncoderReranker` (`rerank(candidates: list[DOMCandidate], req: JobSearchRequirement, top_k: int = 25) -> list[DOMCandidate]`)

- [ ] **Step 1: Write the failing test in `tests/test_ml_pipeline.py`**

```python
from scraper.schemas import DOMCandidate, JobSearchRequirement
from scraper.ml_classifier import BiEncoderClassifier
from scraper.reranker import CrossEncoderReranker, passes_card_context_prefilter


def test_bi_encoder_separates_jobs_pagination_and_noise():
    clf = BiEncoderClassifier()
    candidates = [
        DOMCandidate(
            url="https://careersat.tech/jobs/honeywell-aerospace-internship-2028",
            url_path="/jobs/honeywell-aerospace-internship-2028",
            text="Honeywell Aerospace Internship 2028 – Software Engineering Intern",
            context="Honeywell Aerospace Bangalore, Hyderabad 1d ago Internship 0–1 yrs Batch 2028 ₹45,000/month",
            region="article",
            tag_path="main > div > article > a",
            sibling_count=20,
            source_portal="careersat",
        ),
        DOMCandidate(
            url="https://thejobcompany.co.in/job-category/batch/2028/page/2",
            url_path="/job-category/batch/2028/page/2",
            text="Next Page 2 - 2028 Batch Jobs",
            context="Pagination Page 1 2 3 Next Filter by Batch 2027 2028",
            region="main",
            tag_path="main > nav > a",
            sibling_count=5,
            source_portal="thejobcompany",
        ),
        DOMCandidate(
            url="https://whatsapp.com/channel/0029VaUJgMW2kNFx7ABlpx2y",
            url_path="/channel/0029VaUJgMW2kNFx7ABlpx2y",
            text="Join WhatsApp channel",
            context="WhatsApp channel 52,000+ followers Verified fresher jobs, internships and study resources. Free, no spam.",
            region="main",
            tag_path="main > div > a",
            sibling_count=1,
            source_portal="careersat",
        ),
        DOMCandidate(
            url="https://careersat.tech/interview-questions/dsa",
            url_path="/interview-questions/dsa",
            text="Free interview prep — 400 questions freshers get asked",
            context="Practice mock interviews, resume prompts and coding questions from our blog",
            region="main",
            tag_path="main > div > a",
            sibling_count=1,
            source_portal="careersat",
        ),
        DOMCandidate(
            url="https://careersat.tech/privacy-policy",
            url_path="/privacy-policy",
            text="Privacy Policy & Terms of Service",
            context="About Contact Privacy Terms DMCA Copyright 2026",
            region="footer",
            tag_path="footer > nav > a",
            sibling_count=8,
            source_portal="careersat",
        ),
    ]

    classified, embeddings = clf.classify_candidates(candidates)
    assert embeddings.shape == (5, 384)
    assert classified[0].predicted_label == "job_listing"
    assert classified[1].predicted_label == "pagination_or_filter"
    assert classified[2].predicted_label == "noise"
    assert classified[3].predicted_label == "noise"
    assert classified[4].predicted_label == "noise"


def test_context_prefilter_and_cross_encoder_reranker():
    req = JobSearchRequirement(
        raw_query="2028 internship software engineering bangalore",
        role_keywords=["Software Engineering", "SDE"],
        job_type="internship",
        target_batches=["2028"],
        experience_level="0 years / Student",
        locations=["Bengaluru"],
    )
    senior_job = DOMCandidate(
        url="https://careersat.tech/jobs/senior-java-architect",
        url_path="/jobs/senior-java-architect",
        text="Senior Staff Java Architect",
        context="5-8 yrs experience required | Batch 2018-2021 only | Full-time",
        region="article",
        sibling_count=15,
        predicted_label="job_listing",
    )
    matching_intern = DOMCandidate(
        url="https://careersat.tech/jobs/rubrik-swe-intern-2028",
        url_path="/jobs/rubrik-swe-intern-2028",
        text="Rubrik Software Engineering Intern 2028",
        context="Bengaluru | Internship | Batch 2028 | Python, C++ | ₹1,00,000/month",
        region="article",
        sibling_count=15,
        predicted_label="job_listing",
    )
    unrelated_intern = DOMCandidate(
        url="https://careersat.tech/jobs/hr-marketing-intern-mumbai",
        url_path="/jobs/hr-marketing-intern-mumbai",
        text="HR & Social Media Marketing Intern",
        context="Mumbai | Internship | Batch 2028 | Non-tech role",
        region="article",
        sibling_count=15,
        predicted_label="job_listing",
    )

    assert passes_card_context_prefilter(senior_job, req) is False
    assert passes_card_context_prefilter(matching_intern, req) is True

    reranker = CrossEncoderReranker()
    ranked = reranker.rerank([unrelated_intern, matching_intern], req, top_k=2)
    assert ranked[0].url == matching_intern.url
    assert ranked[0].relevance_score > ranked[1].relevance_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scraper.ml_classifier'`

- [ ] **Step 3: Implement `scraper/ml_classifier.py`, `scraper/reranker.py`, and `scripts/train_classifier.py`**

In `scraper/ml_classifier.py`:
- Load `sentence-transformers/all-MiniLM-L6-v2` as a singleton so repeated calls reuse the in-memory model.
- Define 20 rich seed prototypes for each of `"job_listing"`, `"pagination_or_filter"`, and `"noise"` (including WhatsApp/Telegram channels, interview prep promos, blog links, resume builders, footers, login/signup links, and sitemaps).
- Combine cosine similarity to the 3 class centroids + trained `LogisticRegression` probabilities (if `models/noise_classifier.pkl` exists) + structural region/URL guardrails (e.g., `whatsapp.com`, `t.me`, `/interview-questions`, `/resume-prompts`, `/blog/`, `/privacy`, `/terms`, `/login` strongly boost `noise`).
- In `scripts/train_classifier.py`, fit a calibrated `LogisticRegression` on the seed prototypes + any labeled samples in `data/labeled_candidates.json` and save `models/noise_classifier.pkl` and `models/centroids.npy`.

In `scraper/reranker.py`:
- Implement `passes_card_context_prefilter(candidate, req)` to reject candidates whose text/context explicitly requires `3+ years` experience (e.g., `\b[3-9]\s*[-–to]+\s*\d+\s*(?:yrs?|years?)\b`) or exclusively old batches (`2020`–`2025`) when the user is searching for a fresher/internship batch like `2028`.
- Implement `CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2` to score and sort surviving candidates.

- [ ] **Step 4: Run `scripts/train_classifier.py` and run tests to verify they pass**

Run: `python -m scripts.train_classifier && pytest tests/test_ml_pipeline.py -v`
Expected: `models/noise_classifier.pkl` created and PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scraper/ml_classifier.py scraper/reranker.py scripts/train_classifier.py models/ tests/test_ml_pipeline.py
git commit -m "feat: add Stage 1 Bi-Encoder classifier and Stage 2 Cross-Encoder reranker"
```

---

### Task 4: 4-Layer Data Cleaner, JSON-LD / Trafilatura Detail Extractor & Batched `glm-5.3-flash` Extractor

**Files:**
- Create: `scraper/cleaner.py`
- Create: `scraper/detail_extractor.py`
- Create: `scraper/llm_extractor.py`
- Test: `tests/test_cleaner_and_detail_extractor.py`

**Interfaces:**
- Consumes: `JobRecord`, `JobSearchRequirement` from `scraper.schemas`, `Settings` from `scraper.config`
- Produces:
  - `scraper.cleaner.clean_text_block(text: str) -> str`
  - `scraper.cleaner.normalize_locations(locations: list[str] | str) -> list[str]`
  - `scraper.cleaner.normalize_job_record(record: JobRecord) -> Optional[JobRecord]`
  - `scraper.detail_extractor.RawJobDetail` (`source_url`, `source_portal`, `title`, `company`, `json_ld_data`, `cleaned_markdown`, `outbound_apply_url`, `relevance_score`)
  - `scraper.detail_extractor.extract_job_detail_from_html(html: str, source_url: str, source_portal: str, relevance_score: float = 0.0) -> RawJobDetail`
  - `scraper.llm_extractor.StructuredJobExtractor` (`extract_batch(details: list[RawJobDetail], req: JobSearchRequirement) -> list[JobRecord]`)

- [ ] **Step 1: Write the failing test in `tests/test_cleaner_and_detail_extractor.py`**

```python
from scraper.cleaner import clean_text_block, normalize_locations, normalize_job_record
from scraper.detail_extractor import extract_job_detail_from_html
from scraper.llm_extractor import StructuredJobExtractor
from scraper.schemas import JobSearchRequirement


DETAIL_HTML_WITH_JSONLD_AND_PROMO = """
<html>
<head>
  <title>Rubrik Software Engineering Intern 2028 | TheJobCompany</title>
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "JobPosting",
        "title": "Software Engineering Intern (2028 Batch)",
        "hiringOrganization": {"@type": "Organization", "name": "Rubrik Hiring Off Campus"},
        "jobLocation": {"@type": "Place", "address": {"addressLocality": "Bangalore"}},
        "baseSalary": {"@type": "MonetaryAmount", "value": "₹1,00,000 / month"},
        "employmentType": "INTERN",
        "datePosted": "2026-11-04",
        "description": "<p>Design and build distributed systems in Python and C++.</p><p>Join our WhatsApp group for daily job updates!</p>"
      }
    ]
  }
  </script>
</head>
<body>
  <article>
    <h1>Rubrik Software Engineering Intern (2028 Batch)</h1>
    <p>Eligible Batch: 2028 Passouts | Degree: B.E./B.Tech CS/IT</p>
    <p>Skills: Python, C++, Distributed Systems, Linux</p>
    <p>Join our Telegram channel https://t.me/fresherjobs for instant alerts!</p>
    <a href="https://www.rubrik.com/careers/job/12345">Apply Now on Company Site</a>
  </article>
</body>
</html>
"""


def test_clean_text_strips_promos_and_html():
    dirty = "<p>Build backend APIs.</p>\nJoin our WhatsApp channel for daily updates!\nFollow us on Telegram @jobs"
    cleaned = clean_text_block(dirty)
    assert "Build backend APIs." in cleaned
    assert "WhatsApp" not in cleaned
    assert "Telegram" not in cleaned
    assert "<p>" not in cleaned


def test_extract_job_detail_and_structured_record():
    detail = extract_job_detail_from_html(
        DETAIL_HTML_WITH_JSONLD_AND_PROMO,
        source_url="https://thejobcompany.co.in/jobs/rubrik-intern-2028",
        source_portal="thejobcompany",
        relevance_score=0.95,
    )
    assert detail.title == "Software Engineering Intern (2028 Batch)"
    assert detail.outbound_apply_url == "https://www.rubrik.com/careers/job/12345"
    assert "WhatsApp" not in detail.cleaned_markdown

    req = JobSearchRequirement(raw_query="2028 internship software engineering india", target_batches=["2028"])
    extractor = StructuredJobExtractor(use_llm=False)  # tests deterministic + JSON-LD extraction path
    records = extractor.extract_batch([detail], req)
    assert len(records) == 1
    rec = records[0]
    assert rec.company == "Rubrik"
    assert rec.location == ["Bengaluru"]
    assert "2028" in rec.batch_eligible
    assert rec.apply_url == "https://www.rubrik.com/careers/job/12345"
    assert "WhatsApp" not in rec.job_description
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cleaner_and_detail_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scraper.cleaner'`

- [ ] **Step 3: Implement `scraper/cleaner.py`, `scraper/detail_extractor.py`, and `scraper/llm_extractor.py`**

In `scraper/cleaner.py`:
- Strip HTML tags, unescape entities, remove zero-width chars, and strip promotional lines matching WhatsApp/Telegram/Instagram/Newsletter/Interview-prep boilerplate regexes.
- Normalize Indian city names (`"bangalore"` -> `"Bengaluru"`, `"gurgaon"` -> `"Gurugram"`, `"bombay"` -> `"Mumbai"`, `"ncr"` -> `"Delhi NCR"`).
- Strip company suffixes like `"Hiring"`, `"Off Campus Drive"`, `"Recruitment"`.
- Validate `JobRecord` in `normalize_job_record`: reject empty/noise titles (`"Home"`, `"Jobs"`, `"Login"`), fill missing fields with `"Not Disclosed"` or `"Not Specified"`, and recompute `job_id`.

In `scraper/detail_extractor.py`:
- Parse `<script type="application/ld+json">` (including `@graph` arrays and lists) for `@type == "JobPosting"`.
- Extract main body text via `trafilatura.extract` + BeautifulSoup `<article>`/`<main>` fallback so batch/skills/stipend lines in the body are always captured.
- Detect external apply links (`<a>` with text matching `Apply` or pointing to ATS/external career domains, excluding social media/WhatsApp/Telegram links).

In `scraper/llm_extractor.py`:
- Implement `StructuredJobExtractor` using `openai.OpenAI(api_key=settings.glm_api_key, base_url=settings.glm_base_url)` with model `glm-5.3-flash` to batch-extract 5–8 `RawJobDetail` items into `JobRecord` objects, plus a deterministic JSON-LD + regex parser fallback when `use_llm=False` or when no API key is configured.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cleaner_and_detail_extractor.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scraper/cleaner.py scraper/detail_extractor.py scraper/llm_extractor.py tests/test_cleaner_and_detail_extractor.py
git commit -m "feat: add 4-layer data cleaner, JSON-LD/trafilatura detail extractor, and GLM-5.3-flash extractor"
```

---

### Task 5: Stage 0 Conversational CLI Intake Agent, 16-Portal Discovery & End-to-End Pipeline

**Files:**
- Create: `scraper/cli_agent.py`
- Create: `scraper/discovery.py`
- Create: `scraper/pipeline.py`
- Create: `main.py`
- Test: `tests/test_discovery_and_agent.py`

**Interfaces:**
- Consumes: `TARGET_PORTALS`, `Settings` from `scraper.config`, `JobSearchRequirement`, `JobRecord` from `scraper.schemas`, `SeenJobsStore` from `scraper.dedup`, `HybridFetcher` from `scraper.fetcher`, `extract_dom_candidates` from `scraper.extractor`, `BiEncoderClassifier` from `scraper.ml_classifier`, `CrossEncoderReranker` from `scraper.reranker`, `extract_job_detail_from_html` from `scraper.detail_extractor`, `StructuredJobExtractor` from `scraper.llm_extractor`
- Produces:
  - `scraper.cli_agent.IntakeAgent` (`parse_initial_query(user_input: str) -> JobSearchRequirement`, `get_missing_questions(req: JobSearchRequirement) -> Optional[str]`, `apply_user_clarification(req: JobSearchRequirement, reply: str) -> JobSearchRequirement`)
  - `scraper.discovery.build_16_site_search_queries(req: JobSearchRequirement) -> list[str]`
  - `scraper.discovery.build_direct_portal_urls(req: JobSearchRequirement) -> list[tuple[str, str]]`
  - `scraper.discovery.discover_candidate_urls(req: JobSearchRequirement, seen_store: SeenJobsStore, max_ddg_results_per_query: int = 8) -> list[tuple[str, str]]`
  - `scraper.pipeline.JobScraperPipeline` (`async run(req: JobSearchRequirement, max_detail_pages: int = 20) -> dict`)

- [ ] **Step 1: Write the failing test in `tests/test_discovery_and_agent.py`**

```python
from scraper.cli_agent import IntakeAgent
from scraper.discovery import build_16_site_search_queries, build_direct_portal_urls
from scraper.config import TARGET_PORTALS


def test_intake_agent_parses_query_and_asks_clarification():
    agent = IntakeAgent(use_llm=False)
    req = agent.parse_initial_query("2028 internship software engineering india")
    assert req.job_type == "internship"
    assert "2028" in req.target_batches
    assert any("software" in r.lower() for r in req.role_keywords)

    # Should ask about preferred cities/work mode or tech stack/stipend
    question = agent.get_missing_questions(req)
    assert question is not None

    updated = agent.apply_user_clarification(req, "Bangalore or Gurgaon, Remote ok, Python and C++, min 30k stipend")
    assert "Bengaluru" in updated.locations
    assert "Gurugram" in updated.locations
    assert "Python" in updated.skills


def test_discovery_queries_cover_all_16_sites():
    agent = IntakeAgent(use_llm=False)
    req = agent.parse_initial_query("2028 internship software engineering bangalore")
    queries = build_16_site_search_queries(req)
    combined_queries = " ".join(queries)

    # Every single one of the 16 portal domains or IDs must appear in the generated search queries
    for portal in TARGET_PORTALS:
        assert (portal.domain in combined_queries) or (portal.id in combined_queries)

    direct_urls = build_direct_portal_urls(req)
    assert len(direct_urls) >= 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery_and_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scraper.cli_agent'`

- [ ] **Step 3: Implement `scraper/cli_agent.py`, `scraper/discovery.py`, `scraper/pipeline.py`, and `main.py`**

In `scraper/cli_agent.py`:
- Parse initial query using `glm-5.3-flash` (with deterministic regex/keyword fallback when no API key is set) into `JobSearchRequirement`.
- Check if specific cities, work mode, skills, or minimum stipend were omitted; generate a concise 1-turn follow-up prompt so the user can specify them or press Enter/`"any"`.

In `scraper/discovery.py`:
- Implement `build_16_site_search_queries(req)` generating grouped `site:` DuckDuckGo queries across all 16 `TARGET_PORTALS` (4 portals per query = 4 grouped `site:` queries) + portal-name keyword queries + `build_direct_portal_urls(req)` returning direct search/entry URLs for all 16 portals.
- Implement `discover_candidate_urls` using `ddgs` (`DDGS().text(...)`) + direct portal URLs, deduplicating against `SeenJobsStore`.

In `scraper/pipeline.py` & `main.py`:
- Wire Stages 1–4 together in `JobScraperPipeline.run(req)`:
  1. Discover URLs across all 16 portals.
  2. Fetch listing pages, extract `DOMCandidate` items, classify with `BiEncoderClassifier`, pre-filter card context, and rank with `CrossEncoderReranker`.
  3. Follow top surviving `job_listing` URLs (and 1-hop batch/category `pagination_or_filter` URLs when needed), skipping any URL already in `SeenJobsStore`.
  4. Extract detail pages via `extract_job_detail_from_html` and structure via `StructuredJobExtractor`.
  5. Save deduplicated clean records to `jobs.json` and update `data/seen_jobs.json`.
- Build `main.py` with an interactive CLI loop (`--query` optional flag for one-shot mode, or interactive chat mode by default).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_discovery_and_agent.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scraper/cli_agent.py scraper/discovery.py scraper/pipeline.py main.py tests/test_discovery_and_agent.py
git commit -m "feat: add CLI intake agent, 16-portal discovery engine, and end-to-end scraper pipeline"
```

---

### Task 6: 2D Vector Space Visualizer & Live Portal Benchmark Script

**Files:**
- Create: `scripts/visualize_vector_space.py`
- Create: `scripts/benchmark_sites.py`
- Test: `tests/test_visualizer.py`

**Interfaces:**
- Consumes: `BiEncoderClassifier` from `scraper.ml_classifier`, `DOMCandidate` from `scraper.schemas`, `extract_dom_candidates` from `scraper.extractor`, `HybridFetcher` from `scraper.fetcher`
- Produces:
  - `scripts.visualize_vector_space.generate_vector_space_plots(candidates: list[DOMCandidate], embeddings: np.ndarray, centroids: dict[str, np.ndarray], html_path: Path, png_path: Path) -> tuple[Path, Path]`
  - `scripts.benchmark_sites.run_benchmark(urls: list[str]) -> dict`

- [ ] **Step 1: Write the failing test in `tests/test_visualizer.py`**

```python
from pathlib import Path
from scraper.ml_classifier import BiEncoderClassifier
from scraper.extractor import extract_dom_candidates
from scripts.visualize_vector_space import generate_vector_space_plots
from tests.test_extractor_and_fetcher import SAMPLE_CAREERSAT_HTML


def test_generate_vector_space_plots_creates_html_and_png(tmp_path: Path):
    clf = BiEncoderClassifier()
    candidates = extract_dom_candidates(SAMPLE_CAREERSAT_HTML, "https://careersat.tech/jobs", "careersat")
    classified, embeddings = clf.classify_candidates(candidates)
    centroids = clf.get_centroids()

    html_out = tmp_path / "vector_space.html"
    png_out = tmp_path / "vector_space.png"
    generate_vector_space_plots(classified, embeddings, centroids, html_out, png_out)

    assert html_out.exists() and html_out.stat().st_size > 500
    assert png_out.exists() and png_out.stat().st_size > 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.visualize_vector_space'`

- [ ] **Step 3: Implement `scripts/visualize_vector_space.py` and `scripts/benchmark_sites.py`**

In `scripts/visualize_vector_space.py`:
- Project candidate 384-d embeddings + the 3 class centroids (`job_listing`, `pagination_or_filter`, `noise`) into 2D using `PCA(n_components=2)` (or UMAP when `n_samples >= 15` and `umap-learn` is installed).
- Render an interactive Plotly scatter plot (`plots/vector_space.html`) with hover tooltips (`text`, `region`, `url_path`, `confidence`) and a high-resolution Matplotlib scatter plot (`plots/vector_space.png`) with green (`job_listing`), blue (`pagination_or_filter`), and red (`noise`) clusters + star markers for centroids.

In `scripts/benchmark_sites.py`:
- Fetch live pages from target portals, run Stage 1 & Stage 2 ML classification, print a clean before/after table of raw links vs. kept `job_listing` vs. `pagination_or_filter` vs. dropped `noise`, and save `plots/vector_space.html` and `plots/vector_space.png`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/visualize_vector_space.py scripts/benchmark_sites.py tests/test_visualizer.py
git commit -m "feat: add 2D vector space visualizer (HTML/PNG) and live site benchmark script"
```

