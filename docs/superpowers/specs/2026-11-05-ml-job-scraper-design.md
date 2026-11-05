# Design Document: ML-Powered Job & Internship Web Scraper + Conversational CLI Agent

**Date:** 2026-11-05  
**Target Workspace:** `/Users/khush/Documents/Experiments/InterAT 2.0`  
**Primary LLM:** `glm-5.3-flash` (via OpenAI-compatible SDK, configurable via environment variables)  
**Local ML Models:**
- Stage 1 Bi-Encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-d embeddings)
- Stage 2 Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 1. Intent & Why

Job and internship seekers in India must navigate 16+ fragmented portals (`careersat.tech`, `thejobcompany.co.in`, `internshala.com`, `unstop.com`, `cuvette.tech`, `wellfound.com`, etc.) that mix real job listings with aggressive in-feed noise: WhatsApp/Telegram channel promos, interview prep ads, resume builder banners, blog links, duplicate mobile/desktop navbars, and massive SEO footers. Traditional rule-based CSS/DOM cleaners break whenever a site changes its markup or Tailwind utility hashes.

We are building an end-to-end **Conversational CLI AI Agent & Two-Stage ML Job Scraper** that:
1. Talks with the user in a terminal CLI chat to gather all necessary job search criteria before scraping.
2. Searches across all 16 target job portals (combining site-specific DuckDuckGo queries + direct portal URLs) while tracking previously scraped jobs so no listing is ever scraped or returned twice.
3. Uses a **Two-Stage Local ML Pipeline** (Bi-Encoder Vector Space Classifier + Cross-Encoder Re-Ranker) to separate real `job_listing` and `pagination_or_filter` links from `noise` with high accuracy and zero per-link LLM cost.
4. Enforces **strict multi-stage data cleaning** at every step—from DOM extraction to JSON-LD/Markdown sanitization to Pydantic post-validation—so that `jobs.json` contains only clean, verified, deduplicated, and human-readable job records.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Interactive CLI Agent (`glm-5.3-flash`):** Understands initial user queries (e.g., `"2028 internship software engineering india"`), asks concise follow-up questions for missing parameters (preferred cities/remote, minimum stipend/CTC, specific sub-roles or skills, batch/experience confirmation), and proceeds once confirmed.
- **16-Portal Hybrid Discovery:** Incorporates all 16 target portals into search queries (via DuckDuckGo search + direct portal entry/search templates) to discover relevant listing and job detail URLs in ~2–4 seconds.
- **Cross-Run & Intra-Run Deduplication:** Persists seen job URLs and normalized `(company, title, portal)` fingerprints in `data/seen_jobs.json` so already-scraped jobs from a site are never re-fetched, re-processed by the LLM, or re-listed.
- **Stage 1 Bi-Encoder Vector Space Classifier (`all-MiniLM-L6-v2`):** Maps DOM candidates into a 384-d vector space and classifies them into `job_listing`, `pagination_or_filter`, or `noise` using prototype centroids + a trainable classifier head (`LogisticRegression` / `MLPClassifier`).
- **Stage 2 Cross-Encoder Re-Ranker (`ms-marco-MiniLM-L-6-v2`):** Scores surviving `job_listing` and `pagination_or_filter` candidates against the user's exact query and pre-filters obvious mismatches in card context (e.g., skipping `"3-5 yrs"` or `"Batch 2024"` when the user asked for `"2028 internship"`).
- **Deep Clean Data Pipeline:** Extracts full job details via `application/ld+json` (`JobPosting`) first, falling back to `trafilatura` + BeautifulSoup main-content extraction, strips all promotional boilerplate, and uses batched `glm-5.3-flash` structured extraction with strict Pydantic validation to produce clean records in `jobs.json`.
- **2D Vector Space Visualizer:** Provides `scripts/visualize_vector_space.py` and `scripts/benchmark_sites.py` to plot the 384-d embeddings in 2D (PCA/UMAP HTML + PNG) showing the separation of `job_listing`, `pagination_or_filter`, and `noise` clusters.

### Non-Goals
- Building a web frontend UI or browser extension (interaction is via terminal CLI chat for now).
- Automating login/authenticated scraping behind hard login walls or solving CAPTCHAs (per security/anti-bot guidelines, any CAPTCHA or auth wall is skipped cleanly and logged).

### Anti-Goals (What Would Make This a Failure Even If It Runs)
- **Polluted / Dirty Output Data:** Letting HTML tags, raw Unicode garbage, `"Join our WhatsApp/Telegram channel"` text, cookie banners, or `"Apply on company website"` placeholder strings leak into `job_description`, `title`, or `company`.
- **Hallucinated Fields:** Having the LLM invent stipends, batches, or company names when not present in the source text (missing fields must be normalized to `"Not Disclosed"` or `null`, never guessed).
- **Re-scraping Seen Jobs:** Scraping and listing the exact same job URL or `(company, title, source_portal)` across repeated runs.
- **Brittle Site-Specific CSS Selectors for Noise Removal:** Hardcoding site-specific CSS class hashes instead of relying on structural DOM features + the Bi-Encoder vector space classifier.

---

## 3. Target Portals Configuration (`scraper/config.py`)

All 16 portals are registered in `TARGET_PORTALS`:

```json
[
  {"id": "thejobcompany", "name": "The Job Company", "domain": "thejobcompany.co.in", "entry_url": "https://thejobcompany.co.in/", "render": "auto"},
  {"id": "careersat", "name": "CareersAt.Tech", "domain": "careersat.tech", "entry_url": "https://careersat.tech/jobs", "render": "always"},
  {"id": "hiredoor", "name": "Hiredoor", "domain": "hiredoor.in", "entry_url": "https://hiredoor.in/", "render": "always"},
  {"id": "freshersworld", "name": "Freshersworld", "domain": "freshersworld.com", "entry_url": "https://www.freshersworld.com/jobs/category/internship-job-vacancies", "render": "always"},
  {"id": "offcampus", "name": "OffCampusJobs4u", "domain": "offcampusjobs4u.com", "entry_url": "https://offcampusjobs4u.com/", "render": "auto"},
  {"id": "talentbattle", "name": "Talent Battle", "domain": "talentbattle.in", "entry_url": "https://talentbattle.in/", "render": "auto"},
  {"id": "internshala", "name": "Internshala", "domain": "internshala.com", "entry_url": "https://internshala.com/internships/", "render": "always"},
  {"id": "unstop", "name": "Unstop", "domain": "unstop.com", "entry_url": "https://unstop.com/internship", "render": "always"},
  {"id": "wellfound", "name": "Wellfound", "domain": "wellfound.com", "entry_url": "https://wellfound.com/jobs", "render": "auto"},
  {"id": "instahyre", "name": "Instahyre", "domain": "instahyre.com", "entry_url": "https://www.instahyre.com/search-jobs/", "render": "always"},
  {"id": "cutshort", "name": "Cutshort", "domain": "cutshort.io", "entry_url": "https://cutshort.io/jobs", "render": "auto"},
  {"id": "hasjob", "name": "Hasjob", "domain": "hasjob.co", "entry_url": "https://hasjob.co/", "render": "always"},
  {"id": "indeed", "name": "Indeed India", "domain": "in.indeed.com", "entry_url": "https://in.indeed.com/jobs", "render": "always"},
  {"id": "linkedin", "name": "LinkedIn Jobs", "domain": "linkedin.com", "entry_url": "https://www.linkedin.com/jobs/search/", "render": "never"},
  {"id": "cuvette", "name": "Cuvette", "domain": "cuvette.tech", "entry_url": "https://cuvette.tech/", "render": "auto"},
  {"id": "prosple", "name": "Prosple India", "domain": "in.prosple.com", "entry_url": "https://in.prosple.com/", "render": "auto"}
]
```

---

## 4. End-to-End Pipeline Architecture

### Stage 0: Conversational CLI Intake Agent (`scraper/cli_agent.py`)
1. User enters a natural language request in the terminal (e.g., `"2028 internship software engineering india"`).
2. The agent (`glm-5.3-flash`) parses the input into a `JobSearchRequirement` state object:
   - `role_keywords`: `list[str]` (e.g., `["Software Engineer", "SDE", "Backend", "Full Stack"]`)
   - `job_type`: `"internship" | "full_time" | "both"`
   - `target_batches`: `list[str]` (e.g., `["2028"]`)
   - `experience_level`: `str` (e.g., `"Fresher / 0 years / Student"`)
   - `locations`: `list[str]` (e.g., `["India", "Bangalore", "Hyderabad", "Gurgaon", "Remote"]`)
   - `work_mode`: `"onsite" | "remote" | "hybrid" | "any"`
   - `min_stipend_or_salary`: `Optional[str]`
   - `skills`: `list[str]`
3. **Interactive Clarification Loop:**
   - If key fields (`locations` specificity, `work_mode`, `skills`, or `min_stipend_or_salary`) were not mentioned, the agent displays what it already understood and asks **one concise question** asking the user to fill in any preferences (or press Enter / type `"any"` to use smart defaults).
   - Once the user replies (or says `"start"` / `"any"`), the finalized `JobSearchRequirement` is locked in.

### Stage 1: Hybrid Link Discovery & Deduplication Gate (`scraper/discovery.py` + `scraper/dedup.py`)
1. **Query Generation (`glm-5.3-flash` + Deterministic Site Templates):**
   - Generates targeted search queries combining the user's `JobSearchRequirement` with all 16 portal domains/names.
   - To maximize recall without hitting search query length limits, the 16 sites are grouped into 4 site-batched DuckDuckGo queries (4 `site:` domains per query) + portal-name keyword queries (e.g., `"thejobcompany" OR "offcampusjobs4u" OR "careersat.tech" 2028 software engineering internship`) + direct entry/search URLs from the 16 portals.
2. **Async DuckDuckGo Search (`duckduckgo-search` / `ddgs` + HTTP fallback):**
   - Executes the queries concurrently and collects candidate URLs.
3. **URL Normalization & Pre-Fetch Deduplication (`scraper/dedup.py`):**
   - Normalizes all URLs: strips tracking parameters (`utm_*`, `ref`, `fbclid`, `gclid`, `trk`), normalizes trailing slashes, lowercases hostnames.
   - Checks `SeenJobsStore` (`data/seen_jobs.json`):
     - If a URL is a **job detail URL** that was already scraped in a previous session, it is skipped immediately before any HTTP request is made.
     - Listing/index URLs (e.g., `https://careersat.tech/jobs`) are always allowed to be fetched so newly posted job cards on those pages can be discovered, while individual job cards pointing to already-scraped detail URLs are filtered out before Stage 3.

### Stage 2: Hybrid Fetcher, Structural DOM Extractor & Two-Stage Local ML (`scraper/fetcher.py`, `scraper/extractor.py`, `scraper/ml_classifier.py`, `scraper/reranker.py`)
1. **Hybrid Fetcher (`scraper/fetcher.py`):**
   - `render="never"`: Fast async `httpx` client with realistic browser headers and retry/backoff.
   - `render="always"`: Shared async Playwright Chromium browser context with route interception blocking heavy media/images/fonts (`*.png`, `*.jpg`, `*.mp4`, `*.woff2`) for 3–5x faster rendering.
   - `render="auto"`: Fetches via `httpx` first; automatically escalates to Playwright if the body text is sparse (< 200 chars), `#root`/`#__next`/`#app` is empty, or a client-side render shell is detected.
2. **Structural DOM Candidate Extractor (`scraper/extractor.py`):**
   - Strips `<script>`, `<style>`, `<noscript>`, `<svg>`, `<canvas>`, `<iframe>`, `<template>`, `[hidden]`, `[aria-hidden="true"]`, and HTML comments.
   - **Outer-First Region Resolution:** Walks each element's ancestors from outermost to innermost with explicit priority (`footer` and `aside` take precedence over nested `nav` or `div`, so `<nav aria-label="Footer">` inside `<footer>` is guaranteed `region="footer"`).
   - Computes structural features for every candidate `<a>` / card block:
     - `region`: `article` | `main` | `aside` | `nav` | `header` | `footer` | `body`
     - `tag_path`: simplified ancestor signature (e.g., `main > div > article > a`)
     - `sibling_count`: count of sibling containers in the same parent sharing the same tag/structural signature
     - `text`: cleaned anchor/heading text
     - `context`: cleaned parent card text (up to 400 chars: company, city, stipend/CTC, batch, experience, posted time)
     - `url_path`: normalized path + non-tracking query string
   - **Page-Level URL Deduplication:** When the same normalized URL appears multiple times on a page (e.g., mobile + desktop nav, or card title + "Apply" button), merges them by keeping the candidate in the highest-priority region (`article` > `main` > `body` > `aside` > `nav` > `header` > `footer`) and combining/retaining the richest `context`.
3. **Stage 1 ML: Bi-Encoder Vector Space Classifier (`scraper/ml_classifier.py`):**
   - Formats each candidate into:
     `"[REGION] {region} | [SIBLINGS] {sibling_count} | [TEXT] {text} | [CONTEXT] {context} | [URL] {url_path}"`
   - Encodes all candidates on a page in a single batched forward pass using `sentence-transformers/all-MiniLM-L6-v2` into L2-normalized 384-d vectors.
   - Classifies each vector into:
     - `job_listing` (0)
     - `pagination_or_filter` (1)
     - `noise` (2)
   - Uses a hybrid scoring mechanism:
     - **Zero-Shot Prototype Centroids:** 20 rich anchor descriptions per class embedded and averaged into normalized class centroids (`models/centroids.npy`), working out-of-the-box immediately.
     - **Trained Classifier Head (`models/noise_classifier.pkl`):** Calibrated `LogisticRegression` / `MLPClassifier` trained via `scripts/train_classifier.py` on embeddings + structural features (`region` one-hot, `sibling_count`, URL depth). If `models/noise_classifier.pkl` exists, blends classifier probabilities with centroid cosine similarities.
4. **Card-Context Pre-Filter & Stage 2 ML Cross-Encoder Re-Ranker (`scraper/reranker.py`):**
   - **Rule + Regex Pre-Filter on Card Context:** Before fetching detail pages, checks if the card context explicitly contradicts hard user constraints (e.g., if the user requested `2028` internship / fresher, drops cards whose context explicitly states `"3-6 yrs"`, `"5+ years"`, or `"2022/2023/2024 passout only"` without mentioning `2028`).
   - **Cross-Encoder Scoring (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** Scores pairs `[user_search_summary, f"{candidate.text} — {candidate.context} ({candidate.url_path})"]` for all surviving `job_listing` and `pagination_or_filter` candidates.
   - If a discovered URL is a listing/category page, optionally follows the top-ranked `pagination_or_filter` links (e.g., `/job-category/batch/2028`) up to 1 hop to collect direct job detail URLs.
   - Filters out any job detail URL already present in `SeenJobsStore`.

### Stage 3: Deep Detail Extraction & Multi-Layer Data Cleaning (`scraper/detail_extractor.py` + `scraper/cleaner.py`)
To guarantee clean data, every surviving job detail page passes through a **4-Layer Data Cleaning Pipeline**:

1. **Layer 1 — Structured `application/ld+json` (`JobPosting`) Extraction:**
   - Parses all `<script type="application/ld+json">` blocks (handling nested `@graph` arrays and malformed trailing commas).
   - If a `@type == "JobPosting"` object is found, extracts `title`, `hiringOrganization.name`, `jobLocation`, `jobLocationType` (`TELECOMMUTE`), `baseSalary`, `employmentType`, `datePosted`, `validThrough`, `experienceRequirements`, and `description` (stripping embedded HTML tags from `description` into clean plain text/Markdown).
2. **Layer 2 — Main Content Markdown Extraction (`trafilatura` + BeautifulSoup Fallback):**
   - Extracts clean article text from the DOM after stripping all `nav`, `footer`, `aside`, social share widgets, related-jobs carousels, and in-post promotional blocks.
   - Combines JSON-LD fields (when available) with the cleaned body text so batch eligibility (e.g., `"Batch: 2027 / 2028"`) and stipend details mentioned in the body text are never missed.
3. **Layer 3 — Promo & Boilerplate Sanitizer (`scraper/cleaner.py`):**
   - Strips common Indian fresher-portal boilerplate lines before LLM extraction:
     - `"Join our WhatsApp / Telegram group for daily updates..."`
     - `"Follow us on Instagram / LinkedIn..."`
     - `"Click here to practice 400+ interview questions..."`
     - `"Subscribe to our newsletter / turn on notifications..."`
   - Normalizes whitespace, removes zero-width characters (`\u200b`, `\ufeff`), fixes mojibake/HTML entities (`&amp;`, `&nbsp;`, `&#8377;` -> `₹`), and truncates repetitive footer text.
   - Resolves the **true `apply_url`**: if the page is an aggregator post (like `thejobcompany.co.in` or `offcampusjobs4u.com`), extracts the external outbound application link (e.g., Greenhouse, Workday, Lever, Unstop, or company career portal link) alongside the `source_url`.

### Stage 4: Batched `glm-5.3-flash` Structured Extraction & Post-Validation (`scraper/llm_extractor.py` + `scraper/schemas.py`)
1. **Batched Extraction (5–8 jobs per LLM call):**
   - Sends cleaned job detail payloads to `glm-5.3-flash` with a strict JSON schema prompt instructing the model **never to hallucinate** missing fields and to extract only verified details.
2. **Complete Job Record Schema (`JobRecord` in `scraper/schemas.py`):**
   - `job_id`: Deterministic SHA-256 hash of normalized `(company, title, apply_url)`
   - `title`: Clean job title (e.g., `"Software Engineering Intern (Summer 2028)"`)
   - `company`: Clean company name (stripped of `"Hiring"`, `"Off Campus Drive"`, etc.)
   - `job_type`: `"Internship" | "Full-Time" | "Contract" | "Trainee"`
   - `batch_eligible`: `list[str]` (e.g., `["2027", "2028"]` or `["Any / Not Specified"]`)
   - `experience_required`: Normalized string (e.g., `"0 years (Fresher / Student)"`, `"0-1 years"`)
   - `location`: `list[str]` of normalized city names (e.g., `["Bengaluru", "Gurugram"]`)
   - `work_mode`: `"Onsite" | "Remote" | "Hybrid" | "Not Specified"`
   - `stipend_or_salary`: Normalized compensation string (e.g., `"₹40,000 / month"`, `"₹12-15 LPA"`, or `"Not Disclosed"`)
   - `duration`: Internship duration if applicable (e.g., `"2 months (May–July)"`, `"6 months"`, or `null`)
   - `skills_required`: `list[str]` of cleaned technical skills (e.g., `["Python", "Java", "Data Structures", "React"]`)
   - `eligibility_criteria`: Clean summary of degree/branch/CGPA requirements (e.g., `"B.E./B.Tech in CS/IT or related fields, 2028 graduating batch"`)
   - `job_description`: Clean, concise 3–6 bullet/sentence summary of responsibilities and role details (guaranteed free of promo/HTML noise)
   - `posted_date`: ISO date or relative string (`"2026-11-04"` / `"1 day ago"` / `"Not Specified"`)
   - `application_deadline`: Deadline if stated, else `"Not Specified"`
   - `apply_url`: Direct application URL
   - `source_url`: The portal page URL where the job was scraped from
   - `source_portal`: Portal ID (`"careersat"`, `"internshala"`, `"thejobcompany"`, etc.)
   - `relevance_score`: Cross-Encoder relevance score (`float`)
   - `scraped_at`: ISO-8601 timestamp
3. **Layer 4 — Post-LLM Deterministic Validator & Field Normalizer (`scraper/cleaner.py`):**
   - City synonym normalization (`"Bangalore"` -> `"Bengaluru"`, `"Gurgaon"` -> `"Gurugram"`).
   - Rejects any record where `title` or `company` is empty, generic (`"Home"`, `"Jobs"`, `"Sign In"`), or matches known noise patterns.
   - Deduplicates across the batch and against `SeenJobsStore`, appends new records to `jobs.json`, and updates `data/seen_jobs.json`.

---

## 5. Project File Structure

```text
InterAT 2.0/
├── pyproject.toml
├── .env.example
├── README.md
├── main.py                          # Entry point for Interactive Terminal CLI Chat Agent
├── scraper/
│   ├── __init__.py
│   ├── config.py                    # 16 Target Portals config, GLM-5.3-flash settings, paths
│   ├── schemas.py                   # Pydantic models: JobSearchRequirement, DOMCandidate, JobRecord
│   ├── cli_agent.py                 # Stage 0: Conversational intake & requirement clarification
│   ├── discovery.py                 # Stage 1: 16-portal DuckDuckGo search + direct URL generator
│   ├── dedup.py                     # Persistent SeenJobsStore (URL + company/title hash dedup)
│   ├── fetcher.py                   # Hybrid async httpx + Playwright Chromium pool
│   ├── extractor.py                 # Outer-first DOM region resolver + structural candidate extractor
│   ├── ml_classifier.py             # Stage 1 ML: Bi-Encoder (all-MiniLM-L6-v2) + Centroids & Classifier Head
│   ├── reranker.py                  # Stage 2 ML: Cross-Encoder (ms-marco-MiniLM-L-6-v2) + context pre-filter
│   ├── detail_extractor.py          # Stage 3: JSON-LD JobPosting parser + trafilatura Markdown extractor
│   ├── cleaner.py                   # 4-layer data sanitizer, promo stripper, city/salary/batch normalizer
│   ├── llm_extractor.py             # Stage 4: Batched glm-5.3-flash structured extraction + fallback parser
│   └── pipeline.py                  # Orchestrates Stages 1 -> 4 end-to-end
├── scripts/
│   ├── train_classifier.py          # Trains & saves LogisticRegression/MLP head on 384-d embeddings
│   ├── visualize_vector_space.py    # Generates 2D PCA/UMAP HTML & PNG plots of the 3 clusters
│   └── benchmark_sites.py           # Live benchmark across target portals with noise-reduction stats
├── models/                          # Saved centroids.npy & noise_classifier.pkl
├── data/                            # seen_jobs.json & cached training samples
├── plots/                           # Generated 2D vector space visualizations (.html, .png)
└── tests/                           # Unit & integration tests for extractor, cleaner, dedup, ML pipeline
```

---

## 6. Design Decisions Left to the Builder (Default Calls)
- **GLM API Client & Fallback:** Uses the official `openai` Python SDK pointed at the configurable GLM base URL (`GLM_BASE_URL`, defaulting to ZhipuAI's OpenAI-compatible endpoint `https://open.bigmodel.cn/api/paas/v4/`, model `glm-5.3-flash`, with `GLM_API_KEY` from `.env`). Also includes a deterministic heuristic + JSON-LD fallback extractor so the pipeline still produces valid `JobRecord` entries during local testing or if an API key is not yet set.
- **2D Dimensionality Reduction:** Supports both `PCA` + `t-SNE` (from `scikit-learn`) and `UMAP` (if `umap-learn` is installed) and outputs both an interactive `plots/vector_space.html` and a static `plots/vector_space.png`.

