# InterAT — AI-Powered Job & Internship Web Scraper

A clean, modular, next-generation **Job & Internship Web Scraper + Conversational CLI Agent (`glm-5.3-flash`)** across 17 Indian job portals using a **Two-Stage Local ML Pipeline**:

1. **Go Engine (`cmd/agent`, `internal/scraper`):**
   - Interactive CLI Agent & 17-Portal Targeted DuckDuckGo Search + Direct Portal URLs
   - Persistent `SeenJobsStore` (`data/seen_jobs.json`) so jobs from any portal are never re-scraped or duplicated
   - High-speed concurrent HTML fetcher & Outer-First DOM Candidate Extractor (`footer` strictly overrides nested `nav`)
2. **Python ML Layer (`scraper/`, `scripts/`):**
   - **Stage 1 Bi-Encoder (`sentence-transformers/all-MiniLM-L6-v2`):** 384-d embeddings fine-tuned in a **PyTorch Contrastive Pair + Triplet Margin Embedding Space (`ContrastiveProjectionNet`)** and classified via **Nearest Centroid Cosine Similarity** into `job_listing`, `pagination_or_filter`, or `noise`
   - **Stage 2 Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** Re-ranks surviving job candidates against the user's exact query + card-context pre-filtering
   - **Stage 3 & 4 Deep Cleaning & Extraction:** `application/ld+json` (`JobPosting`) + `trafilatura` Markdown extraction + 4-layer promo/boilerplate stripper + batched `glm-5.3-flash` structured output into `jobs.json`

---

## Architecture & Pipeline Diagrams

### 1. End-to-End Multi-Stage Scraping Pipeline

```mermaid
flowchart TD
    U["User Natural-Language Query<br/>e.g. '2028 internship software engineering india'"] --> S0

    subgraph S0["Stage 0: Conversational CLI Intake Agent (cli_agent.py)"]
        P0["Regex Heuristics + GLM-5.3-flash Parser"] --> Q0{"Missing Key Criteria?<br/>(Location / Skills)"}
        Q0 -- "Yes" --> C0["Interactive Clarification Prompt"] --> R0["Locked JobSearchRequirement"]
        Q0 -- "No / --yes" --> R0
    end

    R0 --> S1

    subgraph S1["Stage 1: 17-Portal Hybrid Discovery (discovery.py)"]
        D1["Direct Portal Listing & Batch URLs<br/>(thejobcompany, careersat, interndoor, unstop, hiredoor...)"]
        D2["17 Single-Domain DuckDuckGo Queries<br/>site:portal.domain software engineering internship 2028 (India)"]
        D1 & D2 --> DEDUP1["SeenJobsStore URL Gate<br/>(data/seen_jobs.json)"]
    end

    DEDUP1 --> S2

    subgraph S2["Stage 2: Fetcher, Outer-First DOM Extractor & Two-Stage Local ML"]
        F1["HybridFetcher (fetcher.py)<br/>Async HTTPX + JSON Feeds + Playwright Fallback"] --> E1["Outer-First DOM Candidate Extractor (extractor.py)<br/>Resolves Semantic Region & Card Context"]
        E1 --> ML1["Stage 1 ML: Bi-Encoder + ContrastiveProjectionNet (ml_classifier.py)<br/>384-d MiniLM + 11-d Structural Features -> Nearest Centroid Cosine"]
        ML1 -- "noise" --> DROP1["Dropped (Nav / Footer / Social / Promos)"]
        ML1 -- "pagination_or_filter" --> PAG["1-Hop Pagination & Category Expansion"]
        ML1 -- "job_listing" --> PRE["Card-Context Pre-Filter (reranker.py)<br/>Rejects 3+ Yrs Exp, Senior Titles & Non-Tech Roles"]
        PAG --> E1
        PRE --> ML2["Stage 2 ML: Cross-Encoder Re-Ranker (reranker.py)<br/>ms-marco-MiniLM-L-6-v2 Joint Query-Card Scoring"]
    end

    ML2 --> S3

    subgraph S3["Stage 3 & 4: 4-Layer Cleaning, Detail Extraction & Deduplication"]
        DF["Fetch Top Candidate Detail Pages"] --> L1["Layer 1 & 2: JSON-LD JobPosting + Trafilatura Markdown<br/>& Outbound Apply URL Resolution (detail_extractor.py)"]
        L1 --> L3["Layer 3 & 4: Promo/WhatsApp Stripper, City/Company Normalizer<br/>& Structured Extractor (cleaner.py + llm_extractor.py)"]
        L3 --> BF["Strict Post-Detail Batch & Role Verification"]
        BF --> DEDUP2["3-Key Persistent Deduplication<br/>(URL + JobID + Cross-Portal Fingerprint)"]
    end

    DEDUP2 --> OUT["Clean Validated Output<br/>jobs.json + Terminal Top-20 Display"]
```

### 2. Stage 1 & Stage 2 Local ML Architecture

```mermaid
flowchart LR
    subgraph Input["DOM Candidate Representation"]
        A["DOMCandidate<br/>[REGION] article | [SIBLINGS] 20<br/>[TEXT] Anchor Title | [CONTEXT] Card Snippet | [URL] Path"]
    end

    subgraph Stage1["Stage 1: Bi-Encoder + Contrastive Metric Projection"]
        B1["MiniLM Bi-Encoder<br/>(all-MiniLM-L6-v2)<br/>384-d Semantic Vector"]
        B2["DOM Structural Features<br/>Region One-Hot + Sibling Log + URL Priors<br/>11-d Vector"]
        A --> B1 & B2
        B1 & B2 --> CONCAT["Concatenate (395-d)"]
        CONCAT --> PROJ["ContrastiveProjectionNet (PyTorch)<br/>Linear(395->256) + LayerNorm + GELU + Residual<br/>Trained with CosinePair + TripletMargin Loss"]
        PROJ --> CENT["Nearest Centroid Cosine Similarity<br/>(models/centroids.npy)"]
    end

    CENT -- "job_listing" --> Stage2
    CENT -- "pagination_or_filter" --> HOP["Ranked Pagination Queue"]
    CENT -- "noise" --> NOISE["Discarded"]

    subgraph Stage2["Stage 2: Cross-Encoder Re-Ranker"]
        PF["Rule Pre-Filter<br/>Batch / Seniority / Domain Gate"] --> CE["Cross-Encoder<br/>(ms-marco-MiniLM-L-6-v2)<br/>Scores (User Query, Candidate Card)"]
        CE --> TOPK["Top-K Candidates<br/>Sent to Stage 3 Detail Scraper"]
    end
```

### 3. Go + Python Hybrid Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as CLI Agent (main.py / cmd/agent)
    participant Disc as 17-Portal Discovery + DuckDuckGo
    participant Fetch as HybridFetcher & DOM Extractor
    participant ML as Local ML (Bi-Encoder + Cross-Encoder)
    participant Clean as Detail Extractor & GLM-5.3-flash
    participant Store as SeenJobsStore & jobs.json

    User->>CLI: Run query ("2028 internship software engineering india")
    CLI->>CLI: Parse into JobSearchRequirement (Role, Batch, Locations)
    CLI->>Disc: Generate 17 portal URLs + 17 single-domain DDG queries
    Disc->>Store: Filter out already-seen job detail URLs
    Disc-->>Fetch: Unseen listing & candidate URLs
    Fetch->>Fetch: Concurrent fetch (HTTPX / JSON Feeds / Playwright) + Outer-First DOM parse
    Fetch-->>ML: Raw DOMCandidates (~1,900 links/run)
    ML->>ML: Stage 1 Contrastive Bi-Encoder -> Keep job_listing & pagination
    ML->>ML: Stage 2 Cross-Encoder -> Pre-filter & rank top candidates
    ML-->>Clean: Top-K Ranked Job Candidates
    Clean->>Clean: Extract JSON-LD + Trafilatura Markdown + Outbound Apply URL
    Clean->>Clean: 4-Layer Sanitizer + Strict Batch/Role Filter
    Clean->>Store: Deduplicate (URL, JobID, Cross-Portal Fingerprint) & Persist
    Store-->>User: Save to jobs.json & Display Top 20 Jobs
```

---

## Quick Start

### 1. Train the Contrastive Pair / Triplet Embedding Space & Generate 2D Vector Plots
```bash
# Train ContrastiveProjectionNet (CosineEmbeddingLoss + TripletMarginWithDistanceLoss)
python3 -m scripts.train_classifier

# Or scrape live portals first to augment the training set with real DOM candidates:
python3 -m scripts.train_classifier --scrape-live

# Generate Side-by-Side Before vs. After 2D Vector Space Plots (plots/vector_space.html & .png)
python3 -m scripts.visualize_vector_space
```

### 2. Run the Interactive CLI Agent (Go + Python Hybrid or Python Entrypoint)
```bash
# Run the Python CLI entrypoint with an initial query:
/opt/anaconda3/bin/python main.py --query "2028 internship software engineering india" --yes

# Or run interactively:
/opt/anaconda3/bin/python main.py

# Or run the Go + Python Hybrid Agent:
go run ./cmd/agent -query "2028 internship software engineering india"
```

### 3. Run the Full Test Suite (Go + Python)
```bash
go test ./... -v
/opt/anaconda3/bin/python -m pytest -v
```
