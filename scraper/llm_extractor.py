"""Stage 4: Batched `glm-5.3-flash` Structured Job Extraction with Deterministic JSON-LD/Regex Fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from scraper.cleaner import clean_company_name, clean_text_block, normalize_job_record, normalize_locations
from scraper.config import Settings
from scraper.detail_extractor import RawJobDetail
from scraper.schemas import JobRecord, JobSearchRequirement

logger = logging.getLogger(__name__)

KNOWN_SKILLS = [
    "Python",
    "C++",
    "Java",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "React",
    "Node.js",
    "Next.js",
    "FastAPI",
    "Django",
    "Spring Boot",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
    "Linux",
    "Git",
    "Data Structures",
    "Algorithms",
    "Distributed Systems",
    "Machine Learning",
    "PyTorch",
]

BATCH_LINE_KEYWORDS = ("batch", "passout", "pass out", "graduat", "class of", "eligible", "b.tech", "b.e", "student", "yop")
BATCH_RE = re.compile(r"\b(202[4-9]|2030)\b")


def _extract_batches_from_text(title: str, body: str, default_batches: list[str]) -> list[str]:
    """Extract graduation batch years only from batch/eligibility context (ignoring 2026 posting dates)."""
    found: set[str] = set()
    # Check title first (unless it's 'Recruitment 2026' / 'Drive 2026')
    title_clean = re.sub(r"\b(?:recruitment|drive|hiring|off\s*campus|summer|winter)\s+2026\b", "", title, flags=re.I)
    for m in BATCH_RE.findall(title_clean):
        found.add(m)

    for line in body.splitlines():
        ll = line.lower()
        # Skip posted date / copyright lines
        if any(skip in ll for skip in ("posted", "published", "copyright", "updated on", "date:")):
            continue
        if any(kw in ll for kw in BATCH_LINE_KEYWORDS):
            for yr in BATCH_RE.findall(line):
                found.add(yr)

    if found:
        return sorted(found)
    return list(default_batches) if default_batches else ["Not Specified"]
SALARY_RE = re.compile(
    r"((?:₹|INR|Rs\.?)\s*[\d,.]+\s*(?:-\s*(?:₹|INR|Rs\.?)?\s*[\d,.]+)?\s*(?:LPA|Lakh|K|/\s*month|per\s*month|/mo|pm)?)",
    re.IGNORECASE,
)


def _extract_location_from_jsonld(json_ld: dict[str, Any], body_text: str) -> list[str]:
    locs: list[str] = []
    job_loc = json_ld.get("jobLocation")
    items = job_loc if isinstance(job_loc, list) else ([job_loc] if job_loc else [])
    for item in items:
        if isinstance(item, dict):
            addr = item.get("address")
            if isinstance(addr, dict):
                for key in ("addressLocality", "addressRegion"):
                    val = addr.get(key)
                    if isinstance(val, str) and val.strip():
                        locs.append(val.strip())
            elif isinstance(addr, str):
                locs.append(addr.strip())
    if str(json_ld.get("jobLocationType") or "").upper() == "TELECOMMUTE":
        locs.append("Remote")
    if not locs:
        for city in ("Bengaluru", "Bangalore", "Gurugram", "Gurgaon", "Hyderabad", "Pune", "Mumbai", "Chennai", "Noida", "Delhi", "Remote"):
            if re.search(rf"\b{re.escape(city)}\b", body_text, re.IGNORECASE):
                locs.append(city)
    return normalize_locations(locs)


def _extract_salary_from_jsonld(json_ld: dict[str, Any], body_text: str) -> str:
    base_sal = json_ld.get("baseSalary")
    if isinstance(base_sal, dict):
        val = base_sal.get("value")
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, (int, float)):
            return f"₹{val}"
        if isinstance(val, dict):
            min_v = val.get("minValue")
            max_v = val.get("maxValue")
            unit = val.get("unitText", "")
            if min_v and max_v:
                return f"₹{min_v} - ₹{max_v} {unit}".strip()
            if val.get("value"):
                return f"₹{val['value']} {unit}".strip()
    m = SALARY_RE.search(body_text)
    if m and any(ch.isdigit() for ch in m.group(1)):
        return m.group(1).strip()
    return "Not Disclosed"


def _deterministic_extract_single(detail: RawJobDetail, req: JobSearchRequirement) -> JobRecord | None:
    """Extract a clean JobRecord from JSON-LD + cleaned Markdown without requiring an LLM call."""
    jld = detail.json_ld_data
    body = detail.cleaned_markdown
    combined = f"{detail.title}\n{body}"

    title = detail.title or str(jld.get("title") or "").strip()
    company = detail.company
    if not company or company == "Not Disclosed":
        # Check "Company Name - Role Title" pattern first (e.g. "Wells Fargo - Technology Program Intern")
        if " - " in title:
            prefix = title.split(" - ", 1)[0].strip()
            if 1 <= len(prefix.split()) <= 4 and not any(
                w in prefix.lower() for w in ("intern", "software", "sde", "developer", "engineer")
            ):
                company = clean_company_name(prefix)
        if not company or company == "Not Disclosed":
            m_at = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.\s]{1,30})", title)
            if m_at:
                company = clean_company_name(m_at.group(1))
            else:
                first_word = title.split()[0] if title.split() else "Not Disclosed"
                if first_word.lower() not in ("software", "sde", "backend", "frontend", "full", "data", "ai", "ml", "intern", "junior"):
                    company = clean_company_name(first_word)
                else:
                    company = "Not Disclosed"

    emp_type = str(jld.get("employmentType") or "").upper()
    if "INTERN" in emp_type or "intern" in title.lower() or req.job_type == "internship":
        job_type = "Internship"
    else:
        job_type = "Full-Time"

    batches = _extract_batches_from_text(title, body, req.target_batches)

    locs = _extract_location_from_jsonld(jld, combined)
    work_mode = "Remote" if "Remote" in locs or "remote" in combined.lower() else ("Hybrid" if "hybrid" in combined.lower() else "Onsite")
    stipend = _extract_salary_from_jsonld(jld, combined)

    skills: list[str] = []
    for sk in KNOWN_SKILLS:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(sk)}(?![A-Za-z0-9])", combined, re.IGNORECASE):
            skills.append(sk)

    elig_lines = [
        ln for ln in body.splitlines()
        if any(k in ln.lower() for k in ("batch", "degree", "b.tech", "b.e", "eligible", "qualification", "pursuing"))
    ]
    eligibility = elig_lines[0][:240] if elig_lines else "Not Specified"

    desc_lines = [
        ln for ln in body.splitlines()
        if len(ln) > 20 and not ln.lower().startswith(("eligible batch:", "skills:"))
    ]
    description = " ".join(desc_lines[:4])[:600] if desc_lines else clean_text_block(body)[:600]

    posted_date = str(jld.get("datePosted") or "Not Specified").strip()
    deadline = str(jld.get("validThrough") or "Not Specified").strip()

    raw_record = JobRecord(
        title=title,
        company=company,
        job_type=job_type,
        batch_eligible=batches,
        experience_required="0 years (Fresher / Student)" if job_type == "Internship" else "0-2 years",
        location=locs,
        work_mode=work_mode,
        stipend_or_salary=stipend,
        skills_required=skills,
        eligibility_criteria=eligibility,
        job_description=description or title,
        posted_date=posted_date,
        application_deadline=deadline,
        apply_url=detail.outbound_apply_url or detail.source_url,
        source_url=detail.source_url,
        source_portal=detail.source_portal,
        relevance_score=detail.relevance_score,
    )
    return normalize_job_record(raw_record)


def _to_str_list(val: Any, default: list[str] | None = None) -> list[str]:
    """Coerce a string or list from LLM JSON into a clean list[str]."""
    if val is None:
        return list(default) if default else []
    if isinstance(val, str):
        items = [x.strip() for x in re.split(r"[,/|;]+", val) if x.strip()]
        return items if items else (list(default) if default else [])
    if isinstance(val, list):
        items = [str(x).strip() for x in val if str(x).strip()]
        return items if items else (list(default) if default else [])
    return [str(val).strip()]


class StructuredJobExtractor:
    """Batched `glm-5.3-flash` structured job extractor with deterministic JSON-LD/regex fallback."""

    def __init__(self, settings: Settings | None = None, use_llm: bool = True, batch_size: int = 3) -> None:
        self.settings = settings or Settings()
        self.use_llm = use_llm and bool(self.settings.glm_api_key and self.settings.glm_api_key != "your_glm_api_key_here")
        self.batch_size = batch_size

    def _call_glm_batch(self, batch: list[RawJobDetail], req: JobSearchRequirement) -> list[JobRecord]:
        import httpx

        url = f"{self.settings.glm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.glm_api_key}",
            "Content-Type": "application/json",
        }
        items_payload = [
            {
                "index": idx,
                "title": d.title,
                "company": d.company,
                "content": d.cleaned_markdown[:600],
            }
            for idx, d in enumerate(batch)
        ]

        system_prompt = (
            "Extract job records into JSON: {\"jobs\": [{\"title\": str, \"company\": str, "
            "\"job_type\": \"Internship\"|\"Full-Time\", \"batch_eligible\": [str], \"location\": [str], "
            "\"work_mode\": str, \"stipend_or_salary\": str, \"skills_required\": [str], "
            "\"eligibility_criteria\": str, \"job_description\": str}]}. "
            "Be concise. Never invent missing values; use 'Not Disclosed'. Output raw JSON only."
        )
        user_prompt = json.dumps(items_payload, ensure_ascii=False)

        payload = {
            "model": self.settings.glm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=45.0)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or "{}"
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        parsed = json.loads(content)
        jobs_list = parsed.get("jobs", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])

        results: list[JobRecord] = []
        for idx, detail in enumerate(batch):
            raw_item = jobs_list[idx] if idx < len(jobs_list) and isinstance(jobs_list[idx], dict) else {}
            if not raw_item:
                det = _deterministic_extract_single(detail, req)
                if det:
                    results.append(det)
                continue
            dur_val = raw_item.get("duration")
            rec = JobRecord(
                title=str(raw_item.get("title") or detail.title),
                company=str(raw_item.get("company") or detail.company or "Not Disclosed"),
                job_type=str(raw_item.get("job_type") or "Internship"),
                batch_eligible=_to_str_list(raw_item.get("batch_eligible"), req.target_batches or ["Not Specified"]),
                experience_required=str(raw_item.get("experience_required") or "0 years (Fresher / Student)"),
                location=normalize_locations(_to_str_list(raw_item.get("location"), ["India"])),
                work_mode=str(raw_item.get("work_mode") or "Not Specified"),
                stipend_or_salary=str(raw_item.get("stipend_or_salary") or "Not Disclosed"),
                duration=str(dur_val) if dur_val else None,
                skills_required=_to_str_list(raw_item.get("skills_required"), []),
                eligibility_criteria=str(raw_item.get("eligibility_criteria") or "Not Specified"),
                job_description=str(raw_item.get("job_description") or detail.cleaned_markdown[:400]),
                posted_date=str(raw_item.get("posted_date") or "Not Specified"),
                application_deadline=str(raw_item.get("application_deadline") or "Not Specified"),
                apply_url=str(raw_item.get("apply_url") or detail.outbound_apply_url or detail.source_url),
                source_url=detail.source_url,
                source_portal=detail.source_portal,
                relevance_score=detail.relevance_score,
            )
            norm_rec = normalize_job_record(rec)
            if norm_rec:
                results.append(norm_rec)
        return results

    def extract_batch(
        self,
        details: list[RawJobDetail],
        req: JobSearchRequirement,
    ) -> list[JobRecord]:
        """Extract and validate a batch of RawJobDetail items into clean JobRecords.

        Uses the fast JSON-LD / structured path first (0ms latency for the ~70% of sites with
        schema.org JobPosting or clear structured headers), and sends only unstructured pages
        to batched `glm-5.3-flash`.
        """
        if not details:
            return []

        records: list[JobRecord] = []
        needs_llm: list[RawJobDetail] = []

        for d in details:
            det_rec = _deterministic_extract_single(d, req)
            # If the page has JSON-LD JobPosting OR deterministic extraction resolved both title & company cleanly, keep it directly
            if det_rec is not None and (d.json_ld_data or det_rec.company != "Not Disclosed"):
                records.append(det_rec)
            else:
                needs_llm.append(d)

        if not needs_llm:
            return records

        for i in range(0, len(needs_llm), self.batch_size):
            chunk = needs_llm[i : i + self.batch_size]
            if self.use_llm:
                try:
                    records.extend(self._call_glm_batch(chunk, req))
                    continue
                except Exception as exc:
                    logger.warning("GLM-5.3-flash batch call failed (%s), using deterministic fallback", exc)

            for d in chunk:
                rec = _deterministic_extract_single(d, req)
                if rec is not None:
                    records.append(rec)

        return records

