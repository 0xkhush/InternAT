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

BATCH_RE = re.compile(r"\b(202[5-9])\b")
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
        # Try to infer company from title e.g. "Rubrik Software Engineering Intern" or "at Razorpay"
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

    batches = sorted(set(BATCH_RE.findall(combined)))
    if not batches:
        batches = list(req.target_batches) if req.target_batches else ["Not Specified"]

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


class StructuredJobExtractor:
    """Batched `glm-5.3-flash` structured job extractor with deterministic JSON-LD/regex fallback."""

    def __init__(self, settings: Settings | None = None, use_llm: bool = True, batch_size: int = 6) -> None:
        self.settings = settings or Settings()
        self.use_llm = use_llm and bool(self.settings.glm_api_key and self.settings.glm_api_key != "your_glm_api_key_here")
        self.batch_size = batch_size

    def _call_glm_batch(self, batch: list[RawJobDetail], req: JobSearchRequirement) -> list[JobRecord]:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.glm_api_key, base_url=self.settings.glm_base_url)
        items_payload = [
            {
                "index": idx,
                "source_url": d.source_url,
                "source_portal": d.source_portal,
                "outbound_apply_url": d.outbound_apply_url,
                "title": d.title,
                "company": d.company,
                "json_ld": d.json_ld_data,
                "content": d.cleaned_markdown[:1800],
            }
            for idx, d in enumerate(batch)
        ]

        system_prompt = (
            "You are a precise job data extraction engine. Given raw job postings, return a JSON object "
            "with key 'jobs' containing an array of clean job records. Never hallucinate missing fields: "
            "use 'Not Disclosed' for unknown salary/stipend and 'Not Specified' for unknown dates/eligibility. "
            "Strip any promotional WhatsApp/Telegram/social media text from job_description."
        )
        user_prompt = (
            f"User search requirement: {req.to_search_summary()}\n\n"
            f"Extract structured job records for these {len(batch)} items:\n"
            f"{json.dumps(items_payload, ensure_ascii=False)}"
        )

        resp = client.chat.completions.create(
            model=self.settings.glm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
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
            rec = JobRecord(
                title=str(raw_item.get("title") or detail.title),
                company=str(raw_item.get("company") or detail.company or "Not Disclosed"),
                job_type=str(raw_item.get("job_type") or "Internship"),
                batch_eligible=raw_item.get("batch_eligible") or (req.target_batches or ["Not Specified"]),
                experience_required=str(raw_item.get("experience_required") or "0 years (Fresher / Student)"),
                location=raw_item.get("location") or ["India"],
                work_mode=str(raw_item.get("work_mode") or "Not Specified"),
                stipend_or_salary=str(raw_item.get("stipend_or_salary") or "Not Disclosed"),
                duration=raw_item.get("duration"),
                skills_required=raw_item.get("skills_required") or [],
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
        """Extract and validate a batch of RawJobDetail items into clean JobRecords."""
        if not details:
            return []

        records: list[JobRecord] = []
        for i in range(0, len(details), self.batch_size):
            chunk = details[i : i + self.batch_size]
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

