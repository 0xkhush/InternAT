"""Pydantic schemas for search requirements, DOM candidates, and extracted job records."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class JobSearchRequirement(BaseModel):
    """Structured user search criteria gathered by the Stage 0 intake agent."""

    raw_query: str
    role_keywords: list[str] = Field(default_factory=lambda: ["Software Engineer", "SDE"])
    job_type: Literal["internship", "full_time", "both"] = "internship"
    target_batches: list[str] = Field(default_factory=list)
    experience_level: str = "Fresher / 0 years / Student"
    locations: list[str] = Field(default_factory=lambda: ["India"])
    work_mode: Literal["onsite", "remote", "hybrid", "any"] = "any"
    min_stipend_or_salary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)

    def to_search_summary(self) -> str:
        """Build a concise natural-language query for the Cross-Encoder re-ranker."""
        jt = "Internship" if self.job_type == "internship" else ("Full-Time Job" if self.job_type == "full_time" else "Internship or Full-Time")
        roles = ", ".join(self.role_keywords) if self.role_keywords else "Software Engineering"
        batches = f"for {', '.join(self.target_batches)} batch " if self.target_batches else ""
        locs = ", ".join(self.locations) if self.locations else "India"
        skills_part = f" ({', '.join(self.skills)})" if self.skills else ""
        mode_part = f" [{self.work_mode}]" if self.work_mode != "any" else ""
        return f"{batches}{jt} in {roles}{skills_part} located in {locs}{mode_part}".strip()


class DOMCandidate(BaseModel):
    """Structural and contextual representation of a DOM link/card candidate."""

    url: str
    url_path: str
    text: str
    context: str
    region: Literal["article", "main", "aside", "nav", "header", "footer", "body"] = "body"
    tag_path: str = "body > a"
    sibling_count: int = 1
    source_portal: str = "unknown"
    predicted_label: Optional[Literal["job_listing", "pagination_or_filter", "noise"]] = None
    confidence: float = 0.0
    relevance_score: float = 0.0

    def to_feature_string(self) -> str:
        """Serialize into the exact Bi-Encoder input format required by the spec."""
        return (
            f"[REGION] {self.region} | [SIBLINGS] {self.sibling_count} | "
            f"[TEXT] {self.text} | [CONTEXT] {self.context} | [URL] {self.url_path}"
        )


def compute_job_id(company: str, title: str, apply_url: str) -> str:
    """Compute a deterministic SHA-256 hash ID for a job record."""
    norm_company = " ".join(company.lower().strip().split())
    norm_title = " ".join(title.lower().strip().split())
    norm_url = apply_url.strip().rstrip("/").lower()
    payload = f"{norm_company}|{norm_title}|{norm_url}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class JobRecord(BaseModel):
    """Clean, validated job/internship record written to jobs.json."""

    job_id: str = ""
    title: str
    company: str
    job_type: str = "Internship"
    batch_eligible: list[str] = Field(default_factory=lambda: ["Not Specified"])
    experience_required: str = "0 years (Fresher / Student)"
    location: list[str] = Field(default_factory=lambda: ["India"])
    work_mode: str = "Not Specified"
    stipend_or_salary: str = "Not Disclosed"
    duration: Optional[str] = None
    skills_required: list[str] = Field(default_factory=list)
    eligibility_criteria: str = "Not Specified"
    job_description: str
    posted_date: str = "Not Specified"
    application_deadline: str = "Not Specified"
    apply_url: str
    source_url: str
    source_portal: str
    relevance_score: float = 0.0
    scraped_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def _populate_job_id(self) -> "JobRecord":
        if not self.job_id:
            self.job_id = compute_job_id(self.company, self.title, self.apply_url or self.source_url)
        return self

