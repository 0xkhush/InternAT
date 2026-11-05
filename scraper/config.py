"""Configuration for the 16 target job portals, GLM-5.3-flash LLM, and local ML models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PortalConfig(BaseModel):
    """Configuration for a target job or internship portal."""

    id: str
    name: str
    domain: str
    entry_url: str
    render: Literal["auto", "always", "never"] = "auto"


TARGET_PORTALS: list[PortalConfig] = [
    PortalConfig(
        id="thejobcompany",
        name="The Job Company",
        domain="thejobcompany.co.in",
        entry_url="https://thejobcompany.co.in/",
        render="auto",
    ),
    PortalConfig(
        id="careersat",
        name="CareersAt.Tech",
        domain="careersat.tech",
        entry_url="https://careersat.tech/jobs",
        render="always",
    ),
    PortalConfig(
        id="hiredoor",
        name="Hiredoor",
        domain="hiredoor.in",
        entry_url="https://hiredoor.in/",
        render="always",
    ),
    PortalConfig(
        id="freshersworld",
        name="Freshersworld",
        domain="freshersworld.com",
        entry_url="https://www.freshersworld.com/jobs/category/internship-job-vacancies",
        render="always",
    ),
    PortalConfig(
        id="offcampus",
        name="OffCampusJobs4u",
        domain="offcampusjobs4u.com",
        entry_url="https://offcampusjobs4u.com/",
        render="auto",
    ),
    PortalConfig(
        id="talentbattle",
        name="Talent Battle",
        domain="talentbattle.in",
        entry_url="https://talentbattle.in/",
        render="auto",
    ),
    PortalConfig(
        id="internshala",
        name="Internshala",
        domain="internshala.com",
        entry_url="https://internshala.com/internships/",
        render="always",
    ),
    PortalConfig(
        id="unstop",
        name="Unstop",
        domain="unstop.com",
        entry_url="https://unstop.com/internship",
        render="always",
    ),
    PortalConfig(
        id="wellfound",
        name="Wellfound",
        domain="wellfound.com",
        entry_url="https://wellfound.com/jobs",
        render="auto",
    ),
    PortalConfig(
        id="instahyre",
        name="Instahyre",
        domain="instahyre.com",
        entry_url="https://www.instahyre.com/search-jobs/",
        render="always",
    ),
    PortalConfig(
        id="cutshort",
        name="Cutshort",
        domain="cutshort.io",
        entry_url="https://cutshort.io/jobs",
        render="auto",
    ),
    PortalConfig(
        id="hasjob",
        name="Hasjob",
        domain="hasjob.co",
        entry_url="https://hasjob.co/",
        render="always",
    ),
    PortalConfig(
        id="indeed",
        name="Indeed India",
        domain="in.indeed.com",
        entry_url="https://in.indeed.com/jobs",
        render="always",
    ),
    PortalConfig(
        id="linkedin",
        name="LinkedIn Jobs",
        domain="linkedin.com",
        entry_url="https://www.linkedin.com/jobs/search/",
        render="never",
    ),
    PortalConfig(
        id="cuvette",
        name="Cuvette",
        domain="cuvette.tech",
        entry_url="https://cuvette.tech/",
        render="auto",
    ),
    PortalConfig(
        id="prosple",
        name="Prosple India",
        domain="in.prosple.com",
        entry_url="https://in.prosple.com/",
        render="auto",
    ),
]

PORTAL_BY_ID: dict[str, PortalConfig] = {p.id: p for p in TARGET_PORTALS}
PORTAL_BY_DOMAIN: dict[str, PortalConfig] = {p.domain.lower(): p for p in TARGET_PORTALS}


def resolve_portal_for_url(url: str) -> PortalConfig | None:
    """Match a URL against one of the 16 registered portals."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return None
    if host.startswith("www."):
        host = host[4:]
    for domain, portal in PORTAL_BY_DOMAIN.items():
        if host == domain or host.endswith("." + domain):
            return portal
    return None


class Settings(BaseModel):
    """Runtime settings loaded from environment variables with sensible defaults."""

    glm_api_key: str = Field(default_factory=lambda: os.getenv("GLM_API_KEY", ""))
    glm_base_url: str = Field(
        default_factory=lambda: os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    )
    glm_model: str = Field(default_factory=lambda: os.getenv("GLM_MODEL", "glm-5.3-flash"))
    bi_encoder_model: str = Field(
        default_factory=lambda: os.getenv("BI_ENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    cross_encoder_model: str = Field(
        default_factory=lambda: os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    seen_jobs_path: Path = Field(
        default_factory=lambda: Path(os.getenv("SEEN_JOBS_PATH", str(PROJECT_ROOT / "data" / "seen_jobs.json")))
    )
    output_jobs_path: Path = Field(
        default_factory=lambda: Path(os.getenv("OUTPUT_JOBS_PATH", str(PROJECT_ROOT / "jobs.json")))
    )
    classifier_model_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "models" / "noise_classifier.pkl"
    )
    centroids_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "models" / "centroids.npy"
    )
