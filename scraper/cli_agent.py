"""Stage 0: Conversational CLI Intake Agent (`glm-5.3-flash` + Deterministic Parser)."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from scraper.cleaner import CITY_SYNONYMS, normalize_locations
from scraper.config import Settings
from scraper.llm_extractor import KNOWN_SKILLS
from scraper.schemas import JobSearchRequirement

logger = logging.getLogger(__name__)

BATCH_RE = re.compile(r"\b(202[5-9])\b")
STIPEND_RE = re.compile(
    r"((?:min\s*)?(?:₹|rs\.?|inr)?\s*\d+\s*(?:k|lpa|lakh|,\d{3})(?:\s*/?\s*(?:mo|month|pm))?)",
    re.IGNORECASE,
)

ROLE_MAPPING: dict[str, list[str]] = {
    "software": ["Software Engineering", "SDE", "Software Developer"],
    "sde": ["SDE", "Software Development Engineer", "Software Engineering"],
    "swe": ["Software Engineering", "SWE", "SDE"],
    "backend": ["Backend Developer", "Backend Engineering", "SDE"],
    "frontend": ["Frontend Developer", "Frontend Engineering", "React Developer"],
    "full stack": ["Full Stack Developer", "Full Stack Engineer"],
    "fullstack": ["Full Stack Developer", "Full Stack Engineer"],
    "data": ["Data Science", "Data Analyst", "Machine Learning"],
    "ml": ["Machine Learning Engineer", "AI/ML Intern"],
    "ai": ["AI Engineer", "Machine Learning Intern"],
    "devops": ["DevOps Engineer", "Cloud Intern"],
}


def _heuristic_parse_query(text: str) -> dict:
    lower = text.lower()
    batches = sorted(set(BATCH_RE.findall(lower)))

    if "full time" in lower or "full-time" in lower or "fte" in lower:
        job_type = "full_time"
    elif "both" in lower:
        job_type = "both"
    else:
        job_type = "internship"

    roles: list[str] = []
    for key, mapped in ROLE_MAPPING.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            for r in mapped:
                if r not in roles:
                    roles.append(r)

    locs: list[str] = []
    for city_key, norm_city in CITY_SYNONYMS.items():
        if re.search(rf"\b{re.escape(city_key)}\b", lower):
            if norm_city not in locs:
                locs.append(norm_city)

    work_mode = "any"
    if "remote" in lower or "work from home" in lower or "wfh" in lower:
        work_mode = "remote"
    elif "hybrid" in lower:
        work_mode = "hybrid"
    elif "onsite" in lower or "in-office" in lower:
        work_mode = "onsite"

    skills: list[str] = []
    for sk in KNOWN_SKILLS:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(sk)}(?![A-Za-z0-9])", text, re.IGNORECASE):
            skills.append(sk)

    stipend = None
    m_stip = STIPEND_RE.search(text)
    if m_stip:
        stipend = m_stip.group(1).strip()

    return {
        "roles": roles,
        "batches": batches,
        "job_type": job_type,
        "locations": locs,
        "work_mode": work_mode,
        "skills": skills,
        "stipend": stipend,
    }


class IntakeAgent:
    """Conversational intake agent that gathers complete job search criteria before scraping."""

    def __init__(self, settings: Settings | None = None, use_llm: bool = True) -> None:
        self.settings = settings or Settings()
        self.use_llm = use_llm and bool(self.settings.glm_api_key and self.settings.glm_api_key != "your_glm_api_key_here")

    def parse_initial_query(self, user_input: str) -> JobSearchRequirement:
        """Parse user's initial natural-language request into a structured JobSearchRequirement."""
        heur = _heuristic_parse_query(user_input)

        if self.use_llm:
            try:
                import httpx

                url = f"{self.settings.glm_base_url.rstrip('/')}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.settings.glm_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.settings.glm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Extract job search criteria from the user's message into a valid JSON object with keys: "
                                "role_keywords (list[str]), job_type ('internship'|'full_time'|'both'), "
                                "target_batches (list[str]), experience_level (str), locations (list[str]), "
                                "work_mode ('onsite'|'remote'|'hybrid'|'any'), min_stipend_or_salary (str|null), "
                                "skills (list[str]). Output JSON only."
                            ),
                        },
                        {"role": "user", "content": user_input},
                    ],
                    "temperature": 0.1,
                }
                resp = httpx.post(url, headers=headers, json=payload, timeout=25.0)
                resp.raise_for_status()
                raw_content = resp.json()["choices"][0]["message"]["content"] or "{}"
                # Strip markdown code fences if present
                raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.I)
                data = json.loads(raw_content)
                return JobSearchRequirement(
                    raw_query=user_input,
                    role_keywords=data.get("role_keywords") or heur["roles"] or ["Software Engineering", "SDE"],
                    job_type=data.get("job_type") or heur["job_type"],
                    target_batches=[str(b) for b in (data.get("target_batches") or heur["batches"])],
                    experience_level=data.get("experience_level") or "Fresher / 0 years / Student",
                    locations=normalize_locations(data.get("locations") or heur["locations"] or ["India"]),
                    work_mode=data.get("work_mode") or heur["work_mode"],
                    min_stipend_or_salary=data.get("min_stipend_or_salary") or heur["stipend"],
                    skills=data.get("skills") or heur["skills"],
                )
            except Exception as exc:
                logger.warning("GLM intake parse fallback (%s)", exc)

        return JobSearchRequirement(
            raw_query=user_input,
            role_keywords=heur["roles"] or ["Software Engineering", "SDE"],
            job_type=heur["job_type"],
            target_batches=heur["batches"],
            experience_level="Fresher / 0 years / Student" if heur["job_type"] == "internship" else "0-1 years",
            locations=heur["locations"] or ["India"],
            work_mode=heur["work_mode"],
            min_stipend_or_salary=heur["stipend"],
            skills=heur["skills"],
        )

    def get_missing_questions(self, req: JobSearchRequirement) -> Optional[str]:
        """Return a concise follow-up question if specific cities, skills, or stipend preferences are missing."""
        missing_parts: list[str] = []
        if req.locations == ["India"] and req.work_mode == "any":
            missing_parts.append("preferred cities or work mode (e.g., Bangalore, Gurgaon, Hyderabad, Remote, or Any)")
        if not req.skills:
            missing_parts.append("specific tech stack/skills (e.g., Python, C++, React, Backend, or Any)")
        if not req.target_batches:
            missing_parts.append("graduation batch year or experience level (e.g., 2027, 2028, or Fresher)")
        if not req.min_stipend_or_salary:
            missing_parts.append("minimum stipend/CTC expectation (or Any)")

        if not missing_parts:
            return None

        understood = (
            f"Understood so far: {req.job_type.title()} in {', '.join(req.role_keywords)} "
            f"(Batch: {', '.join(req.target_batches) or 'Any'}, Location: {', '.join(req.locations)})."
        )
        numbered = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(missing_parts))
        return (
            f"{understood}\n"
            f"Before I search across the 16 job portals, please share any preferences (or type 'any' / press Enter to start):\n"
            f"{numbered}"
        )

    def apply_user_clarification(self, req: JobSearchRequirement, reply: str) -> JobSearchRequirement:
        """Merge user's follow-up answers into the existing JobSearchRequirement."""
        if not reply or reply.strip().lower() in ("any", "all", "none", "start", "go", "ok", "yes"):
            return req

        parsed = _heuristic_parse_query(reply)
        new_locs = list(req.locations)
        for loc in parsed["locations"]:
            if loc not in new_locs:
                new_locs.append(loc)
        # If specific cities were added, keep India at the end or prioritize the specific cities
        if len(new_locs) > 1 and "India" in new_locs:
            new_locs = [c for c in new_locs if c != "India"] + ["India"]

        new_skills = list(req.skills)
        for sk in parsed["skills"]:
            if sk not in new_skills:
                new_skills.append(sk)

        new_batches = list(req.target_batches)
        for b in parsed["batches"]:
            if b not in new_batches:
                new_batches.append(b)

        new_roles = list(req.role_keywords)
        for r in parsed["roles"]:
            if r not in new_roles:
                new_roles.append(r)

        work_mode = parsed["work_mode"] if parsed["work_mode"] != "any" else req.work_mode
        stipend = parsed["stipend"] or req.min_stipend_or_salary

        return req.model_copy(
            update={
                "raw_query": f"{req.raw_query} | {reply.strip()}",
                "role_keywords": new_roles,
                "target_batches": new_batches,
                "locations": new_locs,
                "work_mode": work_mode,
                "skills": new_skills,
                "min_stipend_or_salary": stipend,
            }
        )

