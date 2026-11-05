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

