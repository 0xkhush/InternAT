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


def test_jsonld_with_unescaped_control_characters_is_parsed():
    html_with_ctrl = """
    <html><head>
    <script type="application/ld+json">
    {
      "@type": "JobPosting",
      "title": "Backend Intern\t2028\nBatch",
      "hiringOrganization": {"name": "Atlassian"},
      "description": "Line 1\nLine 2\twith unescaped newline and tab"
    }
    </script>
    </head><body><article><h1>Backend Intern</h1></article></body></html>
    """
    detail = extract_job_detail_from_html(html_with_ctrl, "https://careersat.tech/jobs/atlassian", "careersat")
    assert detail.company == "Atlassian"
    assert "Backend Intern" in detail.title


