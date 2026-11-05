from scraper.cli_agent import IntakeAgent
from scraper.discovery import build_16_site_search_queries, build_direct_portal_urls
from scraper.config import TARGET_PORTALS
from scraper.ml_bridge import run_ml_bridge_request


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


def test_python_ml_bridge_scores_go_candidates():
    payload = {
        "requirement": {
            "raw_query": "2028 internship software engineering bangalore",
            "role_keywords": ["Software Engineering", "SDE"],
            "job_type": "internship",
            "target_batches": ["2028"],
            "locations": ["Bengaluru"],
        },
        "top_k": 5,
        "candidates": [
            {
                "url": "https://careersat.tech/jobs/rubrik-swe-intern-2028",
                "url_path": "/jobs/rubrik-swe-intern-2028",
                "text": "Rubrik Software Engineering Intern 2028",
                "context": "Bengaluru | Internship | Batch 2028 | Python, C++ | ₹1,00,000/month",
                "region": "article",
                "tag_path": "main > article > a",
                "sibling_count": 20,
                "source_portal": "careersat",
            },
            {
                "url": "https://whatsapp.com/channel/0029VaUJgMW2",
                "url_path": "/channel/0029VaUJgMW2",
                "text": "Join WhatsApp Channel",
                "context": "WhatsApp channel 52,000+ followers Verified fresher jobs",
                "region": "main",
                "tag_path": "main > div > a",
                "sibling_count": 1,
                "source_portal": "careersat",
            },
        ],
    }
    resp = run_ml_bridge_request(payload)
    assert len(resp["job_listings"]) == 1
    assert resp["job_listings"][0]["url"] == "https://careersat.tech/jobs/rubrik-swe-intern-2028"
    assert resp["noise_count"] == 1

