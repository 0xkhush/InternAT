import numpy as np
from scraper.schemas import DOMCandidate, JobSearchRequirement
from scraper.ml_classifier import BiEncoderClassifier
from scraper.reranker import CrossEncoderReranker, passes_card_context_prefilter


def test_bi_encoder_separates_jobs_pagination_and_noise_with_contrastive_projection():
    clf = BiEncoderClassifier()
    candidates = [
        DOMCandidate(
            url="https://careersat.tech/jobs/honeywell-aerospace-internship-2028",
            url_path="/jobs/honeywell-aerospace-internship-2028",
            text="Honeywell Aerospace Internship 2028 – Software Engineering Intern",
            context="Honeywell Aerospace Bangalore, Hyderabad 1d ago Internship 0–1 yrs Batch 2028 ₹45,000/month",
            region="article",
            tag_path="main > div > article > a",
            sibling_count=20,
            source_portal="careersat",
        ),
        DOMCandidate(
            url="https://thejobcompany.co.in/job-category/batch/2028/page/2",
            url_path="/job-category/batch/2028/page/2",
            text="Next Page 2 - 2028 Batch Jobs",
            context="Pagination Page 1 2 3 Next Filter by Batch 2027 2028",
            region="main",
            tag_path="main > nav > a",
            sibling_count=5,
            source_portal="thejobcompany",
        ),
        DOMCandidate(
            url="https://whatsapp.com/channel/0029VaUJgMW2kNFx7ABlpx2y",
            url_path="/channel/0029VaUJgMW2kNFx7ABlpx2y",
            text="Join WhatsApp channel",
            context="WhatsApp channel 52,000+ followers Verified fresher jobs, internships and study resources. Free, no spam.",
            region="main",
            tag_path="main > div > a",
            sibling_count=1,
            source_portal="careersat",
        ),
        DOMCandidate(
            url="https://careersat.tech/interview-questions/dsa",
            url_path="/interview-questions/dsa",
            text="Free interview prep — 400 questions freshers get asked",
            context="Practice mock interviews, resume prompts and coding questions from our blog",
            region="main",
            tag_path="main > div > a",
            sibling_count=1,
            source_portal="careersat",
        ),
        DOMCandidate(
            url="https://careersat.tech/privacy-policy",
            url_path="/privacy-policy",
            text="Privacy Policy & Terms of Service",
            context="About Contact Privacy Terms DMCA Copyright 2026",
            region="footer",
            tag_path="footer > nav > a",
            sibling_count=8,
            source_portal="careersat",
        ),
    ]

    classified, embeddings = clf.classify_candidates(candidates)
    assert embeddings.shape == (5, 384)
    assert classified[0].predicted_label == "job_listing"
    assert classified[1].predicted_label == "pagination_or_filter"
    assert classified[2].predicted_label == "noise"
    assert classified[3].predicted_label == "noise"
    assert classified[4].predicted_label == "noise"

    # Verify Contrastive Metric Projection pulls same-class vectors closer together than raw embeddings
    raw_embs = clf.encode_raw([c.to_feature_string() for c in candidates])
    proj_embs = clf.project_embeddings(raw_embs, candidates)
    assert raw_embs.shape == (5, 384)
    assert proj_embs.shape == (5, 384)
    # Cosine similarity between noise items (index 2 and 3) should increase after contrastive projection
    raw_noise_sim = float(np.dot(raw_embs[2], raw_embs[3]))
    proj_noise_sim = float(np.dot(proj_embs[2], proj_embs[3]))
    assert proj_noise_sim > raw_noise_sim


def test_context_prefilter_and_cross_encoder_reranker():
    req = JobSearchRequirement(
        raw_query="2028 internship software engineering bangalore",
        role_keywords=["Software Engineering", "SDE"],
        job_type="internship",
        target_batches=["2028"],
        experience_level="0 years / Student",
        locations=["Bengaluru"],
    )
    senior_job = DOMCandidate(
        url="https://careersat.tech/jobs/senior-java-architect",
        url_path="/jobs/senior-java-architect",
        text="Senior Staff Java Architect",
        context="5-8 yrs experience required | Batch 2018-2021 only | Full-time",
        region="article",
        sibling_count=15,
        predicted_label="job_listing",
    )
    matching_intern = DOMCandidate(
        url="https://careersat.tech/jobs/rubrik-swe-intern-2028",
        url_path="/jobs/rubrik-swe-intern-2028",
        text="Rubrik Software Engineering Intern 2028",
        context="Bengaluru | Internship | Batch 2028 | Python, C++ | ₹1,00,000/month",
        region="article",
        sibling_count=15,
        predicted_label="job_listing",
    )
    unrelated_intern = DOMCandidate(
        url="https://careersat.tech/jobs/hr-marketing-intern-mumbai",
        url_path="/jobs/hr-marketing-intern-mumbai",
        text="HR & Social Media Marketing Intern",
        context="Mumbai | Internship | Batch 2028 | Non-tech role",
        region="article",
        sibling_count=15,
        predicted_label="job_listing",
    )

    assert passes_card_context_prefilter(senior_job, req) is False
    assert passes_card_context_prefilter(matching_intern, req) is True

    reranker = CrossEncoderReranker()
    ranked = reranker.rerank([unrelated_intern, matching_intern], req, top_k=2)
    assert ranked[0].url == matching_intern.url
    assert ranked[0].relevance_score > ranked[1].relevance_score

