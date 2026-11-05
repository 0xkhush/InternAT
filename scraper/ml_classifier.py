"""Stage 1 ML Layer: Bi-Encoder with PyTorch Contrastive Pair / Triplet Fine-Tuning & Centroid Cosine Classification."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scraper.config import Settings
from scraper.schemas import DOMCandidate

logger = logging.getLogger(__name__)

CLASS_LABELS: list[str] = ["job_listing", "pagination_or_filter", "noise"]
LABEL_TO_IDX: dict[str, int] = {lbl: idx for idx, lbl in enumerate(CLASS_LABELS)}
REGIONS_ORDER: list[str] = ["article", "main", "body", "aside", "nav", "header", "footer"]

SEED_PROTOTYPES: dict[str, list[str]] = {
    "job_listing": [
        "[REGION] article | [SIBLINGS] 20 | [TEXT] Honeywell Aerospace Internship 2027 – Software & Embedded Engineering Intern | [CONTEXT] Honeywell Aerospace Bangalore, Hyderabad 1d ago Internship 0–1 yrs Batch 2026–27 | [URL] /jobs/honeywell-aerospace-internship-2027",
        "[REGION] article | [SIBLINGS] 18 | [TEXT] Rubrik Software Engineering Intern (Winter/Summer 2028) | [CONTEXT] Rubrik Bengaluru Internship 0 yrs Batch 2027, 2028 ₹1,00,000/month Posted 1d ago | [URL] /jobs/rubrik-software-engineering-intern-2028",
        "[REGION] main | [SIBLINGS] 15 | [TEXT] Google STEP Intern 2028 – Software Student Training in Engineering Program | [CONTEXT] Google Bangalore, Hyderabad Internship Batch 2028 B.Tech CS/IT Apply Now | [URL] /jobs/google-step-intern-2028",
        "[REGION] article | [SIBLINGS] 12 | [TEXT] Amazon SDE Intern 2027 / 2028 – Software Development Engineer Internship | [CONTEXT] Amazon India Chennai, Bengaluru, Gurgaon Stipend ₹80,000/month 0 years experience | [URL] /internship/amazon-sde-intern-2028",
        "[REGION] article | [SIBLINGS] 25 | [TEXT] Backend Engineering Intern at Razorpay | [CONTEXT] Razorpay Bengaluru Remote/Hybrid ₹45,000/month Go, Python, PostgreSQL Batch 2027 2028 | [URL] /internships/backend-engineering-internship-at-razorpay",
        "[REGION] main | [SIBLINGS] 10 | [TEXT] Microsoft Software Engineering Internship 2028 Off Campus Drive | [CONTEXT] Microsoft Hyderabad, Noida, Bangalore B.E/B.Tech 2028 Batch Stipend ₹1,25,000/pm | [URL] /microsoft-software-engineering-internship-2028",
        "[REGION] article | [SIBLINGS] 16 | [TEXT] Atlassian Software Engineer Intern – Summer 2028 | [CONTEXT] Atlassian Bengaluru Remote Eligible Batch 2028 Java, Python, React Stipend ₹1.1 Lakh/month | [URL] /jobs/atlassian-swe-intern-2028",
        "[REGION] article | [SIBLINGS] 22 | [TEXT] Flipkart SDE-1 / Software Developer Intern (Runway 2028) | [CONTEXT] Flipkart Bangalore 0-1 yrs experience Batch 2027/2028 ₹50,000 - ₹1,00,000/month | [URL] /unstop/internship/flipkart-runway-2028",
        "[REGION] main | [SIBLINGS] 14 | [TEXT] Tower Research Capital Core Engineering Intern 2028 | [CONTEXT] Gurugram / Gurgaon C++, Python, Low Latency Systems Batch 2027 2028 | [URL] /jobs/tower-research-core-engineering-intern",
        "[REGION] article | [SIBLINGS] 19 | [TEXT] Stripe Software Engineering Intern (India) | [CONTEXT] Stripe Bengaluru Internship Full-time conversion eligible Batch 2027 2028 | [URL] /jobs/stripe-software-engineering-intern",
        "[REGION] article | [SIBLINGS] 15 | [TEXT] Swiggy Associate Software Engineer Intern | [CONTEXT] Swiggy Bangalore Remote 6 months internship ₹35,000/month Java, Spring Boot | [URL] /internships/swiggy-ase-intern",
        "[REGION] main | [SIBLINGS] 12 | [TEXT] Uber UberSTAR / Software Engineering Intern 2028 | [CONTEXT] Uber Bangalore, Hyderabad 2028 Batch B.Tech CS DSA, Algorithms | [URL] /jobs/uber-software-engineering-intern-2028",
        "[REGION] article | [SIBLINGS] 20 | [TEXT] Qualcomm Embedded & Systems Software Intern 2027/2028 | [CONTEXT] Qualcomm Hyderabad, Bangalore, Chennai C/C++, Linux, RTOS Stipend ₹45k/month | [URL] /jobs/qualcomm-embedded-software-intern",
        "[REGION] article | [SIBLINGS] 10 | [TEXT] Zomato SDE Intern – Backend & AI Platform | [CONTEXT] Zomato Gurugram Onsite Batch 2027, 2028 Python, FastAPI, Node.js | [URL] /jobs/zomato-sde-intern-gurugram",
        "[REGION] main | [SIBLINGS] 16 | [TEXT] Salesforce Futureforce Software Engineering Intern Summer 2028 | [CONTEXT] Salesforce Hyderabad, Bengaluru Batch 2028 ₹90,000/month | [URL] /jobs/salesforce-futureforce-intern-2028",
        "[REGION] article | [SIBLINGS] 14 | [TEXT] Goldman Sachs Summer Analyst / Engineering Intern 2028 | [CONTEXT] Goldman Sachs Bengaluru, Hyderabad 2028 graduating batch | [URL] /jobs/goldman-sachs-summer-analyst-2028",
        "[REGION] article | [SIBLINGS] 20 | [TEXT] Full Stack Developer Internship at Startup | [CONTEXT] Remote India ₹20,000-₹30,000/month React, Node.js, TypeScript 3 months | [URL] /internships/full-stack-developer-internship-remote",
        "[REGION] main | [SIBLINGS] 8 | [TEXT] Cisco Technical Undergraduate Intern 2028 | [CONTEXT] Cisco Bangalore Networking, Python, C++ Batch 2028 | [URL] /jobs/cisco-technical-intern-2028",
        "[REGION] article | [SIBLINGS] 12 | [TEXT] Intuit Software Engineer Intern – Co-op 2028 | [CONTEXT] Intuit Bengaluru Java, React, AWS 0 yrs experience | [URL] /jobs/intuit-software-engineer-intern",
        "[REGION] article | [SIBLINGS] 18 | [TEXT] Adobe Research & Product Intern 2028 | [CONTEXT] Adobe Noida, Bangalore B.Tech/M.Tech 2028 Batch ₹1,00,000/month | [URL] /jobs/adobe-product-intern-2028",
    ],
    "pagination_or_filter": [
        "[REGION] main | [SIBLINGS] 6 | [TEXT] Next Page 2 | [CONTEXT] Page 1 2 3 4 5 Next Previous Showing 1-20 of 240 jobs | [URL] /jobs?page=2",
        "[REGION] main | [SIBLINGS] 8 | [TEXT] 2028 Batch Jobs & Internships | [CONTEXT] Filter jobs by graduation batch: 2025 2026 2027 2028 2029 | [URL] /job-category/batch/2028",
        "[REGION] main | [SIBLINGS] 5 | [TEXT] 2027 Batch Off Campus Drives | [CONTEXT] Browse jobs by batch year 2026 2027 2028 | [URL] /job-category/batch/2027",
        "[REGION] main | [SIBLINGS] 7 | [TEXT] Software Development Internships in Bangalore | [CONTEXT] Filter by Location: Bangalore Hyderabad Gurgaon Pune Mumbai Remote | [URL] /internships/software-development-internship-in-bangalore",
        "[REGION] aside | [SIBLINGS] 10 | [TEXT] Work From Home / Remote Internships | [CONTEXT] Category filter: Remote Internships Part-time Internships Engineering | [URL] /internships/work-from-home-internships",
        "[REGION] main | [SIBLINGS] 5 | [TEXT] Load More Jobs (Page 3) | [CONTEXT] Pagination Previous 1 2 3 4 Next Page | [URL] /search-jobs?page=3",
        "[REGION] main | [SIBLINGS] 6 | [TEXT] Next » | [CONTEXT] 1 2 3 … 15 Next » Older Posts | [URL] /page/2",
        "[REGION] aside | [SIBLINGS] 8 | [TEXT] Computer Science Internships | [CONTEXT] Filter by Stream: Computer Science Electronics Mechanical Civil | [URL] /internships/computer-science-internship",
        "[REGION] main | [SIBLINGS] 6 | [TEXT] Internships in Gurgaon / Gurugram | [CONTEXT] Browse internships by city Delhi NCR Gurgaon Noida Bengaluru | [URL] /internships/internship-in-gurgaon",
        "[REGION] main | [SIBLINGS] 4 | [TEXT] See all Fresher Internships | [CONTEXT] View all 150+ active internship openings sorted by newest | [URL] /jobs/category/internship-job-vacancies?page=2",
        "[REGION] main | [SIBLINGS] 6 | [TEXT] Page 2 of 12 | [CONTEXT] Showing results 21 to 40 Next Prev | [URL] /jobs/search/?start=25",
        "[REGION] aside | [SIBLINGS] 9 | [TEXT] Python Internships | [CONTEXT] Skill filter: Python Java C++ React Node.js Machine Learning | [URL] /internships/python-internship",
        "[REGION] main | [SIBLINGS] 5 | [TEXT] Backend Developer Jobs | [CONTEXT] Filter by Role: Frontend Backend Full Stack DevOps Data Science | [URL] /jobs/backend-developer-jobs",
        "[REGION] main | [SIBLINGS] 4 | [TEXT] Previous Page | [CONTEXT] « Previous 1 2 3 Next » | [URL] /jobs?page=1",
        "[REGION] main | [SIBLINGS] 8 | [TEXT] Batch 2028 Internships | [CONTEXT] Tag archive for 2028 batch passouts | [URL] /tag/2028-batch",
        "[REGION] aside | [SIBLINGS] 6 | [TEXT] Stipend > ₹20,000/month | [CONTEXT] Filter by minimum monthly stipend ₹10,000 ₹20,000 ₹40,000 | [URL] /internships?stipend=20000",
        "[REGION] main | [SIBLINGS] 5 | [TEXT] Older Job Postings → | [CONTEXT] Posts navigation Page 1 Page 2 Older posts | [URL] /offcampus/page/2",
        "[REGION] main | [SIBLINGS] 7 | [TEXT] View All Engineering Jobs | [CONTEXT] Explore categories: Engineering MBA Design Product | [URL] /internship?category=engineering",
        "[REGION] main | [SIBLINGS] 6 | [TEXT] Remote Software Internships | [CONTEXT] Location filter Remote Hybrid Onsite India | [URL] /jobs?location=remote&type=internship",
        "[REGION] main | [SIBLINGS] 5 | [TEXT] 3 | [CONTEXT] Pagination 1 2 3 4 5 Next | [URL] /internships/page-3",
    ],
    "noise": [
        "[REGION] main | [SIBLINGS] 1 | [TEXT] Join channel | [CONTEXT] WhatsApp channel 52,000+ followers Verified fresher jobs, internships and study resources. Free, no spam. | [URL] /channel/0029VaUJgMW2kNFx7ABlpx2y",
        "[REGION] main | [SIBLINGS] 1 | [TEXT] Join our Telegram Group | [CONTEXT] Get instant off-campus job alerts on Telegram 100k+ subscribers | [URL] /telegram-group-link",
        "[REGION] main | [SIBLINGS] 1 | [TEXT] Free interview prep — 400 questions freshers get asked | [CONTEXT] Practice DSA, HR, and system design questions asked in top MNC interviews | [URL] /interview-questions/dsa",
        "[REGION] main | [SIBLINGS] 1 | [TEXT] Resume prompts — Tailor your resume with AI | [CONTEXT] Free ATS resume checker and ChatGPT prompts to land more interviews | [URL] /resume-prompts",
        "[REGION] main | [SIBLINGS] 2 | [TEXT] From the blog — AI jobs for freshers in 2026 | [CONTEXT] Read our latest career advice articles, salary guides, and roadmaps | [URL] /blog/ai-jobs-for-freshers",
        "[REGION] footer | [SIBLINGS] 8 | [TEXT] Privacy Policy & Terms of Service | [CONTEXT] About Us Contact Us Privacy Policy Terms & Conditions DMCA Disclaimer | [URL] /privacy-policy",
        "[REGION] footer | [SIBLINGS] 12 | [TEXT] About Us | [CONTEXT] Copyright 2026 All rights reserved Contact Advertise with us Sitemap | [URL] /about-us",
        "[REGION] header | [SIBLINGS] 5 | [TEXT] Login / Sign Up | [CONTEXT] Candidate Login Employer Login Post a Job Register Free | [URL] /login",
        "[REGION] nav | [SIBLINGS] 6 | [TEXT] Post a Job for Free | [CONTEXT] Hire top interns and freshers Employer dashboard Pricing | [URL] /employer/post-job",
        "[REGION] footer | [SIBLINGS] 24 | [TEXT] Jobs in Chennai | [CONTEXT] Footer sitemap links Jobs by City Jobs by Company Copyright | [URL] /footer-sitemap/chennai",
        "[REGION] main | [SIBLINGS] 1 | [TEXT] Follow us on Instagram | [CONTEXT] Daily reels on coding tips, memes, and fresher career guidance | [URL] /instagram",
        "[REGION] aside | [SIBLINGS] 2 | [TEXT] Buy Mock Test Series @ ₹499 | [CONTEXT] Crack TCS NQT, Infosys, Wipro & Cognizant placement papers | [URL] /courses/placement-prep-bundle",
        "[REGION] footer | [SIBLINGS] 6 | [TEXT] Contact Us & Grievance Officer | [CONTEXT] Help Center Report an Issue Refund Policy Cookie Policy | [URL] /contact",
        "[REGION] header | [SIBLINGS] 4 | [TEXT] Download Mobile App | [CONTEXT] Get the Android and iOS app on Google Play and App Store | [URL] /download-app",
        "[REGION] main | [SIBLINGS] 1 | [TEXT] Calculate your In-Hand Salary | [CONTEXT] CTC to In-hand salary calculator tax deductions PF | [URL] /tools/salary-calculator",
        "[REGION] footer | [SIBLINGS] 10 | [TEXT] Terms and Conditions | [CONTEXT] Legal notice User agreement Privacy DMCA | [URL] /terms-of-service",
        "[REGION] nav | [SIBLINGS] 5 | [TEXT] Resume Builder | [CONTEXT] Create a professional resume in 5 minutes | [URL] /resume-builder",
        "[REGION] main | [SIBLINGS] 1 | [TEXT] Subscribe to Newsletter | [CONTEXT] Enter your email to get weekly newsletter and career tips | [URL] /newsletter/subscribe",
        "[REGION] footer | [SIBLINGS] 8 | [TEXT] DMCA Policy | [CONTEXT] Intellectual property and copyright removal requests | [URL] /dmca",
        "[REGION] header | [SIBLINGS] 6 | [TEXT] Home | [CONTEXT] Main navigation bar Home Blog Courses Login | [URL] /",
    ],
}

NOISE_URL_PATTERNS = re.compile(
    r"(whatsapp\.com|wa\.me|t\.me|telegram\.me|instagram\.com|facebook\.com|twitter\.com|x\.com|"
    r"youtube\.com|linkedin\.com/company/|linkedin\.com/in/|linkedin\.com/school/|linkedin\.com/login|"
    r"/privacy|/terms|/dmca|/contact|/about|/login|/signin|/signup|/register|"
    r"/interview-questions|/resume-prompts|/resume-builder|/blog/|/salary-calculator|"
    r"/courses/|/mock-test|/pricing|/employer|mailto:|javascript:)",
    re.IGNORECASE,
)

PAGINATION_URL_PATTERNS = re.compile(
    r"([?&]page=\d+|/page/\d+|/page-\d+|/job-category/|/category/|/tag/|/batch/\d{4}|[?&]start=\d+)",
    re.IGNORECASE,
)

PAGINATION_TEXT_PATTERNS = re.compile(
    r"^(next|prev|previous|page\s*\d+|\d+|load\s*more|older\s*posts|next\s*page.*|.*batch\s*jobs)$",
    re.IGNORECASE,
)

JOB_URL_PATTERNS = re.compile(
    r"(/jobs?/[a-z0-9_-]{6,}|/internships?/[a-z0-9_-]{6,}|/careers?/|/opening/|/vacancy/|jobid=)",
    re.IGNORECASE,
)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def extract_structural_features_single(cand: DOMCandidate) -> np.ndarray:
    """Extract an 11-d numerical structural feature vector for a DOMCandidate."""
    region_onehot = [1.0 if cand.region == r else 0.0 for r in REGIONS_ORDER]
    sib_norm = min(1.0, np.log1p(max(1, cand.sibling_count)) / np.log1p(25.0))
    full_url_and_path = f"{cand.url} {cand.url_path}"
    is_noise_url = 1.0 if NOISE_URL_PATTERNS.search(full_url_and_path) else 0.0
    is_page_url = 1.0 if (
        PAGINATION_URL_PATTERNS.search(full_url_and_path)
        or PAGINATION_TEXT_PATTERNS.match(cand.text.strip())
    ) else 0.0
    is_job_url = 1.0 if (JOB_URL_PATTERNS.search(full_url_and_path) and not is_page_url and not is_noise_url) else 0.0
    return np.array(region_onehot + [sib_norm, is_job_url, is_page_url, is_noise_url], dtype=np.float32)


def candidate_from_feature_string(feat_str: str, label: str | None = None) -> DOMCandidate:
    """Reconstruct a DOMCandidate from its serialized `[REGION] ... | [SIBLINGS] ...` string."""
    region = "body"
    siblings = 1
    text = feat_str
    context = ""
    url_path = "/"
    m_reg = re.search(r"\[REGION\]\s*([a-z]+)", feat_str)
    if m_reg and m_reg.group(1) in REGIONS_ORDER:
        region = m_reg.group(1)
    m_sib = re.search(r"\[SIBLINGS\]\s*(\d+)", feat_str)
    if m_sib:
        siblings = int(m_sib.group(1))
    m_txt = re.search(r"\[TEXT\]\s*(.*?)\s*\|\s*\[CONTEXT\]", feat_str)
    if m_txt:
        text = m_txt.group(1).strip()
    m_ctx = re.search(r"\[CONTEXT\]\s*(.*?)\s*\|\s*\[URL\]", feat_str)
    if m_ctx:
        context = m_ctx.group(1).strip()
    m_url = re.search(r"\[URL\]\s*(.+)$", feat_str)
    if m_url:
        url_path = m_url.group(1).strip()
    return DOMCandidate(
        url=url_path,
        url_path=url_path,
        text=text,
        context=context,
        region=region,  # type: ignore[arg-type]
        sibling_count=siblings,
        predicted_label=label,  # type: ignore[arg-type]
    )


class ContrastiveProjectionNet(nn.Module):
    """PyTorch Metric Projection Network trained with Contrastive Pair + Triplet Margin Loss.

    Takes `[384-d MiniLM vector || 11-d DOM structural features]` (`395-d`) and projects it
    into a L2-normalized `384-d` Contrastive Vector Space where same-class candidates collapse
    tightly around their class centroid and hard negatives are pushed beyond margin.
    """

    def __init__(self, in_dim: int = 395, hidden_dim: int = 256, out_dim: int = 384) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ln1(self.fc1(x)))
        out = self.fc2(h) + 0.5 * self.residual(x)
        return F.normalize(out, p=2, dim=-1)


def _build_contrastive_pairs_and_triplets(
    X: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all Positive Pairs, Negative Pairs, and Hard/Semi-Hard Triplets `(anchor, pos, neg)`."""
    n = X.shape[0]
    anchors_p, others_p, pair_targets = [], [], []
    anchors_t, pos_t, neg_t = [], [], []

    for i in range(n):
        lbl_i = int(y[i].item())
        pos_indices = [j for j in range(n) if j != i and int(y[j].item()) == lbl_i]
        neg_indices = [k for k in range(n) if int(y[k].item()) != lbl_i]

        for j in pos_indices:
            anchors_p.append(X[i])
            others_p.append(X[j])
            pair_targets.append(1.0)

        for k in neg_indices:
            anchors_p.append(X[i])
            others_p.append(X[k])
            pair_targets.append(-1.0)

        # Sample triplets (anchor, positive, negative)
        for j in pos_indices[:6]:
            for k in neg_indices[:8]:
                anchors_t.append(X[i])
                pos_t.append(X[j])
                neg_t.append(X[k])

    return (
        torch.stack(anchors_p),
        torch.stack(others_p),
        torch.tensor(pair_targets, dtype=torch.float32),
        torch.stack(anchors_t),
        torch.stack(pos_t),
        torch.stack(neg_t),
    )


class BiEncoderClassifier:
    """Stage 1 Bi-Encoder (`all-MiniLM-L6-v2`) with PyTorch Contrastive Pair/Triplet Fine-Tuning & Centroid Cosine Classification."""

    _shared_encoder: Any = None
    _shared_model_name: str | None = None

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._encoder = self._get_encoder(self.settings.bi_encoder_model)
        self._raw_centroids: dict[str, np.ndarray] = {}
        self._projected_centroids: dict[str, np.ndarray] = {}
        self._proj_net: ContrastiveProjectionNet | None = None
        self._initialize_centroids_and_projection()

    @classmethod
    def _get_encoder(cls, model_name: str) -> Any:
        if cls._shared_encoder is None or cls._shared_model_name != model_name:
            from sentence_transformers import SentenceTransformer

            cls._shared_encoder = SentenceTransformer(model_name)
            cls._shared_model_name = model_name
        return cls._shared_encoder

    def encode_raw(self, texts: list[str]) -> np.ndarray:
        """Encode feature strings into L2-normalized 384-d raw MiniLM vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = self._encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode feature strings and apply the Contrastive Metric Projection."""
        raw = self.encode_raw(texts)
        cands = [candidate_from_feature_string(t) for t in texts]
        return self.project_embeddings(raw, cands)

    def _initialize_centroids_and_projection(self) -> None:
        for label, examples in SEED_PROTOTYPES.items():
            embs = self.encode_raw(examples)
            centroid = _l2_normalize(np.mean(embs, axis=0, keepdims=True))[0]
            self._raw_centroids[label] = centroid.astype(np.float32)

        if self.settings.classifier_model_path.exists():
            try:
                ckpt = torch.load(self.settings.classifier_model_path, map_location="cpu", weights_only=False)
                net = ContrastiveProjectionNet()
                net.load_state_dict(ckpt["state_dict"])
                net.eval()
                self._proj_net = net
                self._projected_centroids = {
                    k: np.asarray(v, dtype=np.float32) for k, v in ckpt["projected_centroids"].items()
                }
                return
            except Exception as exc:
                logger.info("Training fresh ContrastiveProjectionNet (%s)", exc)

        # If no saved checkpoint exists yet, train the Contrastive Pair/Triplet network on seed prototypes immediately
        texts, labels = [], []
        for lbl, exs in SEED_PROTOTYPES.items():
            for ex in exs:
                texts.append(ex)
                labels.append(lbl)
        self.fit_classifier_head(texts, labels)

    def project_embeddings(
        self,
        raw_embs: np.ndarray,
        candidates: list[DOMCandidate] | None = None,
    ) -> np.ndarray:
        """Pass `[raw_embs || struct_feats]` through the Contrastive Pair/Triplet-trained PyTorch network."""
        if len(raw_embs) == 0:
            return raw_embs

        if candidates is None:
            candidates = [DOMCandidate(url="/", url_path="/", text="", context="") for _ in range(len(raw_embs))]

        struct_feats = np.stack([extract_structural_features_single(c) for c in candidates], axis=0)
        full_feats = np.concatenate([raw_embs, struct_feats], axis=1).astype(np.float32)

        if self._proj_net is None:
            return _l2_normalize(raw_embs)

        self._proj_net.eval()
        with torch.no_grad():
            inp = torch.from_numpy(full_feats)
            proj = self._proj_net(inp).cpu().numpy()
        return _l2_normalize(proj).astype(np.float32)

    def get_raw_centroids(self) -> dict[str, np.ndarray]:
        """Return the unprojected 384-d class centroids."""
        return dict(self._raw_centroids)

    def get_centroids(self) -> dict[str, np.ndarray]:
        """Return the Contrastive Pair/Triplet Projected 384-d class centroids."""
        return dict(self._projected_centroids)

    def classify_candidates(
        self,
        candidates: list[DOMCandidate],
    ) -> tuple[list[DOMCandidate], np.ndarray]:
        """Classify DOMCandidates purely by Nearest Centroid Cosine Similarity in the Contrastive Embedding Space."""
        if not candidates:
            return [], np.zeros((0, 384), dtype=np.float32)

        feature_strings = [c.to_feature_string() for c in candidates]
        raw_embs = self.encode_raw(feature_strings)
        proj_embs = self.project_embeddings(raw_embs, candidates)  # [N, 384]

        centroid_matrix = np.stack([self._projected_centroids[lbl] for lbl in CLASS_LABELS], axis=0)  # [3, 384]
        cos_sims = np.matmul(proj_embs, centroid_matrix.T)  # [N, 3] cosine similarity in [-1, 1]

        # Convert cosine similarities to calibrated confidence probabilities via temperature-scaled softmax
        scaled = cos_sims * 10.0
        scaled -= np.max(scaled, axis=1, keepdims=True)
        exp_scores = np.exp(scaled)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        classified: list[DOMCandidate] = []
        for idx, cand in enumerate(candidates):
            full_url = f"{cand.url} {cand.url_path}"
            if NOISE_URL_PATTERNS.search(full_url) or cand.region == "footer":
                best_idx = 2  # noise
            else:
                best_idx = int(np.argmax(cos_sims[idx]))
            label = CLASS_LABELS[best_idx]
            conf = float(max(probs[idx, best_idx], 0.90 if best_idx == 2 and NOISE_URL_PATTERNS.search(full_url) else 0.0))
            updated = cand.model_copy(
                update={
                    "predicted_label": label,
                    "confidence": round(conf, 4),
                }
            )
            classified.append(updated)

        return classified, proj_embs

    def fit_classifier_head(
        self,
        texts: list[str],
        labels: list[str],
        save_path: Path | None = None,
        epochs: int = 80,
    ) -> dict[str, Any]:
        """Train `ContrastiveProjectionNet` using PyTorch Contrastive Pair (`CosineEmbeddingLoss`) + Triplet (`TripletMarginWithDistanceLoss`) Loss."""
        torch.manual_seed(42)
        cands = [candidate_from_feature_string(t, label=lbl) for t, lbl in zip(texts, labels)]
        raw_embs = self.encode_raw(texts)
        struct_feats = np.stack([extract_structural_features_single(c) for c in cands], axis=0)
        X_np = np.concatenate([raw_embs, struct_feats], axis=1).astype(np.float32)
        y_np = np.array([LABEL_TO_IDX[lbl] for lbl in labels], dtype=np.int64)

        X_t = torch.from_numpy(X_np)
        y_t = torch.from_numpy(y_np)

        anc_p, oth_p, tgt_p, anc_tri, pos_tri, neg_tri = _build_contrastive_pairs_and_triplets(X_t, y_t)

        net = ContrastiveProjectionNet(in_dim=X_np.shape[1], hidden_dim=256, out_dim=384)
        net.train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)

        cosine_pair_loss_fn = nn.CosineEmbeddingLoss(margin=0.15)
        triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda a, b: 1.0 - F.cosine_similarity(a, b),
            margin=0.60,
        )

        for _ in range(epochs):
            optimizer.zero_grad()
            # 1. Contrastive Pair Loss (pulls positive pairs to cos=1.0, pushes negative pairs below margin=0.15)
            z_a = net(anc_p)
            z_b = net(oth_p)
            loss_pair = cosine_pair_loss_fn(z_a, z_b, tgt_p)

            # 2. Triplet Margin Loss on (anchor, positive, hard_negative)
            zt_a = net(anc_tri)
            zt_p = net(pos_tri)
            zt_n = net(neg_tri)
            loss_triplet = triplet_loss_fn(zt_a, zt_p, zt_n)

            loss = loss_pair + loss_triplet
            loss.backward()
            optimizer.step()

        net.eval()
        self._proj_net = net

        with torch.no_grad():
            proj_all = net(X_t).cpu().numpy()

        # Compute the 3 L2-normalized class centroids in the trained Contrastive Embedding Space
        projected_centroids: dict[str, np.ndarray] = {}
        for lbl_idx, lbl_name in enumerate(CLASS_LABELS):
            mask = y_np == lbl_idx
            cls_vecs = proj_all[mask]
            centroid = _l2_normalize(np.mean(cls_vecs, axis=0, keepdims=True))[0].astype(np.float32)
            projected_centroids[lbl_name] = centroid

        self._projected_centroids = projected_centroids

        # Evaluate nearest-centroid cosine accuracy
        centroid_mat = np.stack([projected_centroids[lbl] for lbl in CLASS_LABELS], axis=0)
        preds = np.argmax(np.matmul(proj_all, centroid_mat.T), axis=1)
        train_acc = float(np.mean(preds == y_np))

        target_save = save_path or self.settings.classifier_model_path
        target_save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "loss_type": "ContrastivePair+TripletMarginLoss",
                "state_dict": net.state_dict(),
                "projected_centroids": projected_centroids,
            },
            target_save,
        )

        self.settings.centroids_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.settings.centroids_path, centroid_mat)

        return {
            "num_samples": len(texts),
            "num_pairs": int(anc_p.shape[0]),
            "num_triplets": int(anc_tri.shape[0]),
            "train_accuracy": round(train_acc, 4),
            "model_path": str(target_save),
            "centroids_path": str(self.settings.centroids_path),
        }
