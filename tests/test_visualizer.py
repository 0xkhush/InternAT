from pathlib import Path
from scraper.ml_classifier import BiEncoderClassifier
from scraper.extractor import extract_dom_candidates
from scripts.visualize_vector_space import generate_vector_space_plots
from tests.test_extractor_and_fetcher import SAMPLE_CAREERSAT_HTML


def test_generate_vector_space_plots_creates_html_and_png(tmp_path: Path):
    clf = BiEncoderClassifier()
    candidates = extract_dom_candidates(SAMPLE_CAREERSAT_HTML, "https://careersat.tech/jobs", "careersat")
    classified, embeddings = clf.classify_candidates(candidates)
    centroids = clf.get_centroids()

    html_out = tmp_path / "vector_space.html"
    png_out = tmp_path / "vector_space.png"
    generate_vector_space_plots(classified, embeddings, centroids, html_out, png_out)

    assert html_out.exists() and html_out.stat().st_size > 500
    assert png_out.exists() and png_out.stat().st_size > 1000

