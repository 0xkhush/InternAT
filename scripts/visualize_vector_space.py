"""Side-by-Side 2D Vector Space Visualizer (Before Raw MiniLM vs. After Contrastive Pair/Triplet Projection)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from scraper.config import PROJECT_ROOT
from scraper.ml_classifier import CLASS_LABELS, SEED_PROTOTYPES, BiEncoderClassifier, candidate_from_feature_string
from scraper.schemas import DOMCandidate

CLUSTER_COLORS: dict[str, str] = {
    "job_listing": "#16a34a",         # Green
    "pagination_or_filter": "#2563eb", # Blue
    "noise": "#dc2626",               # Red
}


def _project_to_2d(vectors: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    centroid_list = [centroids[lbl] for lbl in CLASS_LABELS if lbl in centroids]
    stacked = np.vstack([vectors] + centroid_list) if centroid_list else vectors
    n_comp = min(2, stacked.shape[0], stacked.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(stacked)
    if n_comp == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])
    cand_coords = coords[: len(vectors)]
    cent_coords: dict[str, np.ndarray] = {}
    offset = len(vectors)
    for idx, lbl in enumerate(CLASS_LABELS):
        if lbl in centroids:
            cent_coords[lbl] = coords[offset + idx]
    return cand_coords, cent_coords


def generate_vector_space_plots(
    candidates: list[DOMCandidate],
    embeddings: np.ndarray,
    centroids: dict[str, np.ndarray],
    html_path: Path | None = None,
    png_path: Path | None = None,
    raw_embeddings: np.ndarray | None = None,
    raw_centroids: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path]:
    """Generate Side-by-Side Before (Raw MiniLM) vs. After (Contrastive Pair/Triplet Space) 2D scatter plots."""
    html_out = Path(html_path or (PROJECT_ROOT / "plots" / "vector_space.html"))
    png_out = Path(png_path or (PROJECT_ROOT / "plots" / "vector_space.png"))
    html_out.parent.mkdir(parents=True, exist_ok=True)
    png_out.parent.mkdir(parents=True, exist_ok=True)

    if raw_embeddings is None or raw_centroids is None:
        clf = BiEncoderClassifier()
        raw_embeddings = clf.encode_raw([c.to_feature_string() for c in candidates])
        raw_centroids = clf.get_raw_centroids()

    raw_2d, raw_cent_2d = _project_to_2d(raw_embeddings, raw_centroids)
    proj_2d, proj_cent_2d = _project_to_2d(embeddings, centroids)

    # 1. Generate Side-by-Side High-Resolution Matplotlib PNG
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)
    panels = [
        (axes[0], raw_2d, raw_cent_2d, "Before: Raw Pre-Trained MiniLM Embeddings (384-d -> 2D PCA)"),
        (axes[1], proj_2d, proj_cent_2d, "After: Contrastive Pair + Triplet Trained Space (384-d -> 2D PCA)"),
    ]

    for ax, pts, cents, title in panels:
        for lbl in CLASS_LABELS:
            idxs = [i for i, c in enumerate(candidates) if (c.predicted_label or "noise") == lbl]
            if idxs:
                sub = pts[idxs]
                ax.scatter(
                    sub[:, 0],
                    sub[:, 1],
                    c=CLUSTER_COLORS[lbl],
                    label=f"{lbl} ({len(idxs)})",
                    alpha=0.82,
                    s=65,
                    edgecolors="white",
                    linewidth=0.6,
                )
            if lbl in cents:
                cx, cy = cents[lbl]
                ax.scatter(
                    [cx],
                    [cy],
                    c=CLUSTER_COLORS[lbl],
                    marker="*",
                    s=320,
                    edgecolors="black",
                    linewidth=1.3,
                    label=f"{lbl} centroid",
                )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(png_out)
    plt.close(fig)

    # 2. Generate Interactive Side-by-Side Plotly HTML
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pfig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Before: Raw MiniLM Vector Space",
            "After: Contrastive Pair + Triplet Vector Space",
        ),
    )

    for col_idx, (pts, cents) in enumerate([(raw_2d, raw_cent_2d), (proj_2d, proj_cent_2d)], start=1):
        for lbl in CLASS_LABELS:
            idxs = [i for i, c in enumerate(candidates) if (c.predicted_label or "noise") == lbl]
            if idxs:
                sub = pts[idxs]
                hover_texts = [
                    f"<b>{candidates[i].text}</b><br>Region: {candidates[i].region} | Siblings: {candidates[i].sibling_count}"
                    f"<br>URL: {candidates[i].url_path}<br>Confidence: {candidates[i].confidence:.3f}"
                    for i in idxs
                ]
                pfig.add_trace(
                    go.Scatter(
                        x=sub[:, 0],
                        y=sub[:, 1],
                        mode="markers",
                        name=f"{lbl}",
                        legendgroup=lbl,
                        showlegend=(col_idx == 1),
                        marker=dict(size=10, color=CLUSTER_COLORS[lbl], opacity=0.85, line=dict(width=1, color="white")),
                        text=hover_texts,
                        hoverinfo="text",
                    ),
                    row=1,
                    col=col_idx,
                )
            if lbl in cents:
                cx, cy = cents[lbl]
                pfig.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode="markers",
                        name=f"{lbl} centroid",
                        legendgroup=f"{lbl}_centroid",
                        showlegend=(col_idx == 1),
                        marker=dict(
                            size=18,
                            symbol="star",
                            color=CLUSTER_COLORS[lbl],
                            line=dict(width=2, color="black"),
                        ),
                        hoverinfo="name",
                    ),
                    row=1,
                    col=col_idx,
                )

    pfig.update_layout(
        title="Bi-Encoder 384-d Vector Space: Before vs. After Contrastive Pair/Triplet Fine-Tuning",
        template="plotly_white",
        width=1350,
        height=620,
    )
    pfig.write_html(str(html_out), include_plotlyjs="cdn")
    return html_out, png_out


def main() -> None:
    clf = BiEncoderClassifier()
    cands: list[DOMCandidate] = []
    for lbl, examples in SEED_PROTOTYPES.items():
        for ex in examples:
            cands.append(candidate_from_feature_string(ex, label=lbl))
    classified, proj_embs = clf.classify_candidates(cands)
    html_out, png_out = generate_vector_space_plots(classified, proj_embs, clf.get_centroids())
    print(f"Saved interactive HTML plot -> {html_out}")
    print(f"Saved side-by-side PNG plot -> {png_out}")


if __name__ == "__main__":
    main()

