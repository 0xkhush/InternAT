"""Train Stage 1 Bi-Encoder Contrastive Metric Projection & Classifier Head (with optional live scraper dataset collection)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scraper.config import PROJECT_ROOT, TARGET_PORTALS, Settings
from scraper.extractor import extract_dom_candidates
from scraper.fetcher import HybridFetcher
from scraper.ml_classifier import SEED_PROTOTYPES, BiEncoderClassifier

SCRAPED_DATA_PATH = PROJECT_ROOT / "data" / "scraped_training_candidates.json"


async def collect_scraped_training_samples(max_portals: int = 4) -> list[dict[str, str]]:
    """Scrape real DOM candidates from target portals and auto-label high-confidence training examples."""
    fetcher = HybridFetcher(timeout=15.0)
    clf = BiEncoderClassifier()
    collected: list[dict[str, str]] = []

    try:
        for portal in TARGET_PORTALS[:max_portals]:
            try:
                html = await fetcher.fetch(portal.entry_url, render_mode=portal.render)
                if not html:
                    continue
                cands = extract_dom_candidates(html, portal.entry_url, portal.id)
                classified, _ = clf.classify_candidates(cands)
                for c in classified:
                    if c.confidence >= 0.72 and c.predicted_label:
                        collected.append(
                            {
                                "feature_string": c.to_feature_string(),
                                "label": c.predicted_label,
                                "source_portal": portal.id,
                                "confidence": f"{c.confidence:.3f}",
                            }
                        )
            except Exception:
                continue
    finally:
        await fetcher.close()

    if collected:
        SCRAPED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCRAPED_DATA_PATH.write_text(json.dumps(collected, indent=2), encoding="utf-8")
    return collected


def train_and_save(scrape_live: bool = False) -> dict:
    """Combine seed prototypes + any scraped training samples and train the classifier & projection heads."""
    texts: list[str] = []
    labels: list[str] = []

    for label, examples in SEED_PROTOTYPES.items():
        for ex in examples:
            texts.append(ex)
            labels.append(label)

    if scrape_live:
        asyncio.run(collect_scraped_training_samples())

    if SCRAPED_DATA_PATH.exists():
        try:
            extra = json.loads(SCRAPED_DATA_PATH.read_text(encoding="utf-8"))
            for item in extra:
                if item.get("feature_string") and item.get("label") in SEED_PROTOTYPES:
                    texts.append(item["feature_string"])
                    labels.append(item["label"])
        except Exception:
            pass

    settings = Settings()
    clf = BiEncoderClassifier(settings)
    metrics = clf.fit_classifier_head(texts, labels)
    print(f"[train_classifier] Trained on {metrics['num_samples']} samples | Accuracy: {metrics['train_accuracy']*100:.1f}%")
    print(f"[train_classifier] Saved model -> {metrics['model_path']}")
    print(f"[train_classifier] Saved centroids -> {metrics['centroids_path']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 Bi-Encoder Contrastive Metric Projection & Classifier Head")
    parser.add_argument(
        "--scrape-live",
        action="store_true",
        help="Scrape live target portals first to augment training dataset with real DOM candidates",
    )
    args = parser.parse_args()
    train_and_save(scrape_live=args.scrape_live)


if __name__ == "__main__":
    main()

