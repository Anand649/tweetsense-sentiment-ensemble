"""Evaluate a Twitter-specialized ensemble without using the test split to tune it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentiment_benchmark.predict import load_base_model  # noqa: E402
from sentiment_benchmark.twitter_ensemble import TwitterEnsemble  # noqa: E402

REPO = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
FILES = ["config.json", "merges.txt", "pytorch_model.bin", "special_tokens_map.json", "vocab.json"]


def split(name: str) -> tuple[list[str], np.ndarray]:
    folder = ROOT / "data" / "raw" / "tweet_sentiment"
    texts = (folder / f"{name}_text.txt").read_text(encoding="utf-8").splitlines()
    labels = np.array([int(x) for x in (folder / f"{name}_labels.txt").read_text(encoding="utf-8").splitlines()])
    if len(texts) != len(labels):
        raise ValueError(f"{name}: different text and label counts")
    return texts, labels


def transformer_probabilities(model: TwitterEnsemble, texts: list[str]) -> np.ndarray:
    blocks = []
    for start in range(0, len(texts), 240):
        blocks.append(model.transformer_proba(texts[start : start + 240]))
        print(f"Scored {min(start + 240, len(texts)):,}/{len(texts):,} posts", flush=True)
    return np.concatenate(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use previously downloaded transformer files")
    args = parser.parse_args()
    base = load_base_model("tweet_sentiment")
    transformer_dir = ROOT / "models" / "twitter_roberta"
    if not args.offline:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, revision=REVISION, local_dir=transformer_dir, cache_dir=ROOT / ".hf-cache", allow_patterns=FILES)
    if not (transformer_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError("Twitter RoBERTa weights missing; run without --offline")

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    hybrid = TwitterEnsemble(base, transformer_dir, 0.8)

    val_texts, val_labels = split("val")
    base_val = hybrid.base_proba(val_texts)
    transformer_val = transformer_probabilities(hybrid, val_texts)
    # Keep both trained members active; the Twitter encoder can supply at most 95%.
    candidates = np.arange(0.5, 0.951, 0.05)
    scored = [
        (float(f1_score(val_labels, np.argmax((1 - w) * base_val + w * transformer_val, axis=1), average="macro")), round(float(w), 2)) for w in candidates
    ]
    # Prefer the lower transformer weight if validation scores tie.
    val_f1, weight = max(scored, key=lambda item: (item[0], -item[1]))
    print(f"Validation macro F1={val_f1:.4f}; transformer weight={weight:.2f}", flush=True)

    test_texts, test_labels = split("test")
    base_test = hybrid.base_proba(test_texts)
    transformer_test = transformer_probabilities(hybrid, test_texts)
    test_prob = (1 - weight) * base_test + weight * transformer_test
    pred = np.argmax(test_prob, axis=1)
    report = classification_report(test_labels, pred, labels=[0, 1, 2], target_names=base.label_names, output_dict=True, zero_division=0)
    accuracy = float(accuracy_score(test_labels, pred))
    macro_f1 = float(f1_score(test_labels, pred, average="macro"))
    print(f"Held-out test accuracy={accuracy:.4f}; macro F1={macro_f1:.4f}", flush=True)

    card = dict(base.card)
    card.update(
        model="twitter_hybrid_ensemble",
        features="tfidf+twitter_roberta",
        transformer_repo=REPO,
        transformer_revision=REVISION,
        transformer_weight=weight,
        selection="transformer blending weight chosen by macro F1 on the official validation split; test kept separate",
        metrics={"validation_f1": val_f1, "test_accuracy": accuracy, "test_f1": macro_f1},
        per_class=[
            {
                "class": name,
                "support": int(report[name]["support"]),
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
            }
            for name in base.label_names
        ],
    )
    (base.model_dir / "hybrid_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(test_labels, pred, display_labels=base.label_names, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Twitter ensemble · held-out TweetEval test")
    fig.tight_layout()
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "cm_tweet_sentiment_hybrid.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
