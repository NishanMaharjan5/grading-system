"""Hand-crafted features that sit alongside the SBERT embedding.

These exist because frozen all-MiniLM-L6-v2 embeddings encode *topic*, not
*quality*: two sentences on the same subject land in nearly the same place
whether one cites a statistic and the other hedges. Measured over the first 35
labeled examples, the nearest neighbour in embedding space shared the same
score only ~17% of the time, and a classifier trained on embeddings alone did
worse than always guessing the mean score. These features capture what the
rubric actually rewards -- concrete numbers, citations, explicit reasoning --
and what it penalises -- hedging and vagueness.

Must stay identical between training and inference, which is why this lives in
the app package rather than inside the training script.
"""

import re

import numpy as np

HEDGE_RE = re.compile(
    r"\b(many|some|probably|might|could|seems?|various|things|a lot|overall|different|interesting)\b"
)
REASON_RE = re.compile(r"\b(because|therefore|since|which|showing|shows|directly|though|while)\b")
CITE_RE = re.compile(r"\b(study|studies|research|report|survey|found|data|according|percent|analysis)\b")
DIGIT_RE = re.compile(r"\d")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PROPER_RE = re.compile(r"(?<!^)\b[A-Z][a-z]+")

FEATURE_NAMES = [
    "word_count",
    "digit_count",
    "has_percent",
    "year_count",
    "hedge_count",
    "reasoning_count",
    "citation_count",
    "mean_word_length",
    "proper_noun_count",
]


def extract(texts):
    """list[str] -> (n, len(FEATURE_NAMES)) float array."""
    rows = []
    for text in texts:
        lowered = text.lower()
        words = text.split()
        rows.append([
            len(words),
            len(DIGIT_RE.findall(text)),
            1.0 if "%" in text else 0.0,
            len(YEAR_RE.findall(text)),
            len(HEDGE_RE.findall(lowered)),
            len(REASON_RE.findall(lowered)),
            len(CITE_RE.findall(lowered)),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            len(PROPER_RE.findall(text)),
        ])
    return np.array(rows, dtype=float)
