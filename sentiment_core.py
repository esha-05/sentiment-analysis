"""
sentiment_core.py — HuggingFace-powered Sentiment Pipeline
============================================================
Replaces custom sklearn/embedding model with a pretrained
HuggingFace transformer (cardiffnlp/twitter-roberta-base-sentiment-latest).

Handles: emojis, slang, negation, mixed sentiment, multiple languages.
No training required. No .pkl files. No retraining ever.

Install dependencies:
    pip install transformers torch
"""

import json
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# ─── Constants (kept for API compatibility) ───────────────────────────────────
MAX_LEN     = 512    # RoBERTa max tokens
RANDOM_SEED = 42
LABELS      = ['Negative', 'Neutral', 'Positive']

# ─── HuggingFace Model ────────────────────────────────────────────────────────
# cardiffnlp/twitter-roberta-base-sentiment-latest:
#   - Trained on 124M tweets
#   - Natively understands: emojis, slang, negation, mixed sentiment
#   - 3 classes: Negative / Neutral / Positive
#   - ~500MB download on first run, cached locally after that
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

HF_LABEL_MAP = {
    "negative": "Negative",
    "neutral":  "Neutral",
    "positive": "Positive",
}


@lru_cache(maxsize=1)
def _load_hf_pipeline():
    """
    Lazy-load HuggingFace pipeline (singleton, cached).
    Downloads ~500MB model on first call, then loads from local cache.
    """
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading HuggingFace model: %s ...", HF_MODEL)
        pipe = hf_pipeline(
            task="sentiment-analysis",
            model=HF_MODEL,
            top_k=None,       # return ALL class probabilities
            truncation=True,
            max_length=MAX_LEN,
        )
        logger.info("HuggingFace model loaded successfully.")
        return pipe
    except ImportError:
        raise RuntimeError(
            "HuggingFace transformers not installed.\n"
            "Run:  pip install transformers torch"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace model: {e}")


# ─── Main Pipeline Class ──────────────────────────────────────────────────────
class SentimentPipeline:
    """
    Drop-in replacement for the old custom sklearn SentimentPipeline.
    Identical public API — app.py requires zero changes.
    """

    def __init__(self):
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            self._pipe = _load_hf_pipeline()
        return self._pipe

    # ── Core inference ────────────────────────────────────────────────
    def _run(self, texts: list) -> list:
        pipe = self._get_pipe()
        return pipe(texts)   # list of lists when top_k=None

    def _parse_result(self, raw: list) -> tuple:
        """
        Parse HuggingFace output → (label, confidence, probabilities dict).
        raw = [{"label": "positive", "score": 0.92}, ...]
        """
        probs = {}
        for item in raw:
            mapped = HF_LABEL_MAP.get(item["label"].lower(), item["label"].capitalize())
            probs[mapped] = round(float(item["score"]), 4)

        # Ensure all three keys always present
        for lbl in LABELS:
            probs.setdefault(lbl, 0.0)

        best_label = max(probs, key=probs.get)
        confidence = probs[best_label]

        # Confidence threshold: avoid blindly labeling low-confidence predictions
        label = best_label if confidence >= 0.45 else "Uncertain"

        return label, confidence, probs

    # ── Public API ────────────────────────────────────────────────────
    def predict(self, texts: list) -> list:
        results = self._run(texts)
        return [self._parse_result(r)[0] for r in results]

    def predict_proba(self, texts: list) -> list:
        results = self._run(texts)
        out = []
        for raw in results:
            _, _, probs = self._parse_result(raw)
            out.append([probs["Negative"], probs["Neutral"], probs["Positive"]])
        return out

    def predict_single(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text must not be empty.")
        if len(text) > 5000:
            raise ValueError("Input text exceeds maximum length of 5000 characters.")

        raw                      = self._run([text])[0]
        label, confidence, probs = self._parse_result(raw)
        pred                     = LABELS.index(label) if label in LABELS else 0

        return {
            "label":         label,
            "label_index":   pred,
            "confidence":    confidence,
            "probabilities": {
                "Negative": probs["Negative"],
                "Neutral":  probs["Neutral"],
                "Positive": probs["Positive"],
            },
            "text_length": len(text),
        }

    # ── Persistence ───────────────────────────────────────────────────
    def save(self, prefix: str):
        """Save metadata only — no pkl needed, HuggingFace caches automatically."""
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        meta = {
            "model":      HF_MODEL,
            "type":       "huggingface-transformer",
            "labels":     LABELS,
            "max_length": MAX_LEN,
        }
        with open(f"{prefix}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Metadata saved → %s_meta.json", prefix)

    @classmethod
    def load(cls, prefix: str) -> "SentimentPipeline":
        """No pkl to load — create instance and warm up the HF model."""
        instance = cls()
        instance._get_pipe()   # download/load model now so first request is fast
        return instance

    def fit(self, texts, labels):
        raise NotImplementedError(
            "HuggingFace model is pretrained — no fitting needed.\n"
            "You can safely delete train_model.py."
        )