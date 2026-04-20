# -*- coding: utf-8 -*-
"""
download_models.py — Pre-cache ML Models at Docker Build Time
==============================================================
Bias-Free Hiring Pipeline — Sprint 2

This script is executed as a RUN step inside the Docker image build so
that the heavyweight NLP models are baked into the image layer.  The
result is that container startup goes from ~60 s (first-run download)
to ~3 s (models already on disk).

Models downloaded
-----------------
    spaCy   : en_core_web_trf  (transformer-based NER, ~430 MB)
    SBERT   : all-MiniLM-L6-v2 (sentence embeddings, ~90 MB)
    Fallback: en_core_web_sm   (small model, ~12 MB — used if trf fails)

Environment variables
---------------------
    SKIP_LARGE_MODELS=1   Set in CI to skip the 430 MB spaCy transformer
                          and use en_core_web_sm instead.  The pipeline
                          degrades gracefully — NER quality drops but the
                          pipeline still runs end-to-end.

Usage
-----
    # During Docker build (see Dockerfile):
    RUN python download_models.py

    # Locally (one-time setup):
    python download_models.py
"""

from __future__ import annotations

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SKIP_LARGE = os.getenv("SKIP_LARGE_MODELS", "0") == "1"


def download_spacy() -> bool:
    """Download and verify the spaCy NER model.

    Downloads en_core_web_trf by default; falls back to en_core_web_sm
    when SKIP_LARGE_MODELS=1 (CI / resource-constrained environments).

    Returns:
        True if a usable model was installed, False on total failure.
    """
    import subprocess

    if SKIP_LARGE:
        model = "en_core_web_sm"
        logger.info("SKIP_LARGE_MODELS=1 — downloading lightweight model: %s", model)
    else:
        model = "en_core_web_trf"
        logger.info("Downloading spaCy model: %s (~430 MB) …", model)

    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", model],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("spaCy model download failed (exit %d)", result.returncode)
        if not SKIP_LARGE:
            logger.warning("Retrying with fallback model: en_core_web_sm")
            fallback = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=False,
            )
            return fallback.returncode == 0
        return False

    # Verify the model loads correctly
    try:
        import spacy
        nlp = spacy.load(model)
        test_doc = nlp("John Smith applied for the role at Google.")
        persons = [e.text for e in test_doc.ents if e.label_ == "PERSON"]
        logger.info("spaCy model '%s' verified — detected persons: %s", model, persons)
        return True
    except Exception as exc:
        logger.error("spaCy model verification failed: %s", exc)
        return False


def download_sbert() -> bool:
    """Download and verify the sentence-transformers model.

    Caches to the HuggingFace default cache directory
    (~/.cache/huggingface/hub) which is baked into the Docker image layer.

    Returns:
        True if the model loaded and produced embeddings, False otherwise.
    """
    model_name = "all-MiniLM-L6-v2"
    logger.info("Downloading SBERT model: %s (~90 MB) …", model_name)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        # Smoke-test: encode a short sentence
        embedding = model.encode("Data Scientist with Python experience")
        logger.info(
            "SBERT model '%s' verified — embedding shape: %s",
            model_name, embedding.shape,
        )
        return True
    except ImportError:
        logger.error(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )
        return False
    except Exception as exc:
        logger.error("SBERT model download/verification failed: %s", exc)
        return False


def main() -> int:
    """Download all models and return exit code 0 on success, 1 on failure.

    Returns:
        0 if all downloads succeeded, 1 if any failed.
    """
    logger.info("=" * 55)
    logger.info("  Bias-Free Hiring Pipeline — Model Pre-cacher")
    logger.info("=" * 55)

    results = {
        "spacy":  download_spacy(),
        "sbert":  download_sbert(),
    }

    logger.info("")
    logger.info("Download summary:")
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        logger.info("  %-10s %s", name, status)
        if not ok:
            all_ok = False

    if all_ok:
        logger.info("")
        logger.info("All models ready. Container startup will be fast.")
        return 0
    else:
        logger.error("")
        logger.error("One or more models failed to download.")
        logger.error("The pipeline will attempt runtime download as fallback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
