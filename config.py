# -*- coding: utf-8 -*-
"""
config.py — Central Configuration
====================================
All tuneable values live here. Override any setting with an environment
variable — no code changes needed.

    Windows:  set FLASK_PORT=8080
    Mac/Linux: export FLASK_PORT=8080
"""
from __future__ import annotations
import os

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# ── Flask / Dashboard ──────────────────────────────────────────────────────
FLASK_HOST:  str  = os.getenv("FLASK_HOST",       "127.0.0.1")
FLASK_PORT:  int  = int(os.getenv("FLASK_PORT",   "5000"))
FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG",      "0") == "1"
SECRET_KEY:  str  = os.getenv("FLASK_SECRET_KEY", "bias-free-hiring-pipeline-dev-key")

# ── Database ───────────────────────────────────────────────────────────────
DB_PATH: str = os.getenv("PIPELINE_DB_PATH", os.path.join(BASE_DIR, "pipeline.db"))

# ── Pipeline directories ───────────────────────────────────────────────────
RAW_CVS_DIR:      str = os.getenv("RAW_CVS_DIR",      os.path.join(BASE_DIR, "raw_cvs"))
ANONYMIZED_DIR:   str = os.getenv("ANONYMIZED_DIR",   os.path.join(BASE_DIR, "anonymized_cvs"))
PARSED_DIR:       str = os.getenv("PARSED_DIR",       os.path.join(BASE_DIR, "parsed"))
VAULT_DIR:        str = os.getenv("VAULT_DIR",        os.path.join(BASE_DIR, "vault"))
EXPLANATIONS_DIR: str = os.getenv("EXPLANATIONS_DIR", os.path.join(BASE_DIR, "explanations"))
AUDIT_DIR:        str = os.getenv("AUDIT_DIR",        os.path.join(BASE_DIR, "audit"))
LOGS_DIR:         str = os.getenv("LOGS_DIR",         os.path.join(BASE_DIR, "logs"))

# ── ML Models ──────────────────────────────────────────────────────────────
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_trf")
SBERT_MODEL: str = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")

# ── File upload ────────────────────────────────────────────────────────────
MAX_UPLOAD_MB:         int       = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES:      int       = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_CV_EXTENSIONS: frozenset = frozenset({".pdf", ".docx", ".txt"})

# ── Rate limiting ──────────────────────────────────────────────────────────
RATE_LIMIT_PIPELINE: str = os.getenv("RATE_LIMIT_PIPELINE", "5 per hour")
RATE_LIMIT_UPLOAD:   str = os.getenv("RATE_LIMIT_UPLOAD",   "20 per hour")

# ── Role-level ranking thresholds ─────────────────────────────────────────
#   junior: accepts lower scores  |  senior: demands higher scores
VERDICT_THRESHOLDS: dict = {
    "junior": {"strong": 0.60, "good": 0.45, "partial": 0.30},
    "mid":    {"strong": 0.75, "good": 0.60, "partial": 0.45},
    "senior": {"strong": 0.85, "good": 0.70, "partial": 0.55},
}
DEFAULT_ROLE_LEVEL: str = os.getenv("ROLE_LEVEL", "mid")

# ── Bias detection ────────────────────────────────────────────────────────
COMPRESSION_THRESHOLD:     float = float(os.getenv("COMPRESSION_THRESHOLD",     "0.10"))
EXPERIENCE_CORR_THRESHOLD: float = float(os.getenv("EXPERIENCE_CORR_THRESHOLD", "0.85"))
