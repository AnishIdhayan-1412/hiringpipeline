# -*- coding: utf-8 -*-
"""
exceptions.py — Custom Exception Hierarchy
============================================
Bias-Free Hiring Pipeline

Typed exception classes for every stage of the pipeline.
Using typed exceptions instead of bare ``Exception`` gives three benefits:

1. **Granular catching**: ``main.py`` can catch ``RankingError`` without
   accidentally swallowing unrelated ``ValueError``s from third-party code.
2. **Structured context**: every exception carries the ``stage`` and
   optional ``candidate_id`` that caused it, making log messages and
   error responses self-documenting.
3. **HTTP mapping**: ``module5.py`` can map exception types directly to
   HTTP status codes without ad-hoc string matching.

Hierarchy
---------
    PipelineError (base)
    ├── ConfigurationError       — bad arguments, missing files
    ├── AnonymizationError       — module0 failures
    │   └── PIILeakError         — validation found residual PII
    ├── ParsingError             — module0b failures
    ├── RankingError             — module1 failures
    │   └── ModelLoadError       — SBERT / spaCy load failures
    ├── ExplainabilityError      — module2 failures
    ├── AuditError               — module3 failures
    ├── BiasAuditError           — module4 failures
    ├── DashboardError           — module5 / Flask failures
    └── DatabaseError            — SQLite / persistence failures (Sprint 1)
"""

from __future__ import annotations

from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Base exception
# ═══════════════════════════════════════════════════════════════════════════

class PipelineError(Exception):
    """Base class for all Bias-Free Hiring Pipeline exceptions.

    Args:
        message:      Human-readable description of what went wrong.
        stage:        Pipeline stage that raised the error
                      (e.g. ``"module0"``, ``"module1"``).
        candidate_id: The candidate being processed when the error
                      occurred, if applicable.
    """

    def __init__(
        self,
        message:      str,
        stage:        Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> None:
        self.stage        = stage
        self.candidate_id = candidate_id
        super().__init__(message)

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.stage:
            parts.append(f"[stage={self.stage}]")
        if self.candidate_id:
            parts.append(f"[candidate={self.candidate_id}]")
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class ConfigurationError(PipelineError):
    """Raised when the pipeline is invoked with invalid or missing configuration.

    Examples:
        - ``--jd-file`` points to a file that does not exist.
        - A required environment variable is absent.
        - An argument combination is mutually exclusive.
    """


# ═══════════════════════════════════════════════════════════════════════════
# Stage 0 — Anonymization
# ═══════════════════════════════════════════════════════════════════════════

class AnonymizationError(PipelineError):
    """Raised when module0 cannot anonymize a CV.

    Args:
        message:      Description of the failure.
        candidate_id: The candidate whose CV could not be anonymized.
        filepath:     Path to the CV file that caused the error.
    """

    def __init__(
        self,
        message:      str,
        candidate_id: Optional[str] = None,
        filepath:     Optional[str] = None,
    ) -> None:
        self.filepath = filepath
        super().__init__(message, stage="module0", candidate_id=candidate_id)

    def __str__(self) -> str:
        base = super().__str__()
        if self.filepath:
            base += f" [file={self.filepath}]"
        return base


class PIILeakError(AnonymizationError):
    """Raised when post-anonymization validation detects residual PII.

    This is a sub-class of ``AnonymizationError`` but carries additional
    context about what kind of PII was detected, so callers can decide
    whether to abort the pipeline or log a warning and continue.

    Args:
        message:      Description of the validation failure.
        candidate_id: The candidate whose anonymized CV failed validation.
        leaks:        List of leak descriptors from ``validate_pii_removal()``.
        confidence:   Validation confidence score (0–100).
    """

    def __init__(
        self,
        message:      str,
        candidate_id: Optional[str]      = None,
        leaks:        Optional[list]     = None,
        confidence:   float              = 0.0,
    ) -> None:
        self.leaks      = leaks or []
        self.confidence = confidence
        super().__init__(message, candidate_id=candidate_id)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 0b — Parsing
# ═══════════════════════════════════════════════════════════════════════════

class ParsingError(PipelineError):
    """Raised when module0b cannot parse a CV's sections or extract skills.

    Args:
        message:      Description of the failure.
        candidate_id: The candidate whose parsed JSON could not be produced.
    """

    def __init__(
        self,
        message:      str,
        candidate_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, stage="module0b", candidate_id=candidate_id)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Ranking
# ═══════════════════════════════════════════════════════════════════════════

class RankingError(PipelineError):
    """Raised when module1 cannot rank candidates.

    Args:
        message: Description of the failure (e.g. empty CV directory,
                 empty JD, SBERT encoding failure).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="module1")


class ModelLoadError(RankingError):
    """Raised when a required ML model cannot be loaded.

    This is a sub-class of ``RankingError`` so it can be caught at the
    ranking stage, but it also signals an infrastructure problem that may
    require a Docker rebuild or ``pip install`` rather than a user fix.

    Args:
        model_name: The model identifier that failed to load
                    (e.g. ``"all-MiniLM-L6-v2"``).
        reason:     The underlying error message from the ML library.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        super().__init__(f"Could not load model '{model_name}': {reason}")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Explainability
# ═══════════════════════════════════════════════════════════════════════════

class ExplainabilityError(PipelineError):
    """Raised when module2 cannot generate an explanation for a candidate.

    Args:
        message:      Description of the failure.
        candidate_id: The candidate for whom explanation generation failed.
    """

    def __init__(
        self,
        message:      str,
        candidate_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, stage="module2", candidate_id=candidate_id)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — Audit
# ═══════════════════════════════════════════════════════════════════════════

class AuditError(PipelineError):
    """Raised when module3 cannot generate the GDPR audit trail.

    Args:
        message: Description of the failure (e.g. cannot create audit/
                 directory, cannot write audit_log.json).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="module3")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — Bias Audit
# ═══════════════════════════════════════════════════════════════════════════

class BiasAuditError(PipelineError):
    """Raised when module4 cannot complete the bias detection audit.

    Args:
        message: Description of the failure.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="module4")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════

class DashboardError(PipelineError):
    """Raised when module5 (Flask) encounters a fatal startup error.

    Args:
        message: Description of the failure (e.g. port already in use,
                 Flask not installed).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="module5")


# ═══════════════════════════════════════════════════════════════════════════
# Persistence (Sprint 1 — SQLite layer)
# ═══════════════════════════════════════════════════════════════════════════

class DatabaseError(PipelineError):
    """Raised when the SQLite persistence layer encounters an unrecoverable error.

    This exception is reserved for Sprint 1 (database integration).
    It is defined here now so that any module can import it without a
    circular-dependency risk.

    Args:
        message:    Description of the database failure.
        operation:  The SQL operation that failed (e.g. ``"INSERT"``,
                    ``"CREATE TABLE"``).
    """

    def __init__(
        self,
        message:   str,
        operation: Optional[str] = None,
    ) -> None:
        self.operation = operation
        super().__init__(message, stage="database")

    def __str__(self) -> str:
        base = super().__str__()
        if self.operation:
            base += f" [op={self.operation}]"
        return base
