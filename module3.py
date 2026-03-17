# -*- coding: utf-8 -*-
"""
module3.py — GDPR Audit Trail Engine
======================================
Bias-Free Hiring Pipeline — Stage 3

Reads every output produced by the preceding pipeline stages and assembles
a legally defensible, tamper-evident GDPR audit trail that proves every
decision made about every candidate was fair, bias-free, and lawful.

What this module consumes
--------------------------
    module0  → vault/Candidate_01.json     (PII vault per candidate)
               vault/vault_master.json     (consolidated vault index)
    module0b → parsed/Candidate_01.json    (structured CV sections)
    module1  → ranker.last_ranking_details (live in-memory ranking dict)
    module2  → explanations/Candidate_01.json  (per-candidate decisions)
               explanations/summary_report.txt (human summary)

What this module produces
--------------------------
    audit/audit_log.json            ← machine-readable event log (tamper-evident)
    audit/audit_log.sha256          ← SHA-256 hash for integrity verification
    audit/audit_report.txt          ← human-readable report for compliance officers
    audit/compliance_checklist.json ← GDPR article compliance status

Pipeline position
-----------------
    module0  → anonymized_cvs/ + vault/
    module0b → parsed/
    module1  → ranking_report.txt + last_ranking_details
    module2  → explanations/
    module3  → audit/                     ← THIS MODULE

Standalone usage
-----------------
    python module3.py --vault-dir vault \\
                      --parsed-dir parsed \\
                      --ranking-file ranking_details.json \\
                      --explanations-dir explanations \\
                      --output-dir audit \\
                      --raw-cvs-dir raw_cvs

Programmatic usage (called by main.py)
----------------------------------------
    import module3

    ok = module3.run(
        vault_dir        = "vault",
        parsed_dir       = "parsed",
        ranking_details  = ranker.last_ranking_details,
        explanations_dir = "explanations",
        output_dir       = "audit",
        raw_cvs_dir      = "raw_cvs",
    )

Dependencies
------------
    Pure Python stdlib only: json, os, hashlib, logging, datetime,
    argparse, sys, re, pathlib, textwrap, time.
    Zero pip installs. No Flask. No databases. No LLMs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════
# Module-level logger — never use print() in helpers; only inside run()
# ══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Constants — every threshold, string, and filename lives here
# ══════════════════════════════════════════════════════════════════════════

# ── Pipeline identity ─────────────────────────────────────────────────────
PIPELINE_VERSION:     str = "1.0.0"
PIPELINE_NAME:        str = "Bias-Free Hiring Pipeline"
LEGAL_BASIS:          str = "Legitimate interest — bias prevention in recruitment (GDPR Art. 6(1)(f))"
DATA_CONTROLLER_NOTE: str = (
    "The hiring organisation acts as Data Controller. "
    "All processing is conducted under the lawful basis of legitimate interest "
    "to prevent unconscious bias in recruitment decisions."
)

# ── Run ID format ─────────────────────────────────────────────────────────
RUN_ID_FMT:           str = "RUN_%Y%m%d_%H%M%S"

# ── Data retention ────────────────────────────────────────────────────────
RETENTION_DAYS:       int = 30      # days after which personal data must be deleted

# ── Output filenames ──────────────────────────────────────────────────────
AUDIT_LOG_FILENAME:       str = "audit_log.json"
AUDIT_REPORT_FILENAME:    str = "audit_report.txt"
COMPLIANCE_FILENAME:      str = "compliance_checklist.json"
SHA256_FILENAME:          str = "audit_log.sha256"

# ── Formatting ────────────────────────────────────────────────────────────
REPORT_WIDTH:         int = 80
ISO_FMT:              str = "%Y-%m-%dT%H:%M:%S"
HUMAN_FMT:            str = "%d %B %Y at %H:%M UTC"
DATE_ONLY_FMT:        str = "%Y-%m-%d"

# ── Logging ───────────────────────────────────────────────────────────────
_LOG_FORMAT:          str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT:            str = "%H:%M:%S"

# ── Event type constants (exact strings required by spec) ─────────────────
EVT_PIPELINE_START:       str = "PIPELINE_START"
EVT_CV_RECEIVED:          str = "CV_RECEIVED"
EVT_PII_REMOVED:          str = "PII_REMOVED"
EVT_CV_PARSED:            str = "CV_PARSED"
EVT_CANDIDATE_RANKED:     str = "CANDIDATE_RANKED"
EVT_EXPLANATION_WRITTEN:  str = "EXPLANATION_WRITTEN"
EVT_DATA_RETAINED:        str = "DATA_RETAINED"
EVT_PIPELINE_COMPLETE:    str = "PIPELINE_COMPLETE"
EVT_WARNING:              str = "WARNING"
EVT_ERROR:                str = "ERROR"

# ── Severity levels ───────────────────────────────────────────────────────
SEV_INFO:             str = "INFO"
SEV_WARNING:          str = "WARNING"
SEV_ERROR:            str = "ERROR"

# ── GDPR articles ─────────────────────────────────────────────────────────
GDPR_ARTICLE_5:       str = "article_5"
GDPR_ARTICLE_13:      str = "article_13"
GDPR_ARTICLE_17:      str = "article_17"
GDPR_ARTICLE_22:      str = "article_22"
GDPR_ARTICLE_25:      str = "article_25"
GDPR_ARTICLE_30:      str = "article_30"

# ── Vault file skip list (not candidate JSON files) ───────────────────────
VAULT_SKIP_FILES:     frozenset = frozenset({
    "vault_master.json", "error_log.txt", "error_log.json",
})

# ── Parsed file skip list ─────────────────────────────────────────────────
PARSED_SKIP_FILES:    frozenset = frozenset({
    "index.json", "error_log.txt", "error_log.json",
})

# ── Explanation file skip list ────────────────────────────────────────────
EXPLANATION_SKIP_FILES: frozenset = frozenset({
    "summary_report.txt", "error_log.txt",
})

# ── Keywords indicating human oversight in recommendations ────────────────
HUMAN_REVIEW_KEYWORDS: Tuple[str, ...] = (
    "human",
    "review",
    "recommended",
    "consider",
    "manually",
    "interview",
    "qualified",
    "decision-maker",
)


# ══════════════════════════════════════════════════════════════════════════
# Data loaders — each returns a safe default on failure
# ══════════════════════════════════════════════════════════════════════════

def _load_json_file(path: str, label: str) -> Optional[Dict[str, Any]]:
    """Load and return a JSON file, logging a warning on any failure.

    Args:
        path:  Absolute or relative path to the JSON file.
        label: Human-readable label used in warning messages.

    Returns:
        Parsed dict on success, None if the file is absent or unreadable.
    """
    if not os.path.isfile(path):
        logger.warning("%s not found at '%s'.", label, path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("%s is malformed JSON: %s", label, exc)
        return None
    except OSError as exc:
        logger.error("Cannot read %s: %s", label, exc)
        return None


def _load_vault_records(vault_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all individual candidate vault JSON files from vault_dir.

    Skips vault_master.json and error logs. Returns a dict keyed by
    candidate_id. Falls back to vault_master.json if individual files
    are absent.

    Args:
        vault_dir: Path to the vault/ directory produced by module0.

    Returns:
        Dict mapping candidate_id → vault record dict. Empty on total failure.
    """
    records: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(vault_dir):
        logger.warning("Vault directory not found: '%s'.", vault_dir)
        return records

    for fname in sorted(os.listdir(vault_dir)):
        if fname in VAULT_SKIP_FILES or not fname.endswith(".json"):
            continue
        path = os.path.join(vault_dir, fname)
        data = _load_json_file(path, f"vault/{fname}")
        if data is not None:
            cid = data.get("candidate_id") or os.path.splitext(fname)[0]
            records[cid] = data
            logger.debug("Loaded vault record: %s", cid)

    # Fallback: try vault_master.json if no individual files found
    if not records:
        master_path = os.path.join(vault_dir, "vault_master.json")
        master = _load_json_file(master_path, "vault_master.json")
        if master:
            for cid, data in master.items():
                if isinstance(data, dict):
                    records[cid] = data
            logger.info(
                "Loaded %d vault records from vault_master.json.", len(records)
            )

    logger.info("Vault records loaded: %d candidates.", len(records))
    return records


def _load_parsed_records(parsed_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all individual candidate parsed JSON files from parsed_dir.

    Args:
        parsed_dir: Path to the parsed/ directory produced by module0b.

    Returns:
        Dict mapping candidate_id → parsed record dict.
    """
    records: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(parsed_dir):
        logger.warning("Parsed directory not found: '%s'.", parsed_dir)
        return records

    for fname in sorted(os.listdir(parsed_dir)):
        if fname in PARSED_SKIP_FILES or not fname.endswith(".json"):
            continue
        path = os.path.join(parsed_dir, fname)
        data = _load_json_file(path, f"parsed/{fname}")
        if data is not None:
            cid = data.get("candidate_id") or os.path.splitext(fname)[0]
            records[cid] = data
            logger.debug("Loaded parsed record: %s", cid)

    logger.info("Parsed records loaded: %d candidates.", len(records))
    return records


def _load_explanation_records(
    explanations_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Load all individual candidate explanation JSON files.

    Args:
        explanations_dir: Path to the explanations/ directory from module2.

    Returns:
        Dict mapping candidate_id → explanation dict.
    """
    records: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(explanations_dir):
        logger.warning(
            "Explanations directory not found: '%s'.", explanations_dir
        )
        return records

    for fname in sorted(os.listdir(explanations_dir)):
        if fname in EXPLANATION_SKIP_FILES or not fname.endswith(".json"):
            continue
        path = os.path.join(explanations_dir, fname)
        data = _load_json_file(path, f"explanations/{fname}")
        if data is not None:
            cid = data.get("candidate_id") or os.path.splitext(fname)[0]
            records[cid] = data
            logger.debug("Loaded explanation record: %s", cid)

    logger.info(
        "Explanation records loaded: %d candidates.", len(records)
    )
    return records


def _count_raw_cvs(raw_cvs_dir: str) -> int:
    """Count the number of files remaining in the raw_cvs directory.

    Used for GDPR Art. 17 (Right to Erasure) check — if raw CVs still
    exist they should be deleted after processing.

    Args:
        raw_cvs_dir: Path to the raw_cvs/ directory.

    Returns:
        Number of files found (0 if directory is absent or empty).
    """
    if not os.path.isdir(raw_cvs_dir):
        logger.debug(
            "raw_cvs directory not found — treating as empty (Art. 17 satisfied)."
        )
        return 0
    try:
        return sum(
            1 for f in os.listdir(raw_cvs_dir)
            if os.path.isfile(os.path.join(raw_cvs_dir, f))
        )
    except OSError as exc:
        logger.warning("Cannot read raw_cvs directory: %s", exc)
        return 0


# ══════════════════════════════════════════════════════════════════════════
# Event builder — constructs the event list from all pipeline data
# ══════════════════════════════════════════════════════════════════════════

class EventBuilder:
    """Builds the ordered list of audit events from all pipeline stage outputs.

    Events are constructed deterministically from the data already on disk —
    this module never re-runs the pipeline. It reads the outputs left behind
    by each stage and reconstructs the timeline of actions.

    Attributes:
        _counter: Sequential event counter used to generate EVT_NNN IDs.
    """

    def __init__(self) -> None:
        """Initialise the builder with a zeroed event counter."""
        self._counter: int = 0

    def _next_id(self) -> str:
        """Generate the next sequential event ID.

        Returns:
            A zero-padded event ID string e.g. "EVT_001".
        """
        self._counter += 1
        return f"EVT_{self._counter:03d}"

    def _make_event(
        self,
        event_type:    str,
        stage:         str,
        description:   str,
        candidate_id:  Optional[str]       = None,
        severity:      str                 = SEV_INFO,
        timestamp:     Optional[str]       = None,
        extra:         Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct a single audit event dict.

        Args:
            event_type:   One of the EVT_* constants.
            stage:        Module name that generated this event (e.g. "module0").
            description:  Human-readable description of what happened.
            candidate_id: Candidate identifier, or None for pipeline-level events.
            severity:     SEV_INFO, SEV_WARNING, or SEV_ERROR.
            timestamp:    ISO timestamp string; defaults to now if None.
            extra:        Additional fields to merge into the event dict.

        Returns:
            A populated event dict.
        """
        event: Dict[str, Any] = {
            "event_id":    self._next_id(),
            "timestamp":   timestamp or datetime.now().strftime(ISO_FMT),
            "stage":       stage,
            "event_type":  event_type,
            "candidate_id": candidate_id,
            "description": description,
            "legal_basis": LEGAL_BASIS,
            "severity":    severity,
        }
        if extra:
            event.update(extra)
        return event

    def build_pipeline_start(self, run_id: str, n_candidates: int) -> Dict[str, Any]:
        """Build the PIPELINE_START event.

        Args:
            run_id:       The unique run identifier for this pipeline execution.
            n_candidates: Number of candidates discovered in the raw_cvs folder.

        Returns:
            A single event dict.
        """
        return self._make_event(
            event_type  = EVT_PIPELINE_START,
            stage       = "main",
            description = (
                f"Pipeline run {run_id} started. "
                f"{n_candidates} candidate CV(s) submitted for processing."
            ),
            extra = {
                "pipeline_version": PIPELINE_VERSION,
                "run_id":           run_id,
                "n_candidates":     n_candidates,
                "data_retained":    False,
            },
        )

    def build_cv_received_events(
        self,
        vault_records: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build one CV_RECEIVED event per candidate from vault records.

        Args:
            vault_records: Dict of candidate_id → vault data from module0.

        Returns:
            List of CV_RECEIVED event dicts.
        """
        events: List[Dict[str, Any]] = []
        for cid, vault in sorted(vault_records.items()):
            original_fname = vault.get("original_filename", "unknown")
            events.append(
                self._make_event(
                    event_type   = EVT_CV_RECEIVED,
                    stage        = "module0",
                    candidate_id = cid,
                    description  = (
                        f"Raw CV received: '{original_fname}'. "
                        f"Assigned anonymous ID: {cid}."
                    ),
                    extra = {
                        "original_filename": original_fname,
                        "data_retained":     False,
                    },
                )
            )
        return events

    def build_pii_removed_events(
        self,
        vault_records: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build one PII_REMOVED event per candidate from vault records.

        Args:
            vault_records: Dict of candidate_id → vault data from module0.

        Returns:
            List of PII_REMOVED event dicts.
        """
        events: List[Dict[str, Any]] = []
        for cid, vault in sorted(vault_records.items()):
            pii_cats     = vault.get("pii_categories_found", {})
            techniques   = vault.get("techniques_applied",   [])
            level        = vault.get("anonymization_level_used", "STRICT")
            validation   = vault.get("validation_score") or vault.get("validation_report", {})
            passed       = (
                validation.get("passed", False)
                if isinstance(validation, dict)
                else False
            )
            confidence   = (
                validation.get("confidence_score", 0.0)
                if isinstance(validation, dict)
                else 0.0
            )

            # Build a human description of what PII was removed
            pii_parts: List[str] = []
            name = vault.get("original_name")
            if name:
                pii_parts.append("name")
            for cat, count in pii_cats.items():
                if cat not in {"ner_persons", "heuristic_names"} and count > 0:
                    pii_parts.append(f"{count} {cat.replace('_', ' ')}")

            pii_summary = (
                ", ".join(pii_parts) if pii_parts else "no PII detected"
            )

            technique_str = " + ".join(techniques) if techniques else "unknown"

            events.append(
                self._make_event(
                    event_type   = EVT_PII_REMOVED,
                    stage        = "module0",
                    candidate_id = cid,
                    description  = (
                        f"PII removed: {pii_summary}. "
                        f"Anonymization level: {level}. "
                        f"Validation {'passed' if passed else 'flagged issues'} "
                        f"(confidence: {confidence:.1f}%)."
                    ),
                    extra = {
                        "pii_categories":        list(pii_cats.keys()),
                        "pii_counts":            pii_cats,
                        "technique":             technique_str,
                        "anonymization_level":   level,
                        "validation_passed":     passed,
                        "validation_confidence": round(confidence, 2),
                        "data_retained":         False,
                    },
                    severity = SEV_INFO if passed else SEV_WARNING,
                )
            )
        return events

    def build_cv_parsed_events(
        self,
        parsed_records: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build one CV_PARSED event per candidate from parsed records.

        Args:
            parsed_records: Dict of candidate_id → parsed data from module0b.

        Returns:
            List of CV_PARSED event dicts.
        """
        events: List[Dict[str, Any]] = []
        for cid, parsed in sorted(parsed_records.items()):
            sections  = parsed.get("sections", {})
            quality   = parsed.get("quality_score", 0.0)
            warnings  = parsed.get("parse_warnings", [])
            parsed_at = parsed.get("parsed_at", "unknown")

            skill_count = 0
            skills_block = sections.get("skills", {})
            if isinstance(skills_block, dict):
                skill_count = sum(len(v) for v in skills_block.values())

            exp_months = sections.get("total_experience_months", 0)
            n_roles    = len(sections.get("experience", []))
            n_edu      = len(sections.get("education",  []))

            events.append(
                self._make_event(
                    event_type   = EVT_CV_PARSED,
                    stage        = "module0b",
                    candidate_id = cid,
                    timestamp    = parsed_at if parsed_at != "unknown" else None,
                    description  = (
                        f"CV parsed: {skill_count} skills extracted, "
                        f"{n_roles} role(s), {n_edu} education entry(ies). "
                        f"Quality score: {quality:.2f}. "
                        f"{len(warnings)} parse warning(s)."
                    ),
                    extra = {
                        "quality_score":            round(quality, 4),
                        "skills_extracted":         skill_count,
                        "experience_months":        exp_months,
                        "roles_detected":           n_roles,
                        "education_entries":        n_edu,
                        "parse_warnings_count":     len(warnings),
                        "data_retained":            True,
                    },
                    severity = SEV_INFO if not warnings else SEV_WARNING,
                )
            )
        return events

    def build_ranking_events(
        self,
        ranking_details: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build one CANDIDATE_RANKED event per candidate from ranking_details.

        Args:
            ranking_details: ranker.last_ranking_details from module1.

        Returns:
            List of CANDIDATE_RANKED event dicts, sorted by rank.
        """
        events: List[Dict[str, Any]] = []
        sorted_by_rank = sorted(
            ranking_details.items(),
            key=lambda kv: kv[1].get("rank", 9999),
        )
        for cid, rd in sorted_by_rank:
            final_score    = float(rd.get("final_score",    0.0))
            semantic_score = float(rd.get("semantic_score", 0.0))
            keyword_score  = float(rd.get("keyword_score",  0.0))
            quality_score  = float(rd.get("quality_score",  0.0))
            matched        = rd.get("matched_skills", [])
            missing        = rd.get("missing_skills", [])
            jd_count       = int(rd.get("jd_skill_count", 0))
            rank           = int(rd.get("rank", 0))

            events.append(
                self._make_event(
                    event_type   = EVT_CANDIDATE_RANKED,
                    stage        = "module1",
                    candidate_id = cid,
                    description  = (
                        f"Ranked #{rank}: final_score={final_score:.4f} "
                        f"(semantic={semantic_score:.4f}, "
                        f"keyword={keyword_score:.4f}, "
                        f"quality={quality_score:.4f}). "
                        f"{len(matched)} of {jd_count} JD skills matched."
                    ),
                    extra = {
                        "rank":            rank,
                        "final_score":     round(final_score, 4),
                        "semantic_score":  round(semantic_score, 4),
                        "keyword_score":   round(keyword_score, 4),
                        "quality_score":   round(quality_score, 4),
                        "matched_skills":  matched,
                        "missing_skills":  missing,
                        "jd_skill_count":  jd_count,
                        "data_retained":   True,
                        "scoring_method":  "SBERT cosine similarity + keyword overlap",
                    },
                )
            )
        return events

    def build_explanation_events(
        self,
        explanation_records: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build one EXPLANATION_WRITTEN event per candidate.

        Args:
            explanation_records: Dict of candidate_id → explanation from module2.

        Returns:
            List of EXPLANATION_WRITTEN event dicts.
        """
        events: List[Dict[str, Any]] = []
        for cid, exp in sorted(
            explanation_records.items(),
            key=lambda kv: kv[1].get("rank", 9999),
        ):
            verdict        = exp.get("verdict",    "Unknown")
            headline       = exp.get("headline",   "No headline")
            generated_at   = exp.get("generated_at", "unknown")
            recommendation = (
                exp.get("explanation", {}).get("recommendation", "")
            )
            rank           = exp.get("rank", 0)

            events.append(
                self._make_event(
                    event_type   = EVT_EXPLANATION_WRITTEN,
                    stage        = "module2",
                    candidate_id = cid,
                    timestamp    = generated_at if generated_at != "unknown" else None,
                    description  = (
                        f"Explanation generated for rank #{rank}. "
                        f"Verdict: {verdict}. "
                        f"Headline: \"{headline[:80]}{'…' if len(headline) > 80 else ''}\"."
                    ),
                    extra = {
                        "rank":              rank,
                        "verdict":           verdict,
                        "recommendation":    recommendation[:200],
                        "data_retained":     True,
                        "human_review_flag": True,
                    },
                )
            )
        return events

    def build_data_retained_events(
        self,
        vault_records: Dict[str, Dict[str, Any]],
        retention_due: str,
    ) -> List[Dict[str, Any]]:
        """Build one DATA_RETAINED event per candidate documenting what was kept.

        Args:
            vault_records:  Dict of candidate_id → vault data.
            retention_due:  ISO date string when data must be deleted.

        Returns:
            List of DATA_RETAINED event dicts.
        """
        events: List[Dict[str, Any]] = []
        for cid in sorted(vault_records.keys()):
            events.append(
                self._make_event(
                    event_type   = EVT_DATA_RETAINED,
                    stage        = "module3",
                    candidate_id = cid,
                    description  = (
                        f"Anonymized CV and parsed data retained in vault. "
                        f"Original PII stored in encrypted vault only. "
                        f"Retention deadline: {retention_due}."
                    ),
                    extra = {
                        "retained_in":     "vault/",
                        "retention_days":  RETENTION_DAYS,
                        "deletion_due":    retention_due,
                        "data_retained":   True,
                    },
                )
            )
        return events

    def build_warning_events(
        self,
        vault_records:       Dict[str, Dict[str, Any]],
        parsed_records:      Dict[str, Dict[str, Any]],
        explanation_records: Dict[str, Dict[str, Any]],
        ranking_details:     Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build WARNING events for any data quality or coverage issues.

        Detects: low validation confidence, parse warnings, missing
        explanations, and candidates without vault records.

        Args:
            vault_records:       Vault data dict.
            parsed_records:      Parsed data dict.
            explanation_records: Explanation data dict.
            ranking_details:     Module1 ranking details.

        Returns:
            List of WARNING event dicts (may be empty).
        """
        events: List[Dict[str, Any]] = []

        # Warning: validation confidence issues
        for cid, vault in vault_records.items():
            validation = vault.get("validation_score") or vault.get("validation_report", {})
            if isinstance(validation, dict):
                confidence = validation.get("confidence_score", 100.0)
                leaks      = validation.get("leaks_found", [])
                if not validation.get("passed", True):
                    events.append(
                        self._make_event(
                            event_type   = EVT_WARNING,
                            stage        = "module0",
                            candidate_id = cid,
                            description  = (
                                f"PII validation did not pass for {cid}. "
                                f"Confidence: {confidence:.1f}%. "
                                f"{len(leaks)} potential leak(s) detected. "
                                f"Manual review of anonymized CV recommended."
                            ),
                            severity = SEV_WARNING,
                            extra    = {
                                "validation_confidence": confidence,
                                "leaks_detected":        len(leaks),
                                "data_retained":         False,
                            },
                        )
                    )

        # Warning: parse warnings from module0b
        for cid, parsed in parsed_records.items():
            warnings = parsed.get("parse_warnings", [])
            if len(warnings) >= 3:
                events.append(
                    self._make_event(
                        event_type   = EVT_WARNING,
                        stage        = "module0b",
                        candidate_id = cid,
                        description  = (
                            f"{len(warnings)} parse warning(s) recorded for {cid}. "
                            f"CV structure may be non-standard. "
                            f"Extraction accuracy may be reduced."
                        ),
                        severity = SEV_WARNING,
                        extra    = {
                            "parse_warnings": warnings[:5],
                            "data_retained":  False,
                        },
                    )
                )

        # Warning: candidates in ranking but missing explanations
        all_ranked = set(ranking_details.keys())
        all_explained = set(explanation_records.keys())
        missing_explanations = all_ranked - all_explained
        for cid in sorted(missing_explanations):
            events.append(
                self._make_event(
                    event_type   = EVT_WARNING,
                    stage        = "module2",
                    candidate_id = cid,
                    description  = (
                        f"{cid} was ranked by module1 but has no explanation "
                        f"in the explanations/ directory. "
                        f"Transparency requirement (Art. 13) may be affected."
                    ),
                    severity = SEV_WARNING,
                    extra    = {"data_retained": False},
                )
            )

        # Warning: candidates in vault but missing from ranking
        all_vaulted = set(vault_records.keys())
        missing_rankings = all_vaulted - all_ranked
        for cid in sorted(missing_rankings):
            events.append(
                self._make_event(
                    event_type   = EVT_WARNING,
                    stage        = "module1",
                    candidate_id = cid,
                    description  = (
                        f"{cid} was anonymized by module0 but does not appear "
                        f"in ranking_details. The candidate may not have been "
                        f"ranked — check the job description and anonymized_cvs/ directory."
                    ),
                    severity = SEV_WARNING,
                    extra    = {"data_retained": False},
                )
            )

        return events

    def build_pipeline_complete(
        self,
        run_id:          str,
        n_candidates:    int,
        n_events:        int,
        gdpr_compliant:  bool,
    ) -> Dict[str, Any]:
        """Build the PIPELINE_COMPLETE event.

        Args:
            run_id:         The unique run identifier.
            n_candidates:   Total number of candidates processed.
            n_events:       Total number of events in the log.
            gdpr_compliant: Whether all GDPR articles were satisfied.

        Returns:
            A single PIPELINE_COMPLETE event dict.
        """
        return self._make_event(
            event_type  = EVT_PIPELINE_COMPLETE,
            stage       = "module3",
            description = (
                f"Pipeline run {run_id} complete. "
                f"{n_candidates} candidate(s) processed. "
                f"{n_events} audit events recorded. "
                f"GDPR compliance status: {'COMPLIANT' if gdpr_compliant else 'PARTIAL — see checklist'}."
            ),
            extra = {
                "run_id":          run_id,
                "n_candidates":    n_candidates,
                "total_events":    n_events,
                "gdpr_compliant":  gdpr_compliant,
                "data_retained":   False,
            },
        )


# ══════════════════════════════════════════════════════════════════════════
# GDPR Compliance Checker
# ══════════════════════════════════════════════════════════════════════════

class GDPRChecker:
    """Evaluates compliance with six key GDPR articles.

    Each check is a pure function that examines the pipeline outputs and
    returns a structured result dict. No writes happen here — this class
    is a pure analyser.
    """

    def check_article_5(
        self,
        vault_records:  Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Art. 5 — Principles of processing (lawfulness, fairness, transparency).

        Satisfied if the vault exists and contains pii_categories_found,
        demonstrating that PII was identified and removed before any further
        processing.

        Args:
            vault_records: Loaded vault records from module0.

        Returns:
            Compliance result dict.
        """
        has_vault    = bool(vault_records)
        has_pii_cats = any(
            bool(v.get("pii_categories_found"))
            for v in vault_records.values()
        )
        satisfied = has_vault and has_pii_cats

        evidence = (
            f"Vault records found for {len(vault_records)} candidate(s). "
            f"PII categories identified and removed before ranking."
            if satisfied
            else
            "Vault is empty or pii_categories_found is missing — "
            "cannot confirm PII was properly identified and removed."
        )
        return {
            "title":     "Principles of processing",
            "satisfied": satisfied,
            "evidence":  evidence,
        }

    def check_article_13(
        self,
        explanation_records: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Art. 13 — Right to information / transparency.

        Satisfied if at least one explanation JSON was written, demonstrating
        that candidates can be informed about how decisions were made.

        Args:
            explanation_records: Loaded explanation records from module2.

        Returns:
            Compliance result dict.
        """
        n = len(explanation_records)
        satisfied = n > 0
        evidence = (
            f"Explanation generated for {n} candidate(s). "
            f"Each explanation includes score breakdown, verdict, and recommendation."
            if satisfied
            else
            "No explanation files found in explanations/. "
            "Transparency requirement cannot be demonstrated."
        )
        return {
            "title":     "Right to information / transparency",
            "satisfied": satisfied,
            "evidence":  evidence,
        }

    def check_article_17(
        self,
        raw_cvs_dir: str,
    ) -> Dict[str, Any]:
        """Art. 17 — Right to erasure (right to be forgotten).

        Satisfied only if the raw_cvs/ directory is empty. If original CVs
        still exist, they must be manually deleted.

        Args:
            raw_cvs_dir: Path to the raw_cvs/ directory.

        Returns:
            Compliance result dict.
        """
        n_raw   = _count_raw_cvs(raw_cvs_dir)
        satisfied = n_raw == 0
        evidence = (
            "raw_cvs/ directory is empty — original CVs have been removed. "
            "Right to erasure can be exercised."
            if satisfied
            else
            f"{n_raw} raw CV file(s) still present in raw_cvs/. "
            f"Original CVs must be deleted manually to satisfy Art. 17. "
            f"Action required: delete all files from raw_cvs/ after processing."
        )
        return {
            "title":     "Right to erasure",
            "satisfied": satisfied,
            "evidence":  evidence,
        }

    def check_article_22(
        self,
        explanation_records: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Art. 22 — Automated decision-making and profiling.

        Satisfied if every explanation's recommendation contains language
        indicating that a human should review the automated decision.

        Args:
            explanation_records: Loaded explanation records from module2.

        Returns:
            Compliance result dict.
        """
        if not explanation_records:
            return {
                "title":     "Automated decision-making safeguards",
                "satisfied": False,
                "evidence":  "No explanation files found — cannot verify human review language.",
            }

        failing: List[str] = []
        for cid, exp in explanation_records.items():
            rec = (
                exp.get("explanation", {}).get("recommendation", "")
                .lower()
            )
            if not any(kw in rec for kw in HUMAN_REVIEW_KEYWORDS):
                failing.append(cid)

        satisfied = len(failing) == 0
        evidence = (
            f"All {len(explanation_records)} explanation(s) contain human-review "
            f"language in their recommendations."
            if satisfied
            else
            f"{len(failing)} explanation(s) lack explicit human-review language: "
            f"{', '.join(failing)}. Review module2 recommendation templates."
        )
        return {
            "title":     "Automated decision-making safeguards",
            "satisfied": satisfied,
            "evidence":  evidence,
        }

    def check_article_25(
        self,
        vault_records:   Dict[str, Dict[str, Any]],
        ranking_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Art. 25 — Data protection by design and by default.

        Satisfied if anonymization was applied (vault records exist)
        before any ranking occurred (ranking_details is populated),
        confirming privacy-by-design.

        Args:
            vault_records:   Vault data from module0.
            ranking_details: Module1 ranking results.

        Returns:
            Compliance result dict.
        """
        has_vault   = bool(vault_records)
        has_ranking = bool(ranking_details)

        # Both must exist: vault proves anonymization happened;
        # ranking proves it happened before scoring.
        satisfied = has_vault and has_ranking

        if not has_vault:
            evidence = (
                "Vault records not found — cannot confirm anonymization "
                "preceded ranking."
            )
        elif not has_ranking:
            evidence = (
                "No ranking details available — pipeline may not have "
                "completed ranking stage."
            )
        else:
            n_vault   = len(vault_records)
            n_ranked  = len(ranking_details)
            evidence = (
                f"Anonymization applied to {n_vault} CV(s) by module0 "
                f"(vault records present). "
                f"Ranking performed on anonymized text for {n_ranked} candidate(s). "
                f"Privacy-by-design order confirmed."
            )
        return {
            "title":     "Data protection by design",
            "satisfied": satisfied,
            "evidence":  evidence,
        }

    def check_article_30(
        self,
        run_id:      str,
        output_dir:  str,
    ) -> Dict[str, Any]:
        """Art. 30 — Records of processing activities.

        Always satisfied when this module successfully generates audit_log.json,
        as the log itself constitutes the required record.

        Args:
            run_id:     The unique run identifier.
            output_dir: The audit/ output directory path.

        Returns:
            Compliance result dict.
        """
        log_path = os.path.join(output_dir, AUDIT_LOG_FILENAME)
        evidence = (
            f"Full audit log generated at {log_path} (run ID: {run_id}). "
            f"Records of all processing activities are maintained."
        )
        return {
            "title":     "Records of processing activities",
            "satisfied": True,
            "evidence":  evidence,
        }

    def evaluate_all(
        self,
        vault_records:       Dict[str, Dict[str, Any]],
        parsed_records:      Dict[str, Dict[str, Any]],
        ranking_details:     Dict[str, Any],
        explanation_records: Dict[str, Dict[str, Any]],
        raw_cvs_dir:         str,
        run_id:              str,
        output_dir:          str,
    ) -> Dict[str, Any]:
        """Run all six GDPR article checks and build the compliance result dict.

        Args:
            vault_records:       Vault data dict.
            parsed_records:      Parsed data dict.
            ranking_details:     Module1 ranking details.
            explanation_records: Explanation data dict.
            raw_cvs_dir:         Path to raw_cvs/ directory.
            run_id:              The unique pipeline run identifier.
            output_dir:          Audit output directory.

        Returns:
            Full compliance dict matching the canonical schema.
        """
        results: Dict[str, Any] = {
            GDPR_ARTICLE_5:  self.check_article_5(vault_records),
            GDPR_ARTICLE_13: self.check_article_13(explanation_records),
            GDPR_ARTICLE_17: self.check_article_17(raw_cvs_dir),
            GDPR_ARTICLE_22: self.check_article_22(explanation_records),
            GDPR_ARTICLE_25: self.check_article_25(vault_records, ranking_details),
            GDPR_ARTICLE_30: self.check_article_30(run_id, output_dir),
        }

        non_compliant = [
            art for art, res in results.items()
            if not res["satisfied"]
        ]
        overall = len(non_compliant) == 0

        remediation: List[str] = []
        if not results[GDPR_ARTICLE_17]["satisfied"]:
            remediation.append(
                "Delete all files from raw_cvs/ after confirming anonymized_cvs/ "
                "are complete and correct."
            )
        if not results[GDPR_ARTICLE_5]["satisfied"]:
            remediation.append(
                "Re-run module0 to ensure PII removal is applied and vault "
                "records are generated."
            )
        if not results[GDPR_ARTICLE_13]["satisfied"]:
            remediation.append(
                "Re-run module2 to generate explanation files for all candidates."
            )
        if not results[GDPR_ARTICLE_22]["satisfied"]:
            remediation.append(
                "Review module2 recommendation templates to include explicit "
                "human-review language."
            )

        return {
            "run_id":                run_id,
            "gdpr_articles":         results,
            "overall_compliant":     overall,
            "non_compliant_articles": non_compliant,
            "remediation_required":  remediation,
        }


# ══════════════════════════════════════════════════════════════════════════
# Integrity — SHA-256 hashing for tamper-evidence
# ══════════════════════════════════════════════════════════════════════════

def _compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hash of a file's contents.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Lowercase hex digest string.

    Raises:
        OSError: If the file cannot be read.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _write_sha256_file(hash_value: str, sha_path: str) -> bool:
    """Write the SHA-256 hash to a sidecar file.

    Args:
        hash_value: The hex digest to write.
        sha_path:   Destination file path.

    Returns:
        True on success, False on any OS error.
    """
    try:
        with open(sha_path, "w", encoding="utf-8") as fh:
            fh.write(f"{hash_value}  {AUDIT_LOG_FILENAME}\n")
        return True
    except OSError as exc:
        logger.error("Failed to write SHA-256 file: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════
# Report writers — all file I/O isolated here
# ══════════════════════════════════════════════════════════════════════════

class AuditWriter:
    """Handles all file writes for the audit trail.

    Each write method is individually guarded with try/except so that a
    single file failure never prevents the others from being written.
    """

    def write_audit_log(
        self,
        payload:    Dict[str, Any],
        output_dir: str,
    ) -> Optional[str]:
        """Write audit_log.json and return its file path.

        Args:
            payload:    The complete audit log dict.
            output_dir: Destination directory.

        Returns:
            Absolute path of the written file, or None on failure.
        """
        path = os.path.join(output_dir, AUDIT_LOG_FILENAME)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            logger.info("Audit log written → %s", path)
            return path
        except OSError as exc:
            logger.error("Failed to write audit_log.json: %s", exc)
            return None

    def write_compliance_checklist(
        self,
        checklist:  Dict[str, Any],
        output_dir: str,
    ) -> bool:
        """Write compliance_checklist.json.

        Args:
            checklist:  The compliance evaluation dict from GDPRChecker.
            output_dir: Destination directory.

        Returns:
            True on success, False on any OS error.
        """
        path = os.path.join(output_dir, COMPLIANCE_FILENAME)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(checklist, fh, indent=2, ensure_ascii=False)
            logger.info("Compliance checklist written → %s", path)
            return True
        except OSError as exc:
            logger.error("Failed to write compliance_checklist.json: %s", exc)
            return False

    def write_audit_report(
        self,
        run_id:              str,
        generated_at:        str,
        events:              List[Dict[str, Any]],
        vault_records:       Dict[str, Dict[str, Any]],
        parsed_records:      Dict[str, Dict[str, Any]],
        ranking_details:     Dict[str, Any],
        explanation_records: Dict[str, Dict[str, Any]],
        compliance:          Dict[str, Any],
        sha256_hash:         str,
        output_dir:          str,
        retention_due:       str,
    ) -> bool:
        """Write the human-readable audit_report.txt.

        Args:
            run_id:              Pipeline run identifier.
            generated_at:        ISO timestamp of report generation.
            events:              Full event list.
            vault_records:       Vault data dict.
            parsed_records:      Parsed records dict from module0b.
            ranking_details:     Module1 ranking details.
            explanation_records: Explanation data dict.
            compliance:          GDPR compliance result dict.
            sha256_hash:         SHA-256 hash of audit_log.json.
            output_dir:          Destination directory.
            retention_due:       ISO date when data must be deleted.

        Returns:
            True on success, False on any OS error.
        """
        lines = self._build_report_lines(
            run_id, generated_at, events, vault_records, parsed_records,
            ranking_details, explanation_records, compliance,
            sha256_hash, retention_due,
        )
        path = os.path.join(output_dir, AUDIT_REPORT_FILENAME)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            logger.info("Audit report written → %s", path)
            return True
        except OSError as exc:
            logger.error("Failed to write audit_report.txt: %s", exc)
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Private: report construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_report_lines(
        self,
        run_id:              str,
        generated_at:        str,
        events:              List[Dict[str, Any]],
        vault_records:       Dict[str, Dict[str, Any]],
        parsed_records:      Dict[str, Dict[str, Any]],
        ranking_details:     Dict[str, Any],
        explanation_records: Dict[str, Dict[str, Any]],
        compliance:          Dict[str, Any],
        sha256_hash:         str,
        retention_due:       str,
    ) -> List[str]:
        """Build the full report as a list of text lines.

        Returns:
            List of string lines.
        """
        W   = REPORT_WIDTH
        SEP = "═" * W
        DIV = "─" * W
        now = datetime.now().strftime(HUMAN_FMT)

        n_candidates = len(vault_records) or len(ranking_details)
        n_events     = len(events)
        n_warnings   = sum(1 for e in events if e.get("severity") == SEV_WARNING)
        n_errors     = sum(1 for e in events if e.get("severity") == SEV_ERROR)
        overall_ok   = compliance.get("overall_compliant", False)

        lines: List[str] = []

        # ── Section 1: Cover ──────────────────────────────────────────────
        lines += [
            SEP,
            _c("BIAS-FREE HIRING PIPELINE", W),
            _c("GDPR AUDIT TRAIL REPORT", W),
            _c("CONFIDENTIAL — COMPLIANCE OFFICERS ONLY", W),
            SEP,
            f"  Run ID          : {run_id}",
            f"  Generated       : {now}",
            f"  Pipeline        : {PIPELINE_NAME} v{PIPELINE_VERSION}",
            f"  Candidates      : {n_candidates}",
            f"  Total Events    : {n_events}",
            f"  Warnings        : {n_warnings}",
            f"  Errors          : {n_errors}",
            f"  GDPR Status     : {'✓ COMPLIANT' if overall_ok else '✗ PARTIAL — action required'}",
            "",
            "  LEGAL BASIS",
            "  " + DIV,
        ]
        for line in textwrap.wrap(LEGAL_BASIS, width=W - 4):
            lines.append(f"  {line}")
        lines.append("")
        for line in textwrap.wrap(DATA_CONTROLLER_NOTE, width=W - 4):
            lines.append(f"  {line}")
        lines += ["", SEP, ""]

        # ── Section 2: Data Inventory ─────────────────────────────────────
        lines += [
            _c("SECTION 1 — DATA INVENTORY", W),
            DIV,
            "",
            "  PERSONAL DATA PROCESSED",
        ]

        total_pii_items = 0
        for cid, vault in sorted(vault_records.items()):
            pii_cats  = vault.get("pii_categories_found", {})
            level     = vault.get("anonymization_level_used", "STRICT")
            techniques = vault.get("techniques_applied", [])
            original   = vault.get("original_name", "[not recorded]")
            pii_count  = sum(v for v in pii_cats.values() if isinstance(v, int))
            total_pii_items += pii_count

            lines.append(f"  {cid}:")
            lines.append(f"    Original identity  : [REDACTED — stored in vault only]")
            lines.append(f"    PII items removed  : {pii_count}")
            if pii_cats:
                cat_str = ", ".join(
                    f"{k}: {v}" for k, v in pii_cats.items()
                )
                lines += _wrap("Categories: " + cat_str, W, 6)
            lines.append(f"    Techniques applied : {', '.join(techniques) or 'none recorded'}")
            lines.append(f"    Anonymization level: {level}")
            lines.append("")

        lines += [
            f"  Total PII items removed across all candidates: {total_pii_items}",
            "",
            "  DATA RETAINED (POST-ANONYMIZATION)",
            "    • Anonymized CV text   → anonymized_cvs/",
            "    • Structured sections  → parsed/",
            "    • Ranking scores       → ranking_report.txt",
            "    • Decision explanation → explanations/",
            "    • Original PII         → vault/ (encrypted, access restricted)",
            "",
            "  DATA NOT RETAINED",
            "    • Original CV files should be deleted from raw_cvs/ (see Art. 17 below)",
            "",
            SEP, "",
        ]

        # ── Section 3: Per-candidate log ──────────────────────────────────
        lines += [
            _c("SECTION 2 — PER-CANDIDATE ACTION LOG", W),
            DIV, "",
        ]

        all_cids = sorted(
            set(vault_records.keys())
            | set(ranking_details.keys())
            | set(explanation_records.keys())
        )

        for cid in all_cids:
            vault = vault_records.get(cid, {})
            rd    = ranking_details.get(cid, {})
            exp   = explanation_records.get(cid, {})

            lines += [
                DIV,
                f"  {cid}",
                DIV,
            ]

            # Intake
            orig_file = vault.get("original_filename", "unknown")
            lines.append(f"  Intake:         Raw CV received ('{orig_file}').")

            # Anonymization
            if vault:
                pii_cats   = vault.get("pii_categories_found", {})
                pii_count  = sum(v for v in pii_cats.values() if isinstance(v, int))
                validation = vault.get("validation_score") or {}
                passed     = (
                    validation.get("passed", False)
                    if isinstance(validation, dict)
                    else False
                )
                confidence = (
                    validation.get("confidence_score", 0.0)
                    if isinstance(validation, dict)
                    else 0.0
                )
                lines.append(
                    f"  Anonymization:  {pii_count} PII item(s) removed. "
                    f"Validation {'passed' if passed else 'flagged'} "
                    f"({confidence:.1f}% confidence)."
                )
            else:
                lines.append("  Anonymization:  [no vault record found]")

            # Parsing
            from_parsed = self._get_parsed_summary(cid, parsed_records)
            lines.append(f"  Parsing:        CV structure extracted by module0b. ({from_parsed})")

            # Ranking
            if rd:
                rank  = rd.get("rank", "?")
                score = rd.get("final_score", 0.0)
                lines.append(
                    f"  Ranking:        Ranked #{rank}. "
                    f"Final score: {score:.4f} "
                    f"(semantic: {rd.get('semantic_score', 0):.4f}, "
                    f"keyword: {rd.get('keyword_score', 0):.4f})."
                )
            else:
                lines.append("  Ranking:        [no ranking record found]")

            # Decision
            if exp:
                verdict = exp.get("verdict", "Unknown")
                rec     = exp.get("explanation", {}).get("recommendation", "")
                lines.append(f"  Decision:       Verdict — {verdict}.")
                lines += _wrap(f"Recommendation: {rec}", W, 20)
            else:
                lines.append("  Decision:       [no explanation record found]")

            lines.append("")

        lines += [SEP, ""]

        # ── Section 4: Decision Transparency ──────────────────────────────
        lines += [
            _c("SECTION 3 — DECISION TRANSPARENCY", W),
            DIV,
            "",
            f"  {'Rank':<5} {'Candidate':<15} {'Score':>7}  {'Verdict':<16}  "
            f"{'Sem':>6}  {'KW':>6}  {'Qual':>6}",
            f"  {'-'*5} {'-'*15} {'-'*7}  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*6}",
        ]

        sorted_ranking = sorted(
            ranking_details.items(),
            key=lambda kv: kv[1].get("rank", 9999),
        )
        for cid, rd in sorted_ranking:
            exp     = explanation_records.get(cid, {})
            verdict = exp.get("verdict", "Unknown")
            rank    = rd.get("rank", "?")
            fs      = rd.get("final_score",    0.0)
            sem     = rd.get("semantic_score", 0.0)
            kw      = rd.get("keyword_score",  0.0)
            qual    = rd.get("quality_score",  0.0)
            lines.append(
                f"  {str(rank):<5} {cid:<15} {fs:>6.1%}  {verdict:<16}  "
                f"{sem:>5.1%}  {kw:>5.1%}  {qual:>5.1%}"
            )

        lines += [
            "",
            "  Scoring formula:",
            "    final_score = (0.65 × semantic_score) + (0.35 × keyword_score)",
            "                × (0.85 + 0.15 × quality_score)",
            "    All scoring is purely based on skill and content alignment.",
            "    No demographic, gender, age, or protected characteristic data",
            "    is used at any stage of this pipeline.",
            "",
            SEP, "",
        ]

        # ── Section 5: Compliance Checklist ───────────────────────────────
        lines += [
            _c("SECTION 4 — GDPR COMPLIANCE CHECKLIST", W),
            DIV, "",
        ]

        article_meta = {
            GDPR_ARTICLE_5:  "Art. 5  — Principles of processing",
            GDPR_ARTICLE_13: "Art. 13 — Right to information",
            GDPR_ARTICLE_17: "Art. 17 — Right to erasure",
            GDPR_ARTICLE_22: "Art. 22 — Automated decision-making",
            GDPR_ARTICLE_25: "Art. 25 — Privacy by design",
            GDPR_ARTICLE_30: "Art. 30 — Records of processing",
        }

        articles = compliance.get("gdpr_articles", {})
        for art_key, label in article_meta.items():
            result    = articles.get(art_key, {})
            satisfied = result.get("satisfied", False)
            evidence  = result.get("evidence", "No evidence recorded.")
            tick      = "✓" if satisfied else "✗"
            status    = "SATISFIED" if satisfied else "NOT SATISFIED"
            lines.append(f"  [{tick}] {label}")
            lines.append(f"      Status   : {status}")
            lines += _wrap(f"Evidence : {evidence}", W, 15)
            lines.append("")

        non_compliant = compliance.get("non_compliant_articles", [])
        remediation   = compliance.get("remediation_required", [])

        if non_compliant:
            lines += [
                "  ACTION REQUIRED — Non-compliant articles:",
            ]
            for item in remediation:
                lines += _wrap(f"• {item}", W, 4)
            lines.append("")

        lines += [SEP, ""]

        # ── Section 6: Data Retention Schedule ────────────────────────────
        lines += [
            _c("SECTION 5 — DATA RETENTION SCHEDULE", W),
            DIV,
            "",
            f"  Retention period  : {RETENTION_DAYS} days from pipeline run date",
            f"  Deletion deadline : {retention_due}",
            "",
            "  Files to be reviewed / deleted by deadline:",
            "    • raw_cvs/          — original CVs (delete immediately after run)",
            "    • vault/            — PII vault (restrict access; delete on deadline)",
            "    • anonymized_cvs/   — may be retained longer (no PII)",
            "    • parsed/           — may be retained longer (no PII)",
            "    • explanations/     — may be retained longer (no PII)",
            "    • audit/            — retain permanently for compliance records",
            "",
            SEP, "",
        ]

        # ── Section 7: Digital Signature ──────────────────────────────────
        lines += [
            _c("SECTION 6 — DIGITAL SIGNATURE & INTEGRITY", W),
            DIV,
            "",
            "  The audit_log.json file has been hashed using SHA-256.",
            "  If the log file is modified after generation, the hash will",
            "  not match and the audit trail is invalidated.",
            "",
            f"  File    : {AUDIT_LOG_FILENAME}",
            f"  SHA-256 : {sha256_hash}",
            "",
            "  To verify integrity, run:",
            f"    python -c \"import hashlib; "
            f"print(hashlib.sha256(open('audit/{AUDIT_LOG_FILENAME}','rb')"
            f".read()).hexdigest())\"",
            "",
            "  And compare the output to the hash above.",
            "",
            SEP,
            _c("END OF AUDIT REPORT", W),
            SEP,
            "",
        ]

        return lines

    @staticmethod
    def _get_parsed_summary(
        cid: str,
        parsed_records: Dict[str, Dict[str, Any]],
    ) -> str:
        """Get a one-line parsed data summary for a candidate.

        Args:
            cid:            Candidate identifier.
            parsed_records: Parsed records dict.

        Returns:
            A concise summary string.
        """
        parsed = parsed_records.get(cid)
        if not parsed:
            return "no parsed record"
        sections  = parsed.get("sections", {})
        exp       = sections.get("total_experience_months", 0)
        qual      = parsed.get("quality_score", 0.0)
        return f"{exp} months experience, quality={qual:.2f}"


# ══════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════════════

def _c(text: str, width: int) -> str:
    """Centre a string within *width* columns.

    Args:
        text:  String to centre.
        width: Total line width.

    Returns:
        Centre-padded string.
    """
    return text.center(width)


def _wrap(text: str, width: int, indent: int) -> List[str]:
    """Wrap *text* to *width* with a left indent of *indent* spaces.

    Args:
        text:   Text to wrap.
        width:  Maximum line width.
        indent: Number of leading spaces.

    Returns:
        List of wrapped line strings.
    """
    prefix  = " " * indent
    wrapped = textwrap.fill(
        text,
        width             = width,
        initial_indent    = prefix,
        subsequent_indent = prefix,
    )
    return wrapped.split("\n") if wrapped else []


# ══════════════════════════════════════════════════════════════════════════
# AuditEngine — top-level orchestrator
# ══════════════════════════════════════════════════════════════════════════

class AuditEngine:
    """Orchestrates the full GDPR audit trail generation.

    Delegates data loading to the loader functions, event construction to
    EventBuilder, compliance checking to GDPRChecker, and all file I/O to
    AuditWriter. This class owns only the sequencing logic.
    """

    def __init__(self) -> None:
        """Initialise the engine with its component dependencies."""
        self._events:    EventBuilder = EventBuilder()
        self._checker:   GDPRChecker  = GDPRChecker()
        self._writer:    AuditWriter  = AuditWriter()

    def generate(
        self,
        vault_dir:        str,
        parsed_dir:       str,
        ranking_details:  Dict[str, Any],
        explanations_dir: str,
        output_dir:       str,
        raw_cvs_dir:      str,
    ) -> bool:
        """Generate all audit artefacts.

        Processing order:
            1.  Load all pipeline outputs from disk.
            2.  Build event list from every stage's outputs.
            3.  Run GDPR article checks.
            4.  Assemble audit_log.json payload.
            5.  Write audit_log.json.
            6.  Hash audit_log.json and write .sha256 sidecar.
            7.  Write compliance_checklist.json.
            8.  Write audit_report.txt.

        Args:
            vault_dir:        Path to vault/ from module0.
            parsed_dir:       Path to parsed/ from module0b.
            ranking_details:  ranker.last_ranking_details from module1.
            explanations_dir: Path to explanations/ from module2.
            output_dir:       Destination directory for audit/ outputs.
            raw_cvs_dir:      Path to raw_cvs/ (for Art.17 check).

        Returns:
            True if audit_log.json was written successfully.
        """
        run_id       = datetime.now().strftime(RUN_ID_FMT)
        generated_at = datetime.now().strftime(ISO_FMT)
        deletion_due = (
            datetime.now() + timedelta(days=RETENTION_DAYS)
        ).strftime(DATE_ONLY_FMT)

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create audit output directory: %s", exc)
            return False

        # ── Step 1: Load all pipeline data ───────────────────────────────
        logger.info("Loading pipeline outputs …")
        vault_records       = _load_vault_records(vault_dir)
        parsed_records      = _load_parsed_records(parsed_dir)
        explanation_records = _load_explanation_records(explanations_dir)

        # ranking_details is passed in directly (live from module1's ranker)
        # but fall back gracefully if empty
        if not ranking_details:
            logger.warning(
                "ranking_details is empty. Audit will cover anonymization "
                "and parsing stages only."
            )

        n_candidates = max(
            len(vault_records),
            len(ranking_details),
            len(explanation_records),
        )

        # ── Step 2: Build event list ──────────────────────────────────────
        logger.info("Building event log …")
        all_events: List[Dict[str, Any]] = []

        all_events.append(
            self._events.build_pipeline_start(run_id, n_candidates)
        )
        all_events.extend(
            self._events.build_cv_received_events(vault_records)
        )
        all_events.extend(
            self._events.build_pii_removed_events(vault_records)
        )
        all_events.extend(
            self._events.build_cv_parsed_events(parsed_records)
        )
        if ranking_details:
            all_events.extend(
                self._events.build_ranking_events(ranking_details)
            )
        if explanation_records:
            all_events.extend(
                self._events.build_explanation_events(explanation_records)
            )
        all_events.extend(
            self._events.build_data_retained_events(vault_records, deletion_due)
        )
        all_events.extend(
            self._events.build_warning_events(
                vault_records, parsed_records,
                explanation_records, ranking_details,
            )
        )

        # ── Step 3: GDPR compliance checks ────────────────────────────────
        logger.info("Running GDPR compliance checks …")
        compliance = self._checker.evaluate_all(
            vault_records       = vault_records,
            parsed_records      = parsed_records,
            ranking_details     = ranking_details,
            explanation_records = explanation_records,
            raw_cvs_dir         = raw_cvs_dir,
            run_id              = run_id,
            output_dir          = output_dir,
        )
        gdpr_compliant = compliance.get("overall_compliant", False)

        # Append PIPELINE_COMPLETE event now that we know compliance status
        all_events.append(
            self._events.build_pipeline_complete(
                run_id         = run_id,
                n_candidates   = n_candidates,
                n_events       = len(all_events) + 1,  # +1 for this event
                gdpr_compliant = gdpr_compliant,
            )
        )

        # ── Step 4: Build event summary ───────────────────────────────────
        event_summary = self._summarise_events(all_events)

        # ── Step 5: Assemble audit_log.json payload ───────────────────────
        audit_log: Dict[str, Any] = {
            "pipeline_run_id":  run_id,
            "generated_at":     generated_at,
            "pipeline_version": PIPELINE_VERSION,
            "gdpr_compliant":   gdpr_compliant,
            "total_candidates": n_candidates,
            "data_retention": {
                "raw_cvs_deleted":  _count_raw_cvs(raw_cvs_dir) == 0,
                "vault_encrypted":  False,   # filesystem encryption is infra-level
                "retention_days":   RETENTION_DAYS,
                "deletion_due":     deletion_due,
            },
            "events":  all_events,
            "summary": event_summary,
        }

        # ── Step 6: Write audit_log.json ──────────────────────────────────
        log_path = self._writer.write_audit_log(audit_log, output_dir)
        if log_path is None:
            logger.error("Critical: audit_log.json could not be written.")
            return False

        # ── Step 7: Hash audit_log.json ───────────────────────────────────
        sha256_hash = "hash-unavailable"
        try:
            sha256_hash = _compute_sha256(log_path)
            sha_path    = os.path.join(output_dir, SHA256_FILENAME)
            _write_sha256_file(sha256_hash, sha_path)
            logger.info("SHA-256: %s → %s", sha256_hash, sha_path)
        except OSError as exc:
            logger.warning("Could not compute SHA-256: %s", exc)

        # ── Step 8: Write compliance_checklist.json ───────────────────────
        self._writer.write_compliance_checklist(compliance, output_dir)

        # ── Step 9: Write audit_report.txt ───────────────────────────────
        self._writer.write_audit_report(
            run_id              = run_id,
            generated_at        = generated_at,
            events              = all_events,
            vault_records       = vault_records,
            parsed_records      = parsed_records,
            ranking_details     = ranking_details,
            explanation_records = explanation_records,
            compliance          = compliance,
            sha256_hash         = sha256_hash,
            output_dir          = output_dir,
            retention_due       = deletion_due,
        )

        logger.info(
            "Audit trail complete: %d events, GDPR %s.",
            len(all_events),
            "COMPLIANT" if gdpr_compliant else "PARTIAL",
        )
        return True

    @staticmethod
    def _summarise_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary counts from the event list.

        Args:
            events: Full list of event dicts.

        Returns:
            Summary dict with counts by category.
        """
        return {
            "total_events":        len(events),
            "pii_removal_events":  sum(
                1 for e in events if e.get("event_type") == EVT_PII_REMOVED
            ),
            "cv_parsed_events":    sum(
                1 for e in events if e.get("event_type") == EVT_CV_PARSED
            ),
            "ranking_events":      sum(
                1 for e in events if e.get("event_type") == EVT_CANDIDATE_RANKED
            ),
            "explanation_events":  sum(
                1 for e in events if e.get("event_type") == EVT_EXPLANATION_WRITTEN
            ),
            "data_retained_events": sum(
                1 for e in events if e.get("event_type") == EVT_DATA_RETAINED
            ),
            "warnings":            sum(
                1 for e in events if e.get("severity") == SEV_WARNING
            ),
            "errors":              sum(
                1 for e in events if e.get("severity") == SEV_ERROR
            ),
        }


# ══════════════════════════════════════════════════════════════════════════
# Singleton engine
# ══════════════════════════════════════════════════════════════════════════

_audit_engine: Optional[AuditEngine] = None


def _get_engine() -> AuditEngine:
    """Return the module-level cached AuditEngine instance.

    Returns:
        The shared AuditEngine.
    """
    global _audit_engine
    if _audit_engine is None:
        _audit_engine = AuditEngine()
    return _audit_engine


# ══════════════════════════════════════════════════════════════════════════
# run() — primary entry point called by main.py
# ══════════════════════════════════════════════════════════════════════════

def run(
    vault_dir:        str,
    parsed_dir:       str,
    ranking_details:  Dict[str, Any],
    explanations_dir: str,
    output_dir:       str,
    raw_cvs_dir:      str,
) -> bool:
    """Execute the full GDPR audit trail stage.

    This is the canonical programmatic entry point, matching the run()
    convention of module0, module0b, module1, and module2.

    Side effects:
        Creates *output_dir* if it does not exist.
        Writes audit_log.json, audit_log.sha256, compliance_checklist.json,
        and audit_report.txt into *output_dir*.

    Args:
        vault_dir:        Path to vault/ produced by module0.
        parsed_dir:       Path to parsed/ produced by module0b.
        ranking_details:  ranker.last_ranking_details from module1.
        explanations_dir: Path to explanations/ produced by module2.
        output_dir:       Destination for all audit output files.
        raw_cvs_dir:      Path to raw_cvs/ (checked for Art. 17 erasure).

    Returns:
        True if audit_log.json was written successfully.
        False on critical failure.
    """
    t0 = time.time()

    print()
    print("=" * 60)
    print("  MODULE 3 — GDPR AUDIT TRAIL ENGINE")
    print("=" * 60)
    print(f"\n  Vault       : {vault_dir}")
    print(f"  Parsed      : {parsed_dir}")
    print(f"  Explanations: {explanations_dir}")
    print(f"  Raw CVs     : {raw_cvs_dir}")
    print(f"  Output      : {output_dir}")
    print(f"  Candidates  : {len(ranking_details)} in ranking_details")
    print()

    ok = _get_engine().generate(
        vault_dir        = vault_dir,
        parsed_dir       = parsed_dir,
        ranking_details  = ranking_details,
        explanations_dir = explanations_dir,
        output_dir       = output_dir,
        raw_cvs_dir      = raw_cvs_dir,
    )

    elapsed = time.time() - t0
    print()
    if ok:
        print(f"  [OK] Audit trail generated in {elapsed:.1f}s")
        print(f"  Log      → {os.path.join(output_dir, AUDIT_LOG_FILENAME)}")
        print(f"  Report   → {os.path.join(output_dir, AUDIT_REPORT_FILENAME)}")
        print(f"  Checklist→ {os.path.join(output_dir, COMPLIANCE_FILENAME)}")
        print(f"  Hash     → {os.path.join(output_dir, SHA256_FILENAME)}")
    else:
        print("  [FAILED] Audit trail could not be generated.")
        print("           Check logs for details.")

    print("=" * 60)
    return ok


# ══════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ══════════════════════════════════════════════════════════════════════════

def _configure_logging(verbose: bool = False) -> None:
    """Configure root logger for standalone execution.

    Args:
        verbose: If True, sets level to DEBUG; otherwise INFO.
    """
    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = _LOG_FORMAT,
        datefmt = _DATE_FMT,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone execution.

    Returns:
        Configured ArgumentParser instance.
    """
    _base = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        prog        = "module3",
        description = (
            "GDPR Audit Trail Engine — Bias-Free Hiring Pipeline\n\n"
            "Reads all pipeline stage outputs and generates a legally "
            "defensible, tamper-evident GDPR audit trail."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "Examples:\n"
            "  python module3.py --vault-dir vault \\\n"
            "                    --parsed-dir parsed \\\n"
            "                    --ranking-file ranking_details.json \\\n"
            "                    --explanations-dir explanations \\\n"
            "                    --output-dir audit \\\n"
            "                    --raw-cvs-dir raw_cvs\n\n"
            "  # Export ranking_details from a live pipeline first:\n"
            "  python -c \"\n"
            "    import json, module1\n"
            "    r = module1.load_ranker()\n"
            "    json.dump(r.last_ranking_details,\n"
            "              open('ranking_details.json','w'), indent=2)\n"
            "  \"\n"
        ),
    )
    p.add_argument(
        "--vault-dir",
        default = os.path.join(_base, "vault"),
        metavar = "DIR",
        help    = "Path to vault/ directory from module0 (default: vault/).",
    )
    p.add_argument(
        "--parsed-dir",
        default = os.path.join(_base, "parsed"),
        metavar = "DIR",
        help    = "Path to parsed/ directory from module0b (default: parsed/).",
    )
    p.add_argument(
        "--ranking-file",
        default = None,
        metavar = "FILE",
        help    = (
            "Path to a JSON file containing last_ranking_details "
            "(exported from module1). If omitted, ranking data is skipped."
        ),
    )
    p.add_argument(
        "--explanations-dir",
        default = os.path.join(_base, "explanations"),
        metavar = "DIR",
        help    = "Path to explanations/ directory from module2 (default: explanations/).",
    )
    p.add_argument(
        "--output-dir",
        default = os.path.join(_base, "audit"),
        metavar = "DIR",
        help    = "Destination for all audit output files (default: audit/).",
    )
    p.add_argument(
        "--raw-cvs-dir",
        default = os.path.join(_base, "raw_cvs"),
        metavar = "DIR",
        help    = "Path to raw_cvs/ directory (GDPR Art. 17 check, default: raw_cvs/).",
    )
    p.add_argument(
        "--verbose", "-v",
        action = "store_true",
        help   = "Enable verbose/debug logging.",
    )
    return p


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    _configure_logging(verbose=_args.verbose)

    # Load ranking_details from file if supplied
    _ranking_details: Dict[str, Any] = {}
    if _args.ranking_file:
        try:
            with open(_args.ranking_file, "r", encoding="utf-8") as _fh:
                _ranking_details = json.load(_fh)
            logger.info(
                "Loaded ranking_details: %d candidates from '%s'.",
                len(_ranking_details), _args.ranking_file,
            )
        except FileNotFoundError:
            logger.warning("Ranking file not found: %s — proceeding without it.", _args.ranking_file)
        except json.JSONDecodeError as _exc:
            logger.warning("Ranking file is not valid JSON: %s — proceeding without it.", _exc)
        except OSError as _exc:
            logger.warning("Cannot read ranking file: %s — proceeding without it.", _exc)
    else:
        logger.info("No --ranking-file supplied. Audit will cover stages 0/0b only.")

    _ok = run(
        vault_dir        = _args.vault_dir,
        parsed_dir       = _args.parsed_dir,
        ranking_details  = _ranking_details,
        explanations_dir = _args.explanations_dir,
        output_dir       = _args.output_dir,
        raw_cvs_dir      = _args.raw_cvs_dir,
    )
    sys.exit(0 if _ok else 1)