# -*- coding: utf-8 -*-
"""
database.py — SQLite Persistence Layer
=======================================
Bias-Free Hiring Pipeline — Sprint 1

Provides a thread-safe SQLite database layer that stores every pipeline
run, candidate record, ranking result, and audit event.  This replaces
the previous filesystem-only approach (scattered JSON files) with a
structured, queryable store that the Flask dashboard and REST API can
read without walking directories.

Schema
------
    pipeline_runs       — one row per full pipeline execution
    candidates          — one row per candidate per run
    ranking_results     — one row per ranked candidate (scores + skills)
    audit_events        — append-only event log mirroring module3 output
    bias_checks         — one row per fairness check per run

Design decisions
----------------
- WAL mode + check_same_thread=False so the Flask dashboard can read
  while the pipeline writes from a background thread.
- Row factories return sqlite3.Row objects (dict-accessible) so callers
  never have to deal with column indices.
- All writes go through context-managed transactions; exceptions roll back
  automatically.
- The schema is created on first connect (IF NOT EXISTS) so no separate
  migration step is needed for fresh installs.

Usage
-----
    from database import Database

    db = Database()                    # opens pipeline.db in BASE_DIR
    run_id = db.create_run("jd.txt")
    db.upsert_candidate(run_id, "Candidate_01", vault_data, parsed_data)
    db.save_ranking(run_id, ranking_details)
    db.save_audit_events(run_id, events)
    db.save_bias_checks(run_id, bias_results)
    run = db.get_run(run_id)
    candidates = db.list_candidates(run_id)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ── Database location ──────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH: str = os.path.join(BASE_DIR, "pipeline.db")

# ── ISO timestamp format ───────────────────────────────────────────────────
_ISO_FMT: str = "%Y-%m-%dT%H:%M:%S"


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime(_ISO_FMT)


# ═══════════════════════════════════════════════════════════════════════════
# Schema DDL
# ═══════════════════════════════════════════════════════════════════════════

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL UNIQUE,
    jd_file       TEXT,
    jd_text       TEXT,
    status        TEXT    NOT NULL DEFAULT 'running',
    started_at    TEXT    NOT NULL,
    finished_at   TEXT,
    exit_code     INTEGER,
    total_cvs     INTEGER DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id    TEXT    NOT NULL,
    original_file   TEXT,
    anonymized_at   TEXT,
    pii_count       INTEGER DEFAULT 0,
    validation_passed INTEGER DEFAULT 0,
    validation_confidence REAL DEFAULT 0.0,
    quality_score   REAL    DEFAULT 0.0,
    total_exp_months INTEGER DEFAULT 0,
    skills_json     TEXT,
    education_json  TEXT,
    vault_json      TEXT,
    UNIQUE(run_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS ranking_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id    TEXT    NOT NULL,
    rank            INTEGER NOT NULL,
    final_score     REAL    NOT NULL,
    semantic_score  REAL    NOT NULL,
    keyword_score   REAL    NOT NULL,
    quality_score   REAL    NOT NULL,
    verdict         TEXT,
    matched_skills  TEXT,
    missing_skills  TEXT,
    recommendation  TEXT,
    ranked_at       TEXT    NOT NULL,
    UNIQUE(run_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS audit_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL REFERENCES pipeline_runs(run_id),
    event_id     TEXT,
    event_type   TEXT    NOT NULL,
    stage        TEXT,
    candidate_id TEXT,
    severity     TEXT    NOT NULL DEFAULT 'INFO',
    description  TEXT,
    extra_json   TEXT,
    timestamp    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS bias_checks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL REFERENCES pipeline_runs(run_id),
    check_id    TEXT    NOT NULL,
    label       TEXT,
    flagged     INTEGER NOT NULL DEFAULT 0,
    status      TEXT,
    note        TEXT,
    detail_json TEXT,
    checked_at  TEXT    NOT NULL,
    UNIQUE(run_id, check_id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_run
    ON candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_ranking_run_rank
    ON ranking_results(run_id, rank);
CREATE INDEX IF NOT EXISTS idx_audit_run_type
    ON audit_events(run_id, event_type);
"""


# ═══════════════════════════════════════════════════════════════════════════
# Database class
# ═══════════════════════════════════════════════════════════════════════════

class Database:
    """Thread-safe SQLite wrapper for the hiring pipeline.

    A single ``Database`` instance is safe to share across threads because:
    - WAL journal mode allows concurrent readers + one writer.
    - ``check_same_thread=False`` lets Flask worker threads read while the
      background pipeline thread writes.
    - All writes are wrapped in ``_transaction()`` which rolls back on error.

    Args:
        db_path: Path to the SQLite file.  Defaults to ``pipeline.db``
                 in the project root.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        logger.info("Database initialised at %s", db_path)

    # ──────────────────────────────────────────────────────────────────────
    # Connection management
    # ──────────────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Open the SQLite connection and create the schema if needed."""
        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the connection inside a transaction; rollback on error.

        Yields:
            The active ``sqlite3.Connection``.

        Raises:
            sqlite3.Error: Re-raised after rollback so callers can handle it.
        """
        assert self._conn is not None, "Database connection is closed"
        try:
            yield self._conn
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            logger.error("DB transaction rolled back: %s", exc)
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ──────────────────────────────────────────────────────────────────────
    # pipeline_runs
    # ──────────────────────────────────────────────────────────────────────

    def create_run(
        self,
        run_id:   str,
        jd_file:  Optional[str] = None,
        jd_text:  Optional[str] = None,
    ) -> str:
        """Insert a new pipeline run record and return its run_id.

        Args:
            run_id:  Unique identifier (e.g. ``RUN_20240115_143201``).
            jd_file: Path to the JD file, if any.
            jd_text: Full JD text (first 2000 chars stored for quick display).

        Returns:
            The ``run_id`` that was inserted.
        """
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO pipeline_runs
                   (run_id, jd_file, jd_text, status, started_at)
                   VALUES (?, ?, ?, 'running', ?)""",
                (run_id, jd_file, (jd_text or "")[:2000], _now()),
            )
        logger.debug("Run created: %s", run_id)
        return run_id

    def finish_run(
        self,
        run_id:    str,
        exit_code: int,
        total_cvs: int = 0,
        error:     Optional[str] = None,
    ) -> None:
        """Mark a pipeline run as finished.

        Args:
            run_id:    The run to update.
            exit_code: 0 = success, non-zero = failure.
            total_cvs: Number of CVs processed.
            error:     Error message if exit_code != 0.
        """
        status = "completed" if exit_code == 0 else "failed"
        with self._transaction() as conn:
            conn.execute(
                """UPDATE pipeline_runs
                   SET status=?, finished_at=?, exit_code=?,
                       total_cvs=?, error_message=?
                   WHERE run_id=?""",
                (status, _now(), exit_code, total_cvs, error, run_id),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single run record by run_id.

        Args:
            run_id: The run identifier.

        Returns:
            A dict of run fields, or ``None`` if not found.
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent pipeline runs.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            List of run dicts ordered by started_at descending.
        """
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_run_id(self) -> Optional[str]:
        """Return the run_id of the most recently started run.

        Returns:
            The run_id string, or ``None`` if no runs exist.
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT run_id FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    # ──────────────────────────────────────────────────────────────────────
    # candidates
    # ──────────────────────────────────────────────────────────────────────

    def upsert_candidate(
        self,
        run_id:       str,
        candidate_id: str,
        vault_data:   Optional[Dict[str, Any]] = None,
        parsed_data:  Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a candidate record for a given run.

        Extracts the most useful fields from vault_data and parsed_data
        into typed columns for fast dashboard queries, and stores the full
        raw dicts as JSON blobs for completeness.

        Args:
            run_id:       The pipeline run this candidate belongs to.
            candidate_id: Candidate identifier (e.g. ``Candidate_01``).
            vault_data:   Output of module0 vault for this candidate.
            parsed_data:  Output of module0b for this candidate.
        """
        vault = vault_data or {}
        parsed = parsed_data or {}
        sections = parsed.get("sections", {})

        val_report = vault.get("validation_score") or vault.get("validation_report") or {}
        pii_count = sum(
            v for v in vault.get("pii_categories_found", {}).values()
            if isinstance(v, int)
        )

        skills_block = sections.get("skills", {})
        skills_flat = []
        if isinstance(skills_block, dict):
            for lst in skills_block.values():
                skills_flat.extend(lst)

        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO candidates
                   (run_id, candidate_id, original_file, anonymized_at,
                    pii_count, validation_passed, validation_confidence,
                    quality_score, total_exp_months,
                    skills_json, education_json, vault_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                     original_file=excluded.original_file,
                     pii_count=excluded.pii_count,
                     validation_passed=excluded.validation_passed,
                     validation_confidence=excluded.validation_confidence,
                     quality_score=excluded.quality_score,
                     total_exp_months=excluded.total_exp_months,
                     skills_json=excluded.skills_json,
                     education_json=excluded.education_json,
                     vault_json=excluded.vault_json""",
                (
                    run_id,
                    candidate_id,
                    vault.get("original_filename"),
                    _now(),
                    pii_count,
                    1 if val_report.get("passed") else 0,
                    float(val_report.get("confidence_score", 0.0)),
                    float(parsed.get("quality_score", 0.0)),
                    int(sections.get("total_experience_months", 0)),
                    json.dumps(skills_flat),
                    json.dumps(sections.get("education", [])),
                    json.dumps(vault),
                ),
            )

    def list_candidates(self, run_id: str) -> List[Dict[str, Any]]:
        """Return all candidates for a given run.

        Args:
            run_id: The pipeline run identifier.

        Returns:
            List of candidate dicts.
        """
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM candidates WHERE run_id=? ORDER BY candidate_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────────────────────────
    # ranking_results
    # ──────────────────────────────────────────────────────────────────────

    def save_ranking(
        self,
        run_id:          str,
        ranking_details: Dict[str, Any],
        explanation_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the full ranking for a run.

        Args:
            run_id:          The pipeline run identifier.
            ranking_details: ``ranker.last_ranking_details`` from module1.
            explanation_map: Optional dict of candidate_id → explanation
                             dict from module2 (for verdict + recommendation).
        """
        exp_map = explanation_map or {}
        with self._transaction() as conn:
            for cid, rd in ranking_details.items():
                exp = exp_map.get(cid, {}).get("explanation", {})
                conn.execute(
                    """INSERT INTO ranking_results
                       (run_id, candidate_id, rank, final_score,
                        semantic_score, keyword_score, quality_score,
                        verdict, matched_skills, missing_skills,
                        recommendation, ranked_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                         rank=excluded.rank,
                         final_score=excluded.final_score,
                         semantic_score=excluded.semantic_score,
                         keyword_score=excluded.keyword_score,
                         quality_score=excluded.quality_score,
                         verdict=excluded.verdict,
                         matched_skills=excluded.matched_skills,
                         missing_skills=excluded.missing_skills,
                         recommendation=excluded.recommendation""",
                    (
                        run_id,
                        cid,
                        int(rd.get("rank", 0)),
                        float(rd.get("final_score", 0.0)),
                        float(rd.get("semantic_score", 0.0)),
                        float(rd.get("keyword_score", 0.0)),
                        float(rd.get("quality_score", 0.0)),
                        exp_map.get(cid, {}).get("verdict"),
                        json.dumps(rd.get("matched_skills", [])),
                        json.dumps(rd.get("missing_skills", [])),
                        exp.get("recommendation"),
                        _now(),
                    ),
                )
        logger.debug("Saved %d ranking records for run %s", len(ranking_details), run_id)

    def get_rankings(self, run_id: str) -> List[Dict[str, Any]]:
        """Return all ranking results for a run, sorted by rank.

        Args:
            run_id: The pipeline run identifier.

        Returns:
            List of ranking dicts ordered by rank ascending.
        """
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM ranking_results WHERE run_id=? ORDER BY rank",
            (run_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["matched_skills"] = json.loads(d.get("matched_skills") or "[]")
            d["missing_skills"]  = json.loads(d.get("missing_skills") or "[]")
            results.append(d)
        return results

    # ──────────────────────────────────────────────────────────────────────
    # audit_events
    # ──────────────────────────────────────────────────────────────────────

    def save_audit_events(
        self,
        run_id: str,
        events: List[Dict[str, Any]],
    ) -> None:
        """Append a list of audit events to the database.

        Args:
            run_id: The pipeline run these events belong to.
            events: List of event dicts from module3's EventBuilder.
        """
        with self._transaction() as conn:
            conn.executemany(
                """INSERT INTO audit_events
                   (run_id, event_id, event_type, stage, candidate_id,
                    severity, description, extra_json, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        run_id,
                        e.get("event_id"),
                        e.get("event_type", "UNKNOWN"),
                        e.get("stage"),
                        e.get("candidate_id"),
                        e.get("severity", "INFO"),
                        e.get("description"),
                        json.dumps({
                            k: v for k, v in e.items()
                            if k not in ("event_id", "event_type", "stage",
                                         "candidate_id", "severity",
                                         "description", "timestamp",
                                         "legal_basis")
                        }),
                        e.get("timestamp", _now()),
                    )
                    for e in events
                ],
            )
        logger.debug("Saved %d audit events for run %s", len(events), run_id)

    # ──────────────────────────────────────────────────────────────────────
    # bias_checks
    # ──────────────────────────────────────────────────────────────────────

    def save_bias_checks(
        self,
        run_id:  str,
        results: Dict[str, Any],
    ) -> None:
        """Persist the bias audit check results for a run.

        Args:
            run_id:  The pipeline run identifier.
            results: Dict returned by BiasAuditor.run_all_checks().
        """
        checks = results.get("checks", [])
        with self._transaction() as conn:
            for check in checks:
                conn.execute(
                    """INSERT INTO bias_checks
                       (run_id, check_id, label, flagged, status,
                        note, detail_json, checked_at)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(run_id, check_id) DO UPDATE SET
                         flagged=excluded.flagged,
                         status=excluded.status,
                         note=excluded.note,
                         detail_json=excluded.detail_json""",
                    (
                        run_id,
                        check.get("check_id", "unknown"),
                        check.get("label"),
                        1 if check.get("flagged") else 0,
                        check.get("status"),
                        check.get("note"),
                        json.dumps(check.get("detail", {})),
                        _now(),
                    ),
                )

    def get_bias_summary(self, run_id: str) -> Dict[str, Any]:
        """Return a summary of bias check results for a run.

        Args:
            run_id: The pipeline run identifier.

        Returns:
            Dict with keys: total_checks, flags_raised, checks (list).
        """
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM bias_checks WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
        checks = [dict(r) for r in rows]
        return {
            "total_checks": len(checks),
            "flags_raised": sum(1 for c in checks if c.get("flagged")),
            "checks":       checks,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Dashboard helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_dashboard_stats(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregated stats for the dashboard summary cards.

        If ``run_id`` is None, uses the most recent completed run.

        Args:
            run_id: Optional specific run to summarise.

        Returns:
            Dict with keys: total_cvs, top_candidate, top_score,
            flags_raised, run_status, started_at.
        """
        assert self._conn is not None
        rid = run_id or self.get_latest_run_id()
        if not rid:
            return {}

        run = self.get_run(rid)
        rankings = self.get_rankings(rid)
        bias = self.get_bias_summary(rid)

        top = rankings[0] if rankings else {}
        return {
            "run_id":        rid,
            "total_cvs":     run.get("total_cvs", 0) if run else 0,
            "run_status":    run.get("status", "unknown") if run else "unknown",
            "started_at":    run.get("started_at") if run else None,
            "finished_at":   run.get("finished_at") if run else None,
            "top_candidate": top.get("candidate_id"),
            "top_score":     top.get("final_score", 0.0),
            "flags_raised":  bias.get("flags_raised", 0),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

_db_instance: Optional[Database] = None


def get_db(db_path: str = DEFAULT_DB_PATH) -> Database:
    """Return the module-level cached Database instance.

    Creates the database on first call; returns the same object on
    subsequent calls.  Safe to call from any thread.

    Args:
        db_path: Path to the SQLite file (only used on the first call).

    Returns:
        The shared Database instance.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path=db_path)
    return _db_instance
