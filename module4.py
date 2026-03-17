# -*- coding: utf-8 -*-
"""
module4.py — Bias Detection & Fairness Audit
==============================================
Bias-Free Hiring Pipeline — Stage 4

Inspects the numeric ranking outputs produced by module1 (and the structured
CV data produced by module0b) to surface any statistically detectable patterns
that could indicate bias in the scoring pipeline.

What this module consumes
--------------------------
    module1  → ranker.last_ranking_details  (in-memory ranking dict)
    module0b → parsed/Candidate_XX.json     (structured CV sections)

What this module produces
--------------------------
    audit/bias_report.txt  ← human-readable fairness audit report
    audit/bias_audit.json  ← machine-readable metrics + flag details

Pipeline position
-----------------
    module0  → anonymized_cvs/ + vault/
    module0b → parsed/
    module1  → ranking_report.txt + last_ranking_details
    module2  → explanations/
    module3  → audit/audit_log.json  +  audit_report.txt  +  compliance_checklist.json
    module4  → audit/bias_report.txt +  bias_audit.json       ← THIS MODULE

Programmatic usage (called by main.py)
---------------------------------------
    import module4

    ok = module4.run(
        ranking_details = ranker.last_ranking_details,
        parsed_dir      = "parsed",
        output_dir      = "audit",
    )

Standalone usage
----------------
    python module4.py --parsed-dir parsed --output-dir audit \\
                      --ranking-file ranking_details.json

Dependencies
------------
    Pure Python stdlib only: json, os, math, logging, datetime, argparse, sys.
    Zero pip installs. No Flask. No databases. No LLMs. No ML.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════
# Module-level logger
# ══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Constants — every threshold and output filename lives here
# ══════════════════════════════════════════════════════════════════════════

# ── Output filenames ──────────────────────────────────────────────────────
BIAS_REPORT_FILENAME: str = "bias_report.txt"
BIAS_AUDIT_FILENAME:  str = "bias_audit.json"

# ── Thresholds — tune here, nowhere else ─────────────────────────────────
# Score distribution
COMPRESSION_THRESHOLD:    float = 0.10   # max−min below this → "scores too compressed"
LOW_STDDEV_THRESHOLD:     float = 0.05   # std-dev below this → "low score variance"

# Skill-gap disparity
ZERO_MATCH_FLAG:          bool  = True   # flag candidates with 0 matched skills

# Rank vs semantic gap
SEMANTIC_KEYWORD_GAP:     float = 0.40   # |semantic − keyword| above this → flag
TOP_CANDIDATE_GAP_CHECK:  bool  = True   # only check the #1 ranked candidate

# Experience proxy bias
EXPERIENCE_CORRELATION_THRESHOLD: float = 0.85  # Pearson r above this → flag
MIN_CANDIDATES_FOR_CORR:          int   = 3      # need at least this many for correlation

# Quality score spread
QUALITY_SPREAD_THRESHOLD: float = 0.50   # max−min above this → flag CV formatting influence

# ── Formatting ────────────────────────────────────────────────────────────
REPORT_WIDTH:    int = 80
ISO_FMT:         str = "%Y-%m-%dT%H:%M:%S"
HUMAN_FMT:       str = "%d %B %Y at %H:%M UTC"
DIVIDER:         str = "─" * REPORT_WIDTH
HEAVY_DIVIDER:   str = "=" * REPORT_WIDTH

# ── Logging ───────────────────────────────────────────────────────────────
_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT:   str = "%H:%M:%S"


# ══════════════════════════════════════════════════════════════════════════
# Pure math helpers — no I/O, deterministic
# ══════════════════════════════════════════════════════════════════════════

def _mean(values: List[float]) -> float:
    """Return the arithmetic mean of a non-empty list of floats."""
    return sum(values) / len(values) if values else 0.0


def _stddev(values: List[float]) -> float:
    """Return the population standard deviation of a list of floats.

    Returns 0.0 for lists with fewer than two elements.
    """
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length lists.

    Returns 0.0 if the lists are too short or have zero variance.

    Args:
        xs: First variable (e.g. experience months).
        ys: Second variable (e.g. final score).

    Returns:
        Pearson r in [−1, 1], or 0.0 on degenerate input.
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0

    mx, my = _mean(xs), _mean(ys)
    num    = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y  = math.sqrt(sum((y - my) ** 2 for y in ys))

    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _pct(score: float) -> str:
    """Format a 0–1 float as a rounded percentage string."""
    return f"{round(score * 100)}%"


def _flag_label(flagged: bool) -> str:
    """Return a short status label for a boolean flag."""
    return "⚠  FLAGGED" if flagged else "✓  OK"


# ══════════════════════════════════════════════════════════════════════════
# Parsed JSON loader
# ══════════════════════════════════════════════════════════════════════════

def _load_parsed_json(
    parsed_dir:   str,
    candidate_id: str,
) -> Optional[Dict[str, Any]]:
    """Load the module0b parsed JSON for one candidate.

    Never raises — returns None on any failure.

    Args:
        parsed_dir:   Path to the parsed/ directory.
        candidate_id: Candidate identifier (e.g. "Candidate_01").

    Returns:
        Parsed dict on success, None otherwise.
    """
    path = os.path.join(parsed_dir, f"{candidate_id}.json")
    if not os.path.isfile(path):
        logger.warning("module4: parsed JSON not found for %s at '%s'.", candidate_id, path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("module4: cannot read parsed JSON for %s: %s", candidate_id, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════
# BiasAuditor — performs all fairness checks
# ══════════════════════════════════════════════════════════════════════════

class BiasAuditor:
    """Runs a suite of rule-based fairness checks on ranking_details.

    All analysis is deterministic and uses only the data already computed
    by modules 0b and 1.  No additional ML or external calls are made.

    Args:
        ranking_details: ranker.last_ranking_details from module1.
        parsed_records:  Dict of candidate_id → parsed JSON from module0b.
    """

    def __init__(
        self,
        ranking_details: Dict[str, Any],
        parsed_records:  Dict[str, Optional[Dict[str, Any]]],
    ) -> None:
        self._rd      = ranking_details
        self._parsed  = parsed_records
        self._results: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run_all_checks(self) -> Dict[str, Any]:
        """Execute every fairness check and return a combined results dict.

        Returns:
            Dict with keys: checks (list), summary, overall_flag, n_candidates,
            generated_at, flags_raised.
        """
        n = len(self._rd)
        checks: List[Dict[str, Any]] = []

        checks.append(self._check_score_distribution())
        checks.append(self._check_score_compression())
        checks.append(self._check_skill_gap_disparity())
        checks.append(self._check_semantic_keyword_gap())
        checks.append(self._check_experience_correlation())
        checks.append(self._check_quality_score_spread())
        checks.append(self._check_parsed_coverage())

        flags_raised  = sum(1 for c in checks if c.get("flagged", False))
        overall_flag  = flags_raised > 0
        overall_label = "BIAS RISK DETECTED" if overall_flag else "NO BIAS SIGNALS DETECTED"

        self._results = {
            "generated_at":   datetime.utcnow().strftime(ISO_FMT),
            "n_candidates":   n,
            "flags_raised":   flags_raised,
            "overall_flag":   overall_flag,
            "overall_label":  overall_label,
            "checks":         checks,
        }
        return self._results

    # ──────────────────────────────────────────────────────────────────────
    # Individual checks
    # ──────────────────────────────────────────────────────────────────────

    def _scores(self, key: str = "final_score") -> List[float]:
        """Extract a named numeric field from all ranking entries."""
        return [float(v.get(key, 0.0)) for v in self._rd.values()]

    def _sorted_by_rank(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return ranking entries sorted ascending by rank (1 = best)."""
        return sorted(
            self._rd.items(),
            key=lambda kv: kv[1].get("rank", 9999),
        )

    def _check_score_distribution(self) -> Dict[str, Any]:
        """Report min / max / mean / std-dev of final_score."""
        scores = self._scores("final_score")
        if not scores:
            return self._check_result(
                "score_distribution",
                "Score Distribution",
                False,
                "No scores available.",
                {},
            )

        mn    = min(scores)
        mx    = max(scores)
        mu    = _mean(scores)
        sigma = _stddev(scores)

        detail = {
            "min":    round(mn,    4),
            "max":    round(mx,    4),
            "mean":   round(mu,    4),
            "stddev": round(sigma, 4),
            "spread": round(mx - mn, 4),
        }
        note = (
            f"Final scores range {_pct(mn)}–{_pct(mx)}, "
            f"mean {_pct(mu)}, σ={sigma:.3f}."
        )
        return self._check_result(
            "score_distribution", "Score Distribution", False, note, detail
        )

    def _check_score_compression(self) -> Dict[str, Any]:
        """Flag if score spread (max−min) is below COMPRESSION_THRESHOLD."""
        scores = self._scores("final_score")
        if len(scores) < 2:
            return self._check_result(
                "score_compression",
                "Score Compression",
                False,
                "Fewer than 2 candidates — compression check skipped.",
                {},
            )

        spread  = max(scores) - min(scores)
        flagged = spread < COMPRESSION_THRESHOLD
        note = (
            f"Score spread = {spread:.3f} "
            f"({'< ' if flagged else '>= '}{COMPRESSION_THRESHOLD} threshold). "
            + (
                "Scores are too similar to reliably differentiate candidates. "
                "Review JD specificity or ranking weights."
                if flagged else
                "Spread is sufficient to differentiate candidates."
            )
        )
        return self._check_result(
            "score_compression", "Score Compression", flagged, note,
            {"spread": round(spread, 4), "threshold": COMPRESSION_THRESHOLD},
        )

    def _check_skill_gap_disparity(self) -> Dict[str, Any]:
        """Flag candidates with zero matched JD skills."""
        zero_match: List[str] = []
        detail_rows: List[Dict[str, Any]] = []

        for cid, entry in self._sorted_by_rank():
            matched = entry.get("matched_skills", [])
            missing = entry.get("missing_skills", [])
            jd_count = int(entry.get("jd_skill_count", 0))
            row = {
                "candidate_id":  cid,
                "rank":          entry.get("rank", 0),
                "matched_count": len(matched),
                "missing_count": len(missing),
                "jd_skill_count": jd_count,
            }
            detail_rows.append(row)
            if len(matched) == 0 and jd_count > 0:
                zero_match.append(cid)

        flagged = bool(zero_match)
        note = (
            f"{len(zero_match)} candidate(s) matched zero JD skills: "
            f"{', '.join(zero_match)}. "
            "These candidates were ranked using semantic similarity only — "
            "verify that their CVs are not systematically disadvantaged by skill vocabulary."
            if flagged else
            "All candidates matched at least one JD skill."
        )
        return self._check_result(
            "skill_gap_disparity", "Skill-Gap Disparity",
            flagged, note,
            {"zero_match_candidates": zero_match, "per_candidate": detail_rows},
        )

    def _check_semantic_keyword_gap(self) -> Dict[str, Any]:
        """Flag if the top-ranked candidate's semantic and keyword scores diverge widely."""
        ranked = self._sorted_by_rank()
        if not ranked:
            return self._check_result(
                "semantic_keyword_gap",
                "Semantic vs Keyword Gap",
                False,
                "No candidates — check skipped.",
                {},
            )

        top_cid, top_entry = ranked[0]
        sem  = float(top_entry.get("semantic_score", 0.0))
        kw   = float(top_entry.get("keyword_score",  0.0))
        gap  = abs(sem - kw)
        flagged = gap > SEMANTIC_KEYWORD_GAP

        detail = {
            "top_candidate":  top_cid,
            "semantic_score": round(sem, 4),
            "keyword_score":  round(kw,  4),
            "gap":            round(gap, 4),
            "threshold":      SEMANTIC_KEYWORD_GAP,
        }
        note = (
            f"Top candidate {top_cid}: semantic={_pct(sem)}, keyword={_pct(kw)}, "
            f"gap={gap:.3f}. "
            + (
                "Large gap may indicate the top candidate scored high on contextual "
                "relevance but is missing explicit skill keywords — or vice versa. "
                "Manual review recommended."
                if flagged else
                "Semantic and keyword scores are consistent for the top candidate."
            )
        )
        return self._check_result(
            "semantic_keyword_gap", "Semantic vs Keyword Gap", flagged, note, detail
        )

    def _check_experience_correlation(self) -> Dict[str, Any]:
        """Detect if experience months is strongly correlated with rank position."""
        exp_months: List[float] = []
        ranks:      List[float] = []
        cids_used:  List[str]   = []

        for cid, entry in self._rd.items():
            parsed = self._parsed.get(cid)
            if parsed is None:
                continue
            months = int(parsed.get("sections", {}).get("total_experience_months", 0))
            rank   = int(entry.get("rank", 0))
            if rank > 0:
                exp_months.append(float(months))
                ranks.append(float(rank))
                cids_used.append(cid)

        if len(exp_months) < MIN_CANDIDATES_FOR_CORR:
            return self._check_result(
                "experience_correlation",
                "Experience Proxy Bias",
                False,
                f"Fewer than {MIN_CANDIDATES_FOR_CORR} candidates with parsed data "
                "— correlation check skipped.",
                {"candidates_with_parsed_data": len(exp_months)},
            )

        # Correlate experience months vs rank (lower rank = better)
        # Negative r means more experience → better rank (expected benign signal)
        # Positive r means more experience → worse rank (anomalous)
        # We flag |r| > threshold regardless of direction
        r = _pearson_r(exp_months, ranks)
        abs_r   = abs(r)
        flagged = abs_r > EXPERIENCE_CORRELATION_THRESHOLD

        direction = "higher" if r < 0 else "lower"
        detail = {
            "pearson_r":               round(r, 4),
            "abs_r":                   round(abs_r, 4),
            "threshold":               EXPERIENCE_CORRELATION_THRESHOLD,
            "candidates_analysed":     len(exp_months),
            "correlation_direction":   direction,
        }
        note = (
            f"Pearson r(experience, rank) = {r:.3f} "
            f"(|r|={abs_r:.3f}, threshold={EXPERIENCE_CORRELATION_THRESHOLD}). "
            + (
                f"Strong correlation detected: more experience → {direction} rank. "
                "Experience may be acting as an unintended proxy variable. "
                "Verify that ranking weights reflect job requirements, not seniority."
                if flagged else
                f"Experience–rank correlation is within acceptable bounds (|r|={abs_r:.3f})."
            )
        )
        return self._check_result(
            "experience_correlation", "Experience Proxy Bias", flagged, note, detail
        )

    def _check_quality_score_spread(self) -> Dict[str, Any]:
        """Flag if quality_score spread is large enough to materially affect outcomes."""
        q_scores = self._scores("quality_score")
        if len(q_scores) < 2:
            return self._check_result(
                "quality_score_spread",
                "CV Quality Score Influence",
                False,
                "Fewer than 2 candidates — quality spread check skipped.",
                {},
            )

        spread  = max(q_scores) - min(q_scores)
        flagged = spread > QUALITY_SPREAD_THRESHOLD
        note = (
            f"Quality score spread = {spread:.3f} "
            f"({'> ' if flagged else '<= '}{QUALITY_SPREAD_THRESHOLD}). "
            + (
                "Large variation in CV formatting quality is significantly affecting "
                "final scores (quality contributes up to 15% bonus). "
                "Candidates with poorly-formatted CVs may be unfairly penalised."
                if flagged else
                "CV quality score spread is within acceptable bounds."
            )
        )
        return self._check_result(
            "quality_score_spread", "CV Quality Score Influence",
            flagged, note,
            {
                "min":      round(min(q_scores), 4),
                "max":      round(max(q_scores), 4),
                "spread":   round(spread, 4),
                "threshold": QUALITY_SPREAD_THRESHOLD,
            },
        )

    def _check_parsed_coverage(self) -> Dict[str, Any]:
        """Flag candidates in ranking without a corresponding parsed JSON."""
        missing: List[str] = []
        for cid in self._rd:
            if self._parsed.get(cid) is None:
                missing.append(cid)

        flagged = bool(missing)
        note = (
            f"{len(missing)} ranked candidate(s) have no parsed JSON: "
            f"{', '.join(missing)}. "
            "These candidates were scored with default quality (0.5) and zero "
            "keyword matches, which may disadvantage them unfairly."
            if flagged else
            "All ranked candidates have corresponding parsed JSON files."
        )
        return self._check_result(
            "parsed_coverage", "Parsed Data Coverage",
            flagged, note,
            {
                "total_ranked":    len(self._rd),
                "missing_parsed":  missing,
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Factory helper
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_result(
        check_id: str,
        label:    str,
        flagged:  bool,
        note:     str,
        detail:   Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a uniform check-result dict."""
        return {
            "check_id": check_id,
            "label":    label,
            "flagged":  flagged,
            "status":   "FLAGGED" if flagged else "OK",
            "note":     note,
            "detail":   detail,
        }


# ══════════════════════════════════════════════════════════════════════════
# Report writer
# ══════════════════════════════════════════════════════════════════════════

def _build_report_lines(results: Dict[str, Any]) -> List[str]:
    """Convert the BiasAuditor results dict into a human-readable text report.

    Args:
        results: Dict returned by BiasAuditor.run_all_checks().

    Returns:
        List of strings (one per line) ready to be joined with '\\n'.
    """
    now_human = datetime.utcnow().strftime(HUMAN_FMT)
    n         = results["n_candidates"]
    flag_n    = results["flags_raised"]
    overall   = results["overall_label"]
    checks    = results["checks"]

    lines: List[str] = [
        HEAVY_DIVIDER,
        "  BIAS-FREE HIRING PIPELINE — BIAS DETECTION & FAIRNESS AUDIT",
        f"  Generated : {now_human}",
        f"  Candidates: {n}",
        f"  Checks run: {len(checks)}",
        f"  Flags raised: {flag_n}",
        HEAVY_DIVIDER,
        "",
        f"  OVERALL VERDICT: {overall}",
        "",
        DIVIDER,
        "  INDIVIDUAL CHECKS",
        DIVIDER,
    ]

    for i, check in enumerate(checks, start=1):
        label   = check["label"]
        status  = _flag_label(check["flagged"])
        note    = check["note"]

        lines.append("")
        lines.append(f"  {i}. {label}")
        lines.append(f"     Status : {status}")
        # Wrap the note at 74 chars
        words     = note.split()
        line_buf  = "     Note   : "
        indent    = "              "
        first     = True
        for word in words:
            if len(line_buf) + len(word) + 1 > REPORT_WIDTH and not first:
                lines.append(line_buf.rstrip())
                line_buf = indent + word + " "
                first = False
            else:
                line_buf += word + " "
                first = False
        if line_buf.strip():
            lines.append(line_buf.rstrip())

    lines += [
        "",
        DIVIDER,
        "  RECOMMENDATIONS",
        DIVIDER,
        "",
    ]

    if flag_n == 0:
        lines.append(
            "  No fairness flags were raised. The ranking appears consistent "
            "with the configured scoring weights."
        )
        lines.append(
            "  Re-run this audit after any change to job description, scoring "
            "weights, or candidate pool."
        )
    else:
        lines.append(
            f"  {flag_n} fairness flag(s) were raised. Review the flagged checks "
            "above before sharing ranking results with hiring stakeholders."
        )
        lines.append("")
        flagged_checks = [c for c in checks if c["flagged"]]
        for check in flagged_checks:
            lines.append(f"  • {check['label']}: {check['note'][:120]}")
        lines.append("")
        lines.append(
            "  This report does not constitute a definitive finding of bias. "
            "Flags indicate statistical patterns that warrant human review."
        )

    lines += [
        "",
        HEAVY_DIVIDER,
        "  END OF BIAS AUDIT REPORT",
        HEAVY_DIVIDER,
        "",
    ]
    return lines


# ══════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════

def _write_text_report(lines: List[str], output_dir: str) -> str:
    """Write the human-readable bias report to output_dir/bias_report.txt.

    Args:
        lines:      Lines returned by _build_report_lines().
        output_dir: Destination directory.

    Returns:
        Absolute path of the written file.

    Raises:
        OSError: On any filesystem write failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, BIAS_REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _write_json_audit(results: Dict[str, Any], output_dir: str) -> str:
    """Write machine-readable bias audit JSON to output_dir/bias_audit.json.

    Args:
        results:    Dict returned by BiasAuditor.run_all_checks().
        output_dir: Destination directory.

    Returns:
        Absolute path of the written file.

    Raises:
        OSError: On any filesystem write failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, BIAS_AUDIT_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    return path


# ══════════════════════════════════════════════════════════════════════════
# Public entry point — called by main.py Stage 4
# ══════════════════════════════════════════════════════════════════════════

def run(
    ranking_details: Dict[str, Any],
    parsed_dir:      str,
    output_dir:      str,
) -> bool:
    """Execute the full bias detection and fairness audit.

    This is the only function called by main.py:

        ok = module4.run(
            ranking_details = ranker.last_ranking_details,
            parsed_dir      = args.parsed_dir,
            output_dir      = args.audit_dir,
        )

    Args:
        ranking_details: ranker.last_ranking_details dict from module1.
        parsed_dir:      Path to the parsed/ directory from module0b.
        output_dir:      Directory where bias_report.txt and bias_audit.json
                         will be written (normally the same audit/ as module3).

    Returns:
        True on success (both output files written), False on failure.
    """
    import time
    t0 = time.time()

    print()
    print("=" * 60)
    print("  MODULE 4 — BIAS DETECTION & FAIRNESS AUDIT")
    print("=" * 60)

    n_candidates = len(ranking_details)
    print(f"\n  Candidates  : {n_candidates}")
    print(f"  Parsed JSON : {parsed_dir}")
    print(f"  Output      : {output_dir}")

    if n_candidates == 0:
        print("\n  [WARNING] ranking_details is empty — nothing to audit.")
        return True  # not a failure of this module

    # ── Load parsed JSON for every ranked candidate ────────────────────────
    parsed_records: Dict[str, Optional[Dict[str, Any]]] = {}
    for cid in ranking_details:
        parsed_records[cid] = _load_parsed_json(parsed_dir, cid)

    missing_count = sum(1 for v in parsed_records.values() if v is None)
    if missing_count:
        logger.warning(
            "module4: %d of %d parsed JSONs missing — checks will note gaps.",
            missing_count, n_candidates,
        )

    # ── Run fairness checks ────────────────────────────────────────────────
    auditor = BiasAuditor(ranking_details, parsed_records)
    results = auditor.run_all_checks()

    flags    = results["flags_raised"]
    overall  = results["overall_label"]
    elapsed  = time.time() - t0

    # ── Write outputs ──────────────────────────────────────────────────────
    try:
        report_lines = _build_report_lines(results)
        txt_path     = _write_text_report(report_lines, output_dir)
        json_path    = _write_json_audit(results, output_dir)
    except OSError as exc:
        logger.error("module4: failed to write output files: %s", exc)
        print(f"\n  [ERROR] Could not write output files: {exc}")
        return False

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(
        "module4: Bias audit complete — %d candidate(s), %d flag(s). %s",
        n_candidates, flags, overall,
    )
    logger.info("module4: Bias report → %s", txt_path)
    logger.info("module4: Bias audit  → %s", json_path)

    print(f"\n  Bias audit complete in {elapsed:.1f}s")
    print(f"  Checks run    : {len(results['checks'])}")
    print(f"  Flags raised  : {flags}")
    print(f"  Overall verdict: {overall}")
    print(f"  Report → {txt_path}")
    print(f"  JSON   → {json_path}")
    print("=" * 60)

    return True


# ══════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone execution."""
    p = argparse.ArgumentParser(
        prog="module4",
        description="Bias Detection & Fairness Audit — Bias-Free Hiring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python module4.py --ranking-file ranking_details.json "
            "--parsed-dir parsed --output-dir audit\n"
        ),
    )
    _base = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--ranking-file",
        default=None,
        help="Path to a ranking_details JSON file saved from a previous run.",
    )
    p.add_argument(
        "--parsed-dir",
        default=os.path.join(_base, "parsed"),
        help="Path to the parsed/ directory (default: parsed/)",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(_base, "audit"),
        help="Directory for bias report output (default: audit/)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return p


def _configure_logging(verbose: bool = False) -> None:
    """Configure root logger for standalone execution."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_DATE_FMT,
    )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args   = parser.parse_args()
    _configure_logging(args.verbose)

    if not args.ranking_file:
        parser.error("--ranking-file is required for standalone execution.")

    if not os.path.isfile(args.ranking_file):
        print(f"[ERROR] ranking-file not found: {args.ranking_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.ranking_file, "r", encoding="utf-8") as _fh:
            _ranking_details = json.load(_fh)
    except (json.JSONDecodeError, OSError) as _exc:
        print(f"[ERROR] Cannot read ranking file: {_exc}", file=sys.stderr)
        sys.exit(1)

    ok = run(
        ranking_details = _ranking_details,
        parsed_dir      = args.parsed_dir,
        output_dir      = args.output_dir,
    )
    sys.exit(0 if ok else 1)
