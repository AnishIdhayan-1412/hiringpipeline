# -*- coding: utf-8 -*-
"""
module2.py — Explainability Engine
====================================
Bias-Free Hiring Pipeline — Stage 2

Reads the ranking decisions produced by module1 and generates a
plain-English explanation for every candidate: exactly why they ranked
where they did, what their strengths are, what they are missing, and
whether HR should invite them to interview.

Pipeline position
-----------------
    module0  → anonymized_cvs/Candidate_01.txt    (PII stripped)
    module0b → parsed/Candidate_01.json            (structured extraction)
    module1  → ranking_report.txt                  (numeric ranking)
               ranker.last_ranking_details         (in-memory dict)
    module2  → explanations/Candidate_01.json      ← per-candidate output
               explanations/summary_report.txt     ← full run summary

Standalone usage
----------------
    python module2.py --ranking-file ranking_details.json \\
                      --parsed-dir parsed \\
                      --output-dir explanations

    python module2.py --ranking-file ranking_details.json \\
                      --parsed-dir parsed \\
                      --output-dir explanations \\
                      --verbose

Programmatic usage (called by main.py)
---------------------------------------
    import module2

    ok = module2.run(
        ranking_details = ranker.last_ranking_details,
        parsed_dir      = "parsed",
        output_dir      = "explanations",
    )

Dependencies
------------
    Pure Python stdlib only: json, os, logging, datetime, argparse, sys,
    textwrap.  Zero external packages. No LLMs. No ML. No Flask. No databases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════
# Module-level logger — never use print() in helpers; only inside run()
# ══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Constants ─ every threshold, label, and template string lives here.
# Changing a threshold means changing exactly one line.
# ══════════════════════════════════════════════════════════════════════════

# ── Verdict thresholds ────────────────────────────────────────────────────
VERDICT_STRONG:  float = 0.75
VERDICT_GOOD:    float = 0.55
VERDICT_PARTIAL: float = 0.40

# ── Score label thresholds ────────────────────────────────────────────────
LABEL_EXCELLENT: float = 0.80
LABEL_GOOD:      float = 0.60
LABEL_FAIR:      float = 0.40

# ── Strength-trigger thresholds ───────────────────────────────────────────
STRENGTH_KEYWORD_FLOOR:     float = 0.70
STRENGTH_SEMANTIC_FLOOR:    float = 0.70
STRENGTH_EXPERIENCE_SENIOR: int   = 36   # months  → "Extensive experience"
STRENGTH_EXPERIENCE_JUNIOR: int   = 12   # months  → "Relevant experience"
STRENGTH_QUALITY_FLOOR:     float = 0.80

# ── Gap-trigger thresholds ────────────────────────────────────────────────
GAP_EXPERIENCE_FLOOR:  int   = 12      # months below this → limited experience
GAP_KEYWORD_FLOOR:     float = 0.30    # below this → low skill overlap
GAP_QUALITY_FLOOR:     float = 0.50    # below this → CV quality concern
GAP_MAX_SKILL_BULLETS: int   = 5       # max individual skill bullets before overflow note

# ── Degree levels that qualify as "advanced" for strength detection ────────
ADVANCED_DEGREE_LEVELS: frozenset = frozenset({
    "PhD", "Master", "MBA", "M.Sc", "M.A", "M.Tech", "M.E", "MCA", "Postgraduate",
})

# ── Output format ─────────────────────────────────────────────────────────
ISO_DATETIME_FMT:   str = "%Y-%m-%dT%H:%M:%S"
HUMAN_DATETIME_FMT: str = "%d %b %Y at %H:%M"
REPORT_LINE_WIDTH:  int = 78
SUMMARY_FILENAME:   str = "summary_report.txt"
OUTPUT_DIR_DEFAULT: str = "explanations"

# ── Logging ───────────────────────────────────────────────────────────────
_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT:   str = "%H:%M:%S"


# ══════════════════════════════════════════════════════════════════════════
# Pure helpers — no I/O, no side effects, deterministic
# ══════════════════════════════════════════════════════════════════════════

def _verdict(score: float) -> str:
    """Map a final score to its four-tier verdict label.

    Args:
        score: Candidate's final_score in the range 0.0 – 1.0.

    Returns:
        One of: "Strong Match", "Good Match", "Partial Match", "Weak Match".
    """
    if score >= VERDICT_STRONG:
        return "Strong Match"
    if score >= VERDICT_GOOD:
        return "Good Match"
    if score >= VERDICT_PARTIAL:
        return "Partial Match"
    return "Weak Match"


def _score_label(score: float) -> str:
    """Map a numeric sub-score (0–1) to a qualitative label.

    Args:
        score: Any sub-score: semantic, keyword, quality, or final.

    Returns:
        One of: "Excellent", "Good", "Fair", "Poor".
    """
    if score >= LABEL_EXCELLENT:
        return "Excellent"
    if score >= LABEL_GOOD:
        return "Good"
    if score >= LABEL_FAIR:
        return "Fair"
    return "Poor"


def _pct(score: float) -> str:
    """Format a 0–1 float as a whole-number percentage string.

    Args:
        score: Value between 0.0 and 1.0.

    Returns:
        E.g. 0.752 → "75%",  1.0 → "100%".
    """
    return f"{round(score * 100)}%"


def _format_years(months: int) -> str:
    """Express a month count as a concise years string.

    Strips the trailing ".0" so 36 months → "3 years", not "3.0 years".
    Preserves one decimal for fractional years: 18 months → "1.5 years".

    Args:
        months: Non-negative integer number of months.

    Returns:
        E.g. "3 years", "1.5 years", "1 year".
    """
    if months <= 0:
        return "0 years"
    raw   = months / 12
    years = int(raw) if raw == int(raw) else round(raw, 1)
    return f"{years} year{'s' if years != 1 else ''}"


def _join_skills(skills: List[str], limit: int) -> str:
    """Join up to *limit* skills into a comma-separated string.

    Args:
        skills: List of skill name strings.
        limit:  Maximum number of items to include.

    Returns:
        E.g. ["A","B","C"] with limit=2 → "A, B".
    """
    return ", ".join(skills[:limit])


def _highest_education(
    education: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Select the highest-level education entry from a parsed education list.

    Uses an explicit priority ordering so "PhD" always ranks above "B.Tech"
    regardless of list order or alphabetical comparison.

    Args:
        education: List of education entry dicts produced by module0b.
                   Each dict may have keys: "level", "field".

    Returns:
        Tuple (level, field). Both are empty strings if no entries exist
        or none contain recognisable level/field data.
    """
    _PRIORITY: List[str] = [
        "PhD", "MBA", "M.Tech", "M.Sc", "M.E", "M.A", "MCA", "Master",
        "Postgraduate", "B.Tech", "B.E", "B.Sc", "B.A", "BCA", "Bachelor",
        "Undergraduate", "Diploma", "Certificate", "HSC", "SSLC",
        "High School", "Secondary", "CBSE", "ICSE",
    ]
    if not education:
        return "", ""

    best_level:    str = ""
    best_field:    str = ""
    best_priority: int = len(_PRIORITY) + 1

    for entry in education:
        level = entry.get("level", "") or ""
        field = entry.get("field", "") or ""
        try:
            priority = _PRIORITY.index(level)
        except ValueError:
            priority = len(_PRIORITY)

        if priority < best_priority:
            best_priority = priority
            best_level    = level
            best_field    = field

    return best_level, best_field


# ══════════════════════════════════════════════════════════════════════════
# Score-breakdown note generators
# Each function returns a single sentence tied to the actual signal value.
# ══════════════════════════════════════════════════════════════════════════

def _semantic_note(semantic_score: float) -> str:
    """Generate a concise interpretation of the semantic similarity score.

    Args:
        semantic_score: Cosine similarity between JD and CV embeddings (0–1).

    Returns:
        A plain-English note for the score_breakdown JSON field.
    """
    if semantic_score >= LABEL_EXCELLENT:
        return "CV content closely matches the job description"
    if semantic_score >= LABEL_GOOD:
        return "CV content shows good alignment with the job description"
    if semantic_score >= LABEL_FAIR:
        return "CV content partially overlaps with the job description"
    return "CV content shows weak alignment with the job description"


def _keyword_note(matched: int, total: int) -> str:
    """Generate a concrete note for the keyword match score.

    Args:
        matched: Number of JD skills found in the candidate's CV.
        total:   Total number of skills extracted from the JD.

    Returns:
        E.g. "Matched 9 of 12 required skills".
    """
    if total == 0:
        return "No specific skills were extracted from the job description"
    if matched == total:
        return f"Matched all {total} required skill{'s' if total != 1 else ''}"
    return f"Matched {matched} of {total} required skill{'s' if total != 1 else ''}"


def _quality_note(quality_score: float) -> str:
    """Generate a concise interpretation of the CV quality score.

    Args:
        quality_score: CV completeness/structure score from module0b (0–1).

    Returns:
        A plain-English note for the score_breakdown JSON field.
    """
    if quality_score >= LABEL_EXCELLENT:
        return "CV is well-structured and complete"
    if quality_score >= LABEL_GOOD:
        return "CV is reasonably structured with most sections present"
    if quality_score >= LABEL_FAIR:
        return "CV structure is adequate but some sections are sparse"
    return "CV may be incomplete or poorly formatted"


# ══════════════════════════════════════════════════════════════════════════
# Headline generator
# ══════════════════════════════════════════════════════════════════════════

def _generate_headline(
    rank:          int,
    n_candidates:  int,
    verdict:       str,
    total_months:  int,
    keyword_score: float,
) -> str:
    """Generate the one-line headline for a candidate explanation.

    The second clause reflects the single most prominent signal so that
    every headline gives a unique, actionable first impression.

    Args:
        rank:          Candidate's position in the ranking (1-based).
        n_candidates:  Total number of candidates in this run.
        verdict:       Verdict string from _verdict().
        total_months:  Total professional experience in months.
        keyword_score: Fraction of JD skills matched (0–1).

    Returns:
        A one-sentence headline string ready for JSON and report output.
    """
    of_n   = f"of {n_candidates}" if n_candidates > 1 else ""
    prefix = f"Ranked #{rank} {of_n}".strip()

    if verdict == "Strong Match":
        if total_months >= STRENGTH_EXPERIENCE_JUNIOR:
            return f"{prefix} — strong skill alignment and solid experience"
        return f"{prefix} — strong skill alignment with the job requirements"

    if verdict == "Good Match":
        if total_months >= STRENGTH_EXPERIENCE_JUNIOR:
            return f"{prefix} — good overall fit with some skill gaps"
        return f"{prefix} — good overall fit, limited experience on record"

    if verdict == "Partial Match":
        return f"{prefix} — partial match, significant gaps in required skills"

    return f"{prefix} — weak match, skills are largely misaligned with requirements"


# ══════════════════════════════════════════════════════════════════════════
# Summary sentence generators
# ══════════════════════════════════════════════════════════════════════════

def _generate_skills_summary(
    matched_count: int,
    total:         int,
    missing:       List[str],
) -> str:
    """Generate a plain-English skills match summary sentence.

    Args:
        matched_count: Number of JD skills present in the CV.
        total:         Total number of skills extracted from the JD.
        missing:       List of skill names absent from the CV.

    Returns:
        A concise summary sentence that names up to three missing skills.
    """
    if total == 0:
        return "No specific skills were extracted from the job description to compare against."
    if matched_count == total:
        return f"Candidate has all {total} required skill{'s' if total != 1 else ''}."

    shown    = missing[:3]
    overflow = len(missing) - 3
    missing_str = _join_skills(shown, 3)
    if overflow > 0:
        missing_str += f", and {overflow} more"

    return (
        f"Candidate has {matched_count} of {total} required skill{'s' if total != 1 else ''}. "
        f"Missing: {missing_str}."
    )


def _generate_experience_summary(
    total_months:   int,
    n_roles:        int,
    data_available: bool,
) -> str:
    """Generate a plain-English experience summary sentence.

    Args:
        total_months:   Total professional experience in months.
        n_roles:        Number of distinct role entries detected.
        data_available: False if the parsed JSON file was missing.

    Returns:
        A single natural-language sentence describing the experience profile.
    """
    if not data_available:
        return "Detailed experience data unavailable — parsed CV not found."
    if total_months <= 0:
        if n_roles == 0:
            return "No experience entries could be extracted from this CV."
        return "Experience section present but duration could not be determined."

    years_str   = _format_years(total_months)
    role_phrase = (
        f"across {n_roles} role{'s' if n_roles != 1 else ''}"
        if n_roles > 0
        else "on record"
    )
    return f"{years_str.capitalize()} of professional experience {role_phrase}."


def _generate_education_summary(
    level:          str,
    field:          str,
    data_available: bool,
) -> str:
    """Generate a plain-English education summary sentence.

    Args:
        level:          Highest degree level string (may be empty).
        field:          Field of study string (may be empty).
        data_available: False if the parsed JSON file was missing.

    Returns:
        A single natural-language sentence.
    """
    if not data_available:
        return "Detailed education data unavailable — parsed CV not found."
    if level and field:
        return f"{level} in {field}"
    if level:
        return f"{level} (field not specified)"
    if field:
        return f"Degree in {field} (level not recorded)"
    return "Education details could not be extracted from this CV."


# ══════════════════════════════════════════════════════════════════════════
# Strengths generator
# Rules are applied exactly as specified — each fires independently.
# ══════════════════════════════════════════════════════════════════════════

def _generate_strengths(
    keyword_score:  float,
    semantic_score: float,
    total_months:   int,
    quality_score:  float,
    edu_level:      str,
    edu_field:      str,
    missing_skills: List[str],
) -> List[str]:
    """Generate a list of strength statements from exact rule-based conditions.

    Rules:
        keyword_score >= 0.70   → "Strong skill match — X% of required skills present"
        semantic_score >= 0.70  → "High semantic relevance — CV content closely matches job requirements"
        total_months >= 36      → "Extensive experience — X years in the field"
        total_months >= 12      → "Relevant experience — X years in the field"  (only if < 36)
        quality_score >= 0.80   → "Well-structured CV indicating attention to detail"
        edu level PhD/Master/MBA→ "Advanced academic qualification — [level] in [field]"
        no missing skills       → "Complete skill match — all required skills present"

    Args:
        keyword_score:  Fraction of JD skills matched (0–1).
        semantic_score: Cosine similarity between JD and CV (0–1).
        total_months:   Total professional experience in months.
        quality_score:  CV completeness/structure score (0–1).
        edu_level:      Highest detected education level (may be empty).
        edu_field:      Field of study (may be empty).
        missing_skills: Skills from the JD not found in this CV.

    Returns:
        Ordered list of plain-English strength strings. May be empty.
    """
    strengths: List[str] = []

    # Rule: Complete skill match (highest positive signal — listed first)
    if len(missing_skills) == 0:
        strengths.append("Complete skill match — all required skills present")

    # Rule: Strong keyword overlap (only when there are missing skills,
    #       to avoid contradicting "Complete skill match" above)
    elif keyword_score >= STRENGTH_KEYWORD_FLOOR:
        strengths.append(
            f"Strong skill match — {_pct(keyword_score)} of required skills present"
        )

    # Rule: High semantic relevance
    if semantic_score >= STRENGTH_SEMANTIC_FLOOR:
        strengths.append(
            "High semantic relevance — CV content closely matches job requirements"
        )

    # Rule: Experience depth — exactly one of these two fires
    if total_months >= STRENGTH_EXPERIENCE_SENIOR:
        strengths.append(
            f"Extensive experience — {_format_years(total_months)} in the field"
        )
    elif total_months >= STRENGTH_EXPERIENCE_JUNIOR:
        strengths.append(
            f"Relevant experience — {_format_years(total_months)} in the field"
        )

    # Rule: CV quality signal
    if quality_score >= STRENGTH_QUALITY_FLOOR:
        strengths.append("Well-structured CV indicating attention to detail")

    # Rule: Advanced academic qualification
    if edu_level in ADVANCED_DEGREE_LEVELS:
        if edu_field:
            strengths.append(
                f"Advanced academic qualification — {edu_level} in {edu_field}"
            )
        else:
            strengths.append(f"Advanced academic qualification — {edu_level}")

    return strengths


# ══════════════════════════════════════════════════════════════════════════
# Gaps generator
# Rules are applied exactly as specified — each fires independently.
# ══════════════════════════════════════════════════════════════════════════

def _generate_gaps(
    missing_skills: List[str],
    total_months:   int,
    keyword_score:  float,
    quality_score:  float,
    data_available: bool,
) -> List[str]:
    """Generate a list of gap statements from exact rule-based conditions.

    Rules:
        Each missing_skill (max 5) → "Missing [skill] — listed as required in JD"
        If > 5 missing             → "...and N more skills not found in CV"
        total_months < 12          → "Limited professional experience (under 1 year)"
        keyword_score < 0.30       → "Low skill overlap with job requirements"
        quality_score < 0.50       → "CV may be incomplete or poorly structured"

    Args:
        missing_skills: Skills from the JD not found in this CV.
        total_months:   Total professional experience in months.
        keyword_score:  Fraction of JD skills matched (0–1).
        quality_score:  CV completeness/structure score (0–1).
        data_available: False if the parsed JSON file was missing.

    Returns:
        Ordered list of plain-English gap strings. May be empty.
    """
    gaps: List[str] = []

    # Rule: Individual missing skills, capped at GAP_MAX_SKILL_BULLETS
    shown    = missing_skills[:GAP_MAX_SKILL_BULLETS]
    overflow = len(missing_skills) - GAP_MAX_SKILL_BULLETS

    for skill in shown:
        gaps.append(f"Missing {skill} — listed as required in JD")

    if overflow > 0:
        gaps.append(
            f"...and {overflow} more skill{'s' if overflow != 1 else ''} not found in CV"
        )

    # Rule: Limited experience (only when we have parsed data to support it)
    if data_available and total_months < GAP_EXPERIENCE_FLOOR:
        gaps.append("Limited professional experience (under 1 year)")

    # Rule: Low overall skill overlap (only fires when no individual skills
    #       were enumerated above — prevents repeating the same finding)
    if not missing_skills and keyword_score < GAP_KEYWORD_FLOOR:
        gaps.append("Low skill overlap with job requirements")

    # Rule: CV quality concern
    if quality_score < GAP_QUALITY_FLOOR:
        gaps.append("CV may be incomplete or poorly structured")

    return gaps


# ══════════════════════════════════════════════════════════════════════════
# Recommendation generator
# ══════════════════════════════════════════════════════════════════════════

def _generate_recommendation(
    verdict:        str,
    strengths:      List[str],
    missing_skills: List[str],
) -> str:
    """Generate a concrete, actionable recommendation sentence.

    Templates (applied exactly as specified):
        Strong Match + missing   → "Recommended for interview. [top strength]. Suggest probing [first missing] in the interview."
        Strong Match + no missing→ "Recommended for interview. [top strength]. Candidate meets all listed requirements."
        Good Match + missing     → "Consider for interview. Good overall fit. May need support with [top 2 missing]."
        Good Match + no missing  → "Consider for interview. Good overall fit. Candidate meets all listed requirements."
        Partial Match + missing  → "Review manually. Has potential but significant gaps in [top 3 missing]."
        Partial Match + no miss. → "Review manually. Has potential but no specific skill gaps identified."
        Weak Match + missing     → "Not recommended at this stage. Significant skill gaps: [top 3 missing]."
        Weak Match + no missing  → "Not recommended at this stage. Overall profile does not match requirements."

    Args:
        verdict:        Verdict string from _verdict().
        strengths:      Ordered strengths list for this candidate.
        missing_skills: Skills from the JD not found in this CV.

    Returns:
        A 1–2 sentence recommendation string.
    """
    has_missing   = bool(missing_skills)
    top_strength  = strengths[0] if strengths else "Strong overall profile"
    top_missing_1 = missing_skills[0] if has_missing else ""
    top_missing_2 = _join_skills(missing_skills, 2)
    top_missing_3 = _join_skills(missing_skills, 3)

    if verdict == "Strong Match":
        if has_missing:
            return (
                f"Recommended for interview. {top_strength}. "
                f"Suggest probing {top_missing_1} in the interview."
            )
        return (
            f"Recommended for interview. {top_strength}. "
            "Candidate meets all listed requirements."
        )

    if verdict == "Good Match":
        if has_missing:
            return (
                "Consider for interview. Good overall fit. "
                f"May need support with {top_missing_2}."
            )
        return (
            "Consider for interview. Good overall fit. "
            "Candidate meets all listed requirements."
        )

    if verdict == "Partial Match":
        if has_missing:
            return (
                "Review manually. Has potential but significant gaps in "
                f"{top_missing_3}."
            )
        return (
            "Review manually. Has potential but no specific skill gaps identified."
        )

    # Weak Match
    if has_missing:
        return (
            "Not recommended at this stage. "
            f"Significant skill gaps: {top_missing_3}."
        )
    return (
        "Not recommended at this stage. "
        "Overall profile does not match the job requirements."
    )


# ══════════════════════════════════════════════════════════════════════════
# Parsed JSON loader
# ══════════════════════════════════════════════════════════════════════════

def _load_parsed_json(
    parsed_dir:   str,
    candidate_id: str,
) -> Optional[Dict[str, Any]]:
    """Load the module0b parsed JSON for one candidate.

    Never raises — returns None on any failure so the caller can gracefully
    generate a leaner explanation from ranking signals alone.

    Args:
        parsed_dir:   Path to the directory containing parsed JSON files.
        candidate_id: Candidate identifier (e.g. "Candidate_01").

    Returns:
        Parsed dict on success, None if the file is absent or unreadable.
    """
    path = os.path.join(parsed_dir, f"{candidate_id}.json")

    if not os.path.isfile(path):
        logger.warning(
            "%s: parsed JSON not found at '%s'. "
            "Explanation generated from ranking signals only.",
            candidate_id, path,
        )
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except json.JSONDecodeError as exc:
        logger.error(
            "%s: parsed JSON is malformed (%s). Falling back to ranking signals.",
            candidate_id, exc,
        )
        return None
    except OSError as exc:
        logger.error(
            "%s: cannot read parsed JSON (%s). Falling back to ranking signals.",
            candidate_id, exc,
        )
        return None


# ══════════════════════════════════════════════════════════════════════════
# ExplanationBuilder — assembles the canonical output dict for one candidate
# ══════════════════════════════════════════════════════════════════════════

class ExplanationBuilder:
    """Assembles the full explanation JSON dict for a single candidate.

    All text generation is delegated to the pure helper functions above.
    This class holds no state and can be used as a stateless factory.
    """

    def build(
        self,
        candidate_id:  str,
        ranking_entry: Dict[str, Any],
        parsed_data:   Optional[Dict[str, Any]],
        n_candidates:  int,
    ) -> Dict[str, Any]:
        """Build the complete explanation dict matching the canonical schema.

        Args:
            candidate_id:  Candidate identifier string (e.g. "Candidate_01").
            ranking_entry: Sub-dict from ranker.last_ranking_details.
            parsed_data:   Parsed JSON dict from module0b, or None if absent.
            n_candidates:  Total number of candidates in this ranking run.

        Returns:
            A fully populated dict matching the canonical output schema.
        """
        # ── Extract ranking signals — all casted for safety ───────────────
        final_score    = float(ranking_entry.get("final_score",    0.0))
        semantic_score = float(ranking_entry.get("semantic_score", 0.0))
        keyword_score  = float(ranking_entry.get("keyword_score",  0.0))
        quality_score  = float(ranking_entry.get("quality_score",  0.5))
        matched_skills = list(ranking_entry.get("matched_skills",  []))
        missing_skills = list(ranking_entry.get("missing_skills",  []))
        jd_skill_count = int(ranking_entry.get("jd_skill_count",   0))
        rank           = int(ranking_entry.get("rank",              0))

        matched_count  = len(matched_skills)

        # ── Extract structured context from parsed JSON ────────────────────
        data_available: bool                  = parsed_data is not None
        sections:       Dict[str, Any]        = (
            parsed_data.get("sections", {}) if parsed_data else {}
        )

        total_months: int                     = int(sections.get("total_experience_months", 0))
        roles:        List[Dict[str, Any]]    = list(sections.get("experience", []))
        education:    List[Dict[str, Any]]    = list(sections.get("education",  []))

        edu_level, edu_field = _highest_education(education)

        # ── Derived compound values ───────────────────────────────────────
        verdict     = _verdict(final_score)
        match_rate  = _pct(keyword_score) if jd_skill_count > 0 else "N/A"
        total_years = str(round(total_months / 12, 1)) if total_months > 0 else "0"

        # ── Generate all text ─────────────────────────────────────────────
        headline = _generate_headline(
            rank, n_candidates, verdict, total_months, keyword_score
        )
        strengths = _generate_strengths(
            keyword_score, semantic_score, total_months,
            quality_score, edu_level, edu_field, missing_skills,
        )
        gaps = _generate_gaps(
            missing_skills, total_months, keyword_score,
            quality_score, data_available,
        )
        recommendation = _generate_recommendation(verdict, strengths, missing_skills)

        # ── Assemble canonical output dict ────────────────────────────────
        return {
            "candidate_id": candidate_id,
            "rank":         rank,
            "final_score":  round(final_score, 4),
            "verdict":      verdict,
            "headline":     headline,
            "explanation": {
                "score_breakdown": {
                    "semantic_score": {
                        "value": round(semantic_score, 4),
                        "label": _score_label(semantic_score),
                        "note":  _semantic_note(semantic_score),
                    },
                    "keyword_score": {
                        "value": round(keyword_score, 4),
                        "label": _score_label(keyword_score),
                        "note":  _keyword_note(matched_count, jd_skill_count),
                    },
                    "quality_score": {
                        "value": round(quality_score, 4),
                        "label": _score_label(quality_score),
                        "note":  _quality_note(quality_score),
                    },
                    "final_score": {
                        "value": round(final_score, 4),
                        "label": verdict,
                    },
                },
                "skills": {
                    "matched":    matched_skills,
                    "missing":    missing_skills,
                    "match_rate": match_rate,
                    "summary":    _generate_skills_summary(
                        matched_count, jd_skill_count, missing_skills
                    ),
                },
                "experience": {
                    "total_months": total_months,
                    "total_years":  total_years,
                    "summary":      _generate_experience_summary(
                        total_months, len(roles), data_available
                    ),
                },
                "education": {
                    "highest_level": edu_level or None,
                    "field":         edu_field or None,
                    "summary":       _generate_education_summary(
                        edu_level, edu_field, data_available
                    ),
                },
                "strengths":      strengths,
                "gaps":           gaps,
                "recommendation": recommendation,
            },
            "generated_at": datetime.now().strftime(ISO_DATETIME_FMT),
        }


# ══════════════════════════════════════════════════════════════════════════
# ReportWriter — all file I/O isolated here
# ══════════════════════════════════════════════════════════════════════════

class ReportWriter:
    """Handles all file-system writes for the explainability stage.

    Each write operation is individually guarded with try/except so that
    one failure never prevents the others from completing.
    """

    def write_explanation_json(
        self,
        explanation: Dict[str, Any],
        output_dir:  str,
    ) -> bool:
        """Write one candidate's explanation dict as an indented JSON file.

        Args:
            explanation: Assembled dict from ExplanationBuilder.build().
            output_dir:  Directory to write the file into.

        Returns:
            True on success, False on any OS or encoding error.
        """
        cid  = explanation["candidate_id"]
        path = os.path.join(output_dir, f"{cid}.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(explanation, fh, indent=2, ensure_ascii=False)
            logger.debug("Wrote explanation JSON → %s", path)
            return True
        except OSError as exc:
            logger.error("Failed to write explanation JSON for %s: %s", cid, exc)
            return False

    def write_summary_report(
        self,
        explanations:   List[Dict[str, Any]],
        output_dir:     str,
        jd_skill_count: int,
    ) -> bool:
        """Write the aggregate plain-text summary report.

        Structure (in order):
            1. Header   — timestamp, total candidates, JD skill count
            2. Table    — Rank | Candidate | Score | Verdict | Matched | Missing
            3. Narrative — full plain-English paragraph for each candidate
            4. Footer   — pipeline stats and scoring methodology

        Args:
            explanations:   List of assembled explanation dicts, sorted by rank.
            output_dir:     Directory to write summary_report.txt into.
            jd_skill_count: Total skills extracted from the JD.

        Returns:
            True on success, False on any OS error.
        """
        path  = os.path.join(output_dir, SUMMARY_FILENAME)
        lines = self._build_report_lines(explanations, jd_skill_count)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            logger.info("Summary report saved → %s", path)
            return True
        except OSError as exc:
            logger.error("Failed to write summary report: %s", exc)
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Private: report construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_report_lines(
        self,
        explanations:   List[Dict[str, Any]],
        jd_skill_count: int,
    ) -> List[str]:
        """Build the complete report as a list of text lines.

        Args:
            explanations:   Sorted list of explanation dicts.
            jd_skill_count: JD skill count for the header display.

        Returns:
            List of string lines (no trailing newline per element).
        """
        W   = REPORT_LINE_WIDTH
        SEP = "═" * W
        DIV = "─" * W
        now = datetime.now().strftime(HUMAN_DATETIME_FMT)

        n         = len(explanations)
        n_strong  = sum(1 for e in explanations if e["verdict"] == "Strong Match")
        n_good    = sum(1 for e in explanations if e["verdict"] == "Good Match")
        n_partial = sum(1 for e in explanations if e["verdict"] == "Partial Match")
        n_weak    = sum(1 for e in explanations if e["verdict"] == "Weak Match")
        n_rec     = n_strong + n_good
        avg       = sum(e["final_score"] for e in explanations) / n if n > 0 else 0.0

        lines: List[str] = []

        # ── Section 1: Header ─────────────────────────────────────────────
        lines += [
            SEP,
            _centre("BIAS-FREE HIRING PIPELINE — EXPLAINABILITY REPORT", W),
            SEP,
            f"  Generated      : {now}",
            f"  Candidates     : {n}",
            f"  JD Skill Count : {jd_skill_count}",
            f"  Recommended    : {n_rec} of {n}  "
            f"(Strong Match: {n_strong}  |  Good Match: {n_good})",
            f"  Average Score  : {_pct(avg)}",
            SEP,
            "",
        ]

        # ── Section 2: Ranked table ───────────────────────────────────────
        lines.append(_centre("RANKED OVERVIEW", W))
        lines.append(DIV)

        # Column widths
        CW = {"rank": 5, "cand": 22, "score": 7, "verdict": 16, "matched": 20}

        header = (
            f"  {'Rank':<{CW['rank']}} "
            f"{'Candidate':<{CW['cand']}} "
            f"{'Score':>{CW['score']}} "
            f"{'Verdict':<{CW['verdict']}} "
            f"{'Skills Matched':<{CW['matched']}} "
            f"Skills Missing"
        )
        rule = (
            f"  {'-'*CW['rank']} "
            f"{'-'*CW['cand']} "
            f"{'-'*CW['score']} "
            f"{'-'*CW['verdict']} "
            f"{'-'*CW['matched']} "
            f"{'-'*18}"
        )
        lines += [header, rule]

        for e in explanations:
            sk = e["explanation"]["skills"]

            matched_preview = sk["matched"][:3]
            matched_str     = ", ".join(matched_preview)
            if len(sk["matched"]) > 3:
                matched_str += f" (+{len(sk['matched']) - 3})"

            missing_preview = sk["missing"][:2]
            missing_str     = ", ".join(missing_preview)
            if len(sk["missing"]) > 2:
                missing_str += f" (+{len(sk['missing']) - 2})"

            lines.append(
                f"  {e['rank']:<{CW['rank']}} "
                f"{e['candidate_id']:<{CW['cand']}} "
                f"{_pct(e['final_score']):>{CW['score']}} "
                f"{e['verdict']:<{CW['verdict']}} "
                f"{matched_str:<{CW['matched']}} "
                f"{missing_str}"
            )

        lines += ["", DIV, ""]

        # ── Section 3: Full plain-English paragraph per candidate ─────────
        lines.append(_centre("CANDIDATE NARRATIVES", W))
        lines.append("")

        for e in explanations:
            lines.append(DIV)
            lines.append(
                f"  #{e['rank']}  {e['candidate_id']}  "
                f"|  {e['verdict']}  "
                f"|  Final Score: {_pct(e['final_score'])}"
            )
            lines.append(DIV)
            lines += self._narrative_paragraph(e, W)
            lines.append("")

        # ── Section 4: Footer ─────────────────────────────────────────────
        lines += [
            SEP,
            _centre("PIPELINE STATISTICS & SCORING METHODOLOGY", W),
            SEP,
            f"  Strong Match  (score ≥ {int(VERDICT_STRONG  * 100)}%) : {n_strong} candidate(s)",
            f"  Good Match    (score ≥ {int(VERDICT_GOOD    * 100)}%) : {n_good} candidate(s)",
            f"  Partial Match (score ≥ {int(VERDICT_PARTIAL * 100)}%) : {n_partial} candidate(s)",
            f"  Weak Match    (score <  {int(VERDICT_PARTIAL * 100)}%) : {n_weak} candidate(s)",
            f"  Average final score              : {_pct(avg)}",
            f"  Recommended for interview        : {n_rec} of {n} "
            f"({_pct(n_rec / n) if n else '0%'})",
            "",
            "  HOW SCORES ARE CALCULATED",
            "  " + "─" * (W - 2),
            "  final_score  =  (0.65 × semantic_score)",
            "               +  (0.35 × keyword_score)",
            "               ×  (0.85 + 0.15 × quality_score)",
            "",
            "  semantic_score  — cosine similarity (SBERT) between the job",
            "                    description and the anonymized CV text.",
            "  keyword_score   — fraction of JD-extracted skills found in the",
            "                    candidate's structured skill list (module0b).",
            "  quality_score   — CV completeness signal from module0b,",
            "                    reflecting section coverage and skill depth.",
            "  quality_weight  — a ±15% multiplier rewarding complete CVs.",
            "",
            "  All candidate identifiers are anonymized by module0. Scores",
            "  are algorithmic signals only and must be reviewed by a qualified",
            "  human decision-maker before any hiring action is taken.",
            SEP,
            _centre("END OF REPORT", W),
            SEP,
            "",
        ]

        return lines

    def _narrative_paragraph(
        self,
        e:     Dict[str, Any],
        width: int,
    ) -> List[str]:
        """Render the full plain-English narrative section for one candidate.

        Produces a flowing paragraph followed by labelled strengths, gaps,
        and a final recommendation — everything a hiring manager needs to
        make a decision without looking at any raw numbers.

        Args:
            e:     The assembled explanation dict for this candidate.
            width: Maximum line width for text wrapping.

        Returns:
            List of text lines for this candidate's narrative block.
        """
        exp    = e["explanation"]
        sk     = exp["skills"]
        bd     = exp["score_breakdown"]
        lines: List[str] = [""]

        # Headline sentence
        lines += _wrap_block(
            f"{e['headline'].capitalize()}.",
            width, indent=2,
        )
        lines.append("")

        # Score snapshot (inline — avoids a table inside a text report)
        sem_lbl  = bd["semantic_score"]["label"]
        kw_lbl   = bd["keyword_score"]["label"]
        qual_lbl = bd["quality_score"]["label"]
        lines += _wrap_block(
            f"Scores: semantic {_pct(bd['semantic_score']['value'])} ({sem_lbl}), "
            f"keyword {_pct(bd['keyword_score']['value'])} ({kw_lbl}), "
            f"quality {_pct(bd['quality_score']['value'])} ({qual_lbl}).",
            width, indent=2,
        )
        lines.append("")

        # Skills paragraph
        lines += _wrap_block(sk["summary"], width, indent=2)
        if sk["matched"]:
            lines += _wrap_block(
                f"Matched skills: {', '.join(sk['matched'])}.",
                width, indent=2,
            )
        if sk["missing"]:
            lines += _wrap_block(
                f"Missing skills: {', '.join(sk['missing'])}.",
                width, indent=2,
            )
        lines.append("")

        # Experience & Education
        lines += _wrap_block(exp["experience"]["summary"], width, indent=2)
        lines += _wrap_block(exp["education"]["summary"],  width, indent=2)
        lines.append("")

        # Strengths
        if exp["strengths"]:
            lines.append("  Strengths:")
            for s in exp["strengths"]:
                lines += _wrap_block(f"  + {s}", width, indent=6)
            lines.append("")

        # Gaps
        if exp["gaps"]:
            lines.append("  Gaps:")
            for g in exp["gaps"]:
                lines += _wrap_block(f"  - {g}", width, indent=6)
            lines.append("")

        # Recommendation
        lines.append("  Recommendation:")
        lines += _wrap_block(exp["recommendation"], width, indent=4)

        return lines


# ══════════════════════════════════════════════════════════════════════════
# Formatting utilities
# ══════════════════════════════════════════════════════════════════════════

def _centre(text: str, width: int) -> str:
    """Centre *text* within *width* columns.

    Args:
        text:  String to centre.
        width: Total target width.

    Returns:
        Centre-padded string.
    """
    return text.center(width)


def _wrap_block(text: str, width: int, indent: int) -> List[str]:
    """Wrap *text* to *width* columns with a uniform left indent.

    Args:
        text:   Prose to wrap.
        width:  Maximum line width including the indent.
        indent: Number of leading spaces on every line.

    Returns:
        List of wrapped line strings; empty list if *text* is blank.
    """
    if not text or not text.strip():
        return []
    prefix  = " " * indent
    wrapped = textwrap.fill(
        text,
        width             = width,
        initial_indent    = prefix,
        subsequent_indent = prefix,
    )
    return wrapped.split("\n")


# ══════════════════════════════════════════════════════════════════════════
# ExplainabilityEngine — top-level orchestrator
# ══════════════════════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """Orchestrates the full explainability stage for all ranked candidates.

    Delegates business logic to ExplanationBuilder and all I/O to
    ReportWriter, keeping this class focused on sequencing and error handling.
    """

    def __init__(self) -> None:
        """Initialise the engine with its builder and writer dependencies."""
        self._builder: ExplanationBuilder = ExplanationBuilder()
        self._writer:  ReportWriter       = ReportWriter()

    def explain(
        self,
        ranking_details: Dict[str, Dict[str, Any]],
        parsed_dir:      str,
        output_dir:      str,
    ) -> bool:
        """Generate all explanations and write output artefacts.

        Processing order:
            1. Sort candidates by rank for deterministic log output.
            2. Load parsed JSON for each candidate (missing JSON → graceful fallback).
            3. Build explanation dict via ExplanationBuilder.
            4. Write per-candidate JSON file via ReportWriter.
            5. Write aggregate summary_report.txt.

        Args:
            ranking_details: ranker.last_ranking_details from module1.
            parsed_dir:      Path to module0b's parsed/ directory.
            output_dir:      Destination for all output files.

        Returns:
            True if at least one explanation JSON was written successfully.
        """
        if not ranking_details:
            logger.warning("ranking_details is empty — nothing to explain.")
            return False

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "Cannot create output directory '%s': %s", output_dir, exc
            )
            return False

        n_total        = len(ranking_details)
        jd_skill_count = self._resolve_jd_skill_count(ranking_details)

        explanations: List[Dict[str, Any]] = []
        n_written:    int                  = 0

        # Sort by rank so log lines appear in rank order
        sorted_entries = sorted(
            ranking_details.items(),
            key=lambda kv: kv[1].get("rank", 9999),
        )

        for candidate_id, raw_entry in sorted_entries:
            logger.info(
                "Building explanation for %s (rank %s of %d) …",
                candidate_id,
                raw_entry.get("rank", "?"),
                n_total,
            )

            parsed_data = _load_parsed_json(parsed_dir, candidate_id)

            try:
                explanation = self._builder.build(
                    candidate_id  = candidate_id,
                    ranking_entry = raw_entry,
                    parsed_data   = parsed_data,
                    n_candidates  = n_total,
                )
            except Exception as exc:
                logger.error(
                    "ExplanationBuilder failed for %s: %s — skipping.",
                    candidate_id, exc,
                )
                continue

            if self._writer.write_explanation_json(explanation, output_dir):
                n_written += 1

            # Always accumulate for the summary, even if the JSON write failed
            explanations.append(explanation)

        if not explanations:
            logger.error("No explanations could be generated.")
            return False

        # Ensure rank order before writing the summary report
        explanations.sort(key=lambda e: e.get("rank", 9999))

        self._writer.write_summary_report(
            explanations    = explanations,
            output_dir      = output_dir,
            jd_skill_count  = jd_skill_count,
        )

        logger.info(
            "Explainability stage complete — %d/%d JSONs written.",
            n_written, n_total,
        )
        return n_written > 0

    @staticmethod
    def _resolve_jd_skill_count(
        ranking_details: Dict[str, Dict[str, Any]],
    ) -> int:
        """Extract the JD skill count from the first available ranking entry.

        All candidates share the same JD, so any entry's value is representative.

        Args:
            ranking_details: Full ranking_details dict.

        Returns:
            The JD skill count, or 0 if it cannot be determined.
        """
        for data in ranking_details.values():
            count = data.get("jd_skill_count", 0)
            if count:
                return int(count)
        return 0


# ══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ══════════════════════════════════════════════════════════════════════════

_engine: Optional[ExplainabilityEngine] = None


def _get_engine() -> ExplainabilityEngine:
    """Return the cached module-level ExplainabilityEngine instance.

    Returns:
        The shared engine, instantiated on first call.
    """
    global _engine
    if _engine is None:
        _engine = ExplainabilityEngine()
    return _engine


# ══════════════════════════════════════════════════════════════════════════
# run() — primary entry point called by main.py
# ══════════════════════════════════════════════════════════════════════════

def run(
    ranking_details: Dict[str, Dict[str, Any]],
    parsed_dir:      str,
    output_dir:      str,
) -> bool:
    """Execute the full explainability stage.

    This function is the canonical programmatic entry point, matching the
    run() convention established by module0, module0b, and module1.

    Side effects:
        Creates *output_dir* if it does not exist.
        Writes one JSON file per candidate to *output_dir*.
        Writes summary_report.txt to *output_dir*.

    Args:
        ranking_details: The last_ranking_details dict from
                         module1.ScreenRanker.rank_candidates().
        parsed_dir:      Path to the parsed/ directory from module0b.
        output_dir:      Destination directory for all output files.

    Returns:
        True if at least one explanation JSON was written successfully.
        False if ranking_details is empty or every candidate failed.
    """
    import time

    print()
    print("=" * 60)
    print("  MODULE 2 — EXPLAINABILITY ENGINE")
    print("=" * 60)

    if not ranking_details:
        print("\n  [ERROR] ranking_details is empty.")
        print("          Ensure module1 has completed ranking before module2.")
        return False

    n = len(ranking_details)
    print(f"\n  Candidates  : {n}")
    print(f"  Parsed JSON : {parsed_dir}")
    print(f"  Output      : {output_dir}")
    print()

    t0 = time.time()
    ok = _get_engine().explain(
        ranking_details = ranking_details,
        parsed_dir      = parsed_dir,
        output_dir      = output_dir,
    )
    elapsed = time.time() - t0

    print()
    if ok:
        print(f"  [OK] Explanations generated in {elapsed:.1f}s")
        print(f"  JSON   → {output_dir}/Candidate_XX.json")
        print(f"  Report → {os.path.join(output_dir, SUMMARY_FILENAME)}")
    else:
        print("  [FAILED] No explanations could be generated.")
        print("           Check logs for details.")

    print("=" * 60)
    return ok


# ══════════════════════════════════════════════════════════════════════════
# Standalone CLI — mirrors module0 / module0b / module1 conventions exactly
# ══════════════════════════════════════════════════════════════════════════

def _configure_logging(verbose: bool = False) -> None:
    """Configure the root logger for standalone execution.

    Args:
        verbose: If True, sets level to DEBUG; otherwise INFO.
    """
    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = _LOG_FORMAT,
        datefmt = _DATE_FMT,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for standalone execution.

    Returns:
        Configured ArgumentParser with all flags and help text.
    """
    _base = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        prog        = "module2",
        description = (
            "Explainability Engine — Bias-Free Hiring Pipeline\n\n"
            "Reads ranking_details (exported from module1) and generates\n"
            "per-candidate JSON explanation files plus a plain-text summary."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "Examples:\n"
            "  python module2.py --ranking-file ranking_details.json\n\n"
            "  python module2.py --ranking-file ranking_details.json \\\n"
            "                    --parsed-dir parsed \\\n"
            "                    --output-dir explanations \\\n"
            "                    --verbose\n\n"
            "  # Export ranking_details from a live module1 run:\n"
            "  python -c \"\n"
            "    import json, module1\n"
            "    r = module1.load_ranker()\n"
            "    json.dump(r.last_ranking_details,\n"
            "              open('ranking_details.json', 'w'), indent=2)\n"
            "  \"\n"
        ),
    )
    p.add_argument(
        "--ranking-file",
        required = True,
        metavar  = "FILE",
        help     = (
            "Path to a JSON file containing last_ranking_details "
            "(exported from module1.ScreenRanker.last_ranking_details)."
        ),
    )
    p.add_argument(
        "--parsed-dir",
        default = os.path.join(_base, "parsed"),
        metavar = "DIR",
        help    = "Directory of module0b parsed JSON files (default: parsed/).",
    )
    p.add_argument(
        "--output-dir",
        default = os.path.join(_base, OUTPUT_DIR_DEFAULT),
        metavar = "DIR",
        help    = f"Destination for all output files (default: {OUTPUT_DIR_DEFAULT}/).",
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

    try:
        with open(_args.ranking_file, "r", encoding="utf-8") as _fh:
            _ranking_details: Dict[str, Dict[str, Any]] = json.load(_fh)
    except FileNotFoundError:
        logger.critical("Ranking file not found: %s", _args.ranking_file)
        sys.exit(1)
    except json.JSONDecodeError as _exc:
        logger.critical("Ranking file is not valid JSON: %s", _exc)
        sys.exit(1)
    except OSError as _exc:
        logger.critical("Cannot read ranking file: %s", _exc)
        sys.exit(1)

    _ok = run(
        ranking_details = _ranking_details,
        parsed_dir      = _args.parsed_dir,
        output_dir      = _args.output_dir,
    )
    sys.exit(0 if _ok else 1)