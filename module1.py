# -*- coding: utf-8 -*-
"""
module1.py — Hybrid Candidate Screening & Ranking Engine
=========================================================
Bias-Free Hiring Pipeline — Stage 1

Ranks anonymized CVs against a Job Description (JD) using a two-component
hybrid score that combines semantic similarity with structured skill matching.

Scoring formula
---------------
    semantic_score : cosine similarity between SBERT embeddings of JD and CV
    keyword_score  : fraction of JD-required skills present in parsed JSON
    quality_score  : CV quality signal from module0b (completeness, formatting)

    hybrid_score   = (0.65 × semantic_score) + (0.35 × keyword_score)
    final_score    = hybrid_score × (0.85 + 0.15 × quality_score)

    Quality acts as a subtle multiplier (±15 %) so a well-structured CV with
    the same skill match will rank slightly higher — reflecting real-world
    signal about the candidate's attention to detail.

Pipeline position
-----------------
    module0  →  anonymized_cvs/Candidate_01.txt
    module0b →  parsed/Candidate_01.json
    module1  →  ranking_report.txt            ← this module

Standalone usage
----------------
    python module1.py --jd-file jd.txt
    python module1.py --jd-file jd.txt --anonymized-dir anonymized_cvs \\
                      --parsed-dir parsed --verbose

Programmatic usage (called by main.py)
---------------------------------------
    import module1
    ranker   = module1.load_ranker()
    rankings = ranker.rank_candidates(
        anonymized_cv_dir='anonymized_cvs',
        job_description=jd_text,
    )
    # returns: [('Candidate_01', 0.87), ('Candidate_03', 0.72), ...]
    # details: ranker.last_ranking_details  (consumed by module2)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Optional heavyweight imports ───────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
# Module-level logger  (NOT a constant → lowercase name)
# ══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL_NAME:    str   = "all-MiniLM-L6-v2"
SEMANTIC_WEIGHT:       float = 0.65
KEYWORD_WEIGHT:        float = 0.35
QUALITY_BASE:          float = 0.85      # weight when quality_score = 0.0
QUALITY_BONUS:         float = 0.15      # additional weight when quality_score = 1.0
DEFAULT_QUALITY_SCORE: float = 0.5       # used when parsed JSON is absent
FALLBACK_SEMANTIC:     float = 0.5       # used when SBERT is unavailable
ENCODE_BATCH_SIZE:     int   = 32
RANKING_REPORT_NAME:   str   = "ranking_report.txt"
_ISO_FMT:              str   = "%Y-%m-%dT%H:%M:%S"
_LOG_FORMAT:           str   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT:             str   = "%H:%M:%S"

# ══════════════════════════════════════════════════════════════════════════
# SKILLS TAXONOMY — flat list used for JD keyword extraction
# Mirrors the technical + soft + tools categories from module0b.
# Kept as a flat list here for fast O(n) scan against JD text.
# ══════════════════════════════════════════════════════════════════════════
SKILLS_TAXONOMY_FLAT: List[str] = [
    # ── Programming languages ──────────────────────────────────────────
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
    "Go", "Golang", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
    "R", "MATLAB", "Perl", "Shell", "Bash", "PowerShell", "Groovy",
    "Objective-C", "Dart", "Elixir", "Haskell", "Julia", "Lua",
    "Visual Basic", "VBA", "COBOL", "Fortran", "Assembly", "Solidity",
    "PL/SQL", "T-SQL", "GraphQL", "KQL", "ABAP", "Erlang", "OCaml",
    "F#", "Clojure", "Elm", "Nim", "Crystal", "Zig", "Racket",
    # ── Web / Frontend ─────────────────────────────────────────────────
    "HTML", "CSS", "React", "Angular", "Vue", "Svelte", "Next.js",
    "Nuxt.js", "Gatsby", "Redux", "MobX", "RxJS", "jQuery",
    "Bootstrap", "Tailwind CSS", "Material UI", "Chakra UI",
    "Three.js", "D3.js", "WebAssembly", "PWA",
    # ── Backend / API frameworks ───────────────────────────────────────
    "Node.js", "Express", "NestJS", "FastAPI", "Flask", "Django",
    "Spring", "Spring Boot", "Rails", "Laravel", "Symfony",
    "ASP.NET", "ASP.NET Core", "Gin", "Echo", "Fiber",
    "Actix", "Rocket", "Phoenix", "Ktor",
    # ── Data science / ML ──────────────────────────────────────────────
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost",
    "LightGBM", "CatBoost", "spaCy", "NLTK", "Gensim",
    "HuggingFace", "Transformers", "OpenCV", "Pillow",
    "Pandas", "NumPy", "SciPy", "Statsmodels", "Plotly",
    "Matplotlib", "Seaborn", "Bokeh", "Altair", "Dash", "Streamlit",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Reinforcement Learning", "Data Science", "Feature Engineering",
    "Model Deployment", "A/B Testing", "Statistical Analysis",
    # ── Relational databases ───────────────────────────────────────────
    "PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle DB",
    "SQL Server", "DB2", "SQL",
    # ── NoSQL / cache ──────────────────────────────────────────────────
    "MongoDB", "Cassandra", "CouchDB", "DynamoDB", "Firestore",
    "Cosmos DB", "Redis", "Memcached", "Neo4j", "InfluxDB",
    "TimescaleDB", "ClickHouse", "Elasticsearch", "Solr",
    # ── Cloud platforms ────────────────────────────────────────────────
    "AWS", "Azure", "GCP", "Google Cloud", "IBM Cloud",
    "Heroku", "DigitalOcean", "Vercel", "Netlify", "Firebase",
    "Cloudflare", "Linode",
    # ── Cloud services ─────────────────────────────────────────────────
    "S3", "EC2", "Lambda", "EKS", "ECS", "RDS", "CloudFormation",
    "Azure DevOps", "Azure Functions", "Cloud Run", "BigQuery",
    "Redshift", "Snowflake", "Databricks",
    # ── DevOps / Infrastructure ────────────────────────────────────────
    "Docker", "Kubernetes", "Terraform", "Ansible", "Puppet",
    "Chef", "Vagrant", "Helm", "Istio", "Linkerd",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI",
    "Travis CI", "ArgoCD", "Flux", "Tekton", "CI/CD", "Linux",
    # ── Data engineering ───────────────────────────────────────────────
    "Apache Spark", "Spark", "Hadoop", "Kafka", "RabbitMQ", "ActiveMQ",
    "Apache Airflow", "Airflow", "Luigi", "Prefect", "Dagster", "dbt",
    "Flink", "Storm", "NiFi", "Hive", "Pig",
    # ── MLOps ──────────────────────────────────────────────────────────
    "MLflow", "Kubeflow", "BentoML", "Seldon", "DVC",
    "Weights & Biases", "Neptune", "Comet ML",
    # ── Version control ────────────────────────────────────────────────
    "Git", "GitHub", "GitLab", "Bitbucket", "SVN",
    # ── Testing ────────────────────────────────────────────────────────
    "pytest", "JUnit", "Selenium", "Cypress", "Playwright",
    "Jest", "Mocha", "Chai", "TestNG", "Robot Framework",
    "Locust", "k6", "Gatling",
    # ── Security ───────────────────────────────────────────────────────
    "OAuth", "JWT", "SAML", "OpenID Connect", "Vault",
    "OWASP", "Penetration Testing", "SIEM", "IAM",
    # ── Protocols / standards ──────────────────────────────────────────
    "REST", "gRPC", "WebSockets", "SOAP", "MQTT", "AMQP",
    "OpenAPI", "Swagger",
    # ── Mobile ─────────────────────────────────────────────────────────
    "Android", "iOS", "React Native", "Flutter", "Xamarin",
    # ── Embedded / hardware ────────────────────────────────────────────
    "Arduino", "Raspberry Pi", "VHDL", "Verilog", "LabVIEW",
    # ── Game / 3D ──────────────────────────────────────────────────────
    "Unity", "Unreal Engine", "Godot", "OpenGL", "Vulkan",
    # ── IDE / Notebooks ────────────────────────────────────────────────
    "Jupyter", "Google Colab", "VS Code", "IntelliJ IDEA",
    "PyCharm", "Eclipse", "Xcode", "Android Studio",
    # ── Soft skills ────────────────────────────────────────────────────
    "Leadership", "Communication", "Teamwork", "Collaboration",
    "Problem Solving", "Critical Thinking", "Creativity",
    "Adaptability", "Time Management", "Organization",
    "Conflict Resolution", "Negotiation", "Empathy",
    "Decision Making", "Initiative", "Mentoring",
    "Presentation", "Public Speaking", "Coaching",
    "Stakeholder Management", "Project Management",
    "Agile", "Scrum", "Kanban", "Risk Management", "Change Management",
    # ── Tools — project / design / BI / monitoring ─────────────────────
    "Jira", "Confluence", "Trello", "Asana", "Notion", "ClickUp",
    "Figma", "Sketch", "Adobe XD", "InVision", "Miro", "Lucidchart",
    "Tableau", "Power BI", "Qlik", "Looker", "Metabase",
    "Grafana", "Kibana", "Datadog", "Prometheus", "Splunk",
    "New Relic", "Dynatrace", "ELK Stack",
    "Postman", "Insomnia", "SoapUI",
    "SonarQube", "ESLint", "Pylint", "Black", "Prettier",
    "Snyk", "Checkmarx", "Veracode",
    "Salesforce", "HubSpot", "SAP", "ServiceNow", "Zendesk",
    "Slack", "Microsoft Teams", "Zoom", "Discord",
    "Alteryx", "Talend", "Informatica", "Fivetran", "Stitch",
]

# Deduplicate while preserving order
_seen: set = set()
_deduped: List[str] = []
for _s in SKILLS_TAXONOMY_FLAT:
    _key = _s.lower()
    if _key not in _seen:
        _seen.add(_key)
        _deduped.append(_s)
SKILLS_TAXONOMY_FLAT = _deduped
del _seen, _deduped, _s, _key


# ══════════════════════════════════════════════════════════════════════════
# Type aliases
# ══════════════════════════════════════════════════════════════════════════
CandidateId    = str
Score          = float
RankingList    = List[Tuple[CandidateId, Score]]
RankingDetails = Dict[CandidateId, Dict[str, Any]]


# ══════════════════════════════════════════════════════════════════════════
# Helper: JD skill extraction
# ══════════════════════════════════════════════════════════════════════════

def extract_jd_skills(jd_text: str) -> List[str]:
    """Extract skills mentioned in a job description using SKILLS_TAXONOMY_FLAT.

    Performs case-insensitive whole-word matching so that e.g. "Go" does not
    match "Google".  Multi-word skills (e.g. "Machine Learning") are matched
    as a contiguous phrase.

    Args:
        jd_text: Raw text of the job description.

    Returns:
        Deduplicated, sorted list of skill names found in the JD.
    """
    found: List[str] = []
    jd_lower = jd_text.lower()

    for skill in SKILLS_TAXONOMY_FLAT:
        pattern = r"(?<![a-z0-9\-])" + re.escape(skill.lower()) + r"(?![a-z0-9\-])"
        if re.search(pattern, jd_lower):
            found.append(skill)

    return sorted(set(found))


# ══════════════════════════════════════════════════════════════════════════
# Helper: parsed JSON loading
# ══════════════════════════════════════════════════════════════════════════

def _load_parsed_json(parsed_dir: str, candidate_id: str) -> Optional[Dict[str, Any]]:
    """Load the parsed JSON produced by module0b for a single candidate.

    Args:
        parsed_dir:   Path to the directory containing parsed JSON files.
        candidate_id: Candidate identifier (e.g. 'Candidate_01').

    Returns:
        The parsed dict, or None if the file is missing or invalid.
    """
    path = os.path.join(parsed_dir, f"{candidate_id}.json")
    if not os.path.isfile(path):
        logger.warning("Parsed JSON not found for %s at %s", candidate_id, path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read parsed JSON for %s: %s", candidate_id, exc)
        return None


def _extract_candidate_skills(parsed: Dict[str, Any]) -> List[str]:
    """Collect all skills from a parsed JSON's sections.skills block.

    module0b stores skills as::

        {"sections": {"skills": {"technical": [...], "soft": [...], "tools": [...]}}}

    Args:
        parsed: Full parsed JSON dict.

    Returns:
        Flat, lowercased list of all skills for this candidate.
    """
    skills_block = parsed.get("sections", {}).get("skills", {})
    all_skills: List[str] = []
    if isinstance(skills_block, dict):
        for category_skills in skills_block.values():
            if isinstance(category_skills, list):
                all_skills.extend(str(s) for s in category_skills)
    elif isinstance(skills_block, list):
        all_skills.extend(str(s) for s in skills_block)
    return [s.lower() for s in all_skills]


# ══════════════════════════════════════════════════════════════════════════
# Core scoring helpers
# ══════════════════════════════════════════════════════════════════════════

def _compute_keyword_score(
    candidate_skills_lower: List[str],
    jd_skills: List[str],
) -> Tuple[float, List[str], List[str]]:
    """Compute keyword overlap score between candidate skills and JD skills.

    Args:
        candidate_skills_lower: Lowercased skill strings from the candidate's
                                parsed JSON.
        jd_skills:              Skills extracted from the JD (original case).

    Returns:
        Tuple of (score 0–1, matched_skills, missing_skills).
    """
    if not jd_skills:
        return 0.0, [], []

    matched: List[str] = []
    missing: List[str] = []

    for skill in jd_skills:
        if skill.lower() in candidate_skills_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    score = len(matched) / len(jd_skills)
    return min(1.0, score), matched, missing


def _compute_final_score(
    semantic_score: float,
    keyword_score: float,
    quality_score: float,
) -> float:
    """Combine semantic, keyword, and quality signals into a final score.

    Formula::

        hybrid      = (SEMANTIC_WEIGHT × semantic) + (KEYWORD_WEIGHT × keyword)
        final       = hybrid × (QUALITY_BASE + QUALITY_BONUS × quality)

    Args:
        semantic_score: Cosine similarity from SBERT (0–1).
        keyword_score:  Skill-overlap fraction (0–1).
        quality_score:  CV quality score from module0b (0–1).

    Returns:
        Final score, clamped to [0.0, 1.0].
    """
    hybrid = (SEMANTIC_WEIGHT * semantic_score) + (KEYWORD_WEIGHT * keyword_score)
    quality_multiplier = QUALITY_BASE + QUALITY_BONUS * quality_score
    final = hybrid * quality_multiplier
    return min(1.0, max(0.0, final))


# ══════════════════════════════════════════════════════════════════════════
# ScreenRanker class
# ══════════════════════════════════════════════════════════════════════════

class ScreenRanker:
    """Hybrid SBERT + keyword ranker for anonymized CVs.

    Attributes:
        model_name (str): Sentence-Transformers model identifier.
        model:            Loaded SentenceTransformer instance, or None if
                          the package is unavailable.
        last_ranking_details (RankingDetails): Populated after each call to
            rank_candidates(); consumed by module2 (explainability engine).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialise the SBERT model.

        Gracefully degrades to keyword-only mode when sentence-transformers
        is not installed.

        Args:
            model_name: HuggingFace / Sentence-Transformers model identifier.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None  # type: ignore[type-arg]
        self.last_ranking_details: RankingDetails = {}

        if not _SBERT_AVAILABLE:
            logger.warning(
                "sentence-transformers is not installed. "
                "Falling back to keyword-only ranking (semantic_score = %.1f for all). "
                "Install with: pip install sentence-transformers",
                FALLBACK_SEMANTIC,
            )
            return

        logger.info("Loading SBERT model: %s …", model_name)
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("SBERT model loaded successfully.")
        except Exception as exc:  # broad catch — model load can fail in many ways
            logger.error(
                "Failed to load SBERT model '%s': %s. "
                "Falling back to keyword-only ranking.",
                model_name, exc,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def rank_candidates(
        self,
        anonymized_cv_dir: str,
        job_description: str,
        parsed_dir: Optional[str] = None,
    ) -> RankingList:
        """Rank all text CVs in a directory against the supplied JD.

        Reads ``*.txt`` files from *anonymized_cv_dir* and (optionally)
        the corresponding ``*.json`` files from *parsed_dir*.  If
        *parsed_dir* is None, it is inferred as a sibling ``parsed/``
        directory next to *anonymized_cv_dir*.

        Side effect: populates ``self.last_ranking_details``.

        Args:
            anonymized_cv_dir: Path to folder of anonymized ``.txt`` CV files.
            job_description:   Full text of the job description.
            parsed_dir:        Path to the ``parsed/`` directory produced by
                               module0b.  Inferred automatically if None.

        Returns:
            List of ``(candidate_id, final_score)`` tuples sorted by
            descending score.  Returns ``[]`` on total failure.
        """
        if not job_description.strip():
            logger.warning("Job description is empty — nothing to rank against.")
            return []

        # ── Resolve parsed_dir ───────────────────────────────────────────
        if parsed_dir is None:
            parent = Path(anonymized_cv_dir).resolve().parent
            parsed_dir = str(parent / "parsed")
            logger.debug("parsed_dir not supplied; using inferred path: %s", parsed_dir)

        # ── Collect CV files ─────────────────────────────────────────────
        _EXCLUDED_TXT = {"error_log.txt", "index.txt"}
        cv_files = sorted(
            f for f in glob.glob(os.path.join(anonymized_cv_dir, "*.txt"))
            if os.path.basename(f) not in _EXCLUDED_TXT
        )
        if not cv_files:
            logger.warning("No .txt CVs found in %s", anonymized_cv_dir)
            return []

        logger.info("Found %d CVs in %s", len(cv_files), anonymized_cv_dir)

        # ── Read CV text ─────────────────────────────────────────────────
        cv_texts:       List[str]           = []
        candidate_ids:  List[CandidateId]   = []
        parse_cache:    Dict[str, Optional[Dict[str, Any]]] = {}

        for filepath in cv_files:
            cid = os.path.splitext(os.path.basename(filepath))[0]
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as exc:
                logger.warning("Skipping %s — could not read: %s", filepath, exc)
                continue
            cv_texts.append(text)
            candidate_ids.append(cid)
            parse_cache[cid] = _load_parsed_json(parsed_dir, cid)

        if not cv_texts:
            logger.error("All CV files failed to load. Returning empty ranking.")
            return []

        # ── Extract JD skills ─────────────────────────────────────────────
        jd_skills = extract_jd_skills(job_description)
        logger.info(
            "Extracted %d skills from JD: %s%s",
            len(jd_skills),
            ", ".join(jd_skills[:8]),
            " …" if len(jd_skills) > 8 else "",
        )

        # ── Semantic encoding ─────────────────────────────────────────────
        semantic_scores = self._encode_and_score(job_description, cv_texts)

        # ── Build per-candidate scores ────────────────────────────────────
        details: RankingDetails = {}
        results: RankingList    = []

        for i, cid in enumerate(candidate_ids):
            semantic_score = semantic_scores[i]
            parsed         = parse_cache.get(cid)

            # ── Keyword score ─────────────────────────────────────────────
            if parsed is not None:
                cand_skills_lower = _extract_candidate_skills(parsed)
                kw_score, matched, missing = _compute_keyword_score(
                    cand_skills_lower, jd_skills
                )
                quality_score = float(parsed.get("quality_score", DEFAULT_QUALITY_SCORE))
            else:
                kw_score, matched, missing = 0.0, [], list(jd_skills)
                quality_score = DEFAULT_QUALITY_SCORE
                logger.warning(
                    "%s: parsed JSON missing — keyword_score=0.0, quality_score=%.2f (default)",
                    cid, DEFAULT_QUALITY_SCORE,
                )

            # ── Final score ───────────────────────────────────────────────
            final_score = _compute_final_score(semantic_score, kw_score, quality_score)

            details[cid] = {
                "final_score":    round(final_score, 6),
                "semantic_score": round(semantic_score, 6),
                "keyword_score":  round(kw_score, 6),
                "quality_score":  round(quality_score, 6),
                "matched_skills": matched,
                "missing_skills": missing,
                "jd_skill_count": len(jd_skills),
                "rank":           -1,  # filled in after sorting
            }
            results.append((cid, final_score))

        # ── Sort and assign ranks ─────────────────────────────────────────
        results.sort(key=lambda x: x[1], reverse=True)
        for rank_idx, (cid, _) in enumerate(results, start=1):
            details[cid]["rank"] = rank_idx

        self.last_ranking_details = details

        if results:
            top_id, top_score = results[0]
            logger.info(
                "Ranking complete — %d candidates. Top: %s (%.4f)",
                len(results), top_id, top_score,
            )

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _encode_and_score(
        self,
        job_description: str,
        cv_texts: List[str],
    ) -> List[float]:
        """Encode the JD and all CVs, returning cosine similarity scores.

        Falls back to ``FALLBACK_SEMANTIC`` for every candidate when SBERT
        is unavailable or if encoding fails.

        Args:
            job_description: JD text to encode.
            cv_texts:        List of anonymized CV texts.

        Returns:
            List of float similarity scores (same length as *cv_texts*).
        """
        fallback = [FALLBACK_SEMANTIC] * len(cv_texts)

        if not self.model:
            logger.warning(
                "SBERT model unavailable — assigning semantic_score=%.2f to all candidates.",
                FALLBACK_SEMANTIC,
            )
            return fallback

        try:
            logger.info("Encoding JD …")
            jd_embedding = self.model.encode(job_description, convert_to_tensor=True)

            logger.info("Encoding %d CVs (batch_size=%d) …", len(cv_texts), ENCODE_BATCH_SIZE)
            iterator = (
                _tqdm(
                    range(0, len(cv_texts), ENCODE_BATCH_SIZE),
                    desc="Encoding CVs",
                    unit="batch",
                )
                if _TQDM_AVAILABLE
                else range(0, len(cv_texts), ENCODE_BATCH_SIZE)
            )

            all_embeddings = []
            for start in iterator:
                batch = cv_texts[start : start + ENCODE_BATCH_SIZE]
                batch_embs = self.model.encode(batch, convert_to_tensor=True)
                # batch_embs may be a 1-D tensor if batch has a single item
                if len(batch) == 1:
                    batch_embs = batch_embs.unsqueeze(0)  # type: ignore[union-attr]
                all_embeddings.append(batch_embs)

            # Stack all batches along dimension 0
            import torch
            cv_embeddings = torch.cat(all_embeddings, dim=0)

            cosine_scores = st_util.cos_sim(jd_embedding, cv_embeddings)[0]
            return [float(s) for s in cosine_scores]

        except Exception as exc:
            logger.error(
                "SBERT encoding failed: %s. Falling back to semantic_score=%.2f.",
                exc, FALLBACK_SEMANTIC,
            )
            return fallback


# ══════════════════════════════════════════════════════════════════════════
# Singleton helper
# ══════════════════════════════════════════════════════════════════════════

_ranker_instance: Optional[ScreenRanker] = None


def load_ranker(model_name: str = DEFAULT_MODEL_NAME) -> ScreenRanker:
    """Return a module-level cached ScreenRanker instance.

    Creating the ranker is expensive (model download + load ~3–10 s the first
    time).  Subsequent calls return the same object.

    Args:
        model_name: Sentence-Transformers model to load on first call.
                    Ignored on subsequent calls.

    Returns:
        The shared ScreenRanker instance.
    """
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = ScreenRanker(model_name=model_name)
    return _ranker_instance


# ══════════════════════════════════════════════════════════════════════════
# Ranking report writer
# ══════════════════════════════════════════════════════════════════════════

def _write_ranking_report(
    rankings: RankingList,
    details: RankingDetails,
    report_path: str,
    jd_skill_count: int,
) -> None:
    """Persist a plain-text ranking report to *report_path*.

    Args:
        rankings:       Sorted list of (candidate_id, final_score) tuples.
        details:        Per-candidate breakdown dict from last_ranking_details.
        report_path:    Destination file path.
        jd_skill_count: Number of skills extracted from the JD.
    """
    lines: List[str] = [
        "=" * 72,
        "  BIAS-FREE HIRING PIPELINE — CANDIDATE RANKING REPORT",
        f"  Generated : {datetime.now().strftime(_ISO_FMT)}",
        f"  Candidates: {len(rankings)}",
        f"  JD Skills : {jd_skill_count}",
        f"  Weights   : semantic={SEMANTIC_WEIGHT:.0%}  "
        f"keyword={KEYWORD_WEIGHT:.0%}  quality_bonus={QUALITY_BONUS:.0%}",
        "=" * 72,
        "",
        f"  {'Rank':<6} {'Candidate':<22} {'Final':>7}  "
        f"{'Semantic':>8}  {'Keyword':>7}  {'Quality':>7}  "
        f"{'Matched':>7}/{str(jd_skill_count):<5}",
        f"  {'-'*6} {'-'*22} {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*13}",
    ]

    for rank, (cid, final_score) in enumerate(rankings, start=1):
        d = details.get(cid, {})
        sem   = d.get("semantic_score", 0.0)
        kw    = d.get("keyword_score",  0.0)
        qual  = d.get("quality_score",  DEFAULT_QUALITY_SCORE)
        match = len(d.get("matched_skills", []))
        bar   = "█" * int(final_score * 20)
        lines.append(
            f"  {rank:<6} {cid:<22} {final_score:>6.1%}  "
            f"{sem:>7.1%}  {kw:>6.1%}  {qual:>6.1%}  "
            f"{match:>7}       {bar}"
        )

    lines += ["", "─" * 72, "", "  SKILL BREAKDOWN PER CANDIDATE", "─" * 72]

    for rank, (cid, final_score) in enumerate(rankings, start=1):
        d       = details.get(cid, {})
        matched = d.get("matched_skills", [])
        missing = d.get("missing_skills", [])
        lines.append(f"\n  #{rank}  {cid}  (final={final_score:.4f})")
        lines.append(f"    Matched : {', '.join(matched) if matched else '—'}")
        lines.append(f"    Missing : {', '.join(missing) if missing else '—'}")

    lines += ["", "=" * 72, "  END OF REPORT", "=" * 72, ""]

    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("Ranking report saved → %s", report_path)
    except OSError as exc:
        logger.error("Could not write ranking report to %s: %s", report_path, exc)


# ══════════════════════════════════════════════════════════════════════════
# run() — called by main.py for standalone pipeline mode
# ══════════════════════════════════════════════════════════════════════════

def run(
    anonymized_cv_dir: str,
    parsed_dir: str,
    jd_file: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> bool:
    """Execute the full ranking stage and save ranking_report.txt.

    This is the primary entry point called from standalone execution or
    from orchestration scripts that prefer a simple ``run()`` interface.
    (main.py currently calls load_ranker() / rank_candidates() directly
    from its own ``_stage1_ranking`` helper; this function is the self-
    contained equivalent.)

    Args:
        anonymized_cv_dir: Directory of anonymized ``.txt`` CV files.
        parsed_dir:        Directory of parsed ``.json`` files from module0b.
        jd_file:           Path to a plain-text Job Description file.
        model_name:        SBERT model identifier.

    Returns:
        True on success (at least one candidate ranked), False otherwise.
    """
    print()
    print("=" * 60)
    print("  MODULE 1 — HYBRID CANDIDATE SCREENING & RANKING")
    print("=" * 60)

    # ── Validate inputs ───────────────────────────────────────────────────
    if not os.path.isdir(anonymized_cv_dir):
        print(f"\n  [ERROR] anonymized_cv_dir not found: {anonymized_cv_dir}")
        print("          Run module0 first.")
        return False

    if not os.path.isfile(jd_file):
        print(f"\n  [ERROR] JD file not found: {jd_file}")
        return False

    try:
        with open(jd_file, "r", encoding="utf-8") as fh:
            jd_text = fh.read()
    except OSError as exc:
        print(f"\n  [ERROR] Could not read JD file: {exc}")
        return False

    if not jd_text.strip():
        print("\n  [ERROR] JD file is empty.")
        return False

    print(f"\n  Anonymized CVs : {anonymized_cv_dir}")
    print(f"  Parsed JSON    : {parsed_dir}")
    print(f"  Job Description: {jd_file} ({len(jd_text):,} chars)")
    print()

    # ── Rank ──────────────────────────────────────────────────────────────
    t0      = time.time()
    ranker  = load_ranker(model_name=model_name)
    results = ranker.rank_candidates(
        anonymized_cv_dir=anonymized_cv_dir,
        job_description=jd_text,
        parsed_dir=parsed_dir,
    )
    elapsed = time.time() - t0

    if not results:
        print("\n  [FAILED] No candidates ranked. "
              "Check anonymized_cvs/ and the JD file.")
        return False

    # ── Print table ───────────────────────────────────────────────────────
    details = ranker.last_ranking_details
    jd_skill_count = (
        details[results[0][0]]["jd_skill_count"] if details else 0
    )

    print(f"\n  {'Rank':<6} {'Candidate':<22} {'Score':>7}  {'Bar'}")
    print(f"  {'-'*6} {'-'*22} {'-'*7}  {'-'*22}")

    for rank, (cid, score) in enumerate(results, start=1):
        bar = "█" * int(score * 20)
        print(f"  {rank:<6} {cid:<22} {score:>6.1%}  {bar}")

    # ── Save report ───────────────────────────────────────────────────────
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(jd_file)) if os.path.dirname(jd_file) else ".",
        RANKING_REPORT_NAME,
    )
    _write_ranking_report(results, details, report_path, jd_skill_count)

    print(f"\n  {len(results)} candidates ranked in {elapsed:.1f}s")
    print(f"  Top candidate : {results[0][0]} ({results[0][1]:.1%})")
    print(f"  Report saved  → {report_path}")
    print("=" * 60)

    return True


# ══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone execution."""
    p = argparse.ArgumentParser(
        prog="module1",
        description="Hybrid Candidate Screening & Ranking — Bias-Free Hiring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python module1.py --jd-file jd.txt\n"
            "  python module1.py --jd-file jd.txt --anonymized-dir anonymized_cvs "
            "--parsed-dir parsed --verbose\n"
        ),
    )
    _base = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--anonymized-dir",
        default=os.path.join(_base, "anonymized_cvs"),
        help="Directory of anonymized .txt CV files (default: anonymized_cvs/)",
    )
    p.add_argument(
        "--parsed-dir",
        default=os.path.join(_base, "parsed"),
        help="Directory of parsed JSON files from module0b (default: parsed/)",
    )
    p.add_argument(
        "--jd-file",
        required=True,
        help="Path to the plain-text Job Description file",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Sentence-Transformers model name (default: {DEFAULT_MODEL_NAME})",
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
    _args = _build_arg_parser().parse_args()
    _configure_logging(verbose=_args.verbose)
    _ok = run(
        anonymized_cv_dir=_args.anonymized_dir,
        parsed_dir=_args.parsed_dir,
        jd_file=_args.jd_file,
        model_name=_args.model,
    )
    sys.exit(0 if _ok else 1)