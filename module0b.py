# -*- coding: utf-8 -*-
"""
module0b.py — CV Section Parser & Skill Extractor
==================================================
Bias-Free Hiring Pipeline — Stage 2

Reads anonymized CV text files produced by module0 and extracts
structured JSON with skills, experience, education, certifications,
and languages. Outputs one JSON file per candidate plus a master index.

Pipeline position:
    module0  →  anonymized_cvs/Candidate_01.txt
    module0b →  parsed/Candidate_01.json
    module1  →  ranking_report.txt

Standalone usage:
    python module0b.py --input-dir anonymized_cvs --output-dir parsed
    python module0b.py --input-dir anonymized_cvs --output-dir parsed --verbose

Programmatic usage (called by main.py):
    import module0b
    ok = module0b.run(input_dir='anonymized_cvs', output_dir='parsed')
"""

from __future__ import annotations

import json
import logging
import os
import re
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

try:
    from keybert import KeyBERT as _KeyBERT  # optional dependency
    _KEYBERT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _KEYBERT_AVAILABLE = False
    _KeyBERT = None  # type: ignore[assignment,misc]

# ── Module logger (lowercase — loggers are NOT constants) ──────────────────
logger = logging.getLogger(__name__)

# ── File/format constants ──────────────────────────────────────────────────
_LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT    = "%H:%M:%S"
_ISO_FMT     = "%Y-%m-%dT%H:%M:%S"
_SUPPORTED   = {".txt"}

# ── Month abbreviation → int ───────────────────────────────────────────────
_MONTHS: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,  "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ═══════════════════════════════════════════════════════════════════════════
# SKILLS TAXONOMY
# 300+ entries across 4 categories.  All matching is case-insensitive.
# Aliases (JS, k8s, etc.) are resolved during extraction.
# ═══════════════════════════════════════════════════════════════════════════
SKILLS_TAXONOMY: Dict[str, List[str]] = {

    # ── Programming languages, frameworks, databases, cloud, DevOps ────────
    "technical": [
        # Languages
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
        "Go", "Golang", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
        "R", "MATLAB", "Perl", "Shell", "Bash", "PowerShell", "Groovy",
        "Objective-C", "Dart", "Elixir", "Haskell", "Julia", "Lua",
        "Visual Basic", "VBA", "COBOL", "Fortran", "Assembly", "Solidity",
        "PL/SQL", "T-SQL", "GraphQL", "KQL", "ABAP", "Erlang", "OCaml",
        "F#", "Clojure", "Elm", "Nim", "Crystal", "Zig", "Racket",
        # Web / Frontend
        "HTML", "CSS", "React", "Angular", "Vue", "Svelte", "Next.js",
        "Nuxt.js", "Gatsby", "Redux", "MobX", "RxJS", "jQuery",
        "Bootstrap", "Tailwind CSS", "Material UI", "Chakra UI",
        "Three.js", "D3.js", "WebAssembly", "PWA",
        # Backend / API frameworks
        "Node.js", "Express", "NestJS", "FastAPI", "Flask", "Django",
        "Spring", "Spring Boot", "Rails", "Laravel", "Symfony",
        "ASP.NET", "ASP.NET Core", "Gin", "Echo", "Fiber",
        "Actix", "Rocket", "Phoenix", "Ktor",
        # Data science / ML
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost",
        "LightGBM", "CatBoost", "spaCy", "NLTK", "Gensim",
        "HuggingFace", "Transformers", "OpenCV", "Pillow",
        "Pandas", "NumPy", "SciPy", "Statsmodels", "Plotly",
        "Matplotlib", "Seaborn", "Bokeh", "Altair", "Dash", "Streamlit",
        # Databases — relational
        "PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle DB",
        "SQL Server", "DB2",
        # Databases — NoSQL
        "MongoDB", "Cassandra", "CouchDB", "DynamoDB", "Firestore",
        "Cosmos DB", "Redis", "Memcached", "Neo4j", "InfluxDB",
        "TimescaleDB", "ClickHouse", "Elasticsearch", "Solr",
        # Cloud platforms
        "AWS", "Azure", "GCP", "Google Cloud", "IBM Cloud",
        "Heroku", "DigitalOcean", "Vercel", "Netlify", "Firebase",
        "Cloudflare", "Linode", "OVH",
        # Cloud services
        "S3", "EC2", "Lambda", "EKS", "ECS", "RDS", "CloudFormation",
        "Azure DevOps", "Azure Functions", "Cloud Run", "BigQuery",
        "Redshift", "Snowflake", "Databricks",
        # DevOps / Infrastructure
        "Docker", "Kubernetes", "Terraform", "Ansible", "Puppet",
        "Chef", "Vagrant", "Helm", "Istio", "Linkerd",
        "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI",
        "Travis CI", "ArgoCD", "Flux", "Tekton",
        # Data engineering
        "Apache Spark", "Hadoop", "Kafka", "RabbitMQ", "ActiveMQ",
        "Apache Airflow", "Luigi", "Prefect", "Dagster", "dbt",
        "Flink", "Storm", "NiFi", "Hive", "Pig",
        # ML ops
        "MLflow", "Kubeflow", "BentoML", "Seldon", "DVC",
        "Weights & Biases", "Neptune", "Comet ML",
        # Version control / collaboration
        "Git", "GitHub", "GitLab", "Bitbucket", "SVN",
        # Testing
        "pytest", "JUnit", "Selenium", "Cypress", "Playwright",
        "Jest", "Mocha", "Chai", "TestNG", "Robot Framework",
        "Locust", "k6", "Gatling",
        # Security
        "OAuth", "JWT", "SAML", "OpenID Connect", "Vault",
        "OWASP", "Penetration Testing", "SIEM", "IAM",
        # Protocols / standards
        "REST", "gRPC", "WebSockets", "SOAP",
        "MQTT", "AMQP", "OpenAPI", "Swagger",
        # Embedded / hardware
        "Arduino", "Raspberry Pi", "VHDL", "Verilog", "LabVIEW",
        # Mobile
        "Android", "iOS", "React Native", "Flutter", "Xamarin",
        # Game / 3D
        "Unity", "Unreal Engine", "Godot", "OpenGL", "Vulkan",
        # Notebook / IDE (as technical skills)
        "Jupyter", "Google Colab", "VS Code", "IntelliJ IDEA",
        "PyCharm", "Eclipse", "Xcode", "Android Studio",
    ],

    # ── Soft skills ─────────────────────────────────────────────────────────
    "soft": [
        "Leadership", "Communication", "Teamwork", "Collaboration",
        "Problem Solving", "Critical Thinking", "Creativity",
        "Adaptability", "Flexibility", "Time Management",
        "Organization", "Conflict Resolution", "Negotiation",
        "Empathy", "Emotional Intelligence", "Decision Making",
        "Initiative", "Motivation", "Work Ethic",
        "Attention to Detail", "Resilience", "Stress Management",
        "Active Listening", "Presentation", "Public Speaking",
        "Coaching", "Mentoring", "Customer Service",
        "Interpersonal Skills", "Networking", "Self-Management",
        "Accountability", "Integrity", "Patience",
        "Open-mindedness", "Resourcefulness", "Cultural Awareness",
        "Delegation", "Strategic Thinking", "Analytical Thinking",
        "Influence", "Persuasion", "Assertiveness",
        "Self-confidence", "Learning Agility", "Curiosity",
        "Self-awareness", "Inclusivity", "Diversity",
        "Cross-functional Collaboration", "Stakeholder Management",
        "Project Management", "Agile", "Scrum", "Kanban",
        "Risk Management", "Change Management",
    ],

    # ── Dedicated tools (observability, design, PM, BI) ────────────────────
    "tools": [
        # Project / issue tracking
        "Jira", "Confluence", "Trello", "Asana", "Monday.com",
        "Linear", "Notion", "ClickUp", "Basecamp",
        # Design / UX
        "Figma", "Sketch", "Adobe XD", "InVision", "Axure",
        "Zeplin", "Canva", "Miro", "Lucidchart", "Draw.io",
        "Balsamiq", "Marvel",
        # BI / Analytics
        "Tableau", "Power BI", "Qlik", "Looker", "Sisense",
        "Metabase", "Redash", "Superset", "Grafana", "Kibana",
        "MicroStrategy", "Domo", "SAP BusinessObjects",
        # Data integration / ETL
        "Alteryx", "Talend", "Informatica", "Pentaho",
        "Fivetran", "Stitch", "Matillion",
        # Observability / monitoring
        "Datadog", "Prometheus", "Splunk",
        "New Relic", "Dynatrace", "AppDynamics", "PagerDuty",
        "OpsGenie", "ELK Stack",
        # API / dev tools
        "Postman", "Insomnia", "Swagger", "SoapUI",
        "Charles Proxy", "Wireshark",
        # Code quality / security
        "SonarQube", "ESLint", "Pylint", "Black", "Prettier",
        "Snyk", "Checkmarx", "Veracode",
        # Communication
        "Slack", "Microsoft Teams", "Zoom", "Google Meet",
        "Webex", "Discord",
        # CRM / ERP
        "Salesforce", "HubSpot", "SAP", "Oracle ERP",
        "ServiceNow", "Zendesk",
        # Package managers / build
        "npm", "Yarn", "pip", "Poetry", "Conda",
        "Maven", "Gradle", "Make", "CMake",
    ],

    # ── Natural languages ───────────────────────────────────────────────────
    "languages": [
        "English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam",
        "Bengali", "Marathi", "Gujarati", "Punjabi", "Urdu", "Odia",
        "Assamese", "Sanskrit", "French", "German", "Spanish",
        "Italian", "Portuguese", "Russian", "Mandarin", "Cantonese",
        "Chinese", "Japanese", "Korean", "Arabic", "Turkish", "Dutch",
        "Swedish", "Norwegian", "Danish", "Finnish", "Polish", "Czech",
        "Hungarian", "Greek", "Hebrew", "Thai", "Vietnamese",
        "Indonesian", "Malay", "Filipino", "Tagalog", "Swahili",
        "Afrikaans", "Zulu", "Somali", "Amharic", "Yoruba",
        "Hausa", "Sinhala", "Nepali", "Burmese", "Khmer",
        "Mongolian", "Pashto", "Farsi", "Persian", "Kurdish",
        "Armenian", "Georgian", "Azerbaijani", "Uzbek", "Kazakh",
        "Romanian", "Bulgarian", "Serbian", "Croatian", "Slovak",
        "Estonian", "Latvian", "Lithuanian", "Icelandic",
        "Irish", "Welsh", "Catalan", "Basque", "Maltese",
    ],
}

# Pre-build a flat alias → canonical map for O(1) lookup at extraction time.
# Aliases: lowercase canonical, common abbreviations, plural strip.
_SKILL_ALIASES: Dict[str, str] = {}

def _build_alias_map() -> None:
    """Populate _SKILL_ALIASES from SKILLS_TAXONOMY at import time."""
    _manual_aliases: Dict[str, str] = {
        "js":           "JavaScript",
        "ts":           "TypeScript",
        "k8s":          "Kubernetes",
        "tf":           "TensorFlow",
        "sklearn":      "Scikit-learn",
        "scikit":       "Scikit-learn",
        "hf":           "HuggingFace",
        "hugging face": "HuggingFace",
        "node":         "Node.js",
        "nodejs":       "Node.js",
        "vue.js":       "Vue",
        "vuejs":        "Vue",
        "reactjs":      "React",
        "react.js":     "React",
        "angular.js":   "Angular",
        "angularjs":    "Angular",
        "postgres":     "PostgreSQL",
        "mongo":        "MongoDB",
        "mssql":        "SQL Server",
        "gh actions":   "GitHub Actions",
        "gha":          "GitHub Actions",
        "tailwind":     "Tailwind CSS",
        "pb":           "Power BI",
        "powerbi":      "Power BI",
        "spacy":        "spaCy",
        "pytorch":      "PyTorch",
        "tensorflow":   "TensorFlow",
        "gcp":          "GCP",
        "google cloud platform": "GCP",
        "amazon web services":   "AWS",
        "microsoft azure":       "Azure",
    }
    for alias, canonical in _manual_aliases.items():
        _SKILL_ALIASES[alias] = canonical

    for skills in SKILLS_TAXONOMY.values():
        for skill in skills:
            lower = skill.lower()
            _SKILL_ALIASES[lower] = skill
            # strip trailing 's' for simple plurals (frameworks → framework)
            if lower.endswith("s") and len(lower) > 4:
                _SKILL_ALIASES[lower[:-1]] = skill

_build_alias_map()


# ── Section heading keywords ───────────────────────────────────────────────
_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "summary":        ["summary", "profile", "about me", "about",
                       "objective", "career objective", "overview",
                       "professional summary", "personal statement"],
    "skills":         ["skills", "technical skills", "core competencies",
                       "competencies", "expertise", "key skills",
                       "abilities", "proficiencies", "tech stack",
                       "technologies", "tools & technologies"],
    "experience":     ["experience", "work experience", "work history",
                       "employment", "employment history",
                       "professional experience", "career history",
                       "professional background", "internship",
                       "internships"],
    "education":      ["education", "academic background",
                       "academic qualifications", "qualifications",
                       "educational background", "degree", "studies",
                       "academic history"],
    "certifications": ["certifications", "certification", "certificates",
                       "licenses", "accreditations", "awards",
                       "achievements", "accomplishments"],
    "projects":       ["projects", "key projects", "project experience",
                       "portfolio", "personal projects", "open source",
                       "contributions", "side projects"],
    "languages":      ["languages", "language proficiency",
                       "spoken languages", "linguistic skills",
                       "language skills"],
}

# ── Compiled regexes (compiled once at import, reused for all CVs) ─────────
_RE_DATE_RANGE = re.compile(
    r"(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})"
    r"\s*[-–—]\s*"
    r"(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}"
    r"|Present|Current|Now|Ongoing)",
    re.IGNORECASE,
)
_RE_YEAR_RANGE   = re.compile(r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|Present|Current)\b",
                               re.IGNORECASE)
_RE_DURATION     = re.compile(r"(\d+)\s*(?:months?|mos?|yrs?|years?)", re.IGNORECASE)
_RE_YEAR_ONLY    = re.compile(r"\b((?:19|20)\d{2})\b")
_RE_BULLET       = re.compile(r"^[\s\-•*►▪◦‣⁃➤→·]+")
_RE_SECTION_HDR  = re.compile(r"^[\s\-=_*#]{0,4}(.+?)[\s\-=_*#:]{0,4}$")
_RE_YEAR_4       = re.compile(r"\b((?:19|20)\d{2})\b")

_DEGREE_LEVELS = [
    ("phd", "PhD"), ("ph.d", "PhD"), ("doctorate", "PhD"),
    ("doctor of", "PhD"),
    ("master of science", "M.Sc"), ("master of arts", "M.A"),
    ("master of business", "MBA"), ("master of engineering", "M.E"),
    ("master of technology", "M.Tech"), ("mtech", "M.Tech"),
    ("msc", "M.Sc"), ("m.sc", "M.Sc"), ("mba", "MBA"),
    ("m.e.", "M.E"), ("m.e ", "M.E"), ("me ", "M.E"),
    ("master", "Master"),
    ("bachelor of science", "B.Sc"), ("bachelor of arts", "B.A"),
    ("bachelor of engineering", "B.E"), ("bachelor of technology", "B.Tech"),
    ("btech", "B.Tech"), ("b.tech", "B.Tech"),
    ("bsc", "B.Sc"), ("b.sc", "B.Sc"),
    ("b.e.", "B.E"), ("b.e ", "B.E"), ("be ", "B.E"),
    ("bca", "BCA"), ("mca", "MCA"),
    ("bachelor", "Bachelor"),
    ("undergraduate", "Undergraduate"),
    ("postgraduate", "Postgraduate"),
    ("diploma", "Diploma"),
    ("certificate", "Certificate"),
    ("high school", "High School"),
    ("secondary", "Secondary"),
    ("sslc", "SSLC"), ("hsc", "HSC"), ("cbse", "CBSE"), ("icse", "ICSE"),
]

_FIELD_PATTERNS = [
    "Computer Science", "Computer Engineering", "Information Technology",
    "Information Systems", "Software Engineering", "Data Science",
    "Artificial Intelligence", "Machine Learning", "Cyber Security",
    "Electrical Engineering", "Electronics", "Mechanical Engineering",
    "Civil Engineering", "Chemical Engineering", "Aerospace Engineering",
    "Biomedical Engineering", "Industrial Engineering",
    "Business Administration", "Management", "Finance", "Economics",
    "Accounting", "Marketing", "Human Resources", "Supply Chain",
    "Operations Management",
    "Physics", "Chemistry", "Mathematics", "Statistics",
    "Biology", "Biotechnology", "Microbiology",
    "Psychology", "Sociology", "Political Science", "Philosophy",
    "Literature", "History", "Law", "Architecture", "Design",
    "Education", "Nursing", "Medicine",
]
_RE_FIELD = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _FIELD_PATTERNS) + r")\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Section detection
# ═══════════════════════════════════════════════════════════════════════════

def parse_sections(text: str) -> Dict[str, str]:
    """Detect and split CV into named sections using keyword matching.

    Strategy:
      1. Scan every line for a heading match (keyword on its own line,
         possibly decorated with dashes / colons).
      2. Record the character position of each detected heading.
      3. Slice the original text between consecutive heading positions.

    Handles messy formatting: ALL CAPS headings, underlined headings,
    headings with trailing colons, mixed-case, etc.

    Args:
        text: Raw CV text (anonymized by module0).

    Returns:
        Dict mapping section names to their raw text content.
        Only sections that were actually found are included.
    """
    section_positions: Dict[str, int] = {}
    lines = text.split("\n")
    pos = 0  # character position in original text

    for raw_line in lines:
        stripped = raw_line.strip()
        # A heading line is short (≤ 60 chars) and not a bullet/sentence
        if 2 <= len(stripped) <= 60 and not stripped.endswith((".", ",")):
            candidate = stripped.lower().rstrip(":").strip()
            # Remove common decoration characters
            candidate = re.sub(r"^[-=_*#\s]+|[-=_*#\s]+$", "", candidate)
            for section, keywords in _SECTION_KEYWORDS.items():
                if section not in section_positions:
                    for kw in keywords:
                        if candidate == kw or candidate.startswith(kw):
                            section_positions[section] = pos
                            logger.debug("Section '%s' detected at pos %d", section, pos)
                            break
        pos += len(raw_line) + 1  # +1 for the newline

    if not section_positions:
        logger.warning("No CV sections detected — returning full text as 'summary'")
        return {"summary": text}

    sorted_secs = sorted(section_positions.items(), key=lambda x: x[1])
    sections: Dict[str, str] = {}
    for idx, (name, start) in enumerate(sorted_secs):
        end = sorted_secs[idx + 1][1] if idx + 1 < len(sorted_secs) else len(text)
        content = text[start:end].strip()
        # Strip the heading line itself from the content
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline:].strip()
        sections[name] = content

    return sections


# ═══════════════════════════════════════════════════════════════════════════
# Skill extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_skills(section_text: str) -> Dict[str, List[str]]:
    """Extract skills from section text using SKILLS_TAXONOMY + alias map.

    Uses full-text substring search (not token splitting) so multi-word
    skills like "Spring Boot", "Power BI", "Machine Learning" are matched.
    Word-boundary anchors prevent 'R' matching inside 'React', etc.

    If KeyBERT is installed (optional), a second pass is performed: the
    top keyword phrases from KeyBERT are matched against the alias map and
    any skills not already found by the taxonomy pass are merged into the
    result.  If KeyBERT is *not* installed, behaviour is unchanged.

    Args:
        section_text: Raw text of the skills section (or full CV text as
                      fallback when no explicit skills section exists).

    Returns:
        Dict with keys matching SKILLS_TAXONOMY categories, each containing
        a deduplicated list of matched skill strings.
    """
    found: Dict[str, List[str]] = {cat: [] for cat in SKILLS_TAXONOMY}
    text_lower = section_text.lower()

    for category, skills in SKILLS_TAXONOMY.items():
        seen: set[str] = set()
        for skill in skills:
            if skill in seen:
                continue
            # Build search targets: canonical lowercase + any registered aliases
            targets = {skill.lower()}
            for alias, canonical in _SKILL_ALIASES.items():
                if canonical == skill:
                    targets.add(alias)

            matched = False
            for target in targets:
                # Use word boundaries; escape special regex chars in skill names
                pattern = r"(?<![a-zA-Z0-9\.\+#])" + re.escape(target) + r"(?![a-zA-Z0-9\.\+#])"
                if re.search(pattern, text_lower):
                    found[category].append(skill)
                    seen.add(skill)
                    matched = True
                    break

    # ── Optional KeyBERT second pass ─────────────────────────────────────
    if _KEYBERT_AVAILABLE and section_text.strip():
        found = _merge_keybert_skills(section_text, found)

    return found


# Module-level KeyBERT model instance (lazy-loaded on first use).
_keybert_model: Optional[Any] = None


def _get_keybert_model() -> Any:
    """Return (and cache) the KeyBERT model instance."""
    global _keybert_model
    if _keybert_model is None:
        _keybert_model = _KeyBERT()  # uses all-MiniLM-L6-v2 by default
        logger.debug("KeyBERT model initialised")
    return _keybert_model


def _merge_keybert_skills(
    section_text: str,
    found: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Run KeyBERT on *section_text* and merge new skills into *found*.

    Only keywords that resolve to a canonical name in the alias map **AND**
    are not already present in *found* are added.  Unknown free-text phrases
    are silently ignored — this keeps the output schema identical to the
    taxonomy-only result.

    Args:
        section_text: The raw text passed to :func:`extract_skills`.
        found:        The dict already populated by the taxonomy pass
                      (mutated in-place and returned).

    Returns:
        The updated *found* dict.
    """
    # Build a flat set of all canonical skills already captured so we can
    # cheaply skip duplicates.
    already: set[str] = {
        skill
        for skills in found.values()
        for skill in skills
    }

    try:
        kw_model = _get_keybert_model()
        # Extract up to 30 keyword phrases (1- and 2-grams).
        keywords = kw_model.extract_keywords(
            section_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=30,
            use_mmr=True,
            diversity=0.5,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("KeyBERT extraction failed: %s", exc)
        return found

    for keyword, _score in keywords:
        kw_lower = keyword.lower().strip()
        canonical = _SKILL_ALIASES.get(kw_lower)
        if canonical is None:
            continue  # phrase not in taxonomy — skip
        if canonical in already:
            continue  # already caught by taxonomy pass

        # Find which category this canonical skill belongs to and append.
        for category, skills in SKILLS_TAXONOMY.items():
            if canonical in skills:
                found[category].append(canonical)
                already.add(canonical)
                logger.debug(
                    "KeyBERT added '%s' (from keyword '%s') to '%s'",
                    canonical, keyword, category,
                )
                break

    return found


# ═══════════════════════════════════════════════════════════════════════════
# Date / duration helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_date(date_str: str) -> datetime:
    """Parse a date string into a datetime object.

    Handles:
        - "Jan 2022" / "January 2022"
        - "Present" / "Current" / "Now" / "Ongoing"  → today
        - "2022"  → Jan 1 of that year

    Raises:
        ValueError: If the date string cannot be parsed.
    """
    raw = date_str.strip().lower()
    if raw in ("present", "current", "now", "ongoing"):
        return datetime.now()

    # "Month YYYY"
    parts = raw.split()
    if len(parts) == 2:
        month = _MONTHS.get(parts[0][:3])
        if month:
            return datetime(int(parts[1]), month, 1)

    # "YYYY" alone
    if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 4:
        return datetime(int(parts[0]), 1, 1)

    raise ValueError(f"Unrecognised date: '{date_str}'")


def _diff_months(start: datetime, end: datetime) -> int:
    """Return the positive month difference between two datetimes."""
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def _extract_duration_from_line(line: str) -> Tuple[Optional[str], Optional[int], List[str]]:
    """Attempt to extract a date range or explicit duration from one line.

    Returns:
        (period_string, duration_months, warnings_list)
    """
    warnings: List[str] = []

    # Try "Month YYYY – Month YYYY / Present"
    m = _RE_DATE_RANGE.search(line)
    if m:
        period = m.group(0)
        try:
            start_dt = _parse_date(m.group(1))
            end_dt   = _parse_date(m.group(2))
            return period, _diff_months(start_dt, end_dt), warnings
        except ValueError as exc:
            warnings.append(str(exc))
            return period, None, warnings

    # Try "YYYY – YYYY / Present"
    m = _RE_YEAR_RANGE.search(line)
    if m:
        period = m.group(0)
        try:
            start_dt = _parse_date(m.group(1))
            end_dt   = _parse_date(m.group(2))
            return period, _diff_months(start_dt, end_dt), warnings
        except ValueError as exc:
            warnings.append(str(exc))
            return period, None, warnings

    # Try explicit "X months" / "X years"
    m = _RE_DURATION.search(line)
    if m:
        raw = int(m.group(1))
        unit = m.group(0).split()[1].lower()
        months = raw if unit.startswith("mon") else raw * 12
        return m.group(0), months, warnings

    return None, None, warnings


# ═══════════════════════════════════════════════════════════════════════════
# Experience parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_experience(section_text: str) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """Parse the experience section into structured role entries.

    Strategy:
      - Scan each line for a date/duration pattern to anchor a role boundary.
      - Accumulate subsequent lines as the role description until the next
        date anchor or end of section.
      - Job titles are intentionally kept as "[REDACTED]" because module0
        has already stripped employer names; extracting a "title" from
        anonymized text risks surfacing residual PII.

    Args:
        section_text: Raw text of the experience section.

    Returns:
        Tuple of:
          - List of role dicts (title, duration_months, period, description)
          - Total experience in months across all roles
          - List of warning strings
    """
    roles: List[Dict[str, Any]] = []
    total_months = 0
    all_warnings: List[str] = []
    lines = [l.rstrip() for l in section_text.split("\n")]

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        period, months, warnings = _extract_duration_from_line(stripped)
        all_warnings.extend(warnings)

        # If no date found on this line, check the next line (common pattern:
        # role title on line 1, dates on line 2)
        lookahead_period = None
        lookahead_months = None
        if period is None and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            lookahead_period, lookahead_months, lw = _extract_duration_from_line(next_stripped)
            all_warnings.extend(lw)

        if period is not None or lookahead_period is not None:
            # Use lookahead values if current line had no date
            if period is None:
                period = lookahead_period
                months = lookahead_months
                i += 1  # consume the date line

            if months is not None:
                total_months += months

            # Gather description lines until next date anchor or blank+date
            desc_lines: List[str] = [stripped]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                next_period, _, _ = _extract_duration_from_line(next_line)
                if next_period is not None:
                    break
                # Also stop if line looks like a new section heading
                if next_line.lower().rstrip(":") in {
                    kw for kws in _SECTION_KEYWORDS.values() for kw in kws
                }:
                    break
                desc_lines.append(next_line)
                j += 1

            description = " ".join(
                _RE_BULLET.sub("", ln).strip()
                for ln in desc_lines
                if _RE_BULLET.sub("", ln).strip()
            )

            roles.append({
                "title":           "[REDACTED]",  # intentional — module0 has stripped PII
                "duration_months": months if months is not None else 0,
                "period":          period or "",
                "description":     description[:500],  # cap at 500 chars for downstream safety
            })
            i = j
        else:
            i += 1

    # Fallback: if no structured roles detected, try to at least sum up
    # any year-ranges found anywhere in the section
    if not roles:
        logger.debug("Structured role extraction found nothing; falling back to year scan")
        for m in _RE_YEAR_RANGE.finditer(section_text):
            try:
                start_dt = _parse_date(m.group(1))
                end_dt   = _parse_date(m.group(2))
                dur = _diff_months(start_dt, end_dt)
                total_months += dur
                roles.append({
                    "title":           "[REDACTED]",
                    "duration_months": dur,
                    "period":          m.group(0),
                    "description":     "",
                })
            except ValueError:
                pass

    if not roles:
        all_warnings.append("No experience entries could be parsed from this section")

    return roles, total_months, all_warnings


# ═══════════════════════════════════════════════════════════════════════════
# Education parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_education(section_text: str) -> List[Dict[str, Any]]:
    """Parse education section into structured degree entries.

    Extracts degree level, field of study, institution placeholder, and
    the date period (year range). Institution names are always replaced with
    '[Educational Institution]' — module0 will have already done this, but
    we enforce it here as a second safety layer.

    Args:
        section_text: Raw text of the education section.

    Returns:
        List of education entry dicts.
    """
    entries: List[Dict[str, Any]] = []
    # Group lines into logical blocks separated by blank lines
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in section_text.split("\n"):
        stripped = raw_line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    for block in blocks:
        block_text = " ".join(block)
        lower_block = block_text.lower()

        # ── Degree level ──────────────────────────────────────────────────
        level: Optional[str] = None
        for pattern, label in _DEGREE_LEVELS:
            if pattern in lower_block:
                level = label
                break

        # ── Field of study ────────────────────────────────────────────────
        field: Optional[str] = None
        fm = _RE_FIELD.search(block_text)
        if fm:
            field = fm.group(1)

        # ── Period (year range or single year) ────────────────────────────
        period: Optional[str] = None
        ym = _RE_YEAR_RANGE.search(block_text)
        if ym:
            period = ym.group(0)
        else:
            yonly = _RE_YEAR_4.findall(block_text)
            if yonly:
                period = yonly[-1]  # most recent year

        # ── Institution ───────────────────────────────────────────────────
        # Always anonymized — enforce module0's guarantee
        institution = "[Educational Institution]"

        # Only append if we found at least one meaningful field
        if level or field or period:
            entries.append({
                "level":       level or "",
                "field":       field or "",
                "institution": institution,
                "period":      period or "",
            })

    if not entries:
        logger.debug("No structured education entries found; returning raw block count")

    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Certifications parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_certifications(section_text: str) -> List[str]:
    """Extract certification names from the certifications section.

    Each non-empty line is treated as a separate certification.
    Leading bullets and decorators are stripped.

    Args:
        section_text: Raw text of the certifications section.

    Returns:
        Deduplicated list of certification name strings.
    """
    certs: List[str] = []
    seen: set[str] = set()
    for raw_line in section_text.split("\n"):
        clean = _RE_BULLET.sub("", raw_line).strip()
        if clean and clean not in seen and len(clean) > 3:
            certs.append(clean)
            seen.add(clean)
    return certs


# ═══════════════════════════════════════════════════════════════════════════
# Language parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_languages(section_text: str) -> List[str]:
    """Extract natural languages from the languages section.

    Matches against the 'languages' taxonomy using full-text search
    (same strategy as extract_skills) to handle comma/bullet-separated lists.

    Args:
        section_text: Raw text of the languages section.

    Returns:
        Deduplicated list of language name strings.
    """
    text_lower = section_text.lower()
    found: List[str] = []
    seen: set[str] = set()
    for lang in SKILLS_TAXONOMY["languages"]:
        if lang in seen:
            continue
        pattern = r"(?<![a-zA-Z])" + re.escape(lang.lower()) + r"(?![a-zA-Z])"
        if re.search(pattern, text_lower):
            found.append(lang)
            seen.add(lang)
    return found


# ═══════════════════════════════════════════════════════════════════════════
# Quality score
# ═══════════════════════════════════════════════════════════════════════════

def compute_quality_score(sections: Dict[str, Any]) -> float:
    """Compute a 0.0–1.0 quality score for a parsed CV.

    Scoring model (total weight = 1.0):
        Section completeness   0.50
          summary        0.08
          skills         0.15
          experience     0.15
          education      0.08
          certifications 0.02
          languages      0.02
        Skill depth            0.20
          ≥ 15 skills    0.20
          ≥  8 skills    0.12
          ≥  3 skills    0.06
        Experience clarity     0.15
          ≥ 1 role with months 0.10
          total_months > 0     0.05
        Education clarity      0.10
          ≥ 1 entry with level 0.06
          ≥ 1 entry with field 0.04
        Certifications bonus   0.05
          ≥ 1 cert             0.05

    Args:
        sections: The 'sections' sub-dict from the parsed candidate dict.

    Returns:
        Float clamped to [0.0, 1.0].
    """
    score = 0.0

    _section_weights = {
        "summary": 0.08, "skills": 0.15, "experience": 0.15,
        "education": 0.08, "certifications": 0.02, "languages": 0.02,
    }
    for sec, weight in _section_weights.items():
        val = sections.get(sec)
        if val:
            score += weight

    skills = sections.get("skills", {})
    skill_count = sum(len(v) for v in skills.values()) if isinstance(skills, dict) else 0
    if skill_count >= 15:
        score += 0.20
    elif skill_count >= 8:
        score += 0.12
    elif skill_count >= 3:
        score += 0.06

    experience = sections.get("experience", [])
    total_exp = sections.get("total_experience_months", 0)
    if experience and any(r.get("duration_months", 0) > 0 for r in experience):
        score += 0.10
    if total_exp and total_exp > 0:
        score += 0.05

    education = sections.get("education", [])
    if education:
        if any(e.get("level") for e in education):
            score += 0.06
        if any(e.get("field") for e in education):
            score += 0.04

    certs = sections.get("certifications", [])
    if certs:
        score += 0.05

    return round(min(score, 1.0), 4)


# ═══════════════════════════════════════════════════════════════════════════
# Single-file processor
# ═══════════════════════════════════════════════════════════════════════════

def process_file(
    filepath: str,
    candidate_id: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse one anonymized CV text file into a structured dict.

    Args:
        filepath:     Absolute or relative path to the .txt CV file.
        candidate_id: Identifier string (e.g. 'Candidate_01').

    Returns:
        Tuple of (parsed_dict | None, list_of_warnings).
        Returns (None, warnings) if the file cannot be read or is empty.
    """
    warnings: List[str] = []

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return None, [f"Cannot read file: {exc}"]

    if not text.strip():
        return None, ["File is empty or contains only whitespace"]

    sections_raw = parse_sections(text)
    sections: Dict[str, Any] = {}

    # ── Summary ──────────────────────────────────────────────────────────
    if "summary" in sections_raw:
        sections["summary"] = sections_raw["summary"]
    else:
        warnings.append("Section not found: summary")
        sections["summary"] = ""

    # ── Skills ───────────────────────────────────────────────────────────
    skills_source = sections_raw.get("skills", text)  # fallback: scan full text
    if "skills" not in sections_raw:
        warnings.append("Skills section not found; scanning full text")
    sections["skills"] = extract_skills(skills_source)

    # ── Experience ───────────────────────────────────────────────────────
    if "experience" in sections_raw:
        roles, total_months, exp_warns = parse_experience(sections_raw["experience"])
        warnings.extend(exp_warns)
    else:
        warnings.append("Section not found: experience")
        roles, total_months = [], 0
    sections["experience"]             = roles
    sections["total_experience_months"] = total_months

    # ── Education ────────────────────────────────────────────────────────
    if "education" in sections_raw:
        sections["education"] = parse_education(sections_raw["education"])
    else:
        warnings.append("Section not found: education")
        sections["education"] = []

    # ── Certifications ───────────────────────────────────────────────────
    if "certifications" in sections_raw:
        sections["certifications"] = parse_certifications(sections_raw["certifications"])
    else:
        warnings.append("Section not found: certifications")
        sections["certifications"] = []

    # ── Languages ────────────────────────────────────────────────────────
    if "languages" in sections_raw:
        sections["languages"] = parse_languages(sections_raw["languages"])
    else:
        warnings.append("Section not found: languages")
        sections["languages"] = []

    quality = compute_quality_score(sections)

    parsed: Dict[str, Any] = {
        "candidate_id":  candidate_id,
        "parsed_at":     datetime.now().strftime(_ISO_FMT),
        "sections":      sections,
        "quality_score": quality,
        "parse_warnings": warnings,
    }
    return parsed, warnings


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner — called by main.py
# ═══════════════════════════════════════════════════════════════════════════

def run(input_dir: str, output_dir: str) -> bool:
    """Process all .txt CV files in input_dir and write parsed JSON to output_dir.

    This is the primary entry point called by main.py, matching the
    module0.run() signature convention.

    Directory layout after a successful run::

        parsed/
          Candidate_01.json
          Candidate_02.json
          ...
          index.json        ← master index with quality scores
          error_log.txt     ← only present if errors occurred

    Args:
        input_dir:  Directory produced by module0 containing anonymized .txt files.
        output_dir: Destination directory for parsed JSON files.

    Returns:
        True  if at least one file was successfully parsed.
        False if every file failed or the input directory is empty.
    """
    print()
    print("=" * 60)
    print("  MODULE 0b — CV SECTION PARSER & SKILL EXTRACTOR")
    print("=" * 60)

    if not os.path.isdir(input_dir):
        logger.error("Input directory not found: %s", input_dir)
        print(f"\n  [ERROR] Input directory not found: {input_dir}")
        print("          Run module0 first to generate anonymized CVs.")
        return False

    cv_files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in _SUPPORTED
        and f != "error_log.txt"
    )
    total = len(cv_files)

    if total == 0:
        logger.warning("No .txt files found in: %s", input_dir)
        print(f"\n  [!] No .txt files found in: {input_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, "error_log.txt")

    print(f"\n  Input:   {input_dir}  ({total} files)")
    print(f"  Output:  {output_dir}")
    print()

    success_ids: List[str]         = []
    failed_files: List[str]        = []
    candidate_index: Dict[str, Any] = {}
    t0 = time.time()

    iterator = (
        tqdm(cv_files, desc="Parsing CVs", unit="cv")
        if _TQDM_AVAILABLE
        else cv_files
    )

    with open(error_log_path, "w", encoding="utf-8") as elog:
        for fname in iterator:
            candidate_id = os.path.splitext(fname)[0]
            filepath     = os.path.join(input_dir, fname)

            logger.debug("Processing: %s", fname)

            parsed, warnings = process_file(filepath, candidate_id)

            if parsed is None:
                reason = warnings[0] if warnings else "Unknown error"
                logger.error("Failed to parse '%s': %s", fname, reason)
                elog.write(f"FAILED  [{fname}]: {reason}\n")
                failed_files.append(fname)
                continue

            # Write per-candidate JSON
            out_path = os.path.join(output_dir, f"{candidate_id}.json")
            try:
                with open(out_path, "w", encoding="utf-8") as fout:
                    json.dump(parsed, fout, indent=2, ensure_ascii=False)
            except OSError as exc:
                logger.error("Write failed for '%s': %s", out_path, exc)
                elog.write(f"WRITE_ERR [{fname}]: {exc}\n")
                failed_files.append(fname)
                continue

            success_ids.append(candidate_id)
            candidate_index[candidate_id] = {
                "quality_score":           parsed["quality_score"],
                "parsed_at":               parsed["parsed_at"],
                "total_experience_months": parsed["sections"].get("total_experience_months", 0),
                "skill_count":             sum(
                    len(v) for v in parsed["sections"].get("skills", {}).values()
                ),
                "warnings_count":          len(warnings),
            }

            if warnings:
                logger.debug("%s — %d warning(s): %s", candidate_id, len(warnings), warnings[:3])

    # ── Master index ──────────────────────────────────────────────────────
    index_payload = {
        "generated_at":    datetime.now().strftime(_ISO_FMT),
        "total_processed": len(success_ids),
        "total_failed":    len(failed_files),
        "candidates":      candidate_index,
    }
    index_path = os.path.join(output_dir, "index.json")
    try:
        with open(index_path, "w", encoding="utf-8") as idxf:
            json.dump(index_payload, idxf, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.error("Could not write index.json: %s", exc)

    elapsed = time.time() - t0

    # ── Summary report ────────────────────────────────────────────────────
    print(f"\n  Done: {len(success_ids)}/{total} CVs parsed in {elapsed:.1f}s")
    if failed_files:
        print(f"  Failed ({len(failed_files)}): {', '.join(failed_files)}")
        print(f"  See error log: {error_log_path}")
    if success_ids:
        avg_q = sum(
            candidate_index[c]["quality_score"] for c in success_ids
        ) / len(success_ids)
        print(f"  Average quality score: {avg_q:.2f}")
    print(f"  Index: {index_path}")
    print("=" * 60)

    return bool(success_ids)


# ═══════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════

def _configure_logging(verbose: bool = False) -> None:
    """Configure root logger for standalone execution.

    Args:
        verbose: If True, sets level to DEBUG; otherwise INFO.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_DATE_FMT,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="module0b",
        description="CV Section Parser & Skill Extractor — Bias-Free Hiring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python module0b.py --input-dir anonymized_cvs --output-dir parsed\n"
            "  python module0b.py --input-dir anonymized_cvs --output-dir parsed --verbose\n"
        ),
    )
    p.add_argument(
        "--input-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "anonymized_cvs"),
        help="Directory containing anonymized .txt CV files (default: anonymized_cvs/)",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsed"),
        help="Directory to write parsed JSON files (default: parsed/)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return p

if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    _configure_logging(verbose=_args.verbose)
    _ok = run(input_dir=_args.input_dir, output_dir=_args.output_dir)
    raise SystemExit(0 if _ok else 1)