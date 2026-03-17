"""
Module 0 — Differential Privacy Transformation for CV Anonymization
====================================================================
Strips PII, gender/identity proxies, and contact info from CV text.
Returns anonymized text files + a vault for de-identification.

Can be run independently:
    python module0.py
    python module0.py --input-dir my_resumes
"""

import re
import os
import sys
import json
import time
import logging
import argparse
import spacy
from typing import Dict, Any, List, Tuple, Optional
import pytesseract

# Ensure Tesseract is found on Windows regardless of PATH
_TESSERACT_DEFAULT = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.isfile(_TESSERACT_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT

# ── Module logger ──
logger = logging.getLogger(__name__)


# ── Auto-detect and configure Tesseract ─────
def _setup_tesseract() -> None:
    """Auto-detect Tesseract installation and configure pytesseract."""
    # Common Windows install paths
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'D:\Tesseract-OCR\tesseract.exe',
    ]
    
    for path in common_paths:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.debug("Tesseract found at: %s", path)
            return
    
    # If not found in common paths, try system PATH (Linux/Mac or already in PATH on Windows)
    try:
        result = os.popen('where tesseract' if os.name == 'nt' else 'which tesseract').read().strip()
        if result:
            pytesseract.pytesseract.tesseract_cmd = result
            logger.debug("Tesseract found in PATH: %s", result)
            return
    except Exception:
        pass
    
    logger.warning(
        "Tesseract not found. Install: "
        "choco install tesseract (Windows) or apt-get install tesseract-ocr (Linux/Mac). "
        "Or download from: https://github.com/UB-Mannheim/tesseract/wiki"
    )


# ── Auto-detect and configure Poppler ──────
def _setup_poppler() -> Optional[str]:
    """Auto-detect Poppler installation and return path for pdf2image."""
    # Common Windows install paths
    common_paths = [
        r'C:\Program Files\poppler\Library\bin',
        r'C:\Program Files (x86)\poppler\Library\bin',
        r'D:\poppler\Library\bin',
        r'C:\Program Files\poppler-0.68.0\bin',
    ]
    
    for path in common_paths:
        if os.path.isdir(path):
            logger.debug("Poppler found at: %s", path)
            return path
    
    # Try system PATH
    try:
        result = os.popen('where pdfinfo' if os.name == 'nt' else 'which pdfinfo').read().strip()
        if result:
            poppler_dir = os.path.dirname(result)
            logger.debug("Poppler found in PATH: %s", poppler_dir)
            return poppler_dir
    except Exception:
        pass
    
    logger.warning(
        "Poppler not found. Install: "
        "pip install poppler-utils or download from: https://github.com/oschwartz10612/poppler-windows/releases"
    )
    return None

_poppler_path = _setup_poppler()


_setup_tesseract()


# ──────────────────────────────────────────────
# Pronoun / Gender Patterns (case-sensitive)
# ──────────────────────────────────────────────
_PRONOUN_MAP = {
    r'\bHe\b':   'The Candidate',
    r'\bShe\b':  'The Candidate',
    r'\bhe\b':   'the candidate',
    r'\bshe\b':  'the candidate',
    r'\bHim\b':  'Them',
    r'\bHer\b':  'Them',
    r'\bhim\b':  'them',
    r'\bher\b':  'them',
    r'\bHis\b':  'Their',
    r'\bHers\b': 'Theirs',
    r'\bhis\b':  'their',
    r'\bhers\b': 'theirs',
    r'\bHimself\b':  'Themselves',
    r'\bHerself\b':  'Themselves',
    r'\bhimself\b':  'themselves',
    r'\bherself\b':  'themselves',
    r'\bMr\.?\b': '', r'\bMrs\.?\b': '', r'\bMs\.?\b': '',
    r'\bMiss\b': '', r'\bSir\b': '', r'\bMadam\b': '',
}

# ──────────────────────────────────────────────
# Identity-Proxy Association Patterns
# ──────────────────────────────────────────────
_PROXY_REPLACEMENTS = [
    (re.compile(r'Society\s+of\s+Women\s+Engineers',        re.I), 'Professional Engineering Society'),
    (re.compile(r'Women\s+in\s+Tech(?:nology)?',            re.I), 'Technology Professional Network'),
    (re.compile(r'Women\s+in\s+Computing',                  re.I), 'Computing Professional Network'),
    (re.compile(r'Women\s+in\s+STEM',                       re.I), 'STEM Professional Network'),
    (re.compile(r'Girls?\s+Who\s+Code',                     re.I), 'Coding Education Program'),
    (re.compile(r'Women\'?s\s+Engineering\s+Society',       re.I), 'Professional Engineering Society'),
    (re.compile(r'Association\s+for\s+Women\s+in\s+\w+',    re.I), 'Professional Association'),
    (re.compile(r'Men\'?s\s+Club',                          re.I), 'Social Club'),
    (re.compile(r'National\s+Society\s+of\s+Black\s+Engineers',            re.I), 'Professional Engineering Society'),
    (re.compile(r'Society\s+of\s+Hispanic\s+Professional\s+Engineers',     re.I), 'Professional Engineering Society'),
    (re.compile(r'Asian\s+American\s+Engineers?\s+(?:Society|Association)', re.I), 'Professional Engineering Society'),
    (re.compile(r'(?:Black|African\s+American)\s+Student\s+(?:Union|Association)', re.I), 'Student Association'),
    (re.compile(r'Latino\s+Student\s+(?:Union|Association)',               re.I), 'Student Association'),
    (re.compile(r'Native\s+American\s+Student\s+(?:Union|Association)',    re.I), 'Student Association'),
    (re.compile(r'Muslim\s+Students?\s+Association',        re.I), 'Student Association'),
    (re.compile(r'Hindu\s+Students?\s+(?:Council|Association)', re.I), 'Student Association'),
    (re.compile(r'Jewish\s+Student\s+(?:Union|Organization|Association)', re.I), 'Student Association'),
    (re.compile(r'Christian\s+(?:Fellowship|Association|Union)', re.I), 'Student Fellowship'),
    (re.compile(r'Sikh\s+Students?\s+Association',          re.I), 'Student Association'),
    (re.compile(r'Buddhist\s+Students?\s+Association',      re.I), 'Student Association'),
    (re.compile(r'Catholic\s+Students?\s+Association',      re.I), 'Student Association'),
    (re.compile(r'Hillel',                                  re.I), 'Student Association'),
    (re.compile(r'(?:Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Omicron|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega)\s+'
                r'(?:Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Omicron|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega)'
                r'(?:\s+(?:Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Omicron|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega))?',
                re.I), 'Student Organization'),
    (re.compile(r'(?:LGBTQ?\+?|Queer)\s+(?:Alliance|Association|Society|Network|Club)', re.I), 'Student Alliance'),
    (re.compile(r'Pride\s+(?:Alliance|Network|Club|Society)', re.I), 'Student Alliance'),
    (re.compile(r"(?:Men's|Women's)\s+(?:Basketball|Football|Soccer|Tennis|Swimming|Volleyball|Hockey|Cricket|Rugby|Baseball|Softball|Lacrosse|Track|Cross\s+Country)\s*(?:Team|Club)?", re.I),
     'Varsity Sports Team'),
]

# ──────────────────────────────────────────────
# Contact-info regexes
# ──────────────────────────────────────────────
_EMAIL_RE    = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')

# Phone regex — requires proper format; negative lookbehind/lookahead to
# skip alphanumeric strings like serial numbers "REF-123-456-7890"
_PHONE_RE    = re.compile(
    r'(?<![A-Za-z0-9\-])'
    r'(?:\+\d{1,3}[\s\-.]?)?'
    r'(?:\(?\d{2,5}\)?[\s\-.]?)?'
    r'\d{3,5}[\s\-.]?'
    r'\d{3,5}'
    r'(?![A-Za-z0-9])'
)

_URL_RE      = re.compile(r'https?://[^\s,]+', re.I)

# URL without protocol — catches github.com/user, portfolio.dev, bit.ly/abc
_VALID_TLDS = {'com', 'org', 'net', 'dev', 'io', 'co', 'me', 'ly', 'in',
               'edu', 'gov', 'app', 'xyz', 'tech', 'info', 'biz', 'us', 'uk'}
_URL_NO_PROTO_RE = re.compile(
    r'(?<![@A-Za-z0-9])(?:[a-zA-Z0-9\-]+\.)+('
    + '|'.join(_VALID_TLDS)
    + r')(?:/[^\s,)]*)?',
    re.I
)
# GitHub profile pattern
_GITHUB_RE = re.compile(
    r'(?:github\.com/|GitHub:\s*)([A-Za-z0-9\-_.]+)', re.I
)

_LINKEDIN_RE = re.compile(r'(?:linkedin\.com/in/|linkedin:\s*)\S+', re.I)
_LINKEDIN_PATH_RE = re.compile(r'/in/[a-zA-Z0-9\-]+', re.I)
# LinkedIn handle without full URL: "LinkedIn: johnsmith", "@user (LinkedIn)"
_LINKEDIN_HANDLE_RE = re.compile(
    r'(?:LinkedIn\s*[:\-]\s*|@)([A-Za-z0-9._\-]+)(?:\s*\(?LinkedIn\)?)?', re.I
)

_ADDRESS_RE  = re.compile(
    r'\d{1,5}\s+[\w\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\.?'
    r'(?:\s*,?\s*(?:Apt|Suite|Unit|#)\s*\w+)?'
    r'(?:\s*,\s*[\w\s]+)?(?:\s*,\s*[A-Z]{2})?\s*\d{5}(?:-\d{4})?',
    re.I
)

# ──────────────────────────────────────────────
# Experience duration — age proxy detection
# ──────────────────────────────────────────────
_EXPERIENCE_DURATION_RE = re.compile(
    r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|expertise|professional)',
    re.I
)

# ──────────────────────────────────────────────
# Publication / co-author detection
# ──────────────────────────────────────────────
_PUBLICATION_RE = re.compile(
    r"(?:co-?authored|published|authored|wrote)\s+"
    r"(?:(?:a\s+)?(?:paper|article|book|chapter|research|study)"
    r"|(?:['\"].*?['\"]))"
    r"(?:\s+(?:with|by|and)\s+(?:(?:Dr|Prof|Mr|Ms|Mrs)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)*)?",
    re.I
)

# ──────────────────────────────────────────────
# Degree + institution: "Bachelor's from Stanford"
# ──────────────────────────────────────────────
_DEGREE_CONTEXT_RE = re.compile(
    r"(?:Bachelor'?s?|Master'?s?|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?"
    r"|B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|Ph\.?D\.?|MBA|BCA|MCA|Diploma)"
    r"\s+(?:degree\s+)?(?:from|at|in)\s+"
    r"([A-Z][A-Za-z\s&'.]+?)(?=\s*[,.\n(]|\s+(?:in|with|majoring|specializing))",
    re.I
)

# ──────────────────────────────────────────────
# Full dates (with day) — remove; month-year = keep
# ──────────────────────────────────────────────
_FULL_DATE_RE = re.compile(
    r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b'
)
_NAMED_FULL_DATE_RE = re.compile(
    r'\b(?:(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r'\s+\d{1,2},?\s+\d{4}'
    r'|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r',?\s+\d{4})\b',
    re.I
)

# ──────────────────────────────────────────────
# Country whitelist for GPE granularity
# ──────────────────────────────────────────────
_COUNTRY_NAMES = {
    'india', 'usa', 'uk', 'canada', 'australia', 'germany', 'france',
    'japan', 'china', 'singapore', 'brazil', 'united states',
    'united kingdom', 'south korea', 'netherlands', 'switzerland',
    'sweden', 'norway', 'denmark', 'finland', 'ireland', 'new zealand',
    'israel', 'uae', 'saudi arabia', 'russia', 'mexico', 'italy', 'spain',
}

# ──────────────────────────────────────────────
# Education keywords
# ──────────────────────────────────────────────
_EDUCATION_KEYWORDS = [
    'university', 'college', 'institute', 'school', 'academy',
    'polytechnic', 'conservatory', 'seminary', 'lyceum',
    'matric', 'matriculation', 'hr. sec.', 'hr sec',
    'higher secondary', 'cbse', 'icse', 'isc', 'sslc',
    'convent', 'vidyalaya', 'vidya', 'vidhyalaya', 'gurukul',
    'madrasa', 'iit', 'iim', 'nit', 'iiit', 'bits',
    'mit', 'caltech', 'stanford', 'harvard', 'oxford', 'cambridge',
    'yale', 'princeton', 'berkeley', 'eth', 'epfl',
]

# ──────────────────────────────────────────────
# Age / DOB patterns
# ──────────────────────────────────────────────
_DOB_RE = re.compile(
    r'(?:Date\s+of\s+Birth|DOB|D\.O\.B\.?|Birth\s*date|Born)\s*[:\-]?\s*'
    r'(?:\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}[\-/]\d{2}[\-/]\d{2})',
    re.I
)
_AGE_RE = re.compile(r'(?:Age\s*[:\-]?\s*\d{1,3})', re.I)

# ──────────────────────────────────────────────
# Education years — passing years reveal age → bias
# ──────────────────────────────────────────────
_EDUCATION_YEAR_RE = re.compile(
    r'(?:Passing\s+Year\s*[-:–—]?\s*)\d{4}(?:\s*[-–—]\s*\d{4})?',
    re.I
)

# ──────────────────────────────────────────────
# Tech-term whitelist (NER false-positive guard)
# ──────────────────────────────────────────────
_TECH_WHITELIST = {
    'react', 'node', 'node.js', 'django', 'flask', 'fastapi', 'vue',
    'angular', 'express', 'spring', 'rails', 'ruby', 'rust', 'swift',
    'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'puppet',
    'datadog', 'grafana', 'prometheus', 'elasticsearch', 'kafka', 'redis',
    'postgres', 'mongodb', 'mysql', 'sqlite', 'cassandra', 'hadoop',
    'spark', 'airflow', 'tableau', 'git', 'jira', 'slack', 'figma',
    'aws', 'gcp', 'azure', 'heroku', 'vercel', 'netlify',
    'typescript', 'javascript', 'python', 'golang', 'java', 'kotlin',
    'pytorch', 'tensorflow', 'keras', 'scikit', 'pandas', 'numpy',
    'matplotlib', 'jupyter', 'latex', 'linux', 'ubuntu', 'debian',
    'centos', 'windows', 'macos',
}

_NORP_WHITELIST = {
    'c', 'c++', 'c#', 'go', 'r', 'chinese', 'english', 'hindi', 'tamil',
    'french', 'german', 'spanish', 'japanese', 'korean', 'arabic',
}

_GPE_WHITELIST = {
    'flask', 'spring', 'dart', 'swift', 'ruby', 'rust', 'go',
    'bootstrap', 'angular', 'metro', 'oracle',
    'openmp', 'open', 'networking', 'docker', 'linux',
}

# ──────────────────────────────────────────────
# Regex-based education institution patterns
# ──────────────────────────────────────────────
_EDUCATION_INST_RE = re.compile(
    r"(?:[A-Z][\w'.]+\s+)*"
    r"(?:Matric(?:ulation)?|Hr\.?\s*Sec\.?|Higher\s+Secondary|High|Public|Central|Model|Convent|International)"
    r"\s+(?:School|Academy|Vidyalaya)"
    r"(?:\s*,\s*[A-Za-z]+)?",
    re.I
)
_GENERIC_SCHOOL_RE = re.compile(
    r"(?:St\.?\s+)?(?:[A-Za-z][a-zA-Z'.]+\s+){1,4}(?:School|College|Academy|Vidyalaya)\b",
    re.I
)
_ST_PREFIX_RE = re.compile(
    r"St\.?\s+[A-Z][a-z]+(?:'s)?\s*",
)

_ADDRESS_LOCATION_RE = re.compile(
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?=\s*,)',
    re.MULTILINE
)

# Education-line place-name pattern — named groups for clarity & correctness
_EDUCATION_LINE_RE = re.compile(
    r'^(?P<prefix>.*(?:School|College|University|Institute|Matric|Hr\.?\s*Sec|Academy|Vidyalaya'
    r'|Educational Institution'
    r'|B\.Tech|B\.E|B\.Sc|M\.Tech|M\.E|M\.Sc|MBA|BCA|MCA|SSLC|CBSE|ICSE'
    r'|Higher Secondary|Secondary).*?)'
    r',\s*(?P<place>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    r'(?P<suffix>\s*[-–—]\s|\s*,)',
    re.I | re.MULTILINE
)

# .doc removed — python-docx only supports .docx, not legacy .doc
SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx'}


# ═════════════════════════════════════════════
# CVAnonymizer — core engine
# ═════════════════════════════════════════════

class CVAnonymizer:
    """Anonymizes CV text by stripping PII and identity proxies."""

    # Common CV header titles that should NOT be treated as names
    _HEADER_TITLES = frozenset({
        'RESUME', 'CV', 'CURRICULUM VITAE', 'CAREER OBJECTIVE',
        'OBJECTIVE', 'PROFILE', 'SUMMARY', 'ABOUT ME',
    })

    def __init__(self, spacy_model: str = 'en_core_web_trf'):
        """Load spaCy NER model with automatic fallback chain.
        If no model is found, falls back to heuristics-only mode.
        """
        models_to_try = [spacy_model]
        # Prioritize transformer model, then large, then small
        for fallback in ('en_core_web_trf', 'en_core_web_lg', 'en_core_web_sm'):
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        self.nlp = None  # default: heuristics-only mode
        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                if model_name != spacy_model:
                    logger.warning("Primary model '%s' unavailable — loaded '%s'",
                                   spacy_model, model_name)
                else:
                    logger.info("Loaded NER model: %s", model_name)
                return
            except OSError:
                continue

        logger.warning(
            "No usable spaCy model found (tried: %s). "
            "Running in heuristics-only mode.", models_to_try
        )

    # ── Main entry point ──────────────────────────

    def anonymize(self, cv_text: str, candidate_id: str = 'Candidate_01',
                  level: str = 'STRICT') -> Dict[str, Any]:
        """Anonymize a single CV. Returns dict with 'anonymized_cv' and 'vault_data'.

        Args:
            level: 'STRICT' (remove all), 'MODERATE' (keep countries/tech orgs),
                   'MINIMAL' (only direct identifiers).
        """
        level = level.upper()
        if level not in ('STRICT', 'MODERATE', 'MINIMAL'):
            logger.warning("Unknown level '%s', defaulting to STRICT", level)
            level = 'STRICT'

        vault: Dict[str, Any] = {
            'candidate_id': candidate_id,
            'original_name': None,
            'contact_info': {},
            'publications': [],
            'pii_categories_found': {},
            'anonymization_level_used': level,
            'techniques_applied': [],
            'validation_score': None,
        }
        text = cv_text

        # 0. Detect and redact candidate name (heuristic + NER cross-validation)
        text, vault = self._detect_and_redact_name(text, candidate_id, vault)

        # 1–2. Strip contacts, remove DOB/age
        text, vault = self._strip_contacts(text, vault)
        text = self._remove_dob_age(text)

        # 3. Replace identity proxies
        text = self._replace_proxies(text)

        # 4. Remove education passing years (age → bias)
        text = _EDUCATION_YEAR_RE.sub('[YEAR_REDACTED]', text)

        # 4.5. Experience duration → age proxy replacement
        text = self._redact_experience_duration(text, vault)

        # 4.6. Publication / co-author redaction
        text = self._redact_publications(text, vault)

        # 4.7. Full-date removal (keep month-year for timelines)
        text = self._redact_full_dates(text, vault)

        # 5. Regex-based education institution replacement
        text = _EDUCATION_INST_RE.sub('[Educational Institution]', text)
        text = _GENERIC_SCHOOL_RE.sub('[Educational Institution]', text)

        # 5.5. Degree-context institution detection
        text = self._redact_degree_institutions(text, vault)

        # 6. Redact place names on education lines (named groups — no index arithmetic)
        text = _EDUCATION_LINE_RE.sub(
            lambda m: m.group('prefix') + ', [LOCATION_REDACTED]' + m.group('suffix'),
            text,
        )

        # 7. NER pass (persons, orgs, dates, locations, NORP)
        if self.nlp is not None:
            text, vault = self._ner_pass(text, candidate_id, vault, level)
            vault['techniques_applied'].append('spacy_ner')
        else:
            logger.warning("NER skipped (no model loaded), using heuristics only")

        # 7.5. Always run heuristic name detection (improves recall for non-Western names)
        text = self._detect_names_heuristic(text, candidate_id, vault)

        # 8. Neutralize gendered pronouns and honorifics
        text = self._neutralize_pronouns(text)

        # 9. Post-NER cleanup
        text = self._post_cleanup(text, candidate_id, vault)

        # 10. Final whitespace normalization
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        text = text.strip()

        # 11. Validation pass — re-check for leaked PII
        validation = validate_pii_removal(text)
        vault['validation_score'] = validation
        if validation['confidence_score'] < 80:
            logger.warning(
                "Validation confidence %.1f%% (< 80%%) for %s — leaks: %s",
                validation['confidence_score'], candidate_id, validation['leaks_found']
            )

        # Warn if no PII found at all (might be pre-anonymized or parse failure)
        total_pii = sum(vault['pii_categories_found'].values())
        if total_pii == 0 and vault['original_name'] is None:
            logger.warning("No PII detected in %s — may be pre-anonymized or parse failure", candidate_id)

        return {'anonymized_cv': text, 'vault_data': vault}

    # ── Name detection (heuristic + NER cross-validation) ──

    def _detect_and_redact_name(
        self, text: str, candidate_id: str, vault: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the candidate's name using two strategies:
          1. Heuristic: first non-header line (CVs universally start with the name).
          2. NER cross-validation: confirm with spaCy PERSON entity on the first few lines.
        """
        first_line = text.strip().split('\n')[0].strip()
        name_from_first_line: Optional[str] = None

        if (first_line
                and len(first_line) < 60
                and not re.search(r'[@|:;/\\#]', first_line)
                and first_line.upper() not in self._HEADER_TITLES):
            name_from_first_line = first_line

        # NER cross-validation on first 3 lines (skip if no model)
        ner_persons = []
        if self.nlp is not None:
            head = '\n'.join(text.strip().split('\n')[:3])
            doc = self.nlp(head)
            ner_persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

        # Decide which name to use
        chosen_name: Optional[str] = None
        if name_from_first_line and ner_persons:
            # Both agree → high confidence
            chosen_name = name_from_first_line
        elif name_from_first_line:
            # Heuristic only (NER didn't confirm) — stricter validation:
            # require 2+ words and reject single all-caps words (merged headers)
            words = name_from_first_line.split()
            if (2 <= len(words) <= 5
                    and re.match(r'^[A-Za-z\s.\-]+$', name_from_first_line)
                    and not (len(words) == 1 and name_from_first_line.isupper())):
                chosen_name = name_from_first_line
        elif ner_persons:
            # NER found a person in header but first line wasn't a name
            chosen_name = ner_persons[0]

        if chosen_name:
            vault['original_name'] = chosen_name
            text = text.replace(chosen_name, f'[{candidate_id}]')
            # Also replace ALL-CAPS version for CVs that use uppercase names
            if chosen_name != chosen_name.upper():
                text = text.replace(chosen_name.upper(), f'[{candidate_id}]')
            logger.debug("Detected candidate name: '%s'", chosen_name)

        return text, vault

    # ── Contact stripping ─────────────────────────

    def _strip_contacts(
        self, text: str, vault: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Remove emails, phones, LinkedIn URLs, web URLs, GitHub, and physical addresses."""
        vault.setdefault('pii_categories_found', {})
        vault.setdefault('techniques_applied', [])

        # Emails
        emails = _EMAIL_RE.findall(text)
        if emails:
            vault['contact_info']['emails'] = emails
            vault['pii_categories_found']['emails'] = len(emails)
            for e in emails:
                text = text.replace(e, '[EMAIL_REDACTED]')

        # Phone numbers (require 10-15 digits, skip decimal-embedded numbers)
        raw_phones = _PHONE_RE.findall(text)
        phones = [
            p.strip() for p in raw_phones
            if 10 <= len(re.sub(r'\D', '', p)) <= 15
            and not re.search(r'\d+\.\d+', p)
        ]
        if raw_phones and not phones:
            logger.warning("Phone regex matched %d candidates but all rejected by validation", len(raw_phones))
        if phones:
            vault['contact_info']['phones'] = phones
            vault['pii_categories_found']['phones'] = len(phones)
            for p in phones:
                text = text.replace(p, '[PHONE_REDACTED]')
        vault['techniques_applied'].append('contact_regex')

        # LinkedIn (full URLs)
        linkedin = _LINKEDIN_RE.findall(text)
        if linkedin:
            vault['contact_info']['linkedin'] = linkedin
            vault['pii_categories_found']['linkedin'] = len(linkedin)
            for li in linkedin:
                text = text.replace(li, '[LINKEDIN_REDACTED]')
        text = _LINKEDIN_PATH_RE.sub('[LINKEDIN_REDACTED]', text)

        # LinkedIn handles without full URL
        li_handles = _LINKEDIN_HANDLE_RE.findall(text)
        if li_handles:
            vault['contact_info'].setdefault('linkedin_handles', [])
            vault['contact_info']['linkedin_handles'].extend(li_handles)
            vault['pii_categories_found']['linkedin_handles'] = len(li_handles)
            text = _LINKEDIN_HANDLE_RE.sub('[LINKEDIN_REDACTED]', text)

        # GitHub profiles
        gh_users = _GITHUB_RE.findall(text)
        if gh_users:
            vault['contact_info']['github'] = gh_users
            vault['pii_categories_found']['github'] = len(gh_users)
        text = _GITHUB_RE.sub('[URL_REDACTED]', text)

        # Generic URLs (with protocol)
        urls = _URL_RE.findall(text)
        if urls:
            vault['contact_info']['urls'] = urls
            vault['pii_categories_found']['urls'] = len(urls)
            for u in urls:
                text = text.replace(u, '[URL_REDACTED]')

        # URLs without protocol (github.com/user, portfolio.dev, bit.ly/abc)
        no_proto_urls = _URL_NO_PROTO_RE.findall(text)
        if no_proto_urls:
            vault['contact_info'].setdefault('urls_no_proto', [])
            vault['contact_info']['urls_no_proto'].extend(no_proto_urls)
            vault['pii_categories_found']['urls_no_proto'] = len(no_proto_urls)
        text = _URL_NO_PROTO_RE.sub('[URL_REDACTED]', text)

        # Physical addresses
        addresses = _ADDRESS_RE.findall(text)
        if addresses:
            vault['contact_info']['addresses'] = addresses
            vault['pii_categories_found']['addresses'] = len(addresses)
            for a in addresses:
                text = text.replace(a, '[ADDRESS_REDACTED]')

        return text, vault

    # ── DOB / Age removal ─────────────────────────

    @staticmethod
    def _remove_dob_age(text: str) -> str:
        text = _DOB_RE.sub('[DOB_REDACTED]', text)
        text = _AGE_RE.sub('', text)
        return text

    # ── Identity-proxy replacement ────────────────

    @staticmethod
    def _replace_proxies(text: str) -> str:
        for pattern, replacement in _PROXY_REPLACEMENTS:
            text = pattern.sub(replacement, text)
        return text

    # ── Education org helper ──────────────────────

    @staticmethod
    def _is_education_org(name: str) -> bool:
        lower = name.lower()
        return any(kw in lower for kw in _EDUCATION_KEYWORDS)

    def _ner_pass(
        self, text: str, candidate_id: str, vault: Dict[str, Any],
        level: str = 'STRICT'
    ) -> Tuple[str, Dict[str, Any]]:
        """Use spaCy NER to detect and redact PERSON, ORG, DATE, GPE/LOC, NORP entities."""
        doc = self.nlp(text)
        replacements: List[Tuple[int, int, str]] = []

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                ent_lower = ent.text.lower().strip()
                ent_words = {w.strip().lower() for w in re.split(r'[\s,/\\]+', ent_lower) if w.strip()}
                if ent_lower in _TECH_WHITELIST or ent_words & _TECH_WHITELIST:
                    continue
                if vault['original_name'] is None:
                    vault['original_name'] = ent.text
                replacements.append((ent.start_char, ent.end_char, f'[{candidate_id}]'))
                vault['pii_categories_found']['ner_persons'] = \
                    vault['pii_categories_found'].get('ner_persons', 0) + 1

            elif ent.label_ == 'ORG':
                if level == 'MINIMAL':
                    continue  # MINIMAL mode keeps organizations
                if self._is_education_org(ent.text):
                    replacements.append((ent.start_char, ent.end_char, '[Educational Institution]'))

            elif ent.label_ == 'DATE':
                if level == 'MINIMAL':
                    continue  # MINIMAL mode keeps all dates
                if re.search(r'\b(19|20)\d{2}\b', ent.text):
                    if re.search(r'(present|current|to|–|-)', ent.text, re.I):
                        continue
                    # Context-aware: keep month-year formats for work timelines
                    # A date is "month-year only" if it has no day component
                    _month_year_only = re.match(
                        r'^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
                        r'Dec(?:ember)?|Q[1-4])\s+\d{4}$',
                        ent.text.strip(), re.I
                    )
                    if _month_year_only:
                        continue  # preserve month-year for timelines
                    context_start = max(0, ent.start_char - 80)
                    context = text[context_start:ent.start_char].lower()
                    _keep_kw = ('experience', 'work', 'position', 'role', 'employed',
                                'joined', 'education', 'graduated', 'degree', 'bachelor',
                                'master', 'intern', 'project')
                    if any(kw in context for kw in _keep_kw):
                        continue
                    replacements.append((ent.start_char, ent.end_char, ''))

            elif ent.label_ in ('GPE', 'LOC'):
                ent_lower = ent.text.lower().strip()
                if ent_lower in _GPE_WHITELIST or ent_lower in _TECH_WHITELIST:
                    continue
                # GPE granularity by anonymization level
                if level == 'MINIMAL':
                    continue  # MINIMAL keeps all locations
                if level == 'MODERATE' and ent_lower in _COUNTRY_NAMES:
                    continue  # MODERATE keeps country names
                # Boundary safety: ensure entity isn't part of a larger token
                before_char = text[ent.start_char - 1] if ent.start_char > 0 else ' '
                after_char = text[ent.end_char] if ent.end_char < len(text) else ' '
                if before_char.isalnum() or after_char.isalnum():
                    continue
                replacements.append((ent.start_char, ent.end_char, '[LOCATION_REDACTED]'))

            elif ent.label_ == 'NORP':
                ent_stripped = ent.text.lower().strip().rstrip('+#')
                if ent_stripped in _NORP_WHITELIST or ent.text.strip() in ('C', 'C++', 'C#', 'R'):
                    continue
                if level == 'MINIMAL':
                    continue  # MINIMAL keeps NORP
                replacements.append((ent.start_char, ent.end_char, '[REDACTED]'))

        # Deduplicate overlapping spans, then apply in reverse order
        replacements = self._deduplicate_spans(replacements)
        replacements.sort(key=lambda r: r[0], reverse=True)
        for start, end, repl in replacements:
            text = text[:start] + repl + text[end:]

        return text, vault

    # ── Experience duration redaction ─────────────

    @staticmethod
    def _redact_experience_duration(text: str, vault: Dict[str, Any]) -> str:
        """Replace experience duration phrases to prevent age inference.

        Mapping: 0-2y → Early career, 3-7y → Professional,
                 8-15y → Extensive, 16+y → Significant.
        """
        def _map_years(match: re.Match) -> str:
            years = int(match.group(1))
            if years <= 2:
                return "Early career experience"
            if years <= 7:
                return "Professional experience"
            if years <= 15:
                return "Extensive experience"
            return "Significant experience"

        new_text = _EXPERIENCE_DURATION_RE.sub(_map_years, text)
        if new_text != text:
            vault['pii_categories_found']['experience_duration'] = \
                len(_EXPERIENCE_DURATION_RE.findall(text))
            vault['techniques_applied'].append('experience_duration')
        return new_text

    # ── Publication / co-author redaction ─────────

    @staticmethod
    def _redact_publications(text: str, vault: Dict[str, Any]) -> str:
        """Detect and anonymize publication references and co-author names."""
        matches = _PUBLICATION_RE.findall(text)
        if matches:
            vault['publications'].extend(matches)
            vault['pii_categories_found']['publications'] = len(matches)
            vault['techniques_applied'].append('publication_redaction')
        # Replace full match (including co-author names) with generic phrase
        text = _PUBLICATION_RE.sub(
            lambda m: re.sub(
                r'\s+(?:with|by|and)\s+.*$', '',
                m.group(0), flags=re.I
            ) if re.search(r'(?:with|by|and)\s+', m.group(0), re.I)
            else m.group(0),
            text
        )
        return text

    # ── Full-date redaction (keep month-year) ─────

    @staticmethod
    def _redact_full_dates(text: str, vault: Dict[str, Any]) -> str:
        """Remove full dates with day component; keep month-year formats."""
        count = 0
        # dd/mm/yyyy, mm-dd-yyyy style
        new_text = _FULL_DATE_RE.sub('[DATE_REDACTED]', text)
        count += len(_FULL_DATE_RE.findall(text))
        # "March 15, 2020" or "15 March 2020" style
        text = new_text
        new_text = _NAMED_FULL_DATE_RE.sub('[DATE_REDACTED]', text)
        count += len(_NAMED_FULL_DATE_RE.findall(text))
        if count > 0:
            vault['pii_categories_found']['full_dates'] = count
            vault['techniques_applied'].append('full_date_redaction')
        return new_text

    # ── Degree-context institution detection ──────

    @staticmethod
    def _redact_degree_institutions(text: str, vault: Dict[str, Any]) -> str:
        """Detect university names in degree context even if NER misses them.

        E.g. "Bachelor's from Stanford" → "Bachelor's from [Educational Institution]"
        """
        matches = _DEGREE_CONTEXT_RE.findall(text)
        if matches:
            vault['pii_categories_found']['degree_institutions'] = len(matches)
            vault['techniques_applied'].append('degree_context')
        text = _DEGREE_CONTEXT_RE.sub(
            lambda m: m.group(0).replace(m.group(1), '[Educational Institution]'),
            text
        )
        return text

    # ── Heuristic name detection (always-on) ──────

    def _detect_names_heuristic(
        self, text: str, candidate_id: str, vault: Dict[str, Any]
    ) -> str:
        """Scan for capitalized multi-word patterns that look like person names.

        Runs AFTER NER to catch names NER missed (especially non-Western names).
        Deduplicates against already-known name and tech/GPE whitelists.
        """
        # Pattern: 2-4 consecutive capitalized words not at line start after a bullet/header
        _name_pat = re.compile(
            r'(?<=[,;:\-–—]\s)([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15}){1,3})(?=[\s,;.\n])'
        )
        known_name = vault.get('original_name', '')
        all_whitelist = _TECH_WHITELIST | _GPE_WHITELIST | {
            w.lower() for w in self._HEADER_TITLES
        }

        found_names = set()
        for m in _name_pat.finditer(text):
            candidate = m.group(1)
            words = candidate.lower().split()
            # Skip if any word is in whitelist or is the known name
            if any(w in all_whitelist for w in words):
                continue
            if known_name and candidate.lower() == known_name.lower():
                continue
            # Skip common section headers
            if candidate.upper() in self._HEADER_TITLES:
                continue
            found_names.add(candidate)

        for name in found_names:
            text = text.replace(name, f'[{candidate_id}]')

        if found_names:
            vault['pii_categories_found']['heuristic_names'] = len(found_names)
            vault['techniques_applied'].append('heuristic_name_detection')

        return text


    @staticmethod
    def _deduplicate_spans(
        replacements: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """Remove overlapping spans — the longer (or earlier) span wins."""
        if not replacements:
            return replacements
        # Sort by start, then longest span first
        replacements.sort(key=lambda r: (r[0], -(r[1] - r[0])))
        result = [replacements[0]]
        for start, end, repl in replacements[1:]:
            _, prev_end, _ = result[-1]
            if start >= prev_end:
                result.append((start, end, repl))
        return result

    # ── Pronoun neutralization ────────────────────

    @staticmethod
    def _neutralize_pronouns(text: str) -> str:
        for pattern, replacement in _PRONOUN_MAP.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r'  +', ' ', text)
        return text

    # ── Post-NER cleanup ──────────────────────────

    @staticmethod
    def _post_cleanup(text: str, candidate_id: str, vault: Dict[str, Any]) -> str:
        """Final cleanup: St. fragments, doubled tags, address cities, trailing names."""
        # "St. Name's" fragments on education lines
        text = _ST_PREFIX_RE.sub('[Educational Institution] ', text)

        # Doubled [Educational Institution] patterns
        text = re.sub(
            r'\[Educational Institution\]\s*\'?s?\s*\[Educational Institution\]',
            '[Educational Institution]', text
        )
        text = re.sub(
            r"\[Educational Institution\]\s*'s\s*",
            '[Educational Institution] ', text
        )

        # City names on address lines (first 5 lines of CV)
        lines = text.split('\n')
        for i in range(min(5, len(lines))):
            m = _ADDRESS_LOCATION_RE.match(lines[i])
            if (m
                    and m.group(1).lower() not in _GPE_WHITELIST
                    and m.group(1).lower() not in _TECH_WHITELIST):
                lines[i] = _ADDRESS_LOCATION_RE.sub('[LOCATION_REDACTED]', lines[i])
        text = '\n'.join(lines)

        # Standalone education years
        text = re.sub(
            r'((?:Secondary|SSLC|HSC|XII|X|Graduation|B\.?Tech|M\.?Tech|B\.?E|M\.?E|'
            r'B\.?Sc|M\.?Sc|MBA|BCA|MCA|CGPA|GPA|%)[^\n]*?),?\s*\b((?:19|20)\d{2})\b'
            r'(?:\s*[-–—]\s*\b((?:19|20)\d{2})\b)?',
            lambda m: m.group(1) + (', [YEAR_REDACTED]' if m.group(3) else ''),
            text, flags=re.I
        )

        # Location on internship/experience lines
        text = re.sub(
            r'^(.*(?:Networks|Pvt|Ltd|Inc|Corp)\s*\.?),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',
            r'\1, [LOCATION_REDACTED]',
            text, flags=re.MULTILINE
        )

        # Trailing name at end of CV
        last_line = text.strip().rsplit('\n', 1)[-1].strip()
        if (vault.get('original_name')
                and last_line
                and len(last_line) < 50
                and not re.search(r'[@|:;/\\#\[\]]', last_line)
                and re.match(r'^[A-Za-z\s.]+$', last_line)):
            words = last_line.split()
            if 1 <= len(words) <= 5:
                text = f'[{candidate_id}]'.join(text.rsplit(last_line, 1))

        return text


# ═════════════════════════════════════════════
# PII Validation Layer
# ═════════════════════════════════════════════

def validate_pii_removal(anonymized_text: str) -> dict:
    """Re-run PII detection patterns on anonymized text to catch leaks.

    Returns:
        dict with 'passed' (bool), 'leaks_found' (list), 'confidence_score' (float 0-100)
    """
    leaks = []

    # Check for leaked emails
    emails = _EMAIL_RE.findall(anonymized_text)
    if emails:
        leaks.append({'type': 'email', 'matches': emails})

    # Check for leaked phone numbers (10+ digits)
    raw_phones = _PHONE_RE.findall(anonymized_text)
    phones = [p for p in raw_phones if 10 <= len(re.sub(r'\D', '', p)) <= 15]
    if phones:
        leaks.append({'type': 'phone', 'matches': phones})

    # Check for leaked URLs (excluding redaction tags)
    urls = [u for u in _URL_RE.findall(anonymized_text)
            if 'REDACTED' not in u]
    if urls:
        leaks.append({'type': 'url', 'matches': urls})

    # Check for leaked full dates (dd/mm/yyyy style)
    full_dates = _FULL_DATE_RE.findall(anonymized_text)
    if full_dates:
        leaks.append({'type': 'full_date', 'matches': full_dates})

    # Check for potential name patterns (2+ capitalized words in sequence)
    # Only flag if at least 3 such occurrences (to reduce false alarms from headers)
    _name_check = re.findall(
        r'(?<!\[)([A-Z][a-z]{2,15}\s+[A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)(?!\])',
        anonymized_text
    )
    # Filter out common headers and known safe patterns
    _safe = {'Career Objective', 'Technical Skills', 'Work Experience',
             'Professional Experience', 'Early Career', 'Educational Institution',
             'Varsity Sports', 'Student Association', 'Professional Engineering',
             'Technology Professional', 'Computing Professional'}
    suspicious_names = [n for n in _name_check if n not in _safe]
    if len(suspicious_names) > 3:
        leaks.append({'type': 'potential_names', 'matches': suspicious_names[:5]})

    # Confidence scoring: each leak type reduces confidence by 15%
    confidence = max(0.0, 100.0 - (len(leaks) * 15))

    return {
        'passed': len(leaks) == 0,
        'leaks_found': leaks,
        'confidence_score': confidence,
    }


# ═════════════════════════════════════════════
# File readers (TXT, PDF, DOCX)
# ═════════════════════════════════════════════

def read_cv_file(filepath: str) -> str:
    """
    Read CV content from .txt, .pdf, or .docx files.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: for unsupported formats (including legacy .doc).
        ImportError: if a required library is missing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CV file not found: '{filepath}'")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    elif ext == '.pdf':
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF support. Install: pip install PyPDF2>=3.0.0"
            )

        reader = PdfReader(filepath)
        # Try fast text extraction first (works for digitally-created PDFs)
        extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
        if extracted and extracted.strip():
            return extracted

        # Fallback: try Tesseract OCR for scanned/image PDFs
        logger.info("PDF has no embedded text, attempting OCR fallback for: %s", filepath)
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning(
                "OCR dependencies not available (pdf2image, pytesseract, or Pillow). "
                "Install: pip install pytesseract pdf2image pillow poppler-utils"
            )
            return ""  # Return empty string to allow processing to continue

        try:
            images = convert_from_path(filepath, dpi=300, poppler_path=_poppler_path)
        except Exception as exc:
            # Check if it's a Poppler-not-found error
            if "poppler" in str(exc).lower():
                logger.warning(
                    "Poppler not found for OCR fallback on '%s'. "
                    "Download from: https://github.com/oschwartz10612/poppler-windows/releases "
                    "and extract to C:\\Program Files\\poppler",
                    os.path.basename(filepath)
                )
            else:
                logger.error(
                    "Failed to convert PDF to images for OCR on '%s': %s",
                    os.path.basename(filepath), exc
                )
            return ""  # Return empty string to allow processing to continue

        ocr_pages: List[str] = []
        for page_num, img in enumerate(images, 1):
            try:
                ocr_text = pytesseract.image_to_string(img)
                ocr_pages.append(ocr_text)
            except Exception as exc:
                logger.warning("Tesseract OCR failed on page %d of '%s': %s",
                               page_num, os.path.basename(filepath), exc)
                # Continue with remaining pages
                ocr_pages.append('')

        result = '\n'.join(ocr_pages).strip()
        if result:
            logger.info("OCR extracted %d chars from %s", len(result), os.path.basename(filepath))
            return result
        else:
            logger.warning("OCR produced no text from '%s'", os.path.basename(filepath))
            return ""

    elif ext == '.docx':
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX support. Install: pip install python-docx>=0.8.11"
            )
        doc = Document(filepath)
        return '\n'.join(para.text for para in doc.paragraphs)

    elif ext == '.doc':
        raise ValueError(
            f"Legacy .doc format is not supported by python-docx. "
            f"Please convert '{os.path.basename(filepath)}' to .docx first "
            f"(use LibreOffice: soffice --convert-to docx \"{filepath}\")."
        )

    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


# ═════════════════════════════════════════════
# Batch processing — called by main.py or standalone
# ═════════════════════════════════════════════

def run(
    input_dir: str,
    output_dir: str,
    vault_dir: str,
    model: str = 'en_core_web_lg',
    level: str = 'STRICT',
) -> bool:
    """
    Process all CVs in input_dir, save anonymized text to output_dir,
    and vault keys to vault_dir. Returns True on success.
    """
    print("\n" + "=" * 60)
    print("  MODULE 0 — CV ANONYMIZATION")
    print("=" * 60)

    if not os.path.isdir(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        logger.warning("Created input folder: %s", input_dir)
        print(f"\n  [!] Created input folder: {input_dir}")
        print(f"      Place your CV files (.txt/.pdf/.docx) inside and re-run.")
        return False

    cv_files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )
    total = len(cv_files)

    if total == 0:
        logger.warning("No CV files found in: %s", input_dir)
        print(f"\n  [!] No CV files found in: {input_dir}")
        print(f"      Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return False

    print(f"\n  Input:   {input_dir}  ({total} CVs)")
    print(f"  Output:  {output_dir}")
    print(f"  Vault:   {vault_dir}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vault_dir, exist_ok=True)

    print(f"\n  Loading NER model: {model}")
    anonymizer = CVAnonymizer(spacy_model=model)
    print()

    master_vault: Dict[str, Any] = {}
    success = 0
    failed_files: List[str] = []
    t0 = time.time()

    # Progress bar setup
    try:
        from tqdm import tqdm
        progress_iter = tqdm(enumerate(cv_files, start=1), total=total, desc="Anonymizing CVs")
    except ImportError:
        progress_iter = enumerate(cv_files, start=1)

    error_log_path = os.path.join(output_dir, "error_log.txt")
    with open(error_log_path, "w", encoding="utf-8") as error_log:

        for idx, fname in progress_iter:
            candidate_id = f"Candidate_{idx:02d}"
            cv_path = os.path.join(input_dir, fname)
            try:
                cv_text = read_cv_file(cv_path)
                if not cv_text.strip():
                    logger.warning("Skipped empty file: %s", fname)
                    error_log.write(f"SKIPPED (empty): {fname}\n")
                    continue

                result = anonymizer.anonymize(cv_text, candidate_id=candidate_id, level=level)

                with open(os.path.join(output_dir, f"{candidate_id}.txt"), 'w', encoding='utf-8') as f:
                    f.write(result['anonymized_cv'])

                vault_data = result['vault_data']
                vault_data['original_filename'] = fname
                # Add validation report
                vault_data['validation_report'] = validate_pii_removal(result['anonymized_cv'])
                # Standardize vault keys
                for key in ['candidate_id', 'original_name', 'contact_info', 'publications', 'pii_categories_found', 'anonymization_level_used', 'techniques_applied', 'validation_score', 'original_filename', 'validation_report']:
                    if key not in vault_data:
                        vault_data[key] = None
                with open(os.path.join(vault_dir, f"{candidate_id}.json"), 'w', encoding='utf-8') as f:
                    json.dump(vault_data, f, indent=2, ensure_ascii=False)

                master_vault[candidate_id] = vault_data
                success += 1
            except Exception as exc:
                failed_files.append(fname)
                logger.error("Failed to process '%s': %s", fname, exc, exc_info=True)
                error_log.write(f"FAILED ({fname}): {exc}\n")


    # Master vault
    master_path = os.path.join(vault_dir, "vault_master.json")
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(master_vault, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Done: {success}/{total} CVs anonymized in {elapsed:.1f}s")
    if failed_files:
        print(f"  Failed: {', '.join(failed_files)}")
        print(f"  See error log: {error_log_path}")
    print(f"  Vault: {master_path}")
    print("=" * 60)
    return success > 0


# ═════════════════════════════════════════════
# Standalone execution
# ═════════════════════════════════════════════

def _configure_logging(verbose: bool = False) -> None:
    """Configure logging for standalone execution."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='Module 0 — CV Anonymization (standalone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python module0.py --input-dir my_resumes --verbose',
    )
    parser.add_argument('--input-dir',  default=os.path.join(BASE, 'raw_cvs'),
                        help='Directory containing raw CV files (default: raw_cvs/)')
    parser.add_argument('--output-dir', default=os.path.join(BASE, 'anonymized_cvs'),
                        help='Directory for anonymized output (default: anonymized_cvs/)')
    parser.add_argument('--vault-dir',  default=os.path.join(BASE, 'vault'),
                        help='Directory for vault keys (default: vault/)')
    parser.add_argument('--model',      default='en_core_web_trf',
                        help='spaCy model to use (default: en_core_web_trf)')
    parser.add_argument('--level',      default='STRICT', choices=['STRICT', 'MODERATE', 'MINIMAL'],
                        help='Anonymization level: STRICT, MODERATE, MINIMAL (default: STRICT)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    ok = run(args.input_dir, args.output_dir, args.vault_dir, args.model, args.level)
    if not ok:
        sys.exit(1)
