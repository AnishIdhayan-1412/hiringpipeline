# -*- coding: utf-8 -*-
"""
skills_taxonomy.py — Single Source of Truth for Skills Taxonomy
===============================================================
Bias-Free Hiring Pipeline

This module is the authoritative definition of all recognised skills used by
the pipeline.  Both module0b (CV parser / extractor) and module1 (ranker)
import from here so that extraction and scoring always operate on an
identical skill set.

Previously, module0b held SKILLS_TAXONOMY (dict, 481 skills, 4 categories)
and module1 held a hand-written SKILLS_TAXONOMY_FLAT that had already begun
to drift.  This file eliminates that duplication.

Exports
-------
SKILLS_TAXONOMY      : Dict[str, List[str]]
    Full taxonomy keyed by category:
    'technical', 'soft', 'tools', 'languages'.

SKILLS_TAXONOMY_FLAT : List[str]
    Flat, case-preserving, deduplicated list derived automatically from
    SKILLS_TAXONOMY (all four categories).
    Used by module1 for O(n) JD keyword scanning.

_SKILL_ALIASES       : Dict[str, str]
    Alias → canonical-skill mapping populated at import time by
    _build_alias_map().  Used by module0b for fuzzy skill matching.
"""

from __future__ import annotations

from typing import Dict, List

__version__ = "1.0.0"
__all__ = ["SKILLS_TAXONOMY", "SKILLS_TAXONOMY_FLAT", "_SKILL_ALIASES"]

# ═══════════════════════════════════════════════════════════════════════════
# SKILLS TAXONOMY — single source of truth
# Four categories: technical, soft, tools, languages.
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

# ═══════════════════════════════════════════════════════════════════════════
# SKILLS_TAXONOMY_FLAT — auto-generated from SKILLS_TAXONOMY
# Never edit this manually; edit SKILLS_TAXONOMY above instead.
# Preserves first-seen casing; deduplication is case-insensitive.
# ═══════════════════════════════════════════════════════════════════════════
def _make_flat_list() -> List[str]:
    """Return a deduplicated flat list of all skills in SKILLS_TAXONOMY."""
    seen: set = set()
    result: List[str] = []
    for skills in SKILLS_TAXONOMY.values():
        for skill in skills:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                result.append(skill)
    return result


SKILLS_TAXONOMY_FLAT: List[str] = _make_flat_list()

# ═══════════════════════════════════════════════════════════════════════════
# _SKILL_ALIASES — alias → canonical skill name
# Populated at import time by _build_alias_map().
# Used by module0b for fuzzy skill matching during CV parsing.
# ═══════════════════════════════════════════════════════════════════════════
_SKILL_ALIASES: Dict[str, str] = {}


def _build_alias_map() -> None:
    """Populate _SKILL_ALIASES from SKILLS_TAXONOMY at import time."""
    _manual_aliases: Dict[str, str] = {
        "js":                    "JavaScript",
        "ts":                    "TypeScript",
        "k8s":                   "Kubernetes",
        "tf":                    "TensorFlow",
        "sklearn":               "Scikit-learn",
        "scikit":                "Scikit-learn",
        "hf":                    "HuggingFace",
        "hugging face":          "HuggingFace",
        "node":                  "Node.js",
        "nodejs":                "Node.js",
        "vue.js":                "Vue",
        "vuejs":                 "Vue",
        "reactjs":               "React",
        "react.js":              "React",
        "angular.js":            "Angular",
        "angularjs":             "Angular",
        "postgres":              "PostgreSQL",
        "mongo":                 "MongoDB",
        "mssql":                 "SQL Server",
        "gh actions":            "GitHub Actions",
        "gha":                   "GitHub Actions",
        "tailwind":              "Tailwind CSS",
        "pb":                    "Power BI",
        "powerbi":               "Power BI",
        "spacy":                 "spaCy",
        "pytorch":               "PyTorch",
        "tensorflow":            "TensorFlow",
        "gcp":                   "GCP",
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
