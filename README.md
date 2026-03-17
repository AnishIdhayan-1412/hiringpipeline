# Bias-Free Hiring Pipeline

An end-to-end, bias-aware recruitment pipeline that anonymises CVs, ranks candidates against a Job Description, explains rankings, and logs a GDPR-compliant audit trail — all surfaced through a lightweight Flask dashboard.

---

## Pipeline overview

| Stage | Module | What it does |
|-------|--------|--------------|
| 0 | `module0` | PII removal & pronoun neutralisation → `anonymized_cvs/` |
| 0b | `module0b` | Section parsing & skill extraction → `parsed/` |
| 1 | `module1` | SBERT semantic ranking → `ranking_report.txt` |
| 2 | `module2` | Explainability engine → `explanations/` |
| 3 | `module3` | GDPR audit trail → `audit/` |
| 4 | `module4` | Bias detection → `audit/bias_*.txt / .json` |
| 5 | `module5` | Flask dashboard → `http://localhost:5000` |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_trf

# 2. Place raw CVs (PDF or DOCX) in raw_cvs/
# 3. Edit jd.txt with the Job Description
# 4. Run the full pipeline
python main.py --jd-file jd.txt

# 5. (Optional) also launch the dashboard
python main.py --jd-file jd.txt --dashboard
```

### Common flags

```
--input-dir   PATH    Folder containing raw CVs  (default: raw_cvs/)
--jd-file     FILE    Job description text file  (default: jd.txt)
--verbose             Enable DEBUG-level logging
--skip-stage  N       Skip a numbered stage (repeatable)
--dashboard           Launch Flask dashboard after the pipeline finishes
```

---

## Troubleshooting

### Stray file named `argparse.ArgumentParser` in the project root

**What it is**

`argparse.ArgumentParser` is the name of a Python class in the standard
library's `argparse` module — **not** a real filename.  It ends up on disk
when a shell command is mistyped so that the class reference is interpreted
as an output filename, for example:

```powershell
# Accidental redirect — output is written to a file called "argparse.ArgumentParser"
python main.py > argparse.ArgumentParser
```

or when a one-liner is pasted incorrectly:

```cmd
python -c "import argparse; argparse.ArgumentParser" > argparse.ArgumentParser
```

**Why it must not exist**

* It is not a valid Python source file and will confuse editors and import
  resolvers.
* Its presence alongside `argparse` (a stdlib module name root) can shadow
  the real `argparse` module in some environments.
* It clutters the project root with meaningless binary/text noise.

**How to delete it on Windows**

Open **Command Prompt** (`cmd`) in the project root and run:

```cmd
del "argparse.ArgumentParser"
```

Or in **PowerShell**:

```powershell
Remove-Item "argparse.ArgumentParser"
```

Both commands permanently delete the file.  The file is also listed in
`.gitignore` so it will never be committed to the repository even if it is
accidentally re-created.
