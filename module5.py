# -*- coding: utf-8 -*-
"""
module5.py — Web Dashboard
============================
Bias-Free Hiring Pipeline — Stage 5

A production-ready Flask web application that gives HR staff a clean,
browser-based interface to view pipeline results, manage candidate files,
trigger pipeline runs, and verify GDPR compliance — all without touching
the command line.

Pages
-----
    /                       — Dashboard: pipeline status + summary stats
    /rankings               — Ranked candidates table with filters
    /candidate/<id>         — Full candidate detail with score breakdown
    /audit                  — GDPR compliance + bias audit results
    /upload                 — CV/JD upload + live pipeline execution

API endpoints
-------------
    POST /api/upload-cvs        — Save uploaded CVs to raw_cvs/
    POST /api/upload-jd         — Save JD text/file to jd.txt
    POST /api/run-pipeline      — Run main.py, stream logs via SSE
    GET  /api/run-status        — Current pipeline run state
    GET  /api/download/ranking-csv   — Export shortlist as CSV
    GET  /api/download/audit-report  — Download audit_report.txt
    GET  /api/download/bias-report   — Download bias_report.txt

Standalone usage
----------------
    python module5.py
    python module5.py --port 8080
    python module5.py --port 5000 --debug

Programmatic usage (called by main.py)
---------------------------------------
    import module5
    module5.run()                          # blocks — starts the server

Dependencies
------------
    pip install flask
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    from flask import (
        Flask,
        Response,
        jsonify,
        redirect,
        render_template,
        request,
        send_file,
        stream_with_context,
        url_for,
    )
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
# Module-level logger
# ══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR:           str = os.path.dirname(os.path.abspath(__file__))

# ── Pipeline directory paths (all relative to BASE_DIR) ──────────────────
RAW_CVS_DIR:        str = os.path.join(BASE_DIR, "raw_cvs")
ANONYMIZED_DIR:     str = os.path.join(BASE_DIR, "anonymized_cvs")
PARSED_DIR:         str = os.path.join(BASE_DIR, "parsed")
VAULT_DIR:          str = os.path.join(BASE_DIR, "vault")
EXPLANATIONS_DIR:   str = os.path.join(BASE_DIR, "explanations")
AUDIT_DIR:          str = os.path.join(BASE_DIR, "audit")
TEMPLATES_DIR:      str = os.path.join(BASE_DIR, "templates")

# ── Key data files ────────────────────────────────────────────────────────
JD_FILE:            str = os.path.join(BASE_DIR, "jd.txt")
RANKING_REPORT:     str = os.path.join(BASE_DIR, "ranking_report.txt")
PARSED_INDEX:       str = os.path.join(PARSED_DIR,  "index.json")
COMPLIANCE_FILE:    str = os.path.join(AUDIT_DIR,   "compliance_checklist.json")
AUDIT_LOG_FILE:     str = os.path.join(AUDIT_DIR,   "audit_log.json")
AUDIT_REPORT_FILE:  str = os.path.join(AUDIT_DIR,   "audit_report.txt")
BIAS_AUDIT_FILE:    str = os.path.join(AUDIT_DIR,   "bias_audit.json")
BIAS_REPORT_FILE:   str = os.path.join(AUDIT_DIR,   "bias_report.txt")
SHA256_FILE:        str = os.path.join(AUDIT_DIR,   "audit_log.sha256")

# ── Allowed upload extensions ─────────────────────────────────────────────
ALLOWED_CV_EXTENSIONS: frozenset = frozenset({".pdf", ".docx", ".txt"})

# ── Server defaults ───────────────────────────────────────────────────────
DEFAULT_HOST:   str = "127.0.0.1"
DEFAULT_PORT:   int = 5000

# ── Verdict colour mapping ────────────────────────────────────────────────
VERDICT_COLOURS: Dict[str, str] = {
    "Strong Match":  "#4CAF50",
    "Good Match":    "#2196F3",
    "Partial Match": "#FF9800",
    "Weak Match":    "#f44336",
}

# ── Logging ───────────────────────────────────────────────────────────────
_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT:   str = "%H:%M:%S"

# ── Pipeline run state (shared across threads) ────────────────────────────
_pipeline_state: Dict[str, Any] = {
    "running":   False,
    "last_run":  None,
    "exit_code": None,
}
_log_queue: Queue = Queue(maxsize=2000)


# ══════════════════════════════════════════════════════════════════════════
# DataLoader — all JSON reading isolated here, always safe
# ══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Reads all pipeline output files and returns safe, typed structures.

    Every method returns a sane default (empty dict or list) when files are
    absent, malformed, or unreadable.  The dashboard never crashes because a
    pipeline stage hasn't run yet.
    """

    @staticmethod
    def _read_json(path: str) -> Any:
        """Load a JSON file safely.

        Args:
            path: Absolute path to the JSON file.

        Returns:
            Parsed Python object, or None on any failure.
        """
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return None

    def load_rankings(self) -> List[Dict[str, Any]]:
        """Load all candidate explanation JSONs and return a ranked list.

        Reads every Candidate_XX.json from explanations/, skips non-candidate
        files, and sorts by rank ascending.

        Returns:
            List of explanation dicts sorted by rank. Empty list if no files.
        """
        results: List[Dict[str, Any]] = []
        if not os.path.isdir(EXPLANATIONS_DIR):
            return results

        skip = {"summary_report.txt", "error_log.txt", "error_log.json"}
        for fname in os.listdir(EXPLANATIONS_DIR):
            if fname in skip or not fname.endswith(".json"):
                continue
            data = self._read_json(os.path.join(EXPLANATIONS_DIR, fname))
            if isinstance(data, dict) and "candidate_id" in data:
                results.append(data)

        results.sort(key=lambda x: x.get("rank", 9999))
        return results

    def load_candidate(self, candidate_id: str) -> Dict[str, Any]:
        """Load a single candidate's explanation JSON.

        Args:
            candidate_id: Candidate identifier, e.g. "Candidate_01".

        Returns:
            Explanation dict, or empty dict if not found.
        """
        path = os.path.join(EXPLANATIONS_DIR, f"{candidate_id}.json")
        data = self._read_json(path)
        return data if isinstance(data, dict) else {}

    def load_audit(self) -> Dict[str, Any]:
        """Load compliance_checklist.json and audit_log.json.

        Returns:
            Dict with keys: compliance, audit_log, sha256, retention.
        """
        compliance = self._read_json(COMPLIANCE_FILE) or {}
        audit_log  = self._read_json(AUDIT_LOG_FILE)  or {}
        sha256     = ""

        if os.path.isfile(SHA256_FILE):
            try:
                with open(SHA256_FILE, "r", encoding="utf-8") as fh:
                    sha256 = fh.read().split()[0]
            except (OSError, IndexError):
                sha256 = ""

        retention = audit_log.get("data_retention", {})
        return {
            "compliance":  compliance,
            "audit_log":   audit_log,
            "sha256":      sha256,
            "retention":   retention,
        }

    def load_bias(self) -> Dict[str, Any]:
        """Load bias_audit.json.

        Returns:
            Bias audit dict, or empty dict with safe defaults.
        """
        data = self._read_json(BIAS_AUDIT_FILE)
        if not isinstance(data, dict):
            return {
                "overall_flag":  False,
                "overall_label": "No bias audit data",
                "flags_raised":  0,
                "checks":        [],
                "n_candidates":  0,
            }
        return data

    def load_pipeline_status(self) -> Dict[str, Any]:
        """Check which pipeline stages have produced output.

        Returns:
            Dict mapping stage name → {done: bool, path: str, label: str}.
        """
        stages = [
            ("anonymization", ANONYMIZED_DIR,  "CV Anonymization",      True),
            ("parsing",       PARSED_DIR,       "Section Parsing",       True),
            ("ranking",       RANKING_REPORT,   "Candidate Ranking",     False),
            ("explainability",EXPLANATIONS_DIR, "Explainability Engine", True),
            ("audit",         AUDIT_DIR,        "GDPR Audit Trail",      True),
            ("bias",          BIAS_AUDIT_FILE,  "Bias Detection",        False),
        ]
        status: Dict[str, Any] = {}
        for key, path, label, is_dir in stages:
            if is_dir:
                done = os.path.isdir(path) and bool(
                    [f for f in os.listdir(path) if not f.startswith(".")]
                    if os.path.isdir(path) else []
                )
            else:
                done = os.path.isfile(path)
            status[key] = {"done": done, "path": path, "label": label}
        return status

    def load_summary_stats(self) -> Dict[str, Any]:
        """Compute top-level summary statistics for the dashboard header cards.

        Returns:
            Dict with total_cvs, top_score, top_candidate, gdpr_status,
            bias_verdict, last_run, jd_loaded.
        """
        rankings   = self.load_rankings()
        audit_data = self.load_audit()
        bias_data  = self.load_bias()

        total_cvs      = len(rankings)
        top_candidate  = rankings[0].get("candidate_id", "—") if rankings else "—"
        top_score      = rankings[0].get("final_score", 0.0)  if rankings else 0.0
        gdpr_compliant = audit_data["compliance"].get("overall_compliant", None)
        gdpr_status    = (
            "Compliant" if gdpr_compliant is True
            else "Not Compliant" if gdpr_compliant is False
            else "Not Run"
        )
        bias_verdict = bias_data.get("overall_label", "Not Run")
        jd_loaded    = os.path.isfile(JD_FILE)

        # Last run from audit log
        last_run = audit_data["audit_log"].get("generated_at", None)

        return {
            "total_cvs":     total_cvs,
            "top_candidate": top_candidate,
            "top_score":     top_score,
            "gdpr_status":   gdpr_status,
            "bias_verdict":  bias_verdict,
            "last_run":      last_run,
            "jd_loaded":     jd_loaded,
        }


# ══════════════════════════════════════════════════════════════════════════
# Pipeline runner — subprocess + SSE streaming
# ══════════════════════════════════════════════════════════════════════════

def _run_pipeline_subprocess(jd_file: Optional[str] = None) -> None:
    """Run main.py in a background thread, feeding stdout into _log_queue.

    Args:
        jd_file: Path to the JD file, or None to use default jd.txt.
    """
    global _pipeline_state

    _pipeline_state["running"]   = True
    _pipeline_state["exit_code"] = None

    cmd = [sys.executable, os.path.join(BASE_DIR, "main.py")]
    jd  = jd_file or JD_FILE
    if os.path.isfile(jd):
        cmd += ["--jd-file", jd]

    logger.info("Launching pipeline: %s", " ".join(cmd))
    _log_queue.put("data: [PIPELINE STARTED]\n\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout    = subprocess.PIPE,
            stderr    = subprocess.STDOUT,
            text      = True,
            encoding  = "utf-8",
            errors    = "replace",
            cwd       = BASE_DIR,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            clean = line.rstrip()
            if clean:
                _log_queue.put(f"data: {clean}\n\n")

        proc.wait()
        _pipeline_state["exit_code"] = proc.returncode
        _pipeline_state["last_run"]  = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        status = "COMPLETED" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        _log_queue.put(f"data: [PIPELINE {status}]\n\n")
        _log_queue.put("data: __DONE__\n\n")

    except Exception as exc:
        logger.error("Pipeline subprocess error: %s", exc)
        _log_queue.put(f"data: [ERROR] {exc}\n\n")
        _log_queue.put("data: __DONE__\n\n")
    finally:
        _pipeline_state["running"] = False


def _sse_generator() -> Generator[str, None, None]:
    """Yield SSE-formatted log lines from _log_queue until pipeline finishes.

    Yields:
        SSE-formatted strings suitable for text/event-stream response.
    """
    while True:
        try:
            msg = _log_queue.get(timeout=30)
            yield msg
            if "__DONE__" in msg:
                break
        except Empty:
            yield "data: [TIMEOUT — pipeline still running]\n\n"
            break


# ══════════════════════════════════════════════════════════════════════════
# Flask app factory
# ══════════════════════════════════════════════════════════════════════════

def _create_app() -> "Flask":
    """Create and configure the Flask application.

    Returns:
        Configured Flask app instance with all routes registered.
    """
    app = Flask(
        __name__,
        template_folder = TEMPLATES_DIR,
    )
    app.secret_key = "bias-free-hiring-pipeline-2024"

    loader = DataLoader()

    # ──────────────────────────────────────────────────────────────────────
    # Template helpers (injected into every template context)
    # ──────────────────────────────────────────────────────────────────────

    @app.context_processor
    def inject_globals() -> Dict[str, Any]:
        """Inject pipeline status and verdict colours into every template."""
        return {
            "pipeline_status": loader.load_pipeline_status(),
            "verdict_colours": VERDICT_COLOURS,
            "now":             datetime.now().strftime("%d %b %Y %H:%M"),
        }

    # ──────────────────────────────────────────────────────────────────────
    # PAGE 1 — Dashboard
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/")
    def dashboard() -> str:
        """Render the main dashboard page with pipeline status and stats.

        Returns:
            Rendered HTML string.
        """
        stats   = loader.load_summary_stats()
        status  = loader.load_pipeline_status()
        running = _pipeline_state["running"]
        return render_template(
            "dashboard.html",
            stats   = stats,
            status  = status,
            running = running,
        )

    # ──────────────────────────────────────────────────────────────────────
    # PAGE 2 — Rankings
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/rankings")
    def rankings() -> str:
        """Render the candidate rankings table.

        Supports ?filter=<verdict> query parameter to filter by verdict.

        Returns:
            Rendered HTML string.
        """
        all_candidates = loader.load_rankings()
        verdict_filter = request.args.get("filter", "all")

        if verdict_filter != "all":
            filtered = [
                c for c in all_candidates
                if c.get("verdict", "").lower().replace(" ", "_")
                == verdict_filter.lower().replace(" ", "_")
            ]
        else:
            filtered = all_candidates

        verdicts = sorted({c.get("verdict", "") for c in all_candidates if c.get("verdict")})

        return render_template(
            "rankings.html",
            candidates     = filtered,
            all_candidates = all_candidates,
            verdicts       = verdicts,
            active_filter  = verdict_filter,
        )

    # ──────────────────────────────────────────────────────────────────────
    # PAGE 3 — Candidate Detail
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/candidate/<candidate_id>")
    def candidate_detail(candidate_id: str) -> str:
        """Render the full detail page for a single candidate.

        Args:
            candidate_id: URL path parameter, e.g. "Candidate_01".

        Returns:
            Rendered HTML string, or redirect to rankings if not found.
        """
        data = loader.load_candidate(candidate_id)
        if not data:
            return redirect(url_for("rankings"))

        all_candidates = loader.load_rankings()
        current_rank   = data.get("rank", 0)
        total          = len(all_candidates)

        prev_id = next_id = None
        for i, c in enumerate(all_candidates):
            if c.get("candidate_id") == candidate_id:
                if i > 0:
                    prev_id = all_candidates[i - 1].get("candidate_id")
                if i < total - 1:
                    next_id = all_candidates[i + 1].get("candidate_id")
                break

        return render_template(
            "candidate.html",
            candidate    = data,
            total        = total,
            prev_id      = prev_id,
            next_id      = next_id,
        )

    # ──────────────────────────────────────────────────────────────────────
    # PAGE 4 — Audit & Compliance
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/audit")
    def audit() -> str:
        """Render the GDPR compliance and bias audit page.

        Returns:
            Rendered HTML string.
        """
        audit_data = loader.load_audit()
        bias_data  = loader.load_bias()

        gdpr_articles = audit_data["compliance"].get("gdpr_articles", {})
        article_order = [
            ("article_5",  "Art. 5",  "Principles of processing"),
            ("article_13", "Art. 13", "Right to information"),
            ("article_17", "Art. 17", "Right to erasure"),
            ("article_22", "Art. 22", "Automated decision-making"),
            ("article_25", "Art. 25", "Privacy by design"),
            ("article_30", "Art. 30", "Records of processing"),
        ]
        articles = []
        for key, short, title in article_order:
            result = gdpr_articles.get(key, {})
            articles.append({
                "key":       key,
                "short":     short,
                "title":     title,
                "satisfied": result.get("satisfied", None),
                "evidence":  result.get("evidence", "No data available."),
            })

        return render_template(
            "audit.html",
            audit_data = audit_data,
            bias_data  = bias_data,
            articles   = articles,
        )

    # ──────────────────────────────────────────────────────────────────────
    # PAGE 5 — Upload & Run
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/upload")
    def upload() -> str:
        """Render the file upload and pipeline execution page.

        Returns:
            Rendered HTML string.
        """
        jd_exists  = os.path.isfile(JD_FILE)
        jd_text    = ""
        if jd_exists:
            try:
                with open(JD_FILE, "r", encoding="utf-8") as fh:
                    jd_text = fh.read()
            except OSError:
                jd_text = ""

        cv_count = 0
        if os.path.isdir(RAW_CVS_DIR):
            cv_count = sum(
                1 for f in os.listdir(RAW_CVS_DIR)
                if os.path.isfile(os.path.join(RAW_CVS_DIR, f))
                and os.path.splitext(f)[1].lower() in ALLOWED_CV_EXTENSIONS
            )

        return render_template(
            "upload.html",
            jd_exists  = jd_exists,
            jd_text    = jd_text,
            cv_count   = cv_count,
            running    = _pipeline_state["running"],
            last_run   = _pipeline_state.get("last_run"),
            exit_code  = _pipeline_state.get("exit_code"),
        )

    # ──────────────────────────────────────────────────────────────────────
    # API: Upload CVs
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/api/upload-cvs", methods=["POST"])
    def api_upload_cvs() -> Response:
        """Accept multiple CV file uploads and save to raw_cvs/.

        Accepts multipart form data with field name 'cvs'.

        Returns:
            JSON response with success, count, and message.
        """
        files = request.files.getlist("cvs")
        if not files:
            return jsonify({"success": False, "message": "No files received."})

        os.makedirs(RAW_CVS_DIR, exist_ok=True)
        saved   = 0
        skipped = 0

        for f in files:
            if not f.filename:
                continue
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in ALLOWED_CV_EXTENSIONS:
                skipped += 1
                continue
            try:
                safe_name = os.path.basename(f.filename)
                f.save(os.path.join(RAW_CVS_DIR, safe_name))
                saved += 1
            except OSError as exc:
                logger.error("Failed to save CV file '%s': %s", f.filename, exc)
                skipped += 1

        msg = f"Saved {saved} CV file(s)."
        if skipped:
            msg += f" {skipped} file(s) skipped (unsupported format)."

        return jsonify({"success": saved > 0, "message": msg, "count": saved})

    # ──────────────────────────────────────────────────────────────────────
    # API: Upload / save JD
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/api/upload-jd", methods=["POST"])
    def api_upload_jd() -> Response:
        """Accept a JD as text body or file upload and write to jd.txt.

        Accepts either:
            - JSON body: {"text": "..."}
            - Multipart form with field "jd_text" (text area)
            - Multipart file upload with field "jd_file"

        Returns:
            JSON response with success and message.
        """
        text = ""

        # Priority 1: JSON body
        if request.is_json:
            text = request.json.get("text", "").strip()  # type: ignore[union-attr]

        # Priority 2: form text area
        elif request.form.get("jd_text"):
            text = request.form["jd_text"].strip()

        # Priority 3: file upload
        elif "jd_file" in request.files:
            jd_file = request.files["jd_file"]
            try:
                text = jd_file.read().decode("utf-8", errors="replace").strip()
            except Exception as exc:
                return jsonify({"success": False, "message": f"Could not read file: {exc}"})

        if not text:
            return jsonify({"success": False, "message": "No JD content received."})

        try:
            with open(JD_FILE, "w", encoding="utf-8") as fh:
                fh.write(text)
            return jsonify({"success": True, "message": "Job description saved."})
        except OSError as exc:
            return jsonify({"success": False, "message": f"Could not save JD: {exc}"})

    # ──────────────────────────────────────────────────────────────────────
    # API: Run pipeline (SSE streaming)
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/api/run-pipeline", methods=["POST"])
    def api_run_pipeline() -> Response:
        """Trigger a pipeline run and stream stdout as Server-Sent Events.

        If the pipeline is already running, returns an error immediately.

        Returns:
            SSE stream (text/event-stream) with live log lines.
        """
        if _pipeline_state["running"]:
            return Response(
                "data: [Pipeline already running]\n\ndata: __DONE__\n\n",
                mimetype="text/event-stream",
            )

        # Drain stale log queue
        while not _log_queue.empty():
            try:
                _log_queue.get_nowait()
            except Empty:
                break

        thread = threading.Thread(
            target   = _run_pipeline_subprocess,
            kwargs   = {"jd_file": JD_FILE if os.path.isfile(JD_FILE) else None},
            daemon   = True,
            name     = "pipeline-runner",
        )
        thread.start()

        return Response(
            stream_with_context(_sse_generator()),
            mimetype = "text/event-stream",
            headers  = {
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # API: Run status
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/api/run-status")
    def api_run_status() -> Response:
        """Return current pipeline run state as JSON.

        Returns:
            JSON with running, last_run, exit_code.
        """
        return jsonify({
            "running":   _pipeline_state["running"],
            "last_run":  _pipeline_state.get("last_run"),
            "exit_code": _pipeline_state.get("exit_code"),
        })

    # ──────────────────────────────────────────────────────────────────────
    # API: Download ranking CSV
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/api/download/ranking-csv")
    def api_download_ranking_csv() -> Response:
        """Generate and return a CSV shortlist of all ranked candidates.

        Returns:
            CSV file response with Content-Disposition header.
        """
        candidates = loader.load_rankings()
        output     = io.StringIO()
        writer     = csv.writer(output)

        writer.writerow([
            "Rank", "Candidate ID", "Final Score", "Verdict",
            "Semantic Score", "Keyword Score", "Quality Score",
            "Matched Skills", "Missing Skills", "Recommendation",
        ])
        for c in candidates:
            exp  = c.get("explanation", {})
            bd   = exp.get("score_breakdown", {})
            sk   = exp.get("skills", {})
            rec  = exp.get("recommendation", "")
            writer.writerow([
                c.get("rank", ""),
                c.get("candidate_id", ""),
                f"{c.get('final_score', 0.0):.4f}",
                c.get("verdict", ""),
                f"{bd.get('semantic_score', {}).get('value', 0.0):.4f}",
                f"{bd.get('keyword_score',  {}).get('value', 0.0):.4f}",
                f"{bd.get('quality_score',  {}).get('value', 0.0):.4f}",
                "; ".join(sk.get("matched", [])),
                "; ".join(sk.get("missing", [])),
                rec,
            ])

        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Response(
            output.getvalue(),
            mimetype = "text/csv",
            headers  = {
                "Content-Disposition":
                    f'attachment; filename="shortlist_{timestamp}.csv"'
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # API: Download audit report
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/api/download/audit-report")
    def api_download_audit_report() -> Response:
        """Return audit_report.txt as a file download.

        Returns:
            File download response, or 404 JSON if file absent.
        """
        if not os.path.isfile(AUDIT_REPORT_FILE):
            return jsonify({"success": False, "message": "Audit report not found."}), 404  # type: ignore[return-value]
        return send_file(
            AUDIT_REPORT_FILE,
            as_attachment = True,
            download_name = "audit_report.txt",
            mimetype      = "text/plain",
        )

    # ──────────────────────────────────────────────────────────────────────
    # API: Download bias report
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/api/download/bias-report")
    def api_download_bias_report() -> Response:
        """Return bias_report.txt as a file download.

        Returns:
            File download response, or 404 JSON if file absent.
        """
        if not os.path.isfile(BIAS_REPORT_FILE):
            return jsonify({"success": False, "message": "Bias report not found."}), 404  # type: ignore[return-value]
        return send_file(
            BIAS_REPORT_FILE,
            as_attachment = True,
            download_name = "bias_report.txt",
            mimetype      = "text/plain",
        )

    return app


# ══════════════════════════════════════════════════════════════════════════
# run() — primary entry point
# ══════════════════════════════════════════════════════════════════════════

def run(
    host:  str  = DEFAULT_HOST,
    port:  int  = DEFAULT_PORT,
    debug: bool = False,
) -> None:
    """Start the Flask web dashboard server.

    This is the canonical entry point called by main.py with no arguments.
    It blocks until the server is stopped (Ctrl+C).

    Args:
        host:  Host to bind to (default: 127.0.0.1).
        port:  Port to listen on (default: 5000).
        debug: Enable Flask debug mode (default: False).
    """
    if not _FLASK_AVAILABLE:
        print("[ERROR] Flask is not installed.")
        print("        Install it with: pip install flask")
        sys.exit(1)

    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    os.makedirs(RAW_CVS_DIR,   exist_ok=True)

    app = _create_app()

    print()
    print("=" * 60)
    print("  MODULE 5 — BIAS-FREE HIRING DASHBOARD")
    print("=" * 60)
    print(f"  URL     : http://{host}:{port}")
    print(f"  Base    : {BASE_DIR}")
    print(f"  Debug   : {debug}")
    print()
    print("  Open your browser at the URL above.")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    print()

    app.run(host=host, port=port, debug=debug, threaded=True)


# ══════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone execution.

    Returns:
        Configured ArgumentParser.
    """
    p = argparse.ArgumentParser(
        prog        = "module5",
        description = "Web Dashboard — Bias-Free Hiring Pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "Examples:\n"
            "  python module5.py\n"
            "  python module5.py --port 8080\n"
            "  python module5.py --port 5000 --debug\n"
        ),
    )
    p.add_argument(
        "--host",
        default = DEFAULT_HOST,
        help    = f"Host to bind to (default: {DEFAULT_HOST})",
    )
    p.add_argument(
        "--port",
        type    = int,
        default = DEFAULT_PORT,
        help    = f"Port to listen on (default: {DEFAULT_PORT})",
    )
    p.add_argument(
        "--debug",
        action  = "store_true",
        help    = "Enable Flask debug mode (auto-reload on code changes)",
    )
    return p


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


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    _configure_logging()
    run(host=_args.host, port=_args.port, debug=_args.debug)
