"""
Bias-Free Hiring Pipeline — Main Controller
=============================================
Single entry point to run the full pipeline sequentially.

Pipeline stages (run in order):
    Stage 0  — module0   : CV anonymization (PII removal, pronoun neutralization)
    Stage 0b — module0b  : Section parsing & skill extraction → parsed/
    Stage 1  — module1   : Candidate screening & ranking (SBERT)
    Stage 2  — module2   : Explainability engine → explanations/
    Stage 3  — module3   : GDPR audit trail → audit/
    Stage 4  — module4   : Bias detection → audit/bias_*.txt/json
    Stage 5  — module5   : Web dashboard (Flask) → localhost:5000

Usage:
    python main.py --jd-file jd.txt              Full pipeline
    python main.py --jd-file jd.txt --dashboard  Pipeline + launch dashboard
    python main.py --input-dir my_resumes        Custom input folder
    python main.py --verbose                     Enable debug logging
    python main.py --skip-stage 1                Skip ranking
"""

import os
import sys
import logging
import argparse

import module0
import module0b
import module1
import module2
import module3
import module4
from exceptions import ConfigurationError
from pipeline_logging import configure_logging
from database import get_db

# ──────────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
RAW_CVS_DIR       = os.path.join(BASE_DIR, 'raw_cvs')
ANONYMIZED_DIR    = os.path.join(BASE_DIR, 'anonymized_cvs')
PARSED_DIR        = os.path.join(BASE_DIR, 'parsed')
VAULT_DIR         = os.path.join(BASE_DIR, 'vault')
RANKING_REPORT    = os.path.join(BASE_DIR, 'ranking_report.txt')
EXPLANATIONS_DIR  = os.path.join(BASE_DIR, 'explanations')
AUDIT_DIR         = os.path.join(BASE_DIR, 'audit')


def _configure_logging(verbose: bool = False) -> None:
    """Configure root logging for the entire pipeline via centralised module."""
    configure_logging(verbose=verbose, log_dir="logs")


def _run_stage(stage_num, label, fn, *args, skip_stages, **kwargs) -> bool:
    """Run a single pipeline stage with consistent header/footer logging."""
    if str(stage_num) in [str(s) for s in skip_stages]:
        print(f"\n  [SKIP] Stage {stage_num}: {label}")
        return True

    print()
    print(f"  ┌{'─' * 56}┐")
    print(f"  │  STAGE {stage_num:<4} {label:<45}│")
    print(f"  └{'─' * 56}┘")

    try:
        ok = fn(*args, **kwargs)
        if ok:
            print(f"\n  [OK] Stage {stage_num} completed successfully.")
        else:
            print(f"\n  [FAILED] Stage {stage_num} reported failure.")
        return bool(ok)
    except Exception as exc:
        logging.getLogger(__name__).exception("Stage %s crashed: %s", stage_num, exc)
        print(f"\n  [ERROR] Stage {stage_num} crashed: {exc}")
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Bias-Free Hiring Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python main.py --jd-file jd.txt\n'
            '  python main.py --jd-file jd.txt --dashboard\n'
            '  python main.py --input-dir my_resumes --jd-file jd.txt --verbose\n'
            '  python main.py --jd-file jd.txt --skip-stage 1\n'
        ),
    )
    parser.add_argument('--input-dir',      default=RAW_CVS_DIR)
    parser.add_argument('--output-dir',     default=ANONYMIZED_DIR)
    parser.add_argument('--parsed-dir',     default=PARSED_DIR)
    parser.add_argument('--explanations-dir', default=EXPLANATIONS_DIR)
    parser.add_argument('--vault-dir',      default=VAULT_DIR)
    parser.add_argument('--audit-dir',      default=AUDIT_DIR)
    parser.add_argument('--model',          default='en_core_web_trf')
    parser.add_argument('--jd-file',        default=None)
    parser.add_argument('--skip-stage',     dest='skip_stages',
                        action='append', default=[], metavar='STAGE')
    parser.add_argument('--verbose', '-v',  action='store_true')
    parser.add_argument('--dashboard',      action='store_true',
                        help='Launch web dashboard after pipeline completes')
    parser.add_argument('--port',           type=int, default=5000,
                        help='Dashboard port (default: 5000)')
    return parser


def _stage1_ranking(anonymized_dir, jd_file):
    if not jd_file:
        print("\n  [INFO] No --jd-file provided. Skipping Stage 1.")
        return True
    if not os.path.isfile(jd_file):
        print(f"\n  [ERROR] JD file not found: {jd_file}")
        return False
    with open(jd_file, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    ranker   = module1.load_ranker()
    rankings = ranker.rank_candidates(
        anonymized_cv_dir=anonymized_dir, job_description=jd_text)
    if not rankings:
        print("  No candidates ranked.")
        return False
    print()
    print(f"  {'Rank':<6} {'Candidate':<22} {'Score':>8}  {'Bar'}")
    print(f"  {'-'*6} {'-'*22} {'-'*8}  {'-'*20}")
    report_lines = ["Rank | Candidate              | Match Score", "-" * 46]
    for rank, (cid, score) in enumerate(rankings, start=1):
        bar  = '█' * int(score * 20)
        print(f"  {rank:<6} {cid:<22} {score:>7.1%}  {bar}")
        report_lines.append(f"{rank:<5}| {cid:<23}| {score:.4f}")
    with open(RANKING_REPORT, 'w', encoding='utf-8') as rf:
        rf.write("\n".join(report_lines))
    print(f"\n  Ranking report saved → {RANKING_REPORT}")
    return True


def _stage2_explain(parsed_dir, explanations_dir, jd_file):
    if not jd_file:
        print("\n  [INFO] No --jd-file. Skipping Stage 2.")
        return True
    ranker = module1.load_ranker()
    if not ranker.last_ranking_details:
        print("\n  [INFO] No ranking details. Run Stage 1 first.")
        return True
    return module2.run(
        ranking_details=ranker.last_ranking_details,
        parsed_dir=parsed_dir,
        output_dir=explanations_dir,
    )


def _stage3_audit(vault_dir, parsed_dir, explanations_dir, audit_dir,
                  raw_cvs_dir, jd_file):
    ranker          = module1.load_ranker()
    ranking_details = ranker.last_ranking_details or {}
    return module3.run(
        vault_dir=vault_dir, parsed_dir=parsed_dir,
        ranking_details=ranking_details,
        explanations_dir=explanations_dir,
        output_dir=audit_dir, raw_cvs_dir=raw_cvs_dir,
    )


def _stage4_bias(parsed_dir, audit_dir, jd_file):
    if not jd_file:
        print("\n  [INFO] No --jd-file. Skipping Stage 4.")
        return True
    ranker          = module1.load_ranker()
    ranking_details = ranker.last_ranking_details or {}
    if not ranking_details:
        print("\n  [INFO] No ranking details. Skipping Stage 4.")
        return True
    return module4.run(
        ranking_details=ranking_details,
        parsed_dir=parsed_dir,
        output_dir=audit_dir,
    )


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    _configure_logging(verbose=args.verbose)

    print()
    print("*" * 60)
    print("  BIAS-FREE HIRING PIPELINE")
    print("*" * 60)
    print(f"  Input  : {args.input_dir}")
    print(f"  Output : {args.output_dir}")
    print(f"  Parsed : {args.parsed_dir}")
    print(f"  Vault  : {args.vault_dir}")
    print(f"  Explain: {args.explanations_dir}")
    print(f"  Audit  : {args.audit_dir}")
    if args.jd_file:
        print(f"  JD     : {args.jd_file}")
    if args.skip_stages:
        print(f"  Skip   : stages {args.skip_stages}")

    skip = args.skip_stages

    for stage_num, label, fn, kwargs in [
        (0,   "CV Anonymization (module0)",               module0.run,
         dict(input_dir=args.input_dir, output_dir=args.output_dir,
              vault_dir=args.vault_dir, model=args.model)),
        ('0b',"Section Parser & Skill Extractor (module0b)", module0b.run,
         dict(input_dir=args.output_dir, output_dir=args.parsed_dir)),
    ]:
        ok = _run_stage(stage_num, label, fn, skip_stages=skip, **kwargs)
        if not ok:
            print(f"\n  Pipeline stopped at Stage {stage_num}.")
            sys.exit(1)

    ok = _run_stage(1, "Candidate Screening & Ranking (module1)",
                    _stage1_ranking, skip_stages=skip,
                    anonymized_dir=args.output_dir, jd_file=args.jd_file)
    if not ok:
        print("\n  Pipeline stopped at Stage 1."); sys.exit(1)

    ok = _run_stage(2, "Explainability Engine (module2)",
                    _stage2_explain, skip_stages=skip,
                    parsed_dir=args.parsed_dir,
                    explanations_dir=args.explanations_dir,
                    jd_file=args.jd_file)
    if not ok:
        print("\n  Pipeline stopped at Stage 2."); sys.exit(1)

    ok = _run_stage(3, "GDPR Audit Trail (module3)",
                    _stage3_audit, skip_stages=skip,
                    vault_dir=args.vault_dir, parsed_dir=args.parsed_dir,
                    explanations_dir=args.explanations_dir,
                    audit_dir=args.audit_dir, raw_cvs_dir=args.input_dir,
                    jd_file=args.jd_file)
    if not ok:
        print("\n  Pipeline stopped at Stage 3."); sys.exit(1)

    ok = _run_stage(4, "Bias Detection & Fairness Audit (module4)",
                    _stage4_bias, skip_stages=skip,
                    parsed_dir=args.parsed_dir, audit_dir=args.audit_dir,
                    jd_file=args.jd_file)
    if not ok:
        print("\n  Pipeline stopped at Stage 4."); sys.exit(1)

    print()
    print("*" * 60)
    print("  PIPELINE COMPLETE")
    print("*" * 60)
    print()

    # ── Persist run to SQLite database ────────────────────────────────────
    try:
        import module1 as _m1
        _ranker  = _m1.load_ranker()
        _db      = get_db()
        _run_id  = _db.get_latest_run_id()

        # Save ranking + candidate records if a run already exists from module5,
        # otherwise create a new one for CLI invocations.
        if not _run_id or _db.get_run(_run_id).get("status") not in ("running",):
            import datetime as _dt
            _run_id = _dt.datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
            jd_text = ""
            if args.jd_file and os.path.isfile(args.jd_file):
                with open(args.jd_file, "r", encoding="utf-8") as _f:
                    jd_text = _f.read()
            _db.create_run(_run_id, jd_file=args.jd_file, jd_text=jd_text)

        if _ranker.last_ranking_details:
            _db.save_ranking(_run_id, _ranker.last_ranking_details)

        # Count processed CVs
        _total = sum(1 for f in os.listdir(args.output_dir)
                     if f.endswith(".txt")) if os.path.isdir(args.output_dir) else 0
        _db.finish_run(_run_id, exit_code=0, total_cvs=_total)
        print(f"  DB run persisted → {_run_id}")
    except Exception as _db_exc:
        logging.getLogger(__name__).warning("DB persist skipped: %s", _db_exc)

    # ── Stage 5 : Web Dashboard ────────────────────────────────────────
    if args.dashboard:
        _run_stage(5, "Web Dashboard (module5)", _launch_dashboard,
                   skip_stages=skip, port=args.port)


def _launch_dashboard(port: int = 5000) -> bool:
    """Import and launch module5 web dashboard. Blocks until Ctrl+C."""
    try:
        import module5
        module5.run(port=port)
        return True
    except ImportError:
        print("\n  [ERROR] module5.py not found in project directory.")
        return False
    except Exception as exc:
        print(f"\n  [ERROR] Dashboard failed: {exc}")
        return False


if __name__ == '__main__':
    main()
