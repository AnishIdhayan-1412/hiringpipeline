"""
Bias-Free Hiring Pipeline — Main Controller
=============================================
Single entry point to run the full pipeline sequentially.

Pipeline stages (run in order):
    Stage 0  — module0   : CV anonymization (PII removal, pronoun neutralization)
    Stage 0b — module0b  : Section parsing & skill extraction → parsed/
    Stage 1  — module1   : Candidate screening & ranking (SBERT)
    Stage 2  — module2   : Explainability engine → explanations/
    Stage 3  — module3   : GDPR audit trail               [COMING SOON]
    Stage 4  — module4   : Bias detection report          [COMING SOON]
    Stage 5  — module5   : Web dashboard (Flask)          [COMING SOON]

Usage:
    python main.py --jd-file jd.txt              Full pipeline with ranking
    python main.py --input-dir my_resumes        Custom input folder
    python main.py --verbose                     Enable debug logging
    python main.py --skip-stage 1                Skip ranking (useful for testing)
"""

import os
import sys
import logging
import argparse

import module0
import module0b
import module1
import module2

# Future modules — import only when implemented
# import module3
# import module4
# import module5


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


# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
def _configure_logging(verbose: bool = False) -> None:
    """Configure root logging for the entire pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


# ──────────────────────────────────────────────────────────────────────────
# Stage runner helper
# ──────────────────────────────────────────────────────────────────────────
def _run_stage(stage_num: int | str, label: str, fn, *args, skip_stages: list, **kwargs) -> bool:
    """Run a single pipeline stage with consistent header/footer logging.

    Args:
        stage_num:    Stage identifier shown in the header (e.g. 0, '0b', 1).
        label:        Human-readable stage name.
        fn:           The callable to invoke (e.g. module0.run).
        *args:        Positional arguments forwarded to fn.
        skip_stages:  List of stage identifiers to skip.
        **kwargs:     Keyword arguments forwarded to fn.

    Returns:
        True if the stage succeeded or was skipped, False if it failed.
    """
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


# ──────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Bias-Free Hiring Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python main.py --jd-file jd.txt\n'
            '  python main.py --input-dir my_resumes --jd-file jd.txt --verbose\n'
            '  python main.py --jd-file jd.txt --skip-stage 1\n'
        ),
    )
    parser.add_argument(
        '--input-dir', default=RAW_CVS_DIR,
        help='Directory containing raw CV files (default: raw_cvs/)',
    )
    parser.add_argument(
        '--output-dir', default=ANONYMIZED_DIR,
        help='Directory for anonymized output (default: anonymized_cvs/)',
    )
    parser.add_argument(
        '--parsed-dir', default=PARSED_DIR,
        help='Directory for parsed JSON output (default: parsed/)',
    )
    parser.add_argument(
        '--explanations-dir', default=EXPLANATIONS_DIR,
        help='Directory for explanation output (default: explanations/)',
    )
    parser.add_argument(
        '--vault-dir', default=VAULT_DIR,
        help='Directory for vault keys (default: vault/)',
    )
    parser.add_argument(
        '--model', default='en_core_web_trf',
        help='spaCy NER model for module0 (default: en_core_web_trf)',
    )
    parser.add_argument(
        '--jd-file', default=None,
        help='Path to plain-text Job Description file (required for ranking)',
    )
    parser.add_argument(
        '--skip-stage', dest='skip_stages', action='append', default=[],
        metavar='STAGE',
        help='Skip a stage by number (e.g. --skip-stage 1). Repeatable.',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose/debug logging',
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────
# Stage implementations
# ──────────────────────────────────────────────────────────────────────────
def _stage1_ranking(anonymized_dir: str, jd_file: str | None) -> bool:
    """Run module1: SBERT candidate ranking against the job description."""
    if not jd_file:
        print("\n  [INFO] No --jd-file provided. Skipping Stage 1 ranking.")
        print("  [TIP]  Re-run with: python main.py --jd-file path/to/jd.txt")
        return True  # not a failure — just optional

    if not os.path.isfile(jd_file):
        print(f"\n  [ERROR] JD file not found: {jd_file}")
        return False

    with open(jd_file, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    ranker   = module1.load_ranker()
    rankings = ranker.rank_candidates(
        anonymized_cv_dir=anonymized_dir,
        job_description=jd_text,
    )

    if not rankings:
        print("  No candidates ranked. Check anonymized_cvs/ directory.")
        return False

    print()
    print(f"  {'Rank':<6} {'Candidate':<22} {'Score':>8}  {'Bar'}")
    print(f"  {'-'*6} {'-'*22} {'-'*8}  {'-'*20}")

    report_lines = [
        "Rank | Candidate              | Match Score",
        "-" * 46,
    ]
    for rank, (cid, score) in enumerate(rankings, start=1):
        bar  = '█' * int(score * 20)
        line = f"  {rank:<6} {cid:<22} {score:>7.1%}  {bar}"
        print(line)
        report_lines.append(f"{rank:<5}| {cid:<23}| {score:.4f}")

    with open(RANKING_REPORT, 'w', encoding='utf-8') as rf:
        rf.write("\n".join(report_lines))

    print(f"\n  Ranking report saved → {RANKING_REPORT}")
    return True


def _stage2_explain(parsed_dir: str, explanations_dir: str, jd_file: str | None) -> bool:
    """Run module2: generate plain-English explanations for every ranked candidate."""
    if not jd_file:
        print("\n  [INFO] No --jd-file provided. Skipping Stage 2 explainability.")
        return True  # not a failure — ranking didn't run either

    ranker = module1.load_ranker()
    if not ranker.last_ranking_details:
        print("\n  [INFO] No ranking details found. Run Stage 1 first.")
        return True

    return module2.run(
        ranking_details=ranker.last_ranking_details,
        parsed_dir=parsed_dir,
        output_dir=explanations_dir,
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
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
    if args.jd_file:
        print(f"  JD     : {args.jd_file}")
    if args.skip_stages:
        print(f"  Skip   : stages {args.skip_stages}")

    skip = args.skip_stages   # list of stage ids to skip, e.g. ['1', '0b']

    # ── Stage 0 : CV Anonymization ─────────────────────────────────────
    ok = _run_stage(
        0, "CV Anonymization (module0)",
        module0.run,
        skip_stages=skip,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vault_dir=args.vault_dir,
        model=args.model,
    )
    if not ok:
        print("\n  Pipeline stopped at Stage 0. Fix the issue and re-run.")
        sys.exit(1)

    # ── Stage 0b : Section Parsing & Skill Extraction ──────────────────
    ok = _run_stage(
        '0b', "Section Parser & Skill Extractor (module0b)",
        module0b.run,
        skip_stages=skip,
        input_dir=args.output_dir,    # reads anonymized_cvs/
        output_dir=args.parsed_dir,   # writes parsed/
    )
    if not ok:
        print("\n  Pipeline stopped at Stage 0b. Fix the issue and re-run.")
        sys.exit(1)

    # ── Stage 1 : Candidate Ranking ────────────────────────────────────
    ok = _run_stage(
        1, "Candidate Screening & Ranking (module1)",
        _stage1_ranking,
        skip_stages=skip,
        anonymized_dir=args.output_dir,
        jd_file=args.jd_file,
    )
    if not ok:
        print("\n  Pipeline stopped at Stage 1. Fix the issue and re-run.")
        sys.exit(1)

    # ── Stage 2 : Explainability Engine ───────────────────────────────
    ok = _run_stage(
        2, "Explainability Engine (module2)",
        _stage2_explain,
        skip_stages=skip,
        parsed_dir=args.parsed_dir,
        explanations_dir=args.explanations_dir,
        jd_file=args.jd_file,
    )
    if not ok:
        print("\n  Pipeline stopped at Stage 2. Fix the issue and re-run.")
        sys.exit(1)

    # ── Stage 3 : GDPR Audit Trail       [COMING SOON] ─────────────────
    # ok = _run_stage(
    #     3, "GDPR Audit Trail (module3)",
    #     module3.run,
    #     skip_stages=skip,
    #     vault_dir=args.vault_dir,
    # )

    # ── Stage 4 : Bias Detection Report  [COMING SOON] ─────────────────
    # ok = _run_stage(
    #     4, "Bias Detection Report (module4)",
    #     module4.run,
    #     skip_stages=skip,
    #     vault_dir=args.vault_dir,
    #     parsed_dir=args.parsed_dir,
    # )

    # ── Stage 5 : Web Dashboard          [COMING SOON] ─────────────────
    # ok = _run_stage(
    #     5, "Web Dashboard (module5)",
    #     module5.run,
    #     skip_stages=skip,
    # )

    # ── Done ───────────────────────────────────────────────────────────
    print()
    print("*" * 60)
    print("  PIPELINE COMPLETE")
    print("*" * 60)
    print()


if __name__ == '__main__':
    main()