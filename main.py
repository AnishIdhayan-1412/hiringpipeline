"""
Bias-Free Hiring Pipeline — Main Controller
=============================================
Single entry point to run the full anonymization pipeline.

Usage:
    python main.py --jd-file job.txt             Run with a job description
    python main.py --input-dir my_resumes        Custom input folder
    python main.py --verbose                     Enable debug logging
"""

import os
import sys
import logging
import argparse

import module0
import module1


# ──────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_CVS_DIR     = os.path.join(BASE_DIR, 'raw_cvs')
ANONYMIZED_DIR  = os.path.join(BASE_DIR, 'anonymized_cvs')
VAULT_DIR       = os.path.join(BASE_DIR, 'vault')
RANKING_REPORT  = os.path.join(BASE_DIR, 'ranking_report.txt')


def _configure_logging(verbose: bool = False) -> None:
    """Configure root logging for the entire pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def main():
    parser = argparse.ArgumentParser(
        description='Bias-Free Hiring Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python main.py --input-dir my_resumes --verbose',
    )
    parser.add_argument('--input-dir',  default=RAW_CVS_DIR,
                        help='Directory containing raw CV files (default: raw_cvs/)')
    parser.add_argument('--output-dir', default=ANONYMIZED_DIR,
                        help='Directory for anonymized output (default: anonymized_cvs/)')
    parser.add_argument('--vault-dir',  default=VAULT_DIR,
                        help='Directory for vault keys (default: vault/)')
    parser.add_argument('--model',      default='en_core_web_trf',
                        help='spaCy NER model to use (default: en_core_web_trf)')
    parser.add_argument('--jd-file',    default=None,
                        help='Path to a plain-text Job Description file for candidate ranking')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    print()
    print("*" * 60)
    print("  BIAS-FREE HIRING PIPELINE")
    print("*" * 60)

    # ── Module 0: CV Anonymization ──
    ok = module0.run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vault_dir=args.vault_dir,
        model=args.model,
    )
    if not ok:
        print("\n  Pipeline stopped. Fix the issue above and re-run.")
        sys.exit(1)

    # ── Module 1: Automated Screening & Ranking ──
    if args.jd_file:
        if not os.path.isfile(args.jd_file):
            print(f"\n  [ERROR] JD file not found: {args.jd_file}")
            sys.exit(1)

        with open(args.jd_file, 'r', encoding='utf-8') as f:
            jd_text = f.read()

        print()
        print("=" * 60)
        print("  MODULE 1 — Candidate Screening & Ranking")
        print("=" * 60)

        ranker = module1.load_ranker()
        rankings = ranker.rank_candidates(
            anonymized_cv_dir=args.output_dir,
            job_description=jd_text,
        )

        if rankings:
            print()
            print(f"  {'Rank':<6} {'Candidate':<20} {'Match Score':>12}")
            print(f"  {'-'*6} {'-'*20} {'-'*12}")
            report_lines = ["Rank | Candidate            | Match Score",
                            "-" * 44]
            for rank, (cid, score) in enumerate(rankings, start=1):
                bar = '█' * int(score * 20)
                line = f"  {rank:<6} {cid:<20} {score:>10.1%}  {bar}"
                print(line)
                report_lines.append(f"{rank:<5}| {cid:<21}| {score:.4f}")

            # Save report to file
            with open(RANKING_REPORT, 'w', encoding='utf-8') as rf:
                rf.write("\n".join(report_lines))
            print()
            print(f"  Ranking report saved to: {RANKING_REPORT}")
        else:
            print("  No candidates ranked. Check anonymized_cvs/ directory.")
    else:
        print()
        print("  [INFO] No --jd-file provided. Skipping Module 1 ranking.")
        print("  [TIP]  Re-run with: python main.py --jd-file path/to/job.txt")

    # ── Module 2: (will be added here) ──

    print()
    print("*" * 60)
    print("  PIPELINE COMPLETE")
    print("*" * 60)
    print()


if __name__ == '__main__':
    main()
