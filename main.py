#!/usr/bin/env python3
"""
Main entry point for the CV Creation using LLMs capstone project.

This script orchestrates the end-to-end pipeline:
- Read unstructured profile text (required).
- Optionally read job description text.
- Extract structured profile information.
- Extract job keywords and requirements.
- Generate ATS-friendly CV(s).
- Evaluate the generated CV(s).
- Save all outputs to the specified output directory.

Run `python main.py --help` for usage.
"""

import argparse
import os
import sys
from typing import Optional, Dict, Any

import config
import extractor
import jd_parser
import cv_generator
import evaluator
import utils


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Terminal-based multi-LLM CV Creation system using Ollama."
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Path to text file containing unstructured user profile data.",
    )
    parser.add_argument(
        "--jd",
        required=False,
        help="Optional path to text file containing job description.",
    )
    parser.add_argument(
        "--extract-model",
        required=False,
        default=config.DEFAULT_EXTRACTION_MODEL,
        help=f"Model name for extraction/parsing/evaluation (default: {config.DEFAULT_EXTRACTION_MODEL}).",
    )
    parser.add_argument(
        "--generate-model",
        required=False,
        default=config.DEFAULT_GENERATION_MODEL,
        help=f"Model name for CV generation/refinement (default: {config.DEFAULT_GENERATION_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default=config.DEFAULT_OUTPUT_DIR,
        help=f"Output directory for runtime-generated files (default: {config.DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """
    Validate profile/JD file paths and create the output directory.

    Args:
        args (argparse.Namespace): Parsed CLI args.

    Raises:
        SystemExit: If required files are missing or output directory cannot be created.
    """
    if not os.path.isfile(args.profile):
        utils.log_error(f"Profile file not found: {args.profile}")
        sys.exit(1)

    if args.jd and not os.path.isfile(args.jd):
        utils.log_error(f"Job description file not found: {args.jd}")
        sys.exit(1)

    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as exc:
        utils.log_error(f"Failed to create output directory '{args.output_dir}': {exc}")
        sys.exit(1)


def main() -> None:
    """
    Main orchestration function for the CV creation pipeline.

    Steps:
    1. Parse CLI arguments and validate inputs.
    2. Read profile and optional JD text files.
    3. Run extraction/parsing with the extraction model.
    4. Generate general and (optionally) tailored CVs with the generation model.
    5. Evaluate generated CVs.
    6. Save all outputs to disk.
    """
    args = parse_args()
    validate_inputs(args)

    utils.log_info("Starting CV Creation pipeline...")

    # Step 1: Read profile and JD text (supports .txt / .docx / .pdf).
    try:
        profile_text = utils.read_document_text(args.profile, label="profile")
    except Exception as exc:  # noqa: BLE001
        utils.log_error(f"Failed to read profile input: {exc}")
        sys.exit(1)
    if not profile_text.strip():
        utils.log_error("Profile file is empty or contains only whitespace.")
        sys.exit(1)

    jd_text: Optional[str] = None
    if args.jd:
        try:
            jd_text = utils.read_document_text(args.jd, label="job description")
        except Exception as exc:  # noqa: BLE001
            utils.log_warning(f"Failed to read job description input: {exc}; skipping JD tailoring.")
            jd_text = None
        if not jd_text.strip():
            utils.log_warning("Job description file is empty; JD-based tailoring will be skipped.")
            jd_text = None

    # Step 2: Extract structured profile information.
    utils.log_info("Extracting structured profile information...")
    try:
        structured_profile = extractor.extract_profile(
            profile_text=profile_text,
            model_name=args.extract_model,
        )
    except Exception as exc:  # noqa: BLE001
        utils.log_error(f"Profile extraction failed: {exc}")
        sys.exit(1)

    profile_json_path = os.path.join(args.output_dir, config.STRUCTURED_PROFILE_FILENAME)
    utils.write_json_file(profile_json_path, structured_profile)
    utils.log_info(f"Structured profile saved to: {profile_json_path}")

    # Step 3: Extract structured JD information (if JD provided).
    structured_jd: Optional[Dict[str, Any]] = None
    if jd_text:
        utils.log_info("Extracting structured job description information...")
        try:
            structured_jd = jd_parser.parse_job_description(
                jd_text=jd_text,
                model_name=args.extract_model,
            )
        except Exception as exc:  # noqa: BLE001
            utils.log_warning(f"Job description parsing failed; continuing without JD: {exc}")
            structured_jd = None

        if structured_jd is not None:
            jd_json_path = os.path.join(args.output_dir, config.JD_ANALYSIS_FILENAME)
            utils.write_json_file(jd_json_path, structured_jd)
            utils.log_info(f"Job description analysis saved to: {jd_json_path}")

    # Step 4: Generate CV(s).
    utils.log_info("Generating general CV...")
    try:
        cv_general_text = cv_generator.generate_general_cv(
            profile_structured=structured_profile,
            model_name=args.generate_model,
        )
    except Exception as exc:  # noqa: BLE001
        utils.log_error(f"General CV generation failed: {exc}")
        sys.exit(1)

    cv_general_path = os.path.join(args.output_dir, config.CV_GENERAL_FILENAME)
    utils.write_text_file(cv_general_path, cv_general_text)
    utils.log_info(f"General CV saved to: {cv_general_path}")

    # Export DOCX and (best-effort) PDF versions for academic submission.
    try:
        docx_path = os.path.join(args.output_dir, config.CV_GENERAL_DOCX_FILENAME)
        pdf_path = os.path.join(args.output_dir, config.CV_GENERAL_PDF_FILENAME)
        utils.export_cv_docx_and_pdf(
            cv_text=cv_general_text,
            docx_path=docx_path,
            pdf_path=pdf_path,
        )
        utils.log_info(f"General CV exported to DOCX/PDF: {docx_path}, {pdf_path}")
    except Exception as exc:  # noqa: BLE001
        utils.log_warning(f"General CV DOCX/PDF export failed: {exc}")

    cv_tailored_text: Optional[str] = None

    if structured_jd is not None:
        utils.log_info("Generating JD-tailored CV...")
        try:
            cv_tailored_text = cv_generator.generate_tailored_cv(
                profile_structured=structured_profile,
                jd_structured=structured_jd,
                model_name=args.generate_model,
            )
        except Exception as exc:  # noqa: BLE001
            utils.log_warning(f"Tailored CV generation failed; continuing with general CV only: {exc}")
            cv_tailored_text = None

        # FIX: Use .strip() check to guard against whitespace-only LLM output
        if cv_tailored_text and cv_tailored_text.strip():
            cv_tailored_path = os.path.join(args.output_dir, config.CV_TAILORED_FILENAME)
            utils.write_text_file(cv_tailored_path, cv_tailored_text)
            utils.log_info(f"Tailored CV saved to: {cv_tailored_path}")
        else:
            utils.log_warning("Tailored CV text is empty; no tailored CV file will be written.")
            cv_tailored_text = None  # FIX: Normalise to None so evaluator receives clean input
    else:
        utils.log_info("No job description provided; skipping tailored CV generation.")

    # Export tailored DOCX/PDF versions (only if tailored CV exists).
    if cv_tailored_text:
        try:
            tailored_docx = os.path.join(args.output_dir, config.CV_TAILORED_DOCX_FILENAME)
            tailored_pdf = os.path.join(args.output_dir, config.CV_TAILORED_PDF_FILENAME)
            utils.export_cv_docx_and_pdf(
                cv_text=cv_tailored_text,
                docx_path=tailored_docx,
                pdf_path=tailored_pdf,
            )
            utils.log_info(f"Tailored CV exported to DOCX/PDF: {tailored_docx}, {tailored_pdf}")
        except Exception as exc:  # noqa: BLE001
            utils.log_warning(f"Tailored CV DOCX/PDF export failed: {exc}")

    # Step 5: Evaluate generated CV(s).
    utils.log_info("Evaluating generated CV(s)...")
    # FIX: Wrap evaluation in try/except — LLM call inside can fail
    try:
        evaluation_text = evaluator.evaluate_cvs_to_text(
            structured_profile=structured_profile,
            structured_jd=structured_jd,
            cv_general=cv_general_text,
            cv_tailored=cv_tailored_text,
            model_name=args.extract_model,
        )
    except Exception as exc:  # noqa: BLE001
        utils.log_warning(f"CV evaluation failed; skipping evaluation report: {exc}")
        evaluation_text = "Evaluation could not be completed due to an error.\n"

    evaluation_path = os.path.join(args.output_dir, config.CV_EVALUATION_FILENAME)
    utils.write_text_file(evaluation_path, evaluation_text)
    utils.log_info(f"CV evaluation report saved to: {evaluation_path}")

    utils.log_info("CV Creation pipeline completed successfully.")


if __name__ == "__main__":
    main()