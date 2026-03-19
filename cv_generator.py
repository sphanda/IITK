"""
CV generation module.

Generates:
- A general ATS-friendly CV based on structured profile data.
- A JD-tailored ATS-friendly CV based on structured profile and JD data.

Uses a stronger local LLM model (e.g., 'gemma3:4b') via Ollama.
"""

from datetime import datetime
from typing import Dict, Any, List

import config
import llm_client
import prompts
import utils


def generate_general_cv(profile_structured: Dict[str, Any], model_name: str) -> str:
    """
    Generate a general ATS-friendly CV text using structured profile data.

    Args:
        profile_structured (Dict[str, Any]): Structured profile data.
        model_name (str): Generation model name (e.g., 'gemma3:4b').

    Returns:
        str: Generated CV text.

    Raises:
        llm_client.LLMError: If the LLM call fails.
        RuntimeError: If the model returns empty or whitespace-only output.
    """
    # Enforce reverse-chronological experience order before building the prompt.
    profile_structured = _sort_experience_reverse_chronological(profile_structured)
    profile_json_str = utils.to_pretty_json(profile_structured)

    # FIX: prompts.get_general_cv_prompt() now accepts profile_json directly
    # and returns a fully populated prompt. The old .replace() pattern removed.
    prompt = prompts.get_general_cv_prompt(profile_json_str)

    cv_text = llm_client.generate_with_model(
        model_name=model_name,
        prompt=prompt,
        temperature=config.DEFAULT_TEMPERATURE_GENERATION,
        max_tokens=config.DEFAULT_MAX_TOKENS_GENERATION,
    )

    cv_text = cv_text.strip()

    # FIX: Guard against whitespace-only output that slips past llm_client's
    # empty-output check (e.g. a response that is purely newlines).
    if not cv_text:
        raise RuntimeError("General CV generation returned empty output after stripping.")

    # Scrub any hallucinated contact lines (e.g. invented LinkedIn/GitHub URLs).
    cv_text = _scrub_hallucinated_contact_lines(cv_text, profile_structured, label="General CV")
    # Enforce reverse-chronological experience order.
    cv_text = _enforce_experience_order(cv_text, profile_structured)
    # Warn if genuinely critical fields (email, phone) are missing.
    _warn_if_contact_missing(cv_text, profile_structured, label="General CV")

    return cv_text


def generate_tailored_cv(
    profile_structured: Dict[str, Any],
    jd_structured: Dict[str, Any],
    model_name: str,
) -> str:
    """
    Generate a JD-tailored ATS-friendly CV text.

    Args:
        profile_structured (Dict[str, Any]): Structured profile data.
        jd_structured (Dict[str, Any]): Structured job description data.
        model_name (str): Generation model name (e.g., 'gemma3:4b').

    Returns:
        str: Generated tailored CV text.

    Raises:
        llm_client.LLMError: If the LLM call fails.
        RuntimeError: If the model returns empty or whitespace-only output.
    """
    # Enforce reverse-chronological experience order before building the prompt.
    profile_structured = _sort_experience_reverse_chronological(profile_structured)
    profile_json_str = utils.to_pretty_json(profile_structured)
    jd_json_str = utils.to_pretty_json(jd_structured)

    # FIX: prompts.get_tailored_cv_prompt() now accepts both json strings
    # directly. The old chained .replace() pattern is removed.
    prompt = prompts.get_tailored_cv_prompt(profile_json_str, jd_json_str)

    cv_text = llm_client.generate_with_model(
        model_name=model_name,
        prompt=prompt,
        temperature=config.DEFAULT_TEMPERATURE_GENERATION,
        max_tokens=config.DEFAULT_MAX_TOKENS_GENERATION,
    )

    cv_text = cv_text.strip()

    # Guard against whitespace-only output.
    if not cv_text:
        raise RuntimeError("Tailored CV generation returned empty output after stripping.")

    # Scrub any hallucinated contact lines (e.g. invented LinkedIn/GitHub URLs).
    cv_text = _scrub_hallucinated_contact_lines(cv_text, profile_structured, label="Tailored CV")
    # Enforce reverse-chronological experience order.
    cv_text = _enforce_experience_order(cv_text, profile_structured)
    # Warn if genuinely critical fields (email, phone) are missing.
    _warn_if_contact_missing(cv_text, profile_structured, label="Tailored CV")

    return cv_text


def _sort_experience_reverse_chronological(
    profile_structured: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Sort the experience list in the profile dict so the most recent role
    appears first. gemma3:4b sometimes reorders roles when tailoring;
    this enforces correct reverse-chronological order before the prompt
    receives the data.

    Parses start_date strings like "Jun 2023", "2023-01", "2023".
    Roles with unparseable dates are kept at their original position.

    Args:
        profile_structured: Normalized profile dict.

    Returns:
        New profile dict with experience sorted most-recent-first.
    """
    experience: List[Dict] = list(profile_structured.get("experience") or [])
    if len(experience) <= 1:
        return profile_structured

    _MONTH_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    def _parse_date(date_str: str) -> datetime:
        if not date_str or not isinstance(date_str, str):
            return datetime.min
        date_str = date_str.strip().lower()
        # "present" or "current" → treat as most recent
        if date_str in ("present", "current", "now"):
            return datetime.max
        parts = date_str.replace("-", " ").split()
        try:
            if len(parts) == 2:
                # "jun 2023" or "2023 jun"
                if parts[0] in _MONTH_MAP:
                    return datetime(int(parts[1]), _MONTH_MAP[parts[0]], 1)
                if parts[1] in _MONTH_MAP:
                    return datetime(int(parts[0]), _MONTH_MAP[parts[1]], 1)
            if len(parts) == 1 and parts[0].isdigit():
                return datetime(int(parts[0]), 1, 1)
        except (ValueError, KeyError):
            pass
        return datetime.min

    def _sort_key(exp: Dict) -> datetime:
        # Sort by start_date descending — most recent first.
        return _parse_date(exp.get("start_date", ""))

    sorted_exp = sorted(experience, key=_sort_key, reverse=True)
    return {**profile_structured, "experience": sorted_exp}


def _scrub_hallucinated_contact_lines(
    cv_text: str,
    profile_structured: Dict[str, Any],
    label: str,
) -> str:
    """
    Remove lines from the CV header that contain URLs or handles not present
    in the structured profile contact fields.

    gemma3:4b frequently invents plausible LinkedIn/GitHub URLs even when
    the profile JSON has null for those fields. This scrubber removes any
    line in the first 10 lines that looks like a URL (contains '/' or 'http')
    but whose value does not appear in the structured contact dict.

    Args:
        cv_text (str): The generated CV plain text.
        profile_structured (Dict[str, Any]): Normalized profile dict.
        label (str): Short label used in log messages.

    Returns:
        str: CV text with hallucinated contact lines removed.
    """
    contact = profile_structured.get("contact") or {}
    # Build set of all known legitimate contact values (lowercased).
    known_values = {
        (v.lower().strip())
        for v in contact.values()
        if isinstance(v, str) and v.strip()
    }

    lines = cv_text.splitlines()
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        # Only scrub within the first 10 lines (the contact header block).
        if i >= 10:
            cleaned.append(line)
            continue
        # If line looks like a URL or handle (contains '/', 'http', 'linkedin', 'github')
        # verify it matches a known contact value.
        looks_like_url = any(
            marker in stripped
            for marker in ("http", "linkedin", "github", "twitter", ".com/", ".in/")
        )
        if looks_like_url:
            # Accept only if the line content matches a known contact value.
            is_known = any(known in stripped for known in known_values if known)
            if not is_known:
                utils.log_warning(
                    f"{label}: removed hallucinated contact line: '{line.strip()}'"
                )
                continue
        cleaned.append(line)

    return "\n".join(cleaned)



def _enforce_experience_order(
    cv_text: str,
    profile_structured: Dict[str, Any],
) -> str:
    """
    Ensure the EXPERIENCE section in the CV follows the same order as the
    structured profile JSON (which is already reverse-chronological).

    gemma3:4b sometimes reorders experience entries by JD relevance despite
    instructions. This function finds each company name from the profile in
    the CV text and re-emits the EXPERIENCE section in the correct order.

    If any company is not found in the CV text, the original text is returned
    unchanged to avoid corrupting a CV that used different formatting.

    Args:
        cv_text (str): Generated CV plain text.
        profile_structured (Dict[str, Any]): Normalized profile dict.

    Returns:
        str: CV text with experience in correct order.
    """
    experience = profile_structured.get("experience") or []
    if len(experience) < 2:
        return cv_text  # Nothing to reorder.

    # Find EXPERIENCE section boundaries.
    lines = cv_text.splitlines()
    exp_start = None
    next_section_start = None
    section_headings = {"SUMMARY", "EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS", "ACHIEVEMENTS"}

    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped == "EXPERIENCE":
            exp_start = i
        elif exp_start is not None and stripped in section_headings and i > exp_start:
            next_section_start = i
            break

    if exp_start is None:
        return cv_text  # No EXPERIENCE section found.

    exp_end = next_section_start if next_section_start else len(lines)
    exp_lines = lines[exp_start + 1 : exp_end]

    # Split experience block into per-company chunks using company names as anchors.
    companies = [e.get("company", "") for e in experience if e.get("company")]
    if not companies:
        return cv_text

    # Find the line index of each company within exp_lines.
    company_positions = {}
    for company in companies:
        for i, line in enumerate(exp_lines):
            if company.lower() in line.lower():
                company_positions[company] = i
                break

    if len(company_positions) != len(companies):
        # Not all companies found — don't risk corrupting the output.
        utils.log_warning(
            "Experience order check: could not locate all companies in CV text; "
            "skipping reorder."
        )
        return cv_text

    # Check if already in correct order.
    positions = [company_positions[c] for c in companies]
    if positions == sorted(positions):
        return cv_text  # Already correct.

    # Split exp_lines into chunks per company.
    sorted_positions = sorted(company_positions.items(), key=lambda x: x[1])
    chunks = []
    for idx, (company, start_pos) in enumerate(sorted_positions):
        end_pos = sorted_positions[idx + 1][1] if idx + 1 < len(sorted_positions) else len(exp_lines)
        chunks.append((company, exp_lines[start_pos:end_pos]))

    # Reorder chunks to match profile order.
    ordered_chunks = []
    for company in companies:
        for chunk_company, chunk_lines in chunks:
            if chunk_company == company:
                ordered_chunks.append(chunk_lines)
                break

    reordered_exp = []
    for chunk in ordered_chunks:
        reordered_exp.extend(chunk)

    utils.log_warning(
        "Experience order corrected to match profile (reverse-chronological)."
    )

    new_lines = (
        lines[: exp_start + 1]
        + reordered_exp
        + lines[exp_end:]
    )
    return "\n".join(new_lines)

def _warn_if_contact_missing(
    cv_text: str,
    profile_structured: Dict[str, Any],
    label: str,
) -> None:
    """
    Log a warning if critical contact fields from the profile are absent
    from the generated CV text.

    Args:
        cv_text (str): The generated CV plain text.
        profile_structured (Dict[str, Any]): Normalized profile dict.
        label (str): Short label used in log messages (e.g. "General CV").
    """
    contact = profile_structured.get("contact") or {}
    cv_lower = cv_text.lower()

    for field in ("email", "phone"):
        value = contact.get(field)
        if value and value.lower() not in cv_lower:
            utils.log_warning(
                f"{label}: {field} '{value}' from profile is missing in the generated CV. "
                f"Consider re-running or reviewing the output manually."
            )