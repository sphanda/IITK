"""
CV evaluation module.

Reviews generated CV(s) against structured profile data and optional JD,
and returns a human-readable evaluation report, including:

- Contact detail presence check.
- Missing or weak skills.
- Keyword alignment with JD.
- Section completeness and ATS readiness.
- Optional high-level refinement suggestions from the LLM.
"""

import re
from typing import Dict, Any, List, Optional

import llm_client
import prompts
import config
import utils


def evaluate_cvs_to_text(
    structured_profile: Dict[str, Any],
    structured_jd: Optional[Dict[str, Any]],
    cv_general: str,
    cv_tailored: Optional[str],
    model_name: str,
) -> str:
    """
    Evaluate generated CV(s) and return a human-readable multiline string report.

    Args:
        structured_profile (Dict[str, Any]): Structured profile data.
        structured_jd (Optional[Dict[str, Any]]): Structured JD data, if available.
        cv_general (str): General CV text.
        cv_tailored (Optional[str]): Tailored CV text, if any.
        model_name (str): Model name used for evaluation (usually extraction model).

    Returns:
        str: Evaluation report text.
    """
    lines: List[str] = []
    lines.append("CV EVALUATION REPORT")
    lines.append("====================")
    lines.append("")

    # FIX: Check contact detail presence — this was the original reported bug
    # (email etc. going missing). Report it explicitly so it's visible.
    lines.append("1. Contact Details Check")
    contact = structured_profile.get("contact") or {}
    contact_fields = {
        "Email":    contact.get("email"),
        "Phone":    contact.get("phone"),
        "LinkedIn": contact.get("linkedin"),
        "GitHub":   contact.get("github"),
        "Location": contact.get("location"),
    }
    cv_to_check = cv_tailored or cv_general
    for field_name, field_value in contact_fields.items():
        if not field_value:
            lines.append(f"  - {field_name}: Not in profile — skipped.")
            continue
        # Check the actual CV text contains the value verbatim.
        present_in_cv = field_value.lower() in cv_to_check.lower()
        status = "Present in CV" if present_in_cv else "MISSING from CV"
        lines.append(f"  - {field_name} ({field_value}): {status}")
    lines.append("")

    # Section completeness checks.
    lines.append("2. Section Completeness")
    general_sections = _check_section_completeness(cv_general)
    lines.append("  General CV:")
    for sec, present in general_sections.items():
        lines.append(f"    * {sec}: {'Present' if present else 'MISSING'}")

    if cv_tailored:
        tailored_sections = _check_section_completeness(cv_tailored)
        lines.append("  Tailored CV:")
        for sec, present in tailored_sections.items():
            lines.append(f"    * {sec}: {'Present' if present else 'MISSING'}")
    else:
        lines.append("  Tailored CV: Not generated.")
    lines.append("")

    # Keyword coverage if JD is available.
    lines.append("3. Keyword Alignment with Job Description")
    if structured_jd:
        jd_keywords = structured_jd.get("keywords") or []
        if not isinstance(jd_keywords, list):
            jd_keywords = []

        general_kw = _compute_keyword_coverage(cv_general, jd_keywords)
        lines.append("  General CV:")
        # FIX: Show coverage as a percentage — "73%" is clearer than "0.73".
        lines.append(f"    * Coverage: {general_kw['coverage_ratio'] * 100:.1f}%")
        lines.append(f"    * Present keywords: {', '.join(general_kw['present_keywords']) or 'none'}")
        lines.append(f"    * Missing keywords: {', '.join(general_kw['missing_keywords']) or 'none'}")

        if cv_tailored:
            tailored_kw = _compute_keyword_coverage(cv_tailored, jd_keywords)
            lines.append("  Tailored CV:")
            lines.append(f"    * Coverage: {tailored_kw['coverage_ratio'] * 100:.1f}%")
            lines.append(f"    * Present keywords: {', '.join(tailored_kw['present_keywords']) or 'none'}")
            lines.append(f"    * Missing keywords: {', '.join(tailored_kw['missing_keywords']) or 'none'}")
        else:
            lines.append("  Tailored CV: Not generated; no keyword metrics.")
    else:
        lines.append("  No job description provided; skipping JD keyword analysis.")
    lines.append("")

    # Skills alignment (profile vs general CV).
    lines.append("4. Skills Alignment (Profile vs General CV)")
    profile_skills = structured_profile.get("skills") or []
    if not isinstance(profile_skills, list):
        profile_skills = []

    skills_metrics = _compute_keyword_coverage(cv_general, profile_skills)
    lines.append(f"  Skills present in CV:  {', '.join(skills_metrics['present_keywords']) or 'none'}")
    lines.append(f"  Skills missing from CV: {', '.join(skills_metrics['missing_keywords']) or 'none'}")
    lines.append("")

    # LLM-based refinement suggestions.
    lines.append("5. High-Level Refinement Suggestions")
    try:
        suggestions = _llm_based_refinement_suggestions(
            structured_jd=structured_jd,
            cv_text=cv_tailored or cv_general,
            model_name=model_name,
        )
        lines.append(suggestions.strip())
    except Exception as exc:  # noqa: BLE001
        utils.log_warning(f"LLM-based refinement suggestions failed: {exc}")
        lines.append("  Unable to generate refinement suggestions due to an error.")
    lines.append("")

    return "\n".join(lines)


def _check_section_completeness(cv_text: str) -> Dict[str, bool]:
    """
    Check presence of key CV section headings in the text.

    Uses a word-boundary regex so that 'SKILLS' does not falsely match
    inside phrases like 'SOFT SKILLS ASSESSMENT' mid-paragraph.

    Args:
        cv_text (str): CV text.

    Returns:
        Dict[str, bool]: Mapping from section name to presence flag.
    """
    sections = ["SUMMARY", "EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS", "ACHIEVEMENTS"]
    upper_text = cv_text.upper()
    result: Dict[str, bool] = {}
    for sec in sections:
        # FIX: Use word-boundary regex instead of bare substring match.
        # This avoids false positives from section names appearing inside
        # longer words or unrelated phrases.
        result[sec] = bool(re.search(rf"\b{sec}\b", upper_text))
    return result


def _compute_keyword_coverage(cv_text: str, keywords: list) -> Dict[str, Any]:
    """
    Compute simple keyword coverage metrics.

    Args:
        cv_text (str): CV text.
        keywords (list): List of keywords to check.

    Returns:
        Dict[str, Any]: Metrics including coverage ratio, present, and missing lists.
    """
    cv_lower = cv_text.lower()
    present = []
    missing = []

    for kw in keywords:
        if not kw or not isinstance(kw, str):
            continue
        if kw.lower() in cv_lower:
            present.append(kw)
        else:
            missing.append(kw)

    total = len(present) + len(missing)
    coverage = (len(present) / total) if total > 0 else 0.0

    return {
        "total_keywords": total,
        "present_keywords": present,
        "missing_keywords": missing,
        "coverage_ratio": coverage,
    }


def _llm_based_refinement_suggestions(
    structured_jd: Optional[Dict[str, Any]],
    cv_text: str,
    model_name: str,
) -> str:
    """
    Get high-level refinement suggestions from the extraction model.

    Args:
        structured_jd (Optional[Dict[str, Any]]): Structured JD data, if available.
        cv_text (str): CV text to be evaluated.
        model_name (str): Model name to use for evaluation.

    Returns:
        str: Suggestions text (possibly multi-line).
    """
    # FIX: prompts.get_cv_evaluation_prompt() now accepts jd_json and cv_text
    # directly. The old .replace() pattern is removed.
    jd_json_str = utils.to_pretty_json(structured_jd) if structured_jd else "{}"
    prompt = prompts.get_cv_evaluation_prompt(
        jd_json=jd_json_str,
        cv_text=cv_text,
    )

    raw_suggestions = llm_client.generate_with_model(
        model_name=model_name,
        prompt=prompt,
        temperature=config.DEFAULT_TEMPERATURE_EXTRACTION,
        max_tokens=config.DEFAULT_MAX_TOKENS_EXTRACTION,
    )

    # Models may still return Markdown or extra prose; enforce consistent
    # dash-prefixed bullet formatting for report readability.
    return _sanitize_bullets(raw_suggestions, max_bullets=8)


def _sanitize_bullets(raw_text: str, max_bullets: int = 8) -> str:
    """
    Sanitize LLM output into clean single-line dash bullets.

    Args:
        raw_text: Raw LLM output.
        max_bullets: Maximum bullets to keep.

    Returns:
        str: Bullet list as a newline-separated string where each line starts
            with '- '.
    """
    if not raw_text:
        return "- No refinement needed."

    bullets: List[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue

        # Remove common Markdown emphasis markers.
        s = s.replace("**", "").replace("__", "")

        # Match '-' or '*' bullets with content.
        match = re.match(r"^[-*]\s*(.+)$", s)
        if not match:
            continue

        item = match.group(1).strip()
        if not item:
            continue

        bullets.append(item)
        if len(bullets) >= max_bullets:
            break

    if not bullets:
        return "- No refinement needed."

    return "\n".join(f"- {b}" for b in bullets)