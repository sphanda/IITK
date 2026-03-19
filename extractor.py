"""
Profile extraction module.

Takes unstructured user profile text and uses a local LLM (via Ollama)
to extract a structured JSON-like representation of candidate information.
"""

from typing import Dict, Any, Optional

import config
import llm_client
import prompts
import utils


def extract_profile(profile_text: str, model_name: str) -> Dict[str, Any]:
    """
    Extract structured profile information from unstructured text.

    The function:
    - Builds a fully populated prompt via prompts.get_profile_extraction_prompt().
    - Calls the extraction model via llm_client.
    - Attempts to parse the response as JSON.
    - Applies one cleanup/retry pass if parsing fails.
    - Returns a normalized Python dictionary.

    Args:
        profile_text (str): Unstructured profile text.
        model_name (str): Extraction model name (e.g., 'gemma3:1b').

    Returns:
        Dict[str, Any]: Structured profile data.

    Raises:
        ValueError: If profile_text is empty.
        RuntimeError: If extraction or parsing consistently fails.
    """
    if not profile_text.strip():
        raise ValueError("Profile text is empty.")

    prompt = prompts.get_profile_extraction_prompt(profile_text.strip())

    # First attempt.
    raw_output = llm_client.generate_with_model(
        model_name=model_name,
        prompt=prompt,
        temperature=config.DEFAULT_TEMPERATURE_EXTRACTION,
        max_tokens=config.DEFAULT_MAX_TOKENS_EXTRACTION,
    )

    structured = utils.safe_json_parse(raw_output)

    if structured is None or not isinstance(structured, dict):
        utils.log_warning("Initial profile JSON parse failed; attempting strip_non_json cleanup.")
        cleaned_text = utils.strip_non_json(raw_output)
        structured = utils.safe_json_parse(cleaned_text)

        if structured is None or not isinstance(structured, dict):
            utils.log_warning("Cleanup parse failed; retrying with stricter prompt.")
            retry_prompt = (
                prompt
                + "\n\nReturn VALID JSON ONLY. Start with { and end with }. No extra text."
            )
            retry_output = llm_client.generate_with_model(
                model_name=model_name,
                prompt=retry_prompt,
                temperature=config.DEFAULT_TEMPERATURE_EXTRACTION,
                max_tokens=config.DEFAULT_MAX_TOKENS_EXTRACTION,
            )
            structured = utils.safe_json_parse(retry_output)

            if structured is None or not isinstance(structured, dict):
                raise RuntimeError(
                    "Failed to parse structured profile JSON after two recovery attempts."
                )

    normalized = _normalize_profile_structured(structured, profile_text)

    # Warn if extracted project count seems low relative to what the raw
    # profile text mentions — a sign of token truncation mid-JSON.
    raw_project_mentions = profile_text.lower().count("project")
    extracted_projects = len(normalized.get("projects") or [])
    if raw_project_mentions >= 2 and extracted_projects < 2:
        utils.log_warning(
            f"Profile mentions ~{raw_project_mentions} project references but only "
            f"{extracted_projects} project(s) were extracted. The model may have "
            f"truncated. Consider raising DEFAULT_MAX_TOKENS_EXTRACTION in config.py."
        )

    return normalized


def _normalize_contact(raw: Any) -> Dict[str, Optional[str]]:
    """
    Normalize the contact field to the expected nested dict structure.

    Handles three cases the LLM might produce:
    - Correct nested dict  -> extract each sub-field safely.
    - Flat string          -> treat the whole string as "other", all else None.
    - None / missing       -> all sub-fields default to None.

    Args:
        raw (Any): Raw value of the "contact" key from LLM output.

    Returns:
        Dict[str, Optional[str]]: Normalized contact dict with guaranteed keys.
    """
    empty: Dict[str, Optional[str]] = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "location": None,
        "other": None,
    }

    if raw is None:
        return empty

    if isinstance(raw, str):
        return {**empty, "other": raw.strip() or None}

    if isinstance(raw, dict):
        def _clean(val: Any) -> Optional[str]:
            # Strip whitespace the model may have added around verbatim values.
            return val.strip() if isinstance(val, str) and val.strip() else None

        return {
            "email":    _clean(raw.get("email")),
            "phone":    _clean(raw.get("phone")),
            "linkedin": _clean(raw.get("linkedin")),
            "github":   _clean(raw.get("github")),
            "location": _clean(raw.get("location")),
            "other":    _clean(raw.get("other")),
        }

    return empty


def _normalize_profile_structured(data: Dict[str, Any], profile_text: str = "") -> Dict[str, Any]:
    """
    Normalize the structured profile dictionary to ensure all required keys
    exist with correct types, regardless of what the LLM returned.

    Args:
        data (Dict[str, Any]): Raw structured profile dictionary.

    Returns:
        Dict[str, Any]: Normalized structured profile dictionary.
    """
    def _as_list(value: Any) -> list:
        """Return value as a list, or empty list if None or wrong type."""
        if isinstance(value, list):
            return value
        return []

    def _flatten_skills(skills: list) -> list:
        """
        Flatten skills in case the model returned category strings like
        'Programming: Python, SQL' instead of individual skill names.
        Strips category label prefix (text before ':'), splits on commas,
        strips whitespace, and deduplicates while preserving order.
        """
        flat = []
        seen = set()
        for item in skills:
            if not isinstance(item, str):
                continue
            # Strip category label prefix if present (e.g. "Programming: Python")
            part = item.split(":", 1)[-1] if ":" in item else item
            for skill in part.split(","):
                cleaned = skill.strip()
                if cleaned and cleaned.lower() not in seen:
                    seen.add(cleaned.lower())
                    flat.append(cleaned)
        return flat

    def _filter_education(entries: list) -> list:
        """
        Remove coursework/subject list entries that the model sometimes
        extracts as separate education entries despite prompt instructions.
        """
        COURSEWORK_MARKERS = (
            "coursework", "course work", "subjects", "modules",
            "relevant courses", "electives",
        )
        filtered = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            degree_val = (entry.get("degree") or "").lower()
            if any(marker in degree_val for marker in COURSEWORK_MARKERS):
                utils.log_warning(
                    f"Skipping non-degree education entry: '{entry.get('degree')}'"
                )
                continue
            filtered.append(entry)
        return filtered

    def _filter_project_technologies(projects: list) -> list:
        """
        Remove hallucinated technologies from each project by checking that
        every technology string actually appears in the project description.
        gemma3:1b frequently adds frameworks from other projects or invents
        tools not mentioned in the source text despite prompt instructions.
        """
        cleaned = []
        for project in projects:
            if not isinstance(project, dict):
                continue
            description = (project.get("description") or "").lower()
            raw_techs = project.get("technologies") or []
            if not isinstance(raw_techs, list):
                cleaned.append(project)
                continue
            verified = []
            hallucinated = []
            for tech in raw_techs:
                if not isinstance(tech, str):
                    continue
                if tech.lower() in description:
                    verified.append(tech)
                else:
                    hallucinated.append(tech)
            if hallucinated:
                utils.log_warning(
                    f"Project '{project.get('name')}': removed hallucinated "
                    f"technologies not found in description: {hallucinated}"
                )
            cleaned.append({**project, "technologies": verified})
        return cleaned

    def _extract_location_fallback(raw_data: dict, profile_text_ref: str) -> str | None:
        """
        If the LLM left location null, scan the raw profile text for a
        location near the contact block (first 20 lines) as a fallback.
        Looks for 'Location:' label or common Indian city names.
        """
        # First check if any education/experience entry has a city we can reuse.
        for edu in (raw_data.get("education") or []):
            inst = edu.get("institution") or ""
            if "," in inst:
                # e.g. "ABC Institute, Bangalore" — take the part after comma
                city = inst.split(",")[-1].strip()
                if city:
                    return city

        # Scan first 20 lines for explicit Location: label
        for line in profile_text_ref.splitlines()[:20]:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("location:"):
                val = line_stripped.split(":", 1)[-1].strip()
                if val:
                    return val
        return None

    # Extract location fallback before building final dict
    contact = _normalize_contact(data.get("contact"))
    if contact["location"] is None:
        fallback_location = _extract_location_fallback(data, profile_text_ref=profile_text)
        if fallback_location:
            utils.log_warning(
                f"Location was null after extraction; recovered via fallback: '{fallback_location}'"
            )
            contact["location"] = fallback_location

    return {
        "name":         data.get("name") or None,
        "contact":      contact,
        "summary":      data.get("summary") or None,
        # Filter out coursework entries the model adds despite instructions.
        "education":    _filter_education(_as_list(data.get("education"))),
        "experience":   _as_list(data.get("experience")),
        # Flatten in case model returned category-prefixed strings.
        "skills":       _flatten_skills(_as_list(data.get("skills"))),
        # Strip hallucinated technologies not present in project descriptions.
        "projects":     _filter_project_technologies(_as_list(data.get("projects"))),
        "achievements": _as_list(data.get("achievements")),
    }