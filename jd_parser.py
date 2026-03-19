"""
Job description parsing module.

Takes unstructured job description text and uses a local LLM (via Ollama)
to extract role title, skills, keywords, and responsibilities.
"""

from typing import Dict, Any

import config
import llm_client
import prompts
import utils


def parse_job_description(jd_text: str, model_name: str) -> Dict[str, Any]:
    """
    Extract structured job description information from unstructured text.

    Args:
        jd_text (str): Job description text.
        model_name (str): Extraction model name (e.g., 'gemma3:1b').

    Returns:
        Dict[str, Any]: Structured JD data.

    Raises:
        RuntimeError: If parsing consistently fails after recovery attempts.
        ValueError: If jd_text is empty.
    """
    if not jd_text.strip():
        raise ValueError("Job description text is empty.")

    # FIX: prompts.get_jd_parsing_prompt() now accepts jd_text directly and
    # returns a fully populated prompt. The old .replace() pattern is removed.
    prompt = prompts.get_jd_parsing_prompt(jd_text.strip())

    raw_output = llm_client.generate_with_model(
        model_name=model_name,
        prompt=prompt,
        temperature=config.DEFAULT_TEMPERATURE_EXTRACTION,
        max_tokens=config.DEFAULT_MAX_TOKENS_EXTRACTION,
    )

    # FIX: safe_json_parse now returns None on failure (not raw_text), so check
    # for None first. Also use isinstance(x, dict) — lowercase — for runtime
    # checks. typing.Dict is not valid for isinstance() and raises TypeError
    # in Python 3.9+.
    structured = utils.safe_json_parse(raw_output)

    if structured is None or not isinstance(structured, dict):
        utils.log_warning("Initial JD JSON parse failed; attempting strip_non_json cleanup.")
        cleaned_text = utils.strip_non_json(raw_output)
        structured = utils.safe_json_parse(cleaned_text)

        if structured is None or not isinstance(structured, dict):
            utils.log_warning("Cleanup parse failed; retrying with stricter prompt.")
            retry_prompt = prompt + "\n\nReturn VALID JSON ONLY. Start with { and end with }. No extra text."
            retry_output = llm_client.generate_with_model(
                model_name=model_name,
                prompt=retry_prompt,
                temperature=config.DEFAULT_TEMPERATURE_EXTRACTION,
                max_tokens=config.DEFAULT_MAX_TOKENS_EXTRACTION,
            )
            structured = utils.safe_json_parse(retry_output)

            if structured is None or not isinstance(structured, dict):
                raise RuntimeError(
                    "Failed to parse structured JD JSON after two recovery attempts."
                )

    return _normalize_jd_structured(structured)


def _normalize_jd_structured(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the structured JD dictionary to ensure required keys exist
    with correct types regardless of what the LLM returned.

    Args:
        data (Dict[str, Any]): Raw structured JD dictionary.

    Returns:
        Dict[str, Any]: Normalized structured JD dictionary.
    """
    def _as_list(value: Any) -> list:
        """Return value as a list, or empty list if None/wrong type."""
        if isinstance(value, list):
            return value
        return []

    def _dedup(items: list) -> list:
        """Deduplicate case-insensitively, preserving order and original casing."""
        seen: set = set()
        result = []
        for item in items:
            if not isinstance(item, str):
                continue
            key = item.lower().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result

    return {
        "role_title":       data.get("role_title") or None,
        "required_skills":  _dedup(_as_list(data.get("required_skills"))),
        "preferred_skills": _dedup(_as_list(data.get("preferred_skills"))),
        "responsibilities": _dedup(_as_list(data.get("responsibilities"))),
        "keywords":         _dedup(_as_list(data.get("keywords"))),
    }