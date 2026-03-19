"""
LLM client module for interacting with local models via the Ollama HTTP API.

Uses Python's standard library (urllib) to avoid extra dependencies.
Calls the local Ollama /api/generate endpoint directly, which is the only
reliable way to pass inference parameters (temperature, num_predict) across
all Ollama versions. The CLI does not support these flags consistently.
"""

import json
import urllib.error
import urllib.request
from typing import Optional

from utils import log_error


# Default Ollama API base URL. Override via OLLAMA_HOST env var if needed.
OLLAMA_API_BASE = "http://localhost:11434"


class LLMError(Exception):
    """Custom exception for LLM-related failures."""
    pass


def generate_with_model(
    model_name: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Generate text from a local LLM using the Ollama HTTP API.

    Calls POST /api/generate with stream=False to get a single complete
    response. This is more reliable than the CLI for passing inference
    parameters and avoids ANSI escape code pollution in the output.

    Args:
        model_name (str): Ollama model name (e.g., 'gemma3:1b').
        prompt (str): Prompt text to send to the model.
        temperature (float): Sampling temperature for generation.
        max_tokens (Optional[int]): Optional maximum tokens to predict
            (maps to Ollama's num_predict). None means model default.

    Returns:
        str: Generated text output from the model.

    Raises:
        LLMError: If the Ollama API is unreachable, returns an error,
                  or returns empty output.
    """
    if not prompt.strip():
        raise ValueError("Prompt is empty; cannot call LLM with empty prompt.")

    # Build options dict — only include num_predict if explicitly set.
    options: dict = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,  # Get a single complete JSON response.
        "options": options,
    }

    url = f"{OLLAMA_API_BASE}/api/generate"
    request_body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as exc:
        # Covers both connection refused (Ollama not running) and DNS errors.
        log_error(f"Could not connect to Ollama at {OLLAMA_API_BASE}: {exc.reason}")
        raise LLMError(
            f"Cannot reach Ollama at {OLLAMA_API_BASE}. "
            "Is 'ollama serve' running?"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log_error(f"Unexpected error calling Ollama API: {exc}")
        raise LLMError("Unexpected error calling Ollama API.") from exc

    # Parse the JSON response.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_error(f"Failed to parse Ollama API response as JSON: {raw[:200]}")
        raise LLMError("Ollama returned non-JSON response.") from exc

    # Check for API-level errors (e.g. model not found).
    if "error" in data:
        error_msg = data["error"]
        log_error(f"Ollama API error: {error_msg}")
        if "not found" in error_msg.lower():
            raise LLMError(
                f"Model '{model_name}' not found. "
                f"Did you run 'ollama pull {model_name}'?"
            )
        raise LLMError(f"Ollama API returned error: {error_msg}")

    output_text = data.get("response", "").strip()
    if not output_text:
        raise LLMError("Ollama returned empty output.")

    return output_text