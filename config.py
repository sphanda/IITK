"""
Configuration module for the CV Creation using LLMs capstone project.

Holds default model names, file names, and other simple app-level settings.
"""

# Default model names for local Ollama-based LLMs.
DEFAULT_EXTRACTION_MODEL = "gemma3:1b"   # Used for extraction/parsing/evaluation.
DEFAULT_GENERATION_MODEL = "gemma3:4b"   # Used for CV generation/refinement.

# Default output directory if not specified via CLI.
DEFAULT_OUTPUT_DIR = "./outputs"

# Runtime-generated filenames (within the chosen output directory).
STRUCTURED_PROFILE_FILENAME = "structured_profile.json"
JD_ANALYSIS_FILENAME        = "jd_analysis.json"
CV_GENERAL_FILENAME         = "cv_general.txt"
CV_TAILORED_FILENAME        = "cv_tailored.txt"
CV_EVALUATION_FILENAME      = "cv_evaluation.txt"

# Optional document exports (CVs in docx/pdf formats).
CV_GENERAL_DOCX_FILENAME = "cv_general.docx"
CV_GENERAL_PDF_FILENAME = "cv_general.pdf"
CV_TAILORED_DOCX_FILENAME = "cv_tailored.docx"
CV_TAILORED_PDF_FILENAME = "cv_tailored.pdf"

# LLM generation settings.
DEFAULT_TEMPERATURE_EXTRACTION = 0.1   # Low temperature keeps extraction stable.
DEFAULT_TEMPERATURE_GENERATION = 0.3   # Slightly higher for natural CV prose.

# Token limits.
# Extraction (gemma3:1b): capped at 2048 — enough for any structured JSON
# profile or JD, and prevents the model from rambling past the closing brace
# which can truncate or corrupt the JSON object.
# Generation (gemma3:4b): uncapped (None) — CVs vary in length and a hard
# limit risks cutting off the ACHIEVEMENTS or SKILLS section mid-way.
DEFAULT_MAX_TOKENS_EXTRACTION = 4096
DEFAULT_MAX_TOKENS_GENERATION = None

# Miscellaneous.
LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"