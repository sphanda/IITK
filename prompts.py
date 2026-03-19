"""
Prompt templates for the CV Creation capstone project.

Each function accepts the required data as parameters and returns a fully
populated prompt string ready to send to the LLM. Placeholders are filled
here — callers never need to do string replacement themselves, eliminating
the risk of accidentally sending a literal placeholder to the model.
"""


def get_profile_extraction_prompt(profile_text: str) -> str:
    """
    Build the prompt for structured profile extraction.

    Args:
        profile_text (str): Raw unstructured profile text from the user.

    Returns:
        str: Fully populated extraction prompt.
    """
    return f"""You will receive unstructured text describing a person's background.

Your job is to extract information EXACTLY as it appears in the text — do not paraphrase,
summarize, or omit any specific values such as email addresses, phone numbers, URLs, dates,
grades, or company names. Copy them verbatim into the appropriate fields.

Return VALID JSON ONLY.
Return only the JSON object, starting with {{ and ending with }}.
No preamble, no explanation, no markdown fences.

{{
  "name": string or null,
  "contact": {{
    "email": string or null,
    "phone": string or null,
    "linkedin": string or null,
    "github": string or null,
    "location": string or null,
    "other": string or null
  }},
  "summary": string or null,
  "education": [
    {{
      "degree": string or null,
      "institution": string or null,
      "start_year": int or null,
      "end_year": int or null,
      "grade": string or null
    }}
  ],
  "experience": [
    {{
      "title": string or null,
      "company": string or null,
      "start_date": string or null,
      "end_date": string or null,
      "responsibilities": [string]
    }}
  ],
  "skills": [string],
  "projects": [
    {{
      "name": string or null,
      "description": string or null,
      "technologies": [string]
    }}
  ],
  "achievements": [string]
}}

CRITICAL RULES — you must follow all of these:
1. "email": copy the email address exactly as written, including dots, underscores, and domain. Remove any leading or trailing whitespace.
2. "phone": copy the phone number exactly, including country code and formatting.
3. "linkedin": copy ONLY if a LinkedIn URL or handle explicitly appears in the text. If absent, set to null. Do NOT invent or guess a URL.
4. "github": copy ONLY if a GitHub URL or handle explicitly appears in the text. If absent, set to null. Do NOT invent or guess a URL.
5. "location": scan the ENTIRE text for any city, country, or region mentioned near the person's name or contact details and copy it exactly. Do not set to null if a location appears anywhere in the text.
6. "grade": copy GPA, percentage, or classification exactly (e.g. "8.4 CGPA", "First Class").
7. "start_date" / "end_date": copy dates exactly as written (e.g. "Jun 2022", "2021-07").
8. "company" and "institution": copy names exactly — do not abbreviate or expand.
9. "education": include only actual degree or diploma entries (e.g. B.Tech, M.Sc, MBA). Do NOT create a separate entry for coursework, subjects, or modules listed under a degree — these are not separate degrees.
10. "skills": extract as a flat list of individual skill names (e.g. ["Python", "SQL", "Pandas"]). Do NOT include category labels like "Programming:" or "Libraries & Tools:" — just the skill names themselves.
11. "achievements": extract ALL certifications, awards, competition results, publications, and notable accomplishments. Look carefully — they may appear under headings like "Certifications", "Awards", "Achievements", or "Additional Information".
12. "technologies" in projects: list ONLY tools and technologies explicitly mentioned for THAT specific project in the text. Do NOT add tools from other projects or invent frameworks not mentioned.
13. If a field is genuinely absent from the text, set it to null or []. Never invent values.

TEXT:
{profile_text}
"""


def get_jd_parsing_prompt(jd_text: str) -> str:
    """
    Build the prompt for structured job description parsing.

    Args:
        jd_text (str): Raw job description text.

    Returns:
        str: Fully populated JD parsing prompt.
    """
    return f"""You will receive a job description.

Extract key information and return VALID JSON ONLY.
Return only the JSON object, starting with {{ and ending with }}.
No preamble, no explanation, no markdown fences.

{{
  "role_title": string or null,
  "required_skills": [string],
  "preferred_skills": [string],
  "responsibilities": [string],
  "keywords": [string]
}}

RULES:
1. "role_title": copy the exact job title as written.
2. "required_skills": list every explicitly required skill or technology as a separate string.
3. "preferred_skills": list every preferred or nice-to-have skill as a separate string.
4. "responsibilities": copy each responsibility as a short phrase, one per list item.
5. "keywords": include all important technical terms, tools, domain words, and soft skills
   that an ATS system would scan for. These may overlap with required_skills.
6. If any field is absent, use null or an empty list. Never invent content.

TEXT:
{jd_text}
"""


def get_general_cv_prompt(profile_json: str) -> str:
    """
    Build the prompt for general ATS-friendly CV generation.

    Args:
        profile_json (str): Pretty-printed JSON string of the structured profile.

    Returns:
        str: Fully populated general CV generation prompt.
    """
    return f"""You are an expert CV writer. Using the structured profile data below, write a professional, ATS-friendly CV.

CRITICAL — contact details must appear at the top of the CV, exactly as provided:
- Print name on the first line.
- Print each non-null contact field on its own line in this order: email, phone, LinkedIn, GitHub, location.
- Do NOT omit, alter, or paraphrase any contact detail. Copy them verbatim from the JSON.
- If contact.location is not null, it MUST appear in the contact block.

Guidelines:
- Use these exact section headings: SUMMARY, EDUCATION, EXPERIENCE, PROJECTS, SKILLS, ACHIEVEMENTS.
- Use concise bullet points under each section where appropriate.
- SKILLS: group into categories on separate lines, e.g. "Programming: Python, SQL" | "ML & Data Science: scikit-learn, Classification, Regression" | "Visualization: Matplotlib, Seaborn, Power BI". Include every skill from the profile.
- Copy all dates, grades, company names, and institution names exactly as they appear in the JSON.
- Do not invent information not present in the input; you may lightly rephrase responsibilities for clarity and impact.
- ACHIEVEMENTS section: use ONLY items from the "achievements" list in the JSON. If the list is empty, omit the section entirely — do NOT copy bullets from EXPERIENCE or PROJECTS as achievements.
- Output plain text only. No markdown, no JSON, no extra commentary.

PROFILE DATA (JSON):
{profile_json}
"""


def get_tailored_cv_prompt(profile_json: str, jd_json: str) -> str:
    """
    Build the prompt for a JD-tailored ATS-friendly CV.

    Args:
        profile_json (str): Pretty-printed JSON string of the structured profile.
        jd_json (str): Pretty-printed JSON string of the structured job description.

    Returns:
        str: Fully populated tailored CV generation prompt.
    """
    return f"""You are an expert CV writer. Using the structured profile data and job description below, write a professional, ATS-friendly CV tailored to the job.

CRITICAL — contact details must appear at the top of the CV, exactly as provided:
- Print name on the first line.
- Print each non-null contact field on its own line in this order: email, phone, LinkedIn, GitHub, location.
- Do NOT omit, alter, or paraphrase any contact detail. Copy them verbatim from the JSON.
- If contact.location is not null, it MUST appear in the contact block.

Guidelines:
- Use these exact section headings: SUMMARY, EDUCATION, EXPERIENCE, PROJECTS, SKILLS, ACHIEVEMENTS.
- SUMMARY: write 2-3 sentences in professional third-person CV tone (no "I" or "my"). Name the target role_title from the JD, mention 2-3 required skills, and frame experience toward the role. Do NOT write cover-letter language like "this role aligns perfectly".
- EXPERIENCE: keep roles in reverse-chronological order (most recent first). Within each role, reorder bullet points to lead with those most relevant to the JD responsibilities — do not add, remove, or invent bullets.
- PROJECTS: list the most JD-relevant project first.
- SKILLS: group into categories on separate lines: Programming, ML & Data Science, NLP, Visualization, Other. Include ALL skills from the profile JSON — do not omit any.
- Copy all dates, grades, company names, and institution names exactly as they appear in the JSON.
- Do not invent fake roles, degrees, companies, or skills.
- ACHIEVEMENTS section: use ONLY items from the "achievements" list in the JSON. If the list is empty, omit the section entirely — do NOT copy bullets from EXPERIENCE or PROJECTS as achievements.
- Output plain text only. No markdown, no JSON, no extra commentary.

PROFILE DATA (JSON):
{profile_json}

JOB DESCRIPTION DATA (JSON):
{jd_json}
"""


def get_cv_evaluation_prompt(jd_json: str, cv_text: str) -> str:
    """
    Build the prompt for CV evaluation and refinement suggestions.

    Args:
        jd_json (str): Pretty-printed JSON string of the structured JD,
            or '{}' if no JD was provided (evaluation will skip JD alignment).
        cv_text (str): Plain text of the CV to evaluate.

    Returns:
        str: Fully populated evaluation prompt.
    """
    # Explicitly tell the model when no JD is available so it skips
    # JD-alignment feedback rather than hallucinating a role to compare against.
    jd_context = (
        "No job description was provided. Skip JD alignment feedback and focus only on "
        "general CV quality, clarity, bullet point impact, and ATS readiness."
        if jd_json.strip() in ("{}", "")
        else f"JOB DESCRIPTION (JSON):\n{jd_json}"
    )

    return f"""You will evaluate a candidate CV and provide concise refinement suggestions.

Return OUTPUT ONLY as bullet lines.
Rules:
- Output at most 8 bullets.
- Each bullet MUST be exactly one line and MUST start with '-' (dash + space).
- Output no other text before or after the bullets.
- Do not use Markdown (no '**', no bold). Plain text only.
- If there are no useful suggestions, output exactly one bullet: '- No refinement needed.'.

{jd_context}

CV TEXT:
{cv_text}
"""