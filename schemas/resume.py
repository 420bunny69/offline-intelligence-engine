from pydantic import BaseModel
from typing import Optional

class ResumeInfo(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str]
    education: list[str]
    experience: list[str]
    total_years_experience: Optional[float] = None
def build_prompt(raw_text: str) -> str:
    return f"""Extract structured candidate information from the following resume text.
    Identify:
    - name: the candidate's full name
    - email: email address, if present, otherwise null
    - phone: phone number, if present, otherwise null
    - skills: a list of technical skills mentioned (programming languages, tools, frameworks)
    - education: a list of education entries (e.g. "B.Tech ECE, XYZ College, 2027")
    - experience: a list of work/internship experience entries (e.g. "SDE Intern, Company X, 2025")
    - total_years_experience: estimated total years of professional experience as a number, or null if unclear

    Return ONLY valid JSON in this exact format, no other text:
    {{
    "name": "...",
    "email": "...",
    "phone": "...",
    "skills": ["...", "..."],
    "education": ["...", "..."],
    "experience": ["...", "..."],
    "total_years_experience": 0.0
    }}

    Resume text:
    \"\"\"
    {raw_text}
    \"\"\"

    JSON:"""