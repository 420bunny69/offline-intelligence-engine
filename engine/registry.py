from schemas.meeting import MeetingExtraction, build_prompt as meeting_prompt
from schemas.resume import ResumeInfo, build_prompt as resume_prompt
SCHEMA_REGISTRY = {
    "meeting": {
        "model": MeetingExtraction,
        "prompt_builder": meeting_prompt,
    },
    "resume": {
        "model": ResumeInfo,
        "prompt_builder": resume_prompt,
    },
}
#used dict instead of if else because dict lookup is fast
def get_schema(schema_name: str):
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema '{schema_name}'. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[schema_name]