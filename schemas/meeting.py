from pydantic import BaseModel, Field
from typing import Optional

class ActionItem(BaseModel):
    #this represent one action item
    #represents a clear piece of work that needs to be done, who is responsible for doing it, and when it needs to be completed
    task: str = Field(description="The specific task or action to be done")
    owner: Optional[str] = Field(default=None, description="Person responsible, if mentioned")
    due_date: Optional[str] = Field(default=None, description="Deadline, if any")
    priority: Optional[str] = Field(default=None, description="Priority level: high, medium, low, if mentioned")
    #here task is important cause without it, it is not an action item

class MeetingExtraction(BaseModel):
    action_items: list[ActionItem]

def build_prompt(raw_text: str) -> str:
    return f"""Extract all action items from the following meeting notes.

    For each action item, identify:
    - task: what needs to be done
    - owner: who is responsible (if mentioned, otherwise null)
    - due_date: deadline as mentioned in the text (if any, otherwise null)
    - priority: high/medium/low (if stated or clearly implied, otherwise null)

    Return ONLY valid JSON in this exact format, no other text:
    {{
    "action_items": [
        {{"task": "...", "owner": "...", "due_date": "...", "priority": "..."}}
    ]
    }}

    Meeting notes:
    \"\"\"
    {raw_text}
    \"\"\"

    JSON:"""
#we didnt call any model or anything, just built a prompt