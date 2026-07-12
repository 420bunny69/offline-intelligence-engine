import json 
import re 
#re -> regular expression, for pattern matching
from pydantic import ValidationError
from engine.client import query_model
from engine.registry import get_schema

def extract_json_block(text: str)->str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    #this line basically used to find and extract raw json string
    if not match:
        raise ValueError("No JSON object found in model output")
    return match.group(0)
def extract(raw_text: str, schema_name: str, model: str = "llama3.2:3b", temperature: float = 0.0):
    schema = get_schema(schema_name)
    prompt = schema["prompt_builder"](raw_text)
    for attempt in range(2):
        raw_output = query_model(prompt, model=model, temperature=temperature)
        try:
            json_str = extract_json_block(raw_output) #removes junk
            parsed = json.loads(json_str) #transform string to dict
            validated = schema["model"](**parsed) # ** mean unpacking dict
            return {
                "success": True,
                "data": validated,
                "attempts": attempt + 1,
                "raw_output": raw_output,
            }
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            if attempt == 0:
                # Retring with a firmer instruction
                prompt = prompt + "\n\nIMPORTANT: Your previous output was not valid JSON. Return ONLY the JSON object, nothing else."
                continue
            return {
                "success": False,
                "data": None,
                "attempts": attempt + 1,
                "raw_output": raw_output,
                "error": str(e),
            }
