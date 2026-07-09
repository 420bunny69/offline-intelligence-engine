import ollama
import time
CPU_ONLY_MODE={"mistral:7b"}

def query_model(
    prompt: str,
    model: str = "llama3.2:3b",
    # means model which is a string will have default value of llama3.2:3b
    temperature: float = 0.0
) -> str:
    """Send a prompt to a local Ollama model and return the raw text response"""
    options={"temperature": temperature} #  lowercase "temperature" for Ollama API
    if model in CPU_ONLY_MODE:
            options["num_gpu"]=0
    
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options=options,
        )
    # above was sending a request to Ollama
    return response["response"]