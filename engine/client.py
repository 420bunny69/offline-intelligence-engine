import ollama

def query_model(
    prompt: str,
    model: str = "llama3.2:3b",
    # means model which is a string will have default value of llama3.2:3b
    temperature: float = 0.0
) -> str:
    """Send a prompt to a local Ollama model and return the raw text response"""
    
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature},  # Note: lowercase "temperature" for Ollama API
    )
    # above was sending a request to Ollama
    return response["response"]