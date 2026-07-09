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

def query_model_with_timing(prompt:str,model:str="llama3.2:3b",temperature: float = 0.0)->dict:
    """
    Stream the response, capture TTFT and total latency, and token count
    """
    options = {"temperature": temperature}
    if model in CPU_ONLY_MODELS:
        options["num_gpu"] = 0
    
    start=time.perf_counter()
    #intialising var for FTT
    first_token_time=None

    full_response = ""
    eval_count = 0
    stream=ollama.generate(
        model=model,
        prompt=prompt,
        options=options,
        stream=True
    )
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        full_response += chunk.get("response", "")    
        if chunk.get("done"):
            eval_count = chunk.get("eval_count", 0)
            eval_duration_ns = chunk.get("eval_duration", 1)
            eval_duration_secs = eval_duration_ns / 1000000000
    end = time.perf_counter()

    ttft = first_token_time - start if first_token_time else None #if-else is just safety check
    total_latency = end - start
    tokens_per_sec = (eval_count / eval_duration_secs) if eval_count and eval_duration_secs else None
    return {
        "model": model,
        "ttft_sec": round(ttft, 3) if ttft else None,
        "total_latency_sec": round(total_latency, 3),
        "tokens_per_sec": round(tokens_per_sec, 2) ,
        "output_tokens": eval_count,
        "response": full_response,
    }
#eval_count and eval_duration come straight from Ollama's own response, which is more realiable

