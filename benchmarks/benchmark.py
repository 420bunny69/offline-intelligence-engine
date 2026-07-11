import sys
import os 
import csv
import json
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.client import query_model_with_timing
MODELS=["llama3.2:3b","mistral:7b","phi4-mini"]
PROMPTS = {
    "short": "What is the capital of France?",
    "medium": "Summarize the key benefits of local LLM deployment over cloud-based APIs in 3-4 sentences.",
    "long": (
        "Here are notes from a team meeting: 'John will finish the API integration by Friday. "
        "Sarah needs to review the design doc before Thursday, high priority. The team agreed "
        "to push the deployment to next sprint. Mike is blocked on the database migration and "
        "needs help from DevOps by end of week.' Summarize the key action items discussed."
    ),
}
RESULTS_DIR = os.path.dirname(__file__)
def run_benchmark_test():
    results = []
    for model in MODELS:
        for prompt_type, prompt in PROMPTS.items():
            print(f"Running {model} | {prompt_type} prompt...")
            try:
                result = query_model_with_timing(prompt=prompt, model=model)
                result["prompt_type"] = prompt_type
                results.append(result)
                print(f"TTFT: {result['ttft_sec']}s | Tokens/sec: {result['tokens_per_sec']} | Total: {result['total_latency_sec']}s")
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "model": model, "prompt_type": prompt_type,
                    "ttft_sec": None, "total_latency_sec": None,
                    "tokens_per_sec": None, "output_tokens": None, "response": f"ERROR: {e}",
                })

    # Saving raw results in json file
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved raw results to {json_path}")

    return results

def save_summary(results):
    df=pd.DataFrame(results)
    summary=(
        df.groupby("model")[
            ["ttft_sec", "tokens_per_sec", "total_latency_sec"]
        ]
        .mean()
        .round(3)
    )
    print("\nBenchmark Summary")
    print(summary)

    summary.to_csv("benchmarks/summary.csv")

if __name__ == "__main__":
    results=run_benchmark_test()
    save_summary(results)