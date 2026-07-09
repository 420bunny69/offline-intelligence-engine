import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.client import query_model

AVAILABLE_MODELS = ["llama3.2:3b", "mistral:7b", "phi4-mini"]

def main():
    print("Available models:", ", ".join(AVAILABLE_MODELS))
    model = input("Choose a model (default llama3.2:3b): ").strip() or "llama3.2:3b"
    if model not in AVAILABLE_MODELS:
        print(f"Unknown model, defaulting to llama3.2:3b")
        model = "llama3.2:3b"

    print(f"\nChatting with {model}. Type 'exit' to quit.\n")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            break
        response = query_model(prompt, model=model)
        print(f"\n{model}: {response}\n")

if __name__ == "__main__":
    main()
