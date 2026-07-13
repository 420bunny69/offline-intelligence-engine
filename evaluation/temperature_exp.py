import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.extract import extract

MODEL = "llama3.2:3b"
RUNS_PER_TEMP = 5
TEMPERATURES = [0.0, 0.7]
TEST_TEXT = (
    "John will finish the API integration by Friday. Sarah needs to review the "
    "design doc before Thursday, high priority. The team agreed to push the "
    "deployment to next sprint. Mike is blocked on the database migration and "
    "needs help from DevOps by end of week."
)
RESULTS_DIR = os.path.dirname(__file__)
# main function of the experiment
def run_exp():
    all_results={}
    for temp in TEMPERATURES:
        print(f"\nTemperature: {temp}")
        runs=[] # this will contain run1, run2, run3, etc for particular temp
        for i in range(RUNS_PER_TEMP):
            result = extract(TEST_TEXT, "meeting", model=MODEL, temperature=temp)
            if (result["success"]==True):
                items=result["data"].action_items
                run_summary={
                    "run": i + 1,
                    "attempts": result["attempts"],
                    "num_items": len(items),
                    "tasks": [item.task.strip().lower() for item in items],
                    "owners": [item.owner for item in items],
                    "priorities": [item.priority for item in items],
                }
            else:
                run_summary = {
                    "run": i + 1,
                    "attempts": result["attempts"],
                    "num_items": 0,
                    "tasks": [],
                    "owners": [],
                    "priorities": [],
                    "error": result["error"],
                }
            print(f"  Run {i+1}: {run_summary['num_items']} items, "
                  f"{run_summary['attempts']} attempt(s), tasks={run_summary['tasks']}")
            runs.append(run_summary)

        all_results[str(temp)] = runs
        # Saving raw results
    json_path = os.path.join(RESULTS_DIR, "temperature_experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results to {json_path}")

    analyze(all_results)

def analyze(all_results):
    print("\nConsistency Analysis")
    for temp, runs in all_results.items():
        print(f"\nTemp {temp}:")
        item_counts=[r["num_items"] for r in runs] # list comprehension
        print(f"Item counts per run: {item_counts}")
        if len(set(item_counts)) == 1:
            # set removes duplicate
            print("consistent (same count every run)")
        else:
            print("inconsistent (count changed between runs)")
        retries = 0
        for r in runs:
            if r["attempts"] > 1:
                retries += 1
        print(f"\nRuns that needed a retry: {retries} out of {len(runs)}")

        first_run_tasks = set(runs[0]["tasks"])
        print(f"  Tasks in run 1: {first_run_tasks}")
        for i, r in enumerate(runs[1:], start=2):
            this_run_tasks = set(r["tasks"])
            same = this_run_tasks == first_run_tasks
            print(f"  Run {i} tasks match run 1: {same}")

if __name__ == "__main__":
    run_exp()