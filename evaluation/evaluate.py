import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.extract import extract
MODELS = ["llama3.2:3b", "mistral:7b", "phi4-mini"]
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")

# loading dataset
def load_data():
    with open(DATASET_PATH,"r") as f:
        return json.load(f)

def task_matches(expected_task,predicted_task):
    expected_words = expected_task.lower().split()
    predicted_words = predicted_task.lower().split()

    matching_word_count = 0
    for word in expected_words:
        if word in predicted_words:
            matching_word_count += 1

    if len(expected_words) == 0:
        return False

    overlap_ratio = matching_word_count / len(expected_words)
    return overlap_ratio >= 0.6  # at least 60% of expected words must appear

def find_matching_prediction(expected_item, predicted_items):
    """Loop through predictions, return the first one whose task text matches."""
    for pred in predicted_items:
        if task_matches(expected_item["task"], pred.task):
            return pred
    return None
def score_one_sample(expected_items, predicted_items):
    matched = 0
    owner_correct = 0
    due_date_correct = 0

    for expected_item in expected_items:
        pred = find_matching_prediction(expected_item, predicted_items)
        if pred is not None:
            print(f"MATCHED: expected='{expected_item['task']}' <-> predicted='{pred.task}'")
            matched += 1
            if pred.owner == expected_item["owner"]:
                owner_correct += 1
            if pred.due_date == expected_item["due_date"]:
                due_date_correct += 1
        else:
            print(f"NO MATCH for expected='{expected_item['task']}'")
    return {
        "expected_count": len(expected_items),
        "predicted_count": len(predicted_items),
        "matched": matched,
        "owner_correct": owner_correct,
        "due_date_correct": due_date_correct,
    }
def evaluate_model(model, dataset):
    print(f"\n=== Evaluating {model} ===")

    total_expected = 0
    total_predicted = 0
    total_matched = 0
    total_owner_correct = 0
    total_due_date_correct = 0
    json_failures = 0

    for sample in dataset:
        result = extract(sample["text"], "meeting", model=model, temperature=0.0)

        if not result["success"]:
            json_failures += 1
            print(f"  Sample {sample['id']}: extraction FAILED")
            continue

        predicted_items = result["data"].action_items
        scores = score_one_sample(sample["expected"], predicted_items)

        total_expected += scores["expected_count"]
        total_predicted += scores["predicted_count"]
        total_matched += scores["matched"]
        total_owner_correct += scores["owner_correct"]
        total_due_date_correct += scores["due_date_correct"]

        print(f"  Sample {sample['id']}: expected {scores['expected_count']}, "
              f"predicted {scores['predicted_count']}, matched {scores['matched']}")

    # Computing final metrics & guarding against division by zero
    recall = total_matched / total_expected if total_expected > 0 else 0
    precision = total_matched / total_predicted if total_predicted > 0 else 0
    owner_accuracy = total_owner_correct / total_matched if total_matched > 0 else 0
    due_date_accuracy = total_due_date_correct / total_matched if total_matched > 0 else 0
    json_validity_rate = (len(dataset) - json_failures) / len(dataset)

    return {
        "model": model,
        "json_validity_rate": round(json_validity_rate, 3),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "owner_accuracy": round(owner_accuracy, 3),
        "due_date_accuracy": round(due_date_accuracy, 3),
    }
def run_evaluation():
    dataset = load_data()
    all_results = []

    for model in MODELS:
        result = evaluate_model(model, dataset)
        all_results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Final Comparison ===")
    for r in all_results:
        print(r)

    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()