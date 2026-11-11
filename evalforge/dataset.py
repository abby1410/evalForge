import json
from pathlib import Path

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"

def load_dataset(path=None):
    p = Path(path) if path else DEFAULT_DATASET
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    validate_dataset(data)
    return data

def validate_dataset(data):
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty JSON array.")
    ids = set()
    required = {"id", "category", "question", "reference_answer"}
    for i, item in enumerate(data):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Test case at index {i} is missing: {', '.join(sorted(missing))}")
        if item["id"] in ids:
            raise ValueError(f"Duplicate test case id: {item['id']}")
        ids.add(item["id"])
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Test case {item['id']} has an empty question.")
    return True

def export_dataset(data, path):
    validate_dataset(data)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
