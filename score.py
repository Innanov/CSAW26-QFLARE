"""Recompute one attack score from a saved result. Standard library only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RULES = json.loads((ROOT / "rules.json").read_text(encoding="utf-8"))
RELEASE_FILES = (
    "rules.json", "quantum_model.py", "challenge.py", "evaluate.py",
    "score.py", "flower_client.py", "requirements.txt",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_hashes():
    return {name: file_hash(ROOT / name) for name in RELEASE_FILES}


def macro_f1(truth, prediction):
    if not isinstance(truth, list) or not isinstance(prediction, list):
        raise ValueError("Labels and predictions must be lists")
    if not truth or len(truth) != len(prediction):
        raise ValueError("Label and prediction counts differ or are empty")
    if any(type(value) is not int or value not in (0, 1) for value in truth + prediction):
        raise ValueError("Labels and predictions must be binary integers")
    if set(truth) != {0, 1}:
        raise ValueError("The test set must contain both classes")
    scores = []
    for label in (0, 1):
        correct = sum(t == label and p == label for t, p in zip(truth, prediction))
        scores.append(2 * correct / (truth.count(label) + prediction.count(label)))
    return math.fsum(scores) / 2


def compute_score(runs):
    expected = {
        (seed, defense)
        for seed in RULES["public_seeds"] for defense in RULES["defenses"]
    }
    keys = [(row["seed"], row["defense"]) for row in runs]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected exactly one run per public seed and defense")
    details = []
    for row in runs:
        rounds = row["attack_rounds"]
        if not isinstance(rounds, list) or any(
            type(index) is not int or not 0 <= index < RULES["rounds"] for index in rounds
        ) or rounds != sorted(set(rounds)) or len(rounds) > RULES["max_attack_rounds"]:
            raise ValueError("Invalid attack-round record or exceeded round budget")
        clean = macro_f1(row["y_true"], row["clean_predictions"])
        attacked = macro_f1(row["y_true"], row["attacked_predictions"])
        details.append({
            "seed": row["seed"], "defense": row["defense"],
            "clean_f1": clean, "attacked_f1": attacked,
            "drop_pp": 100 * (clean - attacked),
            "attack_rounds_used": len(rounds),
        })
    for seed in RULES["public_seeds"]:
        rows = [row for row in runs if row["seed"] == seed]
        if any(row["y_true"] != rows[0]["y_true"] or row["test_ids"] != rows[0]["test_ids"] for row in rows):
            raise ValueError("Test examples differ between defenses")
        if len(rows[0]["test_ids"]) != len(rows[0]["y_true"]):
            raise ValueError("Test IDs and labels have different lengths")
        ids = rows[0]["test_ids"]
        if any(type(index) is not int or index < 0 for index in ids) or len(set(ids)) != len(ids):
            raise ValueError("Test IDs must be unique nonnegative integers")
    mean_by_defense = {
        defense: math.fsum(row["drop_pp"] for row in details if row["defense"] == defense)
        / len(RULES["public_seeds"])
        for defense in RULES["defenses"]
    }
    return {
        "score": math.fsum(row["drop_pp"] for row in details) / len(details),
        "tie_break": min(mean_by_defense.values()),
        "unit": "macro-F1 percentage points",
        "runs": details,
    }


def load_result(path, submission=None):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("Unsupported result schema")
    if result["rules"] != RULES or result["release_sha256"] != release_hashes():
        raise ValueError("Result belongs to different challenge rules or evaluator files")
    digest = result["submission_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid submission hash")
    if submission is not None and file_hash(submission) != digest:
        raise ValueError("Submission code differs from the saved result")
    return compute_score(result["runs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(load_result(args.result, args.submission), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Cannot score this result. {error}\n")


if __name__ == "__main__":
    main()
