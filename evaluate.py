"""Run a participant's attack against the fixed public benchmark."""
import argparse
from importlib import util
from importlib.metadata import version
import json
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from challenge import prepare, train
from score import RULES, compute_score, file_hash, release_hashes


def load_attack(path):
    specification = util.spec_from_file_location("participant_attack", path)
    if specification is None or specification.loader is None:
        raise ValueError("Submission must be an importable Python file")
    module = util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not callable(getattr(module, "attack", None)):
        raise ValueError("Submission must define a callable named attack")
    return module.attack


def evaluate(submission, output, progress=print):
    submission = Path(submission).resolve()
    code_hash = file_hash(submission)
    release = release_hashes()
    started = time.monotonic()
    runs = []
    with threadpool_limits(limits=1):
        for seed in RULES["public_seeds"]:
            data = prepare(seed)
            for defense in RULES["defenses"]:
                progress(f"Seed {seed}, {defense}: clean training and attacked training", flush=True)
                clean = train(data, defense)
                # Reload the participant module so each run starts independently.
                attacked = train(data, defense, load_attack(submission))
                runs.append({
                    "seed": seed, "defense": defense,
                    "test_ids": data["test_ids"],
                    "y_true": data["labels"][data["test"]].tolist(),
                    "clean_predictions": clean["predictions"],
                    "attacked_predictions": attacked["predictions"],
                    "attack_rounds": attacked["attack_rounds"],
                })
    if release_hashes() != release or file_hash(submission) != code_hash:
        raise ValueError("Submission or evaluator files changed during evaluation")
    result = {
        "schema_version": 1, "rules": RULES, "submission_sha256": code_hash,
        "release_sha256": release,
        "packages": {name: version(name) for name in (
            "PennyLane", "flwr", "autograd", "numpy", "scikit-learn", "scipy"
        )},
        "runs": runs, "summary": compute_score(runs),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the previous complete result if writing a replacement fails.
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("attack.py"))
    parser.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    args = parser.parse_args()
    try:
        result = evaluate(args.submission, args.output)
    except (ValueError, OSError, TypeError, AttributeError) as error:
        parser.exit(1, f"Evaluation failed. {error}\n")
    print(f"Score: {result['summary']['score']:.8f} percentage points")
    print(f"Saved result: {args.output}")


if __name__ == "__main__":
    main()
