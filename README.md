# CSAW26-QFLARE

## Quantum Federated Learning Attack Challenge

Develop an attack that reduces the performance of a federated quantum classifier.

Eight clients train a shared model to recognize handwritten **3** and **8**. You control the quantum circuit updates sent by **two clients**. You may change those updates in **at most 30 of the 60 training rounds**. Your goal is to achieve the highest average drop in test macro-F1 across the evaluation cases.

**Edit only `attack.py`. Submit your attack code and the JSON produced by the evaluator.**

## 1. Set up your environment

Use **Python 3.11 or 3.12**. Extract or clone the repository and open a terminal in its root folder, where `README.md` and `attack.py` are located.

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it using the command for your operating system.

| Operating system | Activation command |
| --- | --- |
| macOS or Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |

Install the dependencies.

```bash
python -m pip install -r requirements.txt
```

## 2. Run the provided starter

```bash
python evaluate.py --submission attack.py --output outputs/result.json
```

The evaluator trains the clean and attacked models for both public seeds and all three defenses. Allow several minutes on a CPU. Progress is printed for each case. When evaluation finishes, the terminal displays your score and the path to the saved result.

The `outputs` folder is created automatically. The released starter's reference score is **0.56530779 percentage points**.

Read the saved result and calculate its score again.

```bash
python score.py outputs/result.json --submission attack.py
```

This second command uses the saved predictions and labels. It runs without retraining and displays the score, the result for each case, and the number of attack rounds used.

## 3. Develop your attack

Open **`attack.py`** and edit its `attack` function. Keep the function name and its four arguments.

```python
def attack(global_params, own_deltas, round_idx, state):
    ...
    return replacement_circuit_updates, new_state
```

An **update**, or **delta**, is a change to model parameters.

| Argument | Shape or type | What you receive |
| --- | --- | --- |
| `global_params` | NumPy array, `(34,)` | The current shared model parameters. |
| `own_deltas` | NumPy array, `(2, 34)` | The honest local updates produced by your two clients. |
| `round_idx` | Integer, 0–59 | The current training round. |
| `state` | Your choice | The state returned by your previous call. Initially `None`. |

Return these two values.

| Return value | Required form |
| --- | --- |
| `replacement_circuit_updates` | A real, finite NumPy array with shape `(2, 24)`. Each row contains the replacement circuit deltas for one client. |
| `new_state` | Any state your method needs in the next round, or `None`. |

The two client rows keep the same order during an evaluation run. The evaluator starts a fresh attack module and resets `state` to `None` for each seed and defense.

### Attack or skip a round

To **skip** a round, return the honest circuit deltas exactly.

```python
return own_deltas[:, :24].copy(), state
```

To **reverse** the circuit deltas in an attack round, return their negatives.

```python
return -own_deltas[:, :24].copy(), state
```

The starter reverses updates in rounds **0, 2, 4, …, 58** and skips the other rounds. This uses 30 attack rounds. You can change the attack direction, coordinate the two clients, and choose a different schedule.

### Parameter layout

The model has 34 parameters.

| Coordinates | Contents |
| --- | --- |
| `0:24` | Quantum circuit weights, reshaped to `(2, 4, 3)` for PennyLane's `StronglyEntanglingLayers`. |
| `24:34` | Classical linear-head weights and biases. |

The circuit weights are ordered by layer, then qubit, then the three angles of `qml.Rot`. Your function returns replacements for the first 24 coordinates. The evaluator preserves the honest head updates and the updates of the six honest clients.

## 4. Follow the attack constraints

| Rule | Limit |
| --- | --- |
| Compromised clients | Two of eight. |
| Attack rounds | At most **30 of 60** per evaluation run. |
| Round counting | A round counts once if either client's applied circuit update differs from its honest update. |
| Skipping after 30 attacks | Return `own_deltas[:, :24].copy()`. |
| Output shape | Exactly `(2, 24)`. |
| Output values | Real and finite. |
| Invalid output or a 31st attack round | Evaluation fails with an error. |

The per-client perturbation is also bounded. For each client, let `d_i` be its honest circuit delta and `a_i` be the applied circuit delta. The evaluator enforces

```text
L2(a_i - d_i) <= 2 * max(L2(d_i), 0.01)
```

Here, `L2(x)` is the Euclidean length of vector `x`.

It projects oversized proposals onto this radius. The 30-round limit resets for each seed and defense.

Use **NumPy and the Python standard library** in a self-contained `attack.py`. Base your method on the function arguments and your state. Initialize any random generator with an explicit fixed seed.

The attack must not read benchmark files, access test labels, inspect the evaluator process, use network or filesystem access, change the training procedure, or embed answers for particular evaluation seeds.

## 5. Understand your score

There are **six cases** — two public seeds, 11 and 29, evaluated against FedAvg, median, and trimmed-mean aggregation.

For each case, the evaluator trains a clean model and an attacked model from identical initial conditions. It measures both models' test macro-F1. The score is the average performance drop in percentage points.

```text
Score = 100 * average(Clean F1 - Attacked F1)
```

The average includes all six evaluation cases.

**Higher scores are better.** For one case, clean F1 of 0.96 and attacked F1 of 0.90 gives a six-percentage-point drop. If your attack improves performance in a case, that negative drop counts in the average.

The `tie_break` field is the lowest mean drop among the three defenses, in percentage points. Full-precision values are used for comparisons.

The saved JSON contains predictions, labels, used attack rounds, evaluation settings, and source-file hashes. The scoring command calculates macro-F1 from the saved predictions and labels. With `--submission attack.py`, it also checks that the result belongs to the submitted code.

## 6. Evaluate your final code and submit

After editing `attack.py`, run these commands again.

```bash
python evaluate.py --submission attack.py --output outputs/result.json
python score.py outputs/result.json --submission attack.py
```

Submit **both files** to the teaching team.

1. `attack.py`
2. `outputs/result.json`

Generate the JSON using the exact code you submit. The teaching team can replay your score and rerun your source to verify the result.

## Files in this repository

| File | Role | Edit? |
| --- | --- | --- |
| `attack.py` | Your attack method. | **Yes** |
| `evaluate.py` | Runs the benchmark and saves results. | No |
| `score.py` | Recalculates a saved score. | No |
| `challenge.py` | Dataset splits, training rounds, and attack constraints. | No |
| `flower_client.py` | Flower client training. | No |
| `quantum_model.py` | PennyLane quantum classifier. | No |
| `rules.json` | Fixed benchmark configuration and limits. | No |
| `requirements.txt` | Dependency versions. | No |

The evaluator files define the shared benchmark and are checked by the scoring script.

## Benchmark details

- Scikit-learn Digits, classes 3 and 8, with a stratified 75% training / 25% test split.
- Four whitened PCA features fitted on the training split, clipped to ±3 and scaled by π/6.
- Four qubits, `AngleEmbedding`, two layers of `StronglyEntanglingLayers`, four Pauli-X expectations, and a two-class linear head.
- PennyLane `default.qubit` with `shots=None`, full statevector simulation, and NumPy/Autograd differentiation.
- Flower clients and built-in FedAvg, FedMedian, and FedTrimmedAvg strategies executed locally.
- Eight mildly non-IID clients, 60 rounds, one local epoch per round, and SGD learning rate 0.5.

The full configuration is in `rules.json`.

## Common errors

| Message | What to check |
| --- | --- |
| Attack round budget exceeded | Limit changed rounds to 30. Return the honest circuit deltas in other rounds. |
| Attack must return a real array with shape (2, 24) | Return the circuit deltas for both clients as a two-dimensional array. |
| Submission code differs from the saved result | Rerun evaluation after editing `attack.py`. |
| Result belongs to different challenge rules or evaluator files | Use the released evaluator files and regenerate the JSON with that version. |
