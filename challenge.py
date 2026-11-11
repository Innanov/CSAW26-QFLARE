"""Fixed dataset preparation, honest-client training, and server defenses."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedMedian, FedTrimmedAvg

from flower_client import QuantumClient
from quantum_model import forward
from score import RULES


def prepare(seed):
    images, original_labels = load_digits(return_X_y=True)
    selected = np.isin(original_labels, RULES["classes"])
    dataset_ids = np.flatnonzero(selected)
    images = images[selected]
    labels = (original_labels[selected] == RULES["positive_class"]).astype(int)
    train, test = train_test_split(
        np.arange(len(labels)), test_size=1 - RULES["train_fraction"],
        stratify=labels, random_state=seed,
    )
    pca = PCA(n_components=RULES["input_features"], whiten=True, svd_solver="full").fit(images[train])
    features = np.clip(pca.transform(images), -3, 3) * (np.pi / 6)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        clients = [[] for _ in range(RULES["clients"])]
        for label in (0, 1):
            members = rng.permutation(train[labels[train] == label])
            proportions = rng.dirichlet(
                np.full(RULES["clients"], RULES["dirichlet_alpha"])
            )
            counts = rng.multinomial(len(members), proportions)
            for client, ids in zip(clients, np.split(members, np.cumsum(counts)[:-1])):
                client.extend(ids)
        if min(map(len, clients)) >= RULES["minimum_client_samples"]:
            break
    else:
        raise RuntimeError("Cannot generate a valid client partition")
    orders = [
        [rng.permutation(client) for client in clients] for _ in range(RULES["rounds"])
    ]
    initial = rng.normal(0, 0.2, RULES["model_parameters"])
    initial[32:] = 0
    compromised = np.sort(rng.choice(
        RULES["clients"], RULES["compromised_clients"], replace=False
    ))
    return {
        "features": features, "labels": labels, "test": test,
        "test_ids": dataset_ids[test].tolist(),
        "train_ids": dataset_ids[train].tolist(),
        "client_ids": [dataset_ids[np.asarray(client)].tolist() for client in clients],
        "orders": orders,
        "initial": initial, "compromised": compromised,
    }


def project(proposals, original):
    raw = np.asarray(proposals)
    expected = (RULES["compromised_clients"], RULES["circuit_parameters"])
    if raw.shape != expected or raw.dtype.kind not in "fiu":
        raise ValueError(f"Attack must return a real array with shape {expected}")
    proposals = raw.astype(float)
    if not np.isfinite(proposals).all():
        raise ValueError("Attack returned nonfinite circuit updates")
    perturbation = proposals - original
    limits = RULES["perturbation_factor"] * np.maximum(
        np.linalg.norm(original, axis=1), RULES["perturbation_floor"]
    )
    magnitude = np.max(np.abs(perturbation), axis=1)
    direction = perturbation / np.maximum(magnitude, 1e-300)[:, None]
    norm = np.linalg.norm(direction, axis=1)
    allowed = limits / np.maximum(norm, 1e-300)
    return original + direction * np.minimum(magnitude, allowed)[:, None]



def enforce_round_budget(proposals, original, attack_rounds, round_idx):
    applied = project(proposals, original)
    if not np.array_equal(applied, original):
        if len(attack_rounds) >= RULES["max_attack_rounds"]:
            raise ValueError(
                f"Attack round budget exceeded at round {round_idx}. "
                f"Use at most {RULES['max_attack_rounds']} rounds and return "
                "own_deltas[:, :24].copy() in other rounds."
            )
        attack_rounds.append(round_idx)
    return applied


def train(data, defense, attack=None):
    settings = {
        "min_fit_clients": RULES["clients"],
        "min_evaluate_clients": RULES["clients"],
        "min_available_clients": RULES["clients"],
        "accept_failures": False,
        "fit_metrics_aggregation_fn": lambda metrics: {},
    }
    strategies = {
        "fedavg": lambda: FedAvg(inplace=False, **settings),
        "median": lambda: FedMedian(**settings),
        "trimmed_mean": lambda: FedTrimmedAvg(beta=0.25, **settings),
    }
    strategy = strategies[defense]()
    clients = [
        QuantumClient(data, client_id).to_client()
        for client_id in range(RULES["clients"])
    ]
    params = data["initial"].copy()
    state = None
    attack_rounds = []
    for round_idx in range(RULES["rounds"]):
        parameters = ndarrays_to_parameters([params])
        responses = [
            client.fit(FitIns(parameters, {"round": round_idx}))
            for client in clients
        ]
        if attack is not None:
            originals = np.stack([
                parameters_to_ndarrays(responses[index].parameters)[0]
                for index in data["compromised"]
            ])
            own_deltas = originals - params
            proposals, state = attack(params.copy(), own_deltas.copy(), round_idx, state)
            attacked_deltas = enforce_round_budget(
                proposals, own_deltas[:, :24], attack_rounds, round_idx
            )
            for row, client_id in enumerate(data["compromised"]):
                updated = originals[row].copy()
                # Applying the perturbation preserves the honest model exactly
                # when the attacker returns the original delta.
                updated[:24] += attacked_deltas[row] - own_deltas[row, :24]
                responses[client_id].parameters = ndarrays_to_parameters([updated])
        aggregated, _ = strategy.aggregate_fit(
            round_idx + 1, list(zip(clients, responses)), []
        )
        if aggregated is None:
            raise RuntimeError("Flower did not return aggregated parameters")
        params = parameters_to_ndarrays(aggregated)[0]
    logits = forward(data["features"][data["test"]], params)
    return {
        "predictions": np.asarray(logits).argmax(axis=1).tolist(),
        "attack_rounds": attack_rounds,
    }
