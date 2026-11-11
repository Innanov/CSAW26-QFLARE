"""Honest local training through Flower's standard NumPyClient API."""
import numpy as np
from flwr.client import NumPyClient

from quantum_model import cross_entropy, forward, loss_gradient
from score import RULES


class QuantumClient(NumPyClient):
    def __init__(self, data, client_id):
        self.data = data
        self.client_id = client_id

    def fit(self, parameters, config):
        local = parameters[0].copy()
        order = self.data["orders"][int(config["round"])][self.client_id]
        for start in range(0, len(order), RULES["batch_size"]):
            batch = order[start:start + RULES["batch_size"]]
            local -= RULES["learning_rate"] * loss_gradient(
                self.data["features"][batch], self.data["labels"][batch], local
            )
        return [local], len(order), {}

    def evaluate(self, parameters, config):
        order = self.data["orders"][0][self.client_id]
        features = self.data["features"][order]
        labels = self.data["labels"][order]
        logits = np.asarray(forward(features, parameters[0]))
        loss = float(cross_entropy(features, labels, parameters[0]))
        accuracy = float(np.mean(logits.argmax(axis=1) == labels))
        return loss, len(order), {"accuracy": accuracy}
