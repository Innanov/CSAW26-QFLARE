"""Standard PennyLane templates with NumPy/Autograd differentiation."""
import numpy as np
import pennylane as qml
from autograd.scipy.special import logsumexp
from pennylane import numpy as pnp

N_WIRES = 4
N_ANGLES = 24
N_PARAMS = 34
DEVICE = qml.device("default.qubit", wires=N_WIRES, shots=None)


@qml.qnode(DEVICE, interface="autograd", diff_method="backprop")
def circuit(features, angles):
    qml.AngleEmbedding(features, wires=range(N_WIRES), rotation="Y")
    qml.StronglyEntanglingLayers(angles.reshape(2, N_WIRES, 3), wires=range(N_WIRES))
    return tuple(qml.expval(qml.PauliX(wire)) for wire in range(N_WIRES))


def forward(features, params):
    readout = pnp.stack(circuit(features, params[:N_ANGLES]), axis=-1)
    return pnp.dot(readout, params[N_ANGLES:32].reshape(N_WIRES, 2)) + params[32:]


def cross_entropy(features, labels, params):
    logits = forward(features, params)
    observed = logits[np.arange(len(labels)), labels]
    return pnp.mean(logsumexp(logits, axis=1) - observed)


loss_derivative = qml.grad(cross_entropy, argnums=2)


def loss_gradient(features, labels, params):
    trainable = pnp.array(params, requires_grad=True)
    return np.asarray(loss_derivative(features, labels, trainable))
