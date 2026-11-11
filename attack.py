"""Edit this file to develop your attack. Keep the function signature below."""
import numpy as np


def attack(global_params, own_deltas, round_idx, state):
    """Return (replacement_circuit_deltas, new_state).

    Inputs
        global_params: current model, shape (34,).
        own_deltas: your two clients' honest updates, shape (2, 34).
        round_idx: current round, 0 through 59.
        state: your previous state, initially None.

    Output
        A finite real array with shape (2, 24), containing replacement
        circuit deltas, and your state for the next round.

    Budget
        Change updates in at most 30 of the 60 rounds per evaluation run.
        Returning own_deltas[:, :24].copy() skips the attack for that round.
        Changing either client counts as one attack round.

    Starter policy
        Reverse the circuit deltas in rounds 0, 2, ..., 58.
        Keep the honest circuit deltas in the other rounds.
        Improve the direction, coordination, or choice of attack rounds.
    """
    circuit_deltas = np.array(own_deltas[:, :24], dtype=float, copy=True)
    if round_idx % 2 == 0:
        return -circuit_deltas, state
    return circuit_deltas, state
