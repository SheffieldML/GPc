"""CIP-0016: gpc.interchange against a numpy oracle (no gp-scuderia needed)."""
import copy

import numpy as np
import pytest

from gpc import interchange as I

X = [[-2.0], [-1.0], [0.0], [1.0], [2.5]]
Y = [[0.3], [-0.4], [0.1], [0.8], [0.2]]

FTC = {
    "schema_version": "0.1", "type": "models.gp_regression",
    "input_dim": 1, "output_dim": 1, "approximation": "ftc",
    "kernel": {"type": "kern.add", "parts": [
        {"type": "kern.rbf", "variance": 1.0, "lengthscale": 1.0, "ARD": False},
        {"type": "kern.white", "variance": 0.1}]},
    "likelihood": {"type": "likelihoods.gaussian", "variance": 0.1},
    "data": {"X": X, "Y": Y},
}


def _ftc_oracle(noise_total):
    x, y = np.array(X), np.array(Y)
    K = np.exp(-0.5 * (x - x.T) ** 2) + noise_total * np.eye(5)
    return (-0.5 * y.T @ np.linalg.solve(K, y)
            - 0.5 * np.linalg.slogdet(K)[1] - 2.5 * np.log(2 * np.pi)).item()


def test_ftc_white_and_likelihood_both_add():
    m = I.from_wire(FTC)
    assert -m.objective() == pytest.approx(_ftc_oracle(0.2), abs=1e-9)


def test_default_noise_ownership_unchanged_without_opt_in():
    m = I.from_wire(FTC)
    m.set_allow_white_and_noise(False)
    assert -m.objective() == pytest.approx(_ftc_oracle(0.1), abs=1e-9)


def test_ard_round_trip_is_exact():
    doc = {
        "schema_version": "0.1", "type": "models.sparse_gp_regression",
        "input_dim": 2, "output_dim": 1, "approximation": "dtcvar",
        "sparse_linalg": "lq", "inducing_inputs": [[0.0, 0.0], [1.0, 0.5]],
        "kernel": {"type": "kern.rbf", "variance": 2.0,
                   "lengthscale": [0.8, 1.5], "ARD": True},
        "likelihood": {"type": "likelihoods.gaussian", "variance": 0.05},
        "data": {"X": [[-1.5, 0.2], [0.1, -0.3], [1.4, 1.1]],
                 "Y": [[0.1], [0.3], [0.4]]},
    }
    back = I.to_wire(I.from_wire(doc), doc)
    assert back["kernel"]["lengthscale"] == pytest.approx([0.8, 1.5], rel=1e-12)
    assert back["likelihood"]["variance"] == pytest.approx(0.05, rel=1e-12)


@pytest.mark.parametrize("mutate", [
    lambda d: d["kernel"]["parts"][0].update(type="kern.nonesuch"),
    lambda d: d["data"].update(X=[[0.0, 1.0]] * 5),
    lambda d: d.update(schema_version="9.9"),
    lambda d: d.pop("data"),
])
def test_refuses_rather_than_guesses(mutate):
    d = copy.deepcopy(FTC)
    mutate(d)
    with pytest.raises(I.InterchangeError):
        I.from_wire(d)
