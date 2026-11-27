"""CIP-0035: gpc.models.DeepGP smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gpc


def _sinc_toy(n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.linspace(-2.0, 2.0, n)[:, None]
    Y = np.sin(1.5 * X) + 0.05 * rng.normal(size=X.shape)
    return X, Y


def test_construct_and_elbo():
    X, Y = _sinc_toy()
    m = gpc.models.DeepGP(X, Y, layers=2, hidden=1, num_inducing=6, chi=0.5)
    assert m.n == X.shape[0]
    assert m.num_layers == 2
    assert m.approx == "deepgp"
    assert m.inducing_type == "points"
    elbo = m.elbo()
    assert np.isfinite(elbo)


def test_optimize_and_predict():
    X, Y = _sinc_toy()
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=1, num_inducing=6, chi=0.5, beta=5.0
    )
    before = m.elbo()
    m.optimize(iters=15, batch_size=X.shape[0], adam_lr=0.05, seed=3)
    after = m.elbo()
    assert np.isfinite(after)
    # Short Adam run should not explode; allow a small drop under noise.
    assert after >= before - 5.0
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert var.shape == Y.shape
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0.0)
    mu_s, var_s = m.predict(X, samples=8, seed=5)
    assert mu_s.shape == Y.shape
    assert np.all(np.isfinite(mu_s))
    assert np.all(var_s > 0.0)


def test_jacobian_spread_prior_draws():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(12, 2))
    Y = rng.normal(size=(12, 2))
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=2, num_inducing=6, chi=0.5, beta=1.0
    )
    spread = m.jacobian_spread(n_probe=8, probe_seed=11)
    assert np.isfinite(spread)
    assert 0.05 < spread < 5.0


def test_probit_likelihood():
    rng = np.random.default_rng(1)
    X = np.linspace(-1.5, 1.5, 20)[:, None]
    Y = (np.sin(X) > 0.0).astype(float) * 2.0 - 1.0
    Y += 0.0 * rng.normal(size=Y.shape)
    m = gpc.models.DeepGP(
        X, Y, layers=1, num_inducing=5, likelihood="probit", beta=1.0
    )
    assert m.likelihood_type == "probit"
    assert np.isfinite(m.elbo())
    m.optimize(iters=8, seed=2, adam_lr=0.05)
    assert np.isfinite(m.elbo())


def test_save_load_roundtrip(tmp_path: Path):
    X, Y = _sinc_toy(n=16)
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=1, num_inducing=4, chi=0.4, drop_residual=True
    )
    m.optimize(iters=5, seed=4, adam_lr=0.05)
    path = tmp_path / "deep.model"
    m.save(str(path))
    loaded = gpc.models.DeepGP.load(str(path))
    assert loaded.num_layers == 2
    assert loaded.drop_residual is True
    # Mean-path predict is deterministic; ELBO is MC so seed/RNG can differ.
    mu0, _ = m.predict(X)
    mu1, _ = loaded.predict(X)
    assert np.allclose(mu0, mu1, atol=1e-6)


def test_residual_off_and_str():
    X, Y = _sinc_toy(n=12)
    m = gpc.models.DeepGP(
        X, Y, layers=2, num_inducing=4, drop_residual=True, chi=0.5
    )
    assert m.drop_residual is True
    text = str(m)
    assert "Deep GP" in text
    assert "deepgp" in m.approx


# ---- CIP-003C façade items -------------------------------------------------


def test_predict_noiseless_removes_gaussian_noise():
    X, Y = _sinc_toy()
    beta = 5.0
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=1, num_inducing=6, chi=0.5, beta=beta
    )
    mu, var = m.predict(X)
    mu_f, var_f = m.predict_noiseless(X)
    assert np.allclose(mu, mu_f)
    assert np.all(var_f > 0.0)
    # Gaussian predict = latent variance + 1/beta (mean path).
    assert np.allclose(var - var_f, 1.0 / beta, rtol=1e-6, atol=1e-6)
    # Sampled path too.
    mu_s, var_s = m.predict(X, samples=6, seed=2)
    mu_sf, var_sf = m.predict_noiseless(X, samples=6, seed=2)
    assert np.all(var_s > var_sf)


def test_jacobian_spread_rectangular():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(14, 3))
    for dout in (1, 2, 5):
        Y = rng.normal(size=(14, dout))
        m = gpc.models.DeepGP(
            X, Y, layers=2, hidden=3, num_inducing=6, chi=0.5, beta=1.0
        )
        s = m.jacobian_spread(n_probe=6, probe_seed=1)
        assert np.isfinite(s) and s >= 0.0
    # One singular value => zero spread.
    m1 = gpc.models.DeepGP(
        X, rng.normal(size=(14, 1)), layers=2, hidden=3, num_inducing=6
    )
    assert m1.jacobian_spread(n_probe=4, probe_seed=1) == pytest.approx(0.0, abs=1e-9)


def test_hidden_width_above_input_dim():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 2))
    Y = np.sin(X[:, :1]) + 0.05 * rng.normal(size=(20, 1))
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=5, num_inducing=6, chi=0.5, beta=5.0
    )
    assert m.hidden_dim == 5
    assert np.isfinite(m.elbo())
    m.optimize(iters=10, adam_lr=0.05, seed=1)
    mu, var = m.predict(X)
    assert np.all(np.isfinite(mu)) and np.all(var > 0.0)


def test_adam_state_persists_across_optimize_calls():
    X, Y = _sinc_toy()

    # Full-batch gradients are deterministic only without hidden-layer noise;
    # use layers=1 so chunked and single runs are comparable exactly.
    def build1():
        return gpc.models.DeepGP(X, Y, layers=1, num_inducing=6, beta=5.0)

    a, b, c = build1(), build1(), build1()
    a.optimize(iters=10, adam_lr=0.05, seed=3)
    b.optimize(iters=5, adam_lr=0.05, seed=3)
    assert b.optimizer_steps == 5
    b.optimize(iters=5, adam_lr=0.05, seed=3)
    assert b.optimizer_steps == 10
    # Equal up to summation order of the (shuffled) full batch.
    assert b.elbo() == pytest.approx(a.elbo(), rel=1e-6)
    # Resetting restarts the moments, so the chunked run differs.
    c.optimize(iters=5, adam_lr=0.05, seed=3)
    c.reset_optimizer()
    assert c.optimizer_steps == 0
    c.optimize(iters=5, adam_lr=0.05, seed=3)
    assert abs(c.elbo() - a.elbo()) > 1e-3


def test_learn_window_widths_on_deepgp():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(24, 1))
    Y = np.tanh(2 * X) + 0.05 * rng.normal(size=(24, 1))
    m = gpc.models.DeepGP(
        X, Y, layers=1, num_inducing=8, kernel="mlp", inducing="weightspace",
        beta=10.0, learn_window_width=True,
    )
    assert m.learn_window_width is True and m.learn_window_widths is False
    plain = gpc.models.DeepGP(
        X, Y, layers=1, num_inducing=8, kernel="mlp", inducing="weightspace",
        beta=10.0,
    )
    assert m.describe().opt_num_params == plain.describe().opt_num_params + 1
    untied = gpc.models.DeepGP(
        X, Y, layers=1, num_inducing=8, kernel="mlp", inducing="weightspace",
        beta=10.0, learn_window_widths=True,
    )
    assert untied.describe().opt_num_params == plain.describe().opt_num_params + 8
    m.optimize(iters=5, adam_lr=0.02, seed=1)
    assert np.isfinite(m.elbo())
    with pytest.raises(Exception):
        gpc.models.DeepGP(X, Y, layers=1, learn_window_width=True)
    with pytest.raises(Exception):
        gpc.models.DeepGP(
            X, Y, layers=1, kernel="mlp", inducing="weightspace",
            learn_window_width=True, learn_window_widths=True,
        )


def test_learned_window_width_changes_the_trained_elbo():
    # Regression (CIP-003C): the width gradient was zero-filled when packed, so
    # learn_window_width=True trained to the same ELBO as the fixed-width model.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    Y = np.sin(X[:, :1])
    elbo = {}
    for flag in (False, True):
        m = gpc.models.DeepGP(
            X, Y, layers=2, hidden=3, num_inducing=16, kernel="mlp",
            inducing="weightspace", learn_window_width=flag,
        )
        m.optimize(iters=100, batch_size=100, samples=1, adam_lr=1e-2, seed=0)
        elbo[flag] = m.elbo()
    assert abs(elbo[True] - elbo[False]) > 1e-3


# ---- CIP-0041 B1: per-layer kernel and inducing family -------------------


def _hd_toy(n: int = 30, d: int = 6, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = np.sin(X[:, :1]) + 0.05 * rng.normal(size=(n, 1))
    return X, Y


def test_per_layer_inducing_constructs_trains_and_roundtrips(tmp_path):
    X, Y = _hd_toy()
    m = gpc.models.DeepGP(
        X, Y, layers=3, hidden=3, num_inducing=6, kernel="mlp",
        inducing=["weightspace", "points", "points"],
    )
    assert m.layer_inducing_types == ["weightspace", "points", "points"]
    assert m.layer_kernel_types == ["mlp", "mlp", "mlp"]
    assert m.inducing_type == "weightspace,points,points"
    assert m.kernel_type == "mlp"
    assert np.isfinite(m.elbo())
    m.optimize(iters=5, batch_size=X.shape[0], adam_lr=0.01, seed=0)
    assert np.isfinite(m.elbo())
    path = tmp_path / "pl.model"
    m.save(str(path))
    m2 = gpc.models.DeepGP.load(str(path))
    assert m2.layer_inducing_types == ["weightspace", "points", "points"]
    # Mean-path predict is deterministic; ELBO is Monte Carlo.
    mu0, _ = m.predict(X)
    mu1, _ = m2.predict(X)
    assert np.allclose(mu0, mu1, atol=1e-6)


def test_per_layer_kernels_mixed():
    X, Y = _hd_toy()
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=3, num_inducing=6, kernel=["mlp", "rbf"],
        inducing=["weightspace", "points"],
    )
    assert m.layer_kernel_types == ["mlp", "rbf"]
    assert m.kernel_type == "mlp,rbf"
    assert np.isfinite(m.elbo())


def test_per_layer_errors_name_the_layer():
    X, Y = _hd_toy()
    with pytest.raises(Exception, match=r"layer 1: .*weightspace.*mlp"):
        gpc.models.DeepGP(
            X, Y, layers=2, hidden=3, num_inducing=6, kernel=["mlp", "rbf"],
            inducing=["points", "weightspace"],
        )
    with pytest.raises(Exception, match=r"layer 0: .*hermite.*mlp"):
        gpc.models.DeepGP(
            X, Y, layers=2, hidden=3, num_inducing=6, kernel=["rbf", "mlp"],
            inducing=["hermite", "points"],
        )
    with pytest.raises(Exception, match=r"layer 1: inducing must be"):
        gpc.models.DeepGP(
            X, Y, layers=2, hidden=3, num_inducing=6, inducing=["points", "bogus"],
        )
    with pytest.raises(Exception, match="length"):
        gpc.models.DeepGP(X, Y, layers=3, hidden=3, num_inducing=6, kernel=["rbf", "rbf"])


def test_scalar_strings_match_single_element_sequences():
    X, Y = _hd_toy()
    a = gpc.models.DeepGP(
        X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp", inducing="weightspace"
    )
    b = gpc.models.DeepGP(
        X, Y, layers=2, hidden=3, num_inducing=6, kernel=["mlp", "mlp"],
        inducing=["weightspace", "weightspace"],
    )
    mu_a, _ = a.predict(X)
    mu_b, _ = b.predict(X)
    assert np.array_equal(mu_a, mu_b)
    assert a.kernel_type == b.kernel_type == "mlp"
    assert a.inducing_type == b.inducing_type == "weightspace"


def test_learn_window_width_applies_to_weightspace_layers_only():
    X, Y = _hd_toy()
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp",
        inducing=["weightspace", "points"], learn_window_width=True,
    )
    m.optimize(iters=3, batch_size=X.shape[0], adam_lr=0.01, seed=0)
    assert np.isfinite(m.elbo())
    with pytest.raises(Exception, match="weightspace"):
        gpc.models.DeepGP(
            X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp",
            inducing=["points", "points"], learn_window_width=True,
        )
