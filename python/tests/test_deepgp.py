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


# ---- CIP-0041 Phase 2 (B3 ARD kernels, B5 thread control) ----------------------------------


@pytest.mark.parametrize("name", ["rbfard", "mlpard"])
def test_ard_kernels_per_layer_describe_and_roundtrip(name, tmp_path):
    X, Y = _sinc_toy(30)
    X2 = np.hstack([X, 0.3 * np.cos(3 * X)])
    m = gpc.models.DeepGP(
        X2, Y, layers=2, hidden=2, num_inducing=6, kernel=[name, "rbf"]
    )
    d = m.describe()
    lines = [l for l in d.header_lines if l.startswith("layer 0 " + name)]
    assert len(lines) == 1, d.header_lines
    assert not any(l.startswith("layer 1 ") and "ard" in l for l in d.header_lines)
    m.optimize(iters=15, batch_size=15, seed=2)
    path = tmp_path / "ard.model"
    m.save(str(path))
    m2 = gpc.models.DeepGP.load(str(path))
    # ELBO is Monte Carlo; the mean-path prediction and the ARD scales are deterministic.
    assert np.allclose(m.predict(X2)[0], m2.predict(X2)[0], atol=1e-6)
    assert m2.describe().header_lines == m.describe().header_lines


def test_ard_rejected_for_weightspace_and_unknown_names():
    X, Y = _sinc_toy(20)
    with pytest.raises(Exception, match="requires kernel='mlp'"):
        gpc.models.DeepGP(X, Y, layers=1, kernel="mlpard", inducing="weightspace")
    with pytest.raises(Exception, match="rbfard"):
        gpc.models.DeepGP(X, Y, layers=1, kernel="matern")


def test_thread_control_roundtrip():
    before = gpc.get_num_threads()
    assert before >= 1
    with pytest.raises(Exception):
        gpc.set_num_threads(0)
    applied = gpc.set_num_threads(1)
    try:
        if applied:
            assert gpc.get_num_threads() == 1
            # OpenBLAS caps the pool at the machine's core count, so a one-core CI
            # runner legitimately reports 1 here.
            gpc.set_num_threads(2)
            assert gpc.get_num_threads() in (1, 2)
        else:
            assert gpc.get_num_threads() == 1  # documented no-op build
    finally:
        gpc.set_num_threads(before)


# ---- CIP-0041 Phase 3 (B2 mlp_init) and Phase 4 (B4 prior-draw Jacobian) -------------------


def _mlp_toy(scale, n=40, d=6, seed=3):
    rng = np.random.default_rng(seed)
    X = scale * rng.normal(size=(n, d))
    Y = np.sin(X[:, :1] / scale) + 0.05 * rng.normal(size=(n, 1))
    return X, Y


def _xhat_sq(m, ell):
    w = m.get_layer_param(ell, "weightVariance")
    b = m.get_layer_param(ell, "biasVariance")
    return w * m.layer_input_mean_sqnorm(ell) + b


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_mlp_init_scaled_normalises_xhat(scale):
    X, Y = _mlp_toy(scale)
    m = gpc.models.DeepGP(X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp",
                          mlp_init="scaled")
    for ell in range(2):
        assert 0.5 <= _xhat_sq(m, ell) <= 2.0
        assert m.get_layer_param(ell, "biasVariance") == pytest.approx(0.1)
    # The unit default does not normalise: ||xhat||^2 follows the input scale.
    mu = gpc.models.DeepGP(X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp")
    assert mu.get_layer_param(0, "weightVariance") == 1.0
    assert mu.get_layer_param(0, "biasVariance") == 1.0
    assert np.isfinite(m.elbo())


def test_mlp_init_unit_is_default_and_identical():
    X, Y = _mlp_toy(2.0)
    a = gpc.models.DeepGP(X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp")
    b = gpc.models.DeepGP(X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp",
                          mlp_init="unit")
    for ell in range(2):
        for p in a.layer_param_paths(ell):
            assert a.get_layer_param(ell, p) == b.get_layer_param(ell, p)
    assert np.array_equal(a.predict(X)[0], b.predict(X)[0])


def test_mlp_init_per_layer_and_errors():
    X, Y = _mlp_toy(1.0)
    m = gpc.models.DeepGP(X, Y, layers=2, hidden=3, num_inducing=6, kernel="mlp",
                          mlp_init=["scaled", "unit"])
    assert m.get_layer_param(1, "biasVariance") == 1.0
    assert m.get_layer_param(0, "biasVariance") == pytest.approx(0.1)
    with pytest.raises(Exception, match="layer 1: mlp_init='scaled' requires kernel='mlp'"):
        gpc.models.DeepGP(X, Y, layers=2, hidden=3, kernel=["mlp", "rbf"],
                          mlp_init=["scaled", "scaled"])
    with pytest.raises(Exception, match="'unit' or 'scaled'"):
        gpc.models.DeepGP(X, Y, layers=1, kernel="mlp", mlp_init="bogus")
    with pytest.raises(Exception, match="length"):
        gpc.models.DeepGP(X, Y, layers=2, hidden=3, kernel="mlp", mlp_init=["scaled"])


def _spread_at_depth(L, var, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(30, d))
    Y = rng.normal(size=(30, d))
    m = gpc.models.DeepGP(X, Y, layers=L, hidden=d, num_inducing=8, kernel="rbf")
    for ell in range(L):
        m.set_layer_param(ell, "variance", var(L))
    return m.jacobian_stats(n_probe=48, probe_seed=5)


def test_jacobian_stats_prior_depth_trend():
    flat = [_spread_at_depth(L, lambda L: 1.0 / L)[0] for L in (2, 4, 6)]
    unit = [_spread_at_depth(L, lambda L: 1.0)[0] for L in (2, 4, 6)]
    # Variance 1/L per layer keeps the spread nearly flat in depth (CIP-0035 V7); unit
    # variance on every layer lets it grow. Compare growth, not level: the level depends
    # on the input dimension and the number of inducing points.
    growth_flat = flat[-1] - flat[0]
    growth_unit = unit[-1] - unit[0]
    assert growth_unit > 0.5
    assert growth_flat < 0.5 * growth_unit


def test_jacobian_spread_signature_and_posterior_mode():
    X, Y = _mlp_toy(1.0, d=3)
    Y = np.hstack([Y, Y**2])
    m = gpc.models.DeepGP(X, Y, layers=2, hidden=3, num_inducing=6)
    s_default = m.jacobian_spread(n_probe=6, probe_seed=2)
    assert s_default == m.jacobian_spread(n_probe=6, probe_seed=2, prior=True)
    assert s_default == m.jacobian_stats(n_probe=6, probe_seed=2)[0]
    m.optimize(iters=40, batch_size=20, adam_lr=0.02, seed=1)
    s_post, ml_post = m.jacobian_stats(n_probe=2, probe_seed=2, prior=False)
    assert np.isfinite(s_post) and np.isfinite(ml_post)
    # The posterior-mean map has no draws: only the probe points depend on the seed.
    assert m.jacobian_stats(n_probe=2, probe_seed=2, prior=False) == (s_post, ml_post)
    with pytest.raises(Exception, match="n_probe"):
        m.jacobian_stats(n_probe=0)
