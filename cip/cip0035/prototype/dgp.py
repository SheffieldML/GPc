"""Deep GP with weight-space (inter-domain) inducing windows — JAX prototype.

Each layer l maps h (B x D_in) -> f (B x D_out):
    f = mean(h) + phi(h) v + r(h),   v ~ q = N(m, S (x) I_{D_out}),  S = L_S L_S^T
    phi(h) = K_fu(h) L_uu^{-T}   (windows fixed => L_uu factorised once, never per step)
    r(h)   ~ N(0, k(h,h) - |phi(h)|^2)  (what the windows miss; independent per point)
Kinds:
    'ridge'   Gaussian ridge windows (CIP-0028), one-sided offsets, directions first
    'hermite' Hermite ridge windows, odd degrees, capacity-aware allocation
    'rf'      Neal random erf units, iid weights, no K_uu^{-1}, no residual (finite Bayesian net)
    'ip'      standard inducing points Z in the layer's input space (Salimbeni & Deisenroth 2017),
              K_uu depends on Z and hyperparameters => Cholesky every step
Kernel: Williams/MLP (erf) kernel, sigma2 * (2/pi) asin(2 x^.x^' / sqrt((1+2|x^|^2)(1+2|x^'|^2))),
x^ = (sqrt(w/D_in) h, sqrt(b)).
"""
from math import comb, factorial, sqrt, pi
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------- kernel pieces
def softplus(x):
    return jnp.logaddexp(x, 0.0)


def inv_softplus(y):
    return float(np.log(np.expm1(y)))


def xhat(h, w, b):
    D = h.shape[1]
    return jnp.concatenate([h * jnp.sqrt(w / D), jnp.full((h.shape[0], 1), jnp.sqrt(b))], axis=1)


def kdiag(xh, sig2):
    n2 = jnp.sum(xh * xh, axis=1)
    return sig2 * (2 / jnp.pi) * jnp.arcsin(2 * n2 / (1 + 2 * n2))


def kcross(x1, x2, sig2):
    n1 = 1 + 2 * jnp.sum(x1 * x1, 1)
    n2 = 1 + 2 * jnp.sum(x2 * x2, 1)
    return sig2 * (2 / jnp.pi) * jnp.arcsin(2 * (x1 @ x2.T) / jnp.sqrt(n1[:, None] * n2[None, :]))


# ---------------------------------------------------------------- window layouts (numpy, fixed)
def orth_dirs(nd, Dp, rng):
    V = []
    while len(V) < nd:
        Q, _ = np.linalg.qr(rng.normal(size=(Dp, Dp)))
        V.extend(list(Q[: min(Dp, nd - len(V))]))
    return np.array(V)


def ridge_kuu(V, a, s):
    M, Dp = V.shape
    s2 = s * s
    alpha = 1.0 / s2 - 0.5
    c = V @ V.T
    lognorm2 = -0.5 * np.log(np.pi * s2) - (Dp - 1) / 2.0 * np.log(2 * np.pi)
    am, an = a[:, None], a[None, :]
    detI = (1 + alpha) ** 2 - (alpha * c) ** 2
    inv11 = (1 + alpha) / detI
    inv12 = -alpha * c / detI
    H11 = inv11 + c * inv12
    H12 = inv12 + c * inv11
    bm, bn = am / s2, an / s2
    quad = H11 * (bm ** 2 + bn ** 2) + 2 * H12 * bm * bn
    logK = lognorm2 + Dp / 2 * np.log(2 * np.pi) - 0.5 * np.log(detI) + 0.5 * quad - (am ** 2 + an ** 2) / (2 * s2)
    K = np.exp(logK)
    return 0.5 * (K + K.T)


def ridge_layout(M, Dp, rng, kappa_max=1e4):
    """Directions first (more directions beat more offsets once d is moderate);
    offsets one-sided (erf is odd, so windows at +-a on one direction give K_fu columns
    that are exact negatives of each other)."""
    if M <= 4 * Dp:
        nd, n_off = M, 1
    else:
        n_off = int(np.ceil(M / (4 * Dp)))
        nd = M // n_off
    V = orth_dirs(nd, Dp, rng)
    a1 = np.array([1.5]) if n_off == 1 else np.linspace(1.0, 2.5, n_off)
    V = np.repeat(V, n_off, axis=0)
    a = np.tile(a1, nd)
    for s in [1.0, 0.8, 0.5, 0.3, 0.2, 0.1]:
        K = ridge_kuu(V, a, s)
        ev = np.linalg.eigvalsh(K)
        if ev.min() > 0 and ev.max() / ev.min() <= kappa_max:
            break
    s2 = s * s
    cs = 2 * s2 / (2 + s2)
    mu = a * cs / s2
    A = (2 * np.pi) ** -0.25 * (np.pi * s2) ** -0.25 * np.sqrt(2 * np.pi * cs) * np.exp(mu ** 2 / (2 * cs) - a ** 2 / (2 * s2))
    L = np.linalg.cholesky(K)
    Linv = np.linalg.inv(L)
    return dict(V=V, a=a, s=s, cs=cs, mu=mu, A=A, LinvT=Linv.T, kappa=ev.max() / ev.min(), M=len(a))


def ridge_kuu_j(V, a, s):
    M, Dp = V.shape
    s2 = s * s
    alpha = 1.0 / s2 - 0.5
    c = jnp.clip(V @ V.T, -1.0, 1.0)
    lognorm2 = -0.5 * jnp.log(jnp.pi * s2) - (Dp - 1) / 2.0 * jnp.log(2 * jnp.pi)
    am, an = a[:, None], a[None, :]
    detI = (1 + alpha) ** 2 - (alpha * c) ** 2
    inv11 = (1 + alpha) / detI
    inv12 = -alpha * c / detI
    H11 = inv11 + c * inv12
    H12 = inv12 + c * inv11
    bm, bn = am / s2, an / s2
    quad = H11 * (bm ** 2 + bn ** 2) + 2 * H12 * bm * bn
    logK = lognorm2 + Dp / 2 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(detI) + 0.5 * quad - (am ** 2 + an ** 2) / (2 * s2)
    K = jnp.exp(logK)
    return 0.5 * (K + K.T)


def c_k(k):
    j = (k - 1) // 2
    return (2 / sqrt(pi)) * (-1) ** j * factorial(2 * j) / (factorial(j) * sqrt(factorial(k)))


def hermite_layout(M, Dp, rng, kmax=9):
    deg1 = min(Dp, M)
    alloc = {1: deg1}
    rest = M - deg1
    degs = list(range(3, kmax + 1, 2))
    cap = {k: comb(Dp + k - 1, k) // 2 for k in degs}
    for k in degs:
        alloc[k] = 0
    while rest > 0:
        moved = False
        for k in degs:
            if rest > 0 and alloc[k] < cap[k]:
                alloc[k] += 1
                rest -= 1
                moved = True
        if not moved:
            break
    Vs, ks = [], []
    for k, n in alloc.items():
        if n == 0:
            continue
        V = orth_dirs(n, Dp, rng) if k == 1 else rng.normal(size=(n, Dp))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        Vs.append(V)
        ks += [k] * n
    V = np.vstack(Vs)
    ks = np.array(ks)
    K = (V @ V.T) ** ks[:, None] * (ks[:, None] == ks[None, :])
    ev = np.linalg.eigvalsh(K)
    L = np.linalg.cholesky(K + 1e-12 * np.eye(len(K)))
    ck = np.array([c_k(int(k)) for k in ks])
    return dict(V=V, ks=ks, ck=ck, LinvT=np.linalg.inv(L).T, kappa=ev.max() / ev.min(), M=len(ks))


# ---------------------------------------------------------------- layers
class Layer:
    def __init__(self, kind, D_in, D_out, M, mean, rng, chi=1.0, Zinit=None, q_scale=1.0, resid=True):
        self.use_resid = resid and kind != "rf"
        self.kind, self.D_in, self.D_out = kind, D_in, D_out
        self.mean = mean  # None (zero) or a fixed D_in x D_out matrix
        Dp = D_in + 1
        if kind in ("ridge", "ridge_learn"):
            self.lay = ridge_layout(M, Dp, rng)
        elif kind == "hermite":
            self.lay = hermite_layout(M, Dp, rng)
        elif kind == "rf":
            self.lay = dict(Om=rng.normal(size=(Dp, M)), M=M)
        elif kind == "ip":
            self.lay = dict(M=M)
        self.M = self.lay["M"]
        self.lay = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in self.lay.items()}
        self.init = dict(
            sig2=inv_softplus(chi), w=inv_softplus(1.0), b=inv_softplus(0.1),
            m=np.zeros((self.M, D_out)),
            Lo=np.zeros((self.M, self.M)), ld=np.full(self.M, np.log(q_scale)),
        )
        if kind == "ip":
            self.init["Z"] = np.asarray(Zinit[: self.M])
        if kind == "ridge_learn":
            self.init["V"] = np.asarray(self.lay["V"]); self.init["a"] = np.asarray(self.lay["a"])
            self.init["ls"] = float(np.log(self.lay["s"]))

    def hyp(self, p):
        return softplus(p["sig2"]), softplus(p["w"]), softplus(p["b"])

    def LS(self, p):
        return jnp.tril(p["Lo"], -1) + jnp.diag(jnp.exp(p["ld"]))

    def kl(self, p):
        LS = self.LS(p)
        return 0.5 * (self.D_out * (jnp.sum(LS * LS) - self.M - 2 * jnp.sum(p["ld"])) + jnp.sum(p["m"] ** 2))

    def features(self, p, h, raw=False):
        """phi (B x M) and prior marginal variance kdiag (B,)."""
        sig2, w, b = self.hyp(p)
        xh = xhat(h, w, b)
        kd = kdiag(xh, sig2)
        L = self.lay
        if self.kind == "ridge":
            t = xh @ L["V"].T
            n2 = jnp.sum(xh * xh, 1)[:, None]
            den = jnp.sqrt(1 + 2 * jnp.maximum(n2 - (1 - L["cs"]) * t * t, 0.0))
            Kfu = jnp.sqrt(sig2) * L["A"][None, :] * erf(L["mu"][None, :] * t / den)
            return (Kfu, L["LinvT"], kd) if raw else (Kfu @ L["LinvT"], kd)
        if self.kind == "ridge_learn":
            V = p["V"] / jnp.linalg.norm(p["V"], axis=1, keepdims=True); a = p["a"]; s_ = jnp.exp(p["ls"])
            s2 = s_ * s_; cs = 2 * s2 / (2 + s2); mu = a * cs / s2
            A = (2 * jnp.pi) ** -0.25 * (jnp.pi * s2) ** -0.25 * jnp.sqrt(2 * jnp.pi * cs) * jnp.exp(mu ** 2 / (2 * cs) - a ** 2 / (2 * s2))
            t = xh @ V.T
            n2 = jnp.sum(xh * xh, 1)[:, None]
            den = jnp.sqrt(1 + 2 * jnp.maximum(n2 - (1 - cs) * t * t, 0.0))
            Kfu = jnp.sqrt(sig2) * A[None, :] * erf(mu[None, :] * t / den)
            Kuu = ridge_kuu_j(V, a, s_) + 1e-8 * jnp.eye(self.M)
            Luu = jnp.linalg.cholesky(Kuu)
            return solve_triangular(Luu, Kfu.T, lower=True).T, kd
        if self.kind == "hermite":
            n2 = jnp.sum(xh * xh, 1)[:, None]
            rho = (xh @ L["V"].T) / jnp.sqrt(1 + 2 * n2)
            Kfu = jnp.sqrt(sig2) * L["ck"][None, :] * rho ** L["ks"][None, :]
            return (Kfu, L["LinvT"], kd) if raw else (Kfu @ L["LinvT"], kd)
        if self.kind == "rf":
            return jnp.sqrt(sig2 / self.M) * erf(xh @ L["Om"]), kd
        if self.kind == "ip":
            zh = xhat(p["Z"], w, b)
            Kuu = kcross(zh, zh, sig2) + 1e-6 * jnp.eye(self.M)
            Luu = jnp.linalg.cholesky(Kuu)  # per step: depends on Z and hyperparameters
            A = solve_triangular(Luu, kcross(zh, xh, sig2), lower=True)
            return A.T, kd
        raise ValueError(self.kind)

    def mean_fn(self, h):
        return 0.0 if self.mean is None else h @ self.mean

    def weight_draw(self, p, h, eps_v):
        """One function draw for the whole minibatch: O(M^2 D_out) + O(B M D_in), plus O(B M^2)
        only if the residual variance (Nystrom diagonal) is kept and K_uu != I."""
        LS = self.LS(p)
        v = p["m"] + LS @ eps_v
        if self.kind in ("ridge", "hermite"):
            Kfu, LinvT, kd = self.features(p, h, raw=True)
            f = Kfu @ (LinvT @ v)
            if self.use_resid:
                phi = Kfu @ LinvT
                return f, jnp.maximum(kd - jnp.sum(phi * phi, 1), 1e-12)
            return f, 0.0
        phi, kd = self.features(p, h)
        resid = jnp.maximum(kd - jnp.sum(phi * phi, 1), 1e-12) if self.use_resid else 0.0
        return phi @ v, resid

    def sample(self, p, h, eps_v, eps_n, mode="local"):
        if mode == "weights":
            f, resid = self.weight_draw(p, h, eps_v)
            return self.mean_fn(h) + f + jnp.sqrt(resid)[:, None] * eps_n if self.use_resid else self.mean_fn(h) + f
        phi, kd = self.features(p, h)
        LS = self.LS(p)
        resid = jnp.maximum(kd - jnp.sum(phi * phi, 1), 1e-12) if self.use_resid else 0.0
        if mode == "weights":       # one function draw per minibatch: an MLP forward pass
            v = p["m"] + LS @ eps_v
            f = phi @ v
            var_extra = resid
        else:                         # local reparameterisation per point
            f = phi @ p["m"]
            var_extra = resid + jnp.sum((phi @ LS) ** 2, 1)
        f = f + jnp.sqrt(var_extra)[:, None] * eps_n
        return self.mean_fn(h) + f

    def moments(self, p, h):
        phi, kd = self.features(p, h)
        LS = self.LS(p)
        resid = jnp.maximum(kd - jnp.sum(phi * phi, 1), 0.0) if self.use_resid else 0.0
        return self.mean_fn(h) + phi @ p["m"], resid + jnp.sum((phi @ LS) ** 2, 1)


# ---------------------------------------------------------------- model
class DeepGP:
    def __init__(self, kind, X, D, L, M, seed=0, chi=1.0, hidden_q=1e-2, resid=True):
        rng = np.random.default_rng(seed)
        d = X.shape[1]
        self.layers = []
        h = X
        Zh = X[rng.permutation(len(X))]
        for l in range(L):
            last = l == L - 1
            D_in = d if l == 0 else D
            D_out = 1 if last else D
            if last:
                mean = None
            elif D_in == D_out:
                mean = np.eye(D)
            else:  # PCA projection, as Salimbeni & Deisenroth
                _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
                mean = Vt[:D_out].T if D_out <= d else np.eye(d, D_out)
            sig2 = 1.0 if last else chi / max(L - 1, 1)
            lay = Layer(kind, D_in, D_out, M, None if mean is None else jnp.asarray(mean), rng,
                        chi=sig2, Zinit=Zh, q_scale=1.0 if last else hidden_q, resid=resid)
            self.layers.append(lay)
            if mean is not None:
                Zh = Zh @ mean
        self.params = dict(layers=[ly.init for ly in self.layers], noise=inv_softplus(0.1))
        self.params = jax.tree_util.tree_map(jnp.asarray, self.params)

    def forward(self, params, X, key, mode="local"):
        h = X
        for ly, p in zip(self.layers, params["layers"]):
            key, k1, k2 = jax.random.split(key, 3)
            eps_v = jax.random.normal(k1, (ly.M, ly.D_out))
            eps_n = jax.random.normal(k2, (h.shape[0], ly.D_out))
            h_last = h
            h = ly.sample(p, h, eps_v, eps_n, mode) if ly is not self.layers[-1] else h
        return h_last, h  # inputs to final layer

    def elbo(self, params, X, y, key, N, mode="local", S=1):
        def one(k):
            hin, _ = self.forward(params, X, k, mode)
            ly, p = self.layers[-1], params["layers"][-1]
            if mode == "weights":
                k1, _ = jax.random.split(k)
                mu, var = ly.weight_draw(p, hin, jax.random.normal(k1, (ly.M, 1)))
            else:
                mu, var = ly.moments(p, hin)      # final layer analytically (Gaussian lik)
            noise = softplus(params["noise"])
            ell = -0.5 * jnp.log(2 * jnp.pi * noise) - 0.5 * ((y - mu[:, 0]) ** 2 + var) / noise
            return jnp.sum(ell)
        keys = jax.random.split(key, S)
        ell = jnp.mean(jax.vmap(one)(keys))
        kl = sum(ly.kl(p) for ly, p in zip(self.layers, params["layers"]))
        return N / X.shape[0] * ell - kl

    def predict_logdensity(self, params, X, y, key, S=64):
        def one(k):
            hin, _ = self.forward(params, X, k, "local")
            mu, var = self.layers[-1].moments(params["layers"][-1], hin)
            var = var + softplus(params["noise"])
            lp = -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * (y - mu[:, 0]) ** 2 / var
            return lp, mu[:, 0]
        lps, mus = jax.vmap(one)(jax.random.split(key, S))
        lpd = jax.scipy.special.logsumexp(lps, 0) - jnp.log(S)
        return lpd, jnp.mean(mus, 0)


def adam_train(model, X, y, steps=4000, B=256, lr=0.01, seed=0, mode="local", S=1, log_every=0):
    import optax
    params = model.params
    opt = optax.adam(lr)
    st = opt.init(params)
    N = len(X)
    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    @jax.jit
    def step(params, st, idx, key):
        loss, g = jax.value_and_grad(lambda p: -model.elbo(p, Xj[idx], yj[idx], key, N, mode, S))(params)
        up, st = opt.update(g, st)
        return optax.apply_updates(params, up), st, loss

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    hist = []
    for t in range(steps):
        idx = rng.choice(N, min(B, N), replace=False)
        key, sk = jax.random.split(key)
        params, st, loss = step(params, st, idx, sk)
        if log_every and t % log_every == 0:
            hist.append(float(loss))
    model.params = params
    return params, hist
