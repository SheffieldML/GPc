"""Reference values for CRobustMaxNoise (CIP-003E Phase 4).

Each row: K c eps mu_1..mu_K var_1..var_K P ell
where P = P(argmax f = c) for independent f_k ~ N(mu_k, var_k) and
ell = E[log p(y=c | f)] with the robust-max likelihood (1-eps on the argmax,
eps/(K-1) elsewhere), as in GPflow's RobustMax.

GPflow was not installed on the development machine, so P is integrated with composite Simpson on
2e5 points over mu_c +- 10 sd using only the standard library, an independent
route from the C++ Gauss-Legendre code. Run from the repository root:
    python3 testdata/robustmax/make_robustmax_fixture.py > testdata/robustmax/robustmax_ref.txt
"""
import math

CASES = [
    (3, 0, 1e-3, [0.3, -0.2, 0.1], [0.5, 0.8, 0.2]),
    (3, 1, 1e-3, [0.3, -0.2, 0.1], [0.5, 0.8, 0.2]),
    (3, 2, 1e-2, [1.5, -1.0, 2.0], [1.0, 1.0, 1.0]),
    (4, 0, 1e-3, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
    (4, 3, 5e-2, [2.0, 1.0, -1.0, 1.5], [0.1, 0.4, 2.0, 0.7]),
    (5, 2, 1e-3, [-1.0, 0.5, 0.4, 0.3, -2.0], [1e-4, 3e-4, 1e-3, 0.02, 4.0]),
    (3, 0, 1e-3, [5.0, 0.0, 0.0], [1e-6, 1e-6, 1e-6]),
    (6, 4, 1e-3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
]


def Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def prob_largest(mu, var, c, n=200000):
    sc = math.sqrt(var[c])
    lo, hi = mu[c] - 10 * sc, mu[c] + 10 * sc
    h = (hi - lo) / n
    total = 0.0
    for i in range(n + 1):
        f = lo + i * h
        g = math.exp(-0.5 * ((f - mu[c]) / sc) ** 2) / (sc * math.sqrt(2 * math.pi))
        for k in range(len(mu)):
            if k != c:
                g *= Phi((f - mu[k]) / math.sqrt(var[k]))
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += w * g
    return total * h / 3.0


for K, c, eps, mu, var in CASES:
    P = prob_largest(mu, var, c)
    ell = P * math.log(1 - eps) + (1 - P) * math.log(eps / (K - 1))
    print(K, c, repr(eps), *[repr(x) for x in mu], *[repr(x) for x in var],
          repr(P), repr(ell))
