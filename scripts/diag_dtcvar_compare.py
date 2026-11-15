#!/usr/bin/env python3
"""Compare GPc DTC/DTCVAR dump against a GPy-aligned VarDTC / DTC reference.

Uses only the Python standard library (no NumPy/GPy required). Algebra follows
GPy.inference.latent_function_inference.var_dtc for spherical Gaussian noise
with fixed inducing inputs and an RBF kernel (inverse-width parameterisation
matching CRbfKern).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def mat_zeros(r: int, c: int) -> list[list[float]]:
    return [[0.0] * c for _ in range(r)]


def mat_eye(n: int) -> list[list[float]]:
    m = mat_zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def mat_add(A, B, alpha=1.0, beta=1.0):
    r, c = len(A), len(A[0])
    out = mat_zeros(r, c)
    for i in range(r):
        for j in range(c):
            out[i][j] = alpha * A[i][j] + beta * B[i][j]
    return out


def mat_scale(A, s):
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_transpose(A):
    r, c = len(A), len(A[0])
    return [[A[i][j] for i in range(r)] for j in range(c)]


def mat_mul(A, B):
    r, k, c = len(A), len(A[0]), len(B[0])
    assert len(B) == k
    out = mat_zeros(r, c)
    for i in range(r):
        for j in range(c):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            out[i][j] = s
    return out


def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def cholesky(A, jitter=1e-8):
    n = len(A)
    L = mat_zeros(n, n)
    for attempt in range(8):
        Aj = [row[:] for row in A]
        for i in range(n):
            Aj[i][i] += jitter * (10**attempt if attempt else 1.0)
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = Aj[i][j] - sum(L[i][t] * L[j][t] for t in range(j))
                if i == j:
                    if s <= 0:
                        ok = False
                        break
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
            if not ok:
                break
        if ok:
            return L
    raise RuntimeError("Cholesky failed")


def solve_lower(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y


def solve_upper_from_lower(L, b):
    # Solve L^T x = b
    n = len(L)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
    return x


def pdinv_via_chol(A):
    L = cholesky(A)
    n = len(A)
    inv = mat_zeros(n, n)
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        y = solve_lower(L, e)
        x = solve_upper_from_lower(L, y)
        for i in range(n):
            inv[i][j] = x[i]
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    return inv, logdet, L


def rbf_K(X1, X2, inverse_width, variance):
    r, c = len(X1), len(X2)
    q = len(X1[0])
    K = mat_zeros(r, c)
    for i in range(r):
        for j in range(c):
            d2 = sum((X1[i][t] - X2[j][t]) ** 2 for t in range(q))
            K[i][j] = variance * math.exp(-0.5 * inverse_width * d2)
    return K


def rbf_diag(X, variance):
    return [variance] * len(X)


def parse_state(path: Path) -> dict:
    lines = path.read_text().strip().splitlines()
    meta = {}
    i = 0

    def take_kv():
        nonlocal i
        parts = lines[i].split()
        i += 1
        return parts[0], parts[1:]

    while i < len(lines) and lines[i] not in ("X", "y", "Xu", "grads_ll"):
        k, rest = take_kv()
        meta[k] = rest[0] if len(rest) == 1 else rest

    def read_matrix(rows, cols):
        nonlocal i
        m = []
        for _ in range(rows):
            vals = [float(x) for x in lines[i].split()]
            assert len(vals) == cols
            m.append(vals)
            i += 1
        return m

    N = int(meta["N"])
    q = int(meta["q"])
    d = int(meta["d"])
    M = int(meta["M"])

    assert lines[i] == "X"
    i += 1
    X = read_matrix(N, q)
    assert lines[i] == "y"
    i += 1
    y = [[float(lines[i + r])] for r in range(N)]
    i += N
    assert lines[i] == "Xu"
    i += 1
    Xu = read_matrix(M, q)
    assert lines[i].startswith("grads_ll")
    ngrads = int(lines[i].split()[1])
    i += 1
    grads = [float(lines[i + j]) for j in range(ngrads)]
    i += ngrads

    return {
        "approx": meta["approx"],
        "N": N,
        "q": q,
        "d": d,
        "M": M,
        "inverseWidth": float(meta["inverseWidth"]),
        "variance": float(meta["variance"]),
        "beta": float(meta["beta"]),
        "ll_cpp": float(meta["ll"]),
        "X": X,
        "y": [row[0] for row in y],
        "Xu": Xu,
        "grads_ll_cpp": grads,
    }


def log_lik_dtc_family(X, y, Xu, inverse_width, variance, beta, variational: bool):
    """GPc / GPmat spherical DTC(+VAR) log likelihood (single output).

    Matches CGp::logLikelihood for DTC/DTCVAR after the final -0.5 scaling
    and HALFLOGTWOPI term (no kernel prior).
    """
    N = len(X)
    M = len(Xu)
    D = 1
    Kuu = rbf_K(Xu, Xu, inverse_width, variance)
    Kuf = rbf_K(Xu, X, inverse_width, variance)  # M x N
    diagK = rbf_diag(X, variance)

    invKuu, logDetKuu, _ = pdinv_via_chol(Kuu)
    # GPc updateAD: A := K_uu; A.gemm(K_uf,K_uf, 1.0, 1/beta, "n","t")
    # => A = K_uf K_uf^T + K_uu / beta  (BLAS beta scales the existing A).
    KufKufT = mat_mul(Kuf, mat_transpose(Kuf))
    A = mat_add(KufKufT, Kuu, 1.0, 1.0 / beta)
    invA, logDetA, _ = pdinv_via_chol(A)

    # L accumulates as in CGp before L*=-0.5 and const terms:
    # L += D * ((M-N)*log(beta) - logDetK_uu + logDetA)
    # for each output: L -= beta*(invAe·e - m·m)
    # DTCVAR: L += D * sum(diagD) with diagD = beta*(diagK - Qff)
    L = D * ((M - N) * math.log(beta) - logDetKuu + logDetA)
    yy = dot(y, y)
    e = mat_vec(Kuf, y)  # M
    invAe = mat_vec(invA, e)
    L -= beta * (dot(invAe, e) - yy)

    if variational:
        # Qff_ii = sum_j Kuf[j,i] * (invKuu * Kuf)[j,i] = col_i^T invKuu col_i
        V = mat_mul(invKuu, Kuf)  # M x N
        # GPc: V.dotMultiply(K_uf) then sumCol -> for each data j, sum_i V[i,j]*Kuf[i,j]
        qff = [sum(V[i][j] * Kuf[i][j] for i in range(M)) for j in range(N)]
        diagD = [beta * (diagK[j] - qff[j]) for j in range(N)]
        L += D * sum(diagD)

    L *= -0.5
    L -= D * N * 0.5 * math.log(2.0 * math.pi)
    return L


def gpy_style_var_dtc(X, y, Xu, inverse_width, variance, beta):
    """GPy VarDTC log_marginal for spherical noise (single output)."""
    N = len(X)
    M = len(Xu)
    output_dim = 1
    psi0 = rbf_diag(X, variance)
    psi1 = rbf_K(X, Xu, inverse_width, variance)  # N x M  (GPy K(X,Z))
    Kmm = rbf_K(Xu, Xu, inverse_width, variance)
    Lm = cholesky(Kmm)

    # tmp = Lm^{-1} * psi1.T * sqrt(beta) ; A = tmp tmp^T
    sqrt_beta = math.sqrt(beta)
    psi1T = mat_transpose(psi1)
    tmp = [solve_lower(Lm, [psi1T[i][j] * sqrt_beta for i in range(M)]) for j in range(N)]
    # tmp currently list over data of M-vectors; build M x N then A = tmp tmp^T
    tmpM = mat_transpose(tmp)  # M x N
    A = mat_mul(tmpM, mat_transpose(tmpM))
    B = mat_add(mat_eye(M), A)
    LB = cholesky(B)

    trYYT = dot(y, y)
    VVT = [beta * yi for yi in y]

    # data_fit path from GPy (simplified single output)
    # _LBi_Lmi_psi1 = LB^{-1} Lm^{-1} psi1.T
    Lmi_psi1 = [solve_lower(Lm, [psi1[n][m] for m in range(M)]) for n in range(N)]  # N of M
    LBi_Lmi_psi1 = [solve_lower(LB, Lmi_psi1[n]) for n in range(N)]
    # data_fit = || LB^{-1} Lm^{-1} psi1.T VVT ||^2 summed
    LBi_Lmi_psi1Vf = [
        sum(LBi_Lmi_psi1[n][m] * VVT[n] for n in range(N)) for m in range(M)
    ]
    # Actually GPy: _LBi_Lmi_psi1Vf = _LBi_Lmi_psi1 @ VVT_factor with shape (M, D)
    data_fit = sum(v * v for v in LBi_Lmi_psi1Vf)

    lik_1 = -0.5 * N * output_dim * (math.log(2.0 * math.pi) - math.log(beta)) - 0.5 * beta * trYYT
    lik_2 = -0.5 * output_dim * (sum(beta * p for p in psi0) - sum(A[i][i] for i in range(M)))
    lik_3 = -output_dim * sum(math.log(LB[i][i]) for i in range(M))
    lik_4 = 0.5 * data_fit
    return lik_1 + lik_2 + lik_3 + lik_4


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "diag_dtc_state.txt", root / "diag_dtcvar_state.txt"]
    if len(argv) > 1:
        paths = [Path(p) for p in argv[1:]]

    fail = 0
    for path in paths:
        if not path.exists():
            print(f"MISSING {path}")
            fail += 1
            continue
        st = parse_state(path)
        approx = st["approx"]
        variational = approx == "dtcvar"
        ll_ref = log_lik_dtc_family(
            st["X"],
            st["y"],
            st["Xu"],
            st["inverseWidth"],
            st["variance"],
            st["beta"],
            variational=variational,
        )
        ll_gpy = None
        if variational:
            ll_gpy = gpy_style_var_dtc(
                st["X"],
                st["y"],
                st["Xu"],
                st["inverseWidth"],
                st["variance"],
                st["beta"],
            )

        diff = abs(ll_ref - st["ll_cpp"])
        print(f"==== {path.name} ({approx}) ====")
        print(f"ll_cpp:              {st['ll_cpp']:.16e}")
        print(f"ll_ref (GPc algebra):{ll_ref:.16e}  |diff|={diff:.3e}")
        if ll_gpy is not None:
            diff_gpy = abs(ll_gpy - st["ll_cpp"])
            print(f"ll_gpy VarDTC:       {ll_gpy:.16e}  |diff|={diff_gpy:.3e}")
            # Pure-Python Cholesky vs LAPACK: expect ~1e-5..1e-6 agreement.
            if diff_gpy > 1e-4:
                fail += 1
                print("FAIL vs GPy-style VarDTC")
            else:
                print("PASS vs GPy-style VarDTC")
        if diff > 1e-4:
            fail += 1
            print("FAIL vs independent GPc-algebra reimplementation")
        else:
            print("PASS vs independent GPc-algebra reimplementation")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
