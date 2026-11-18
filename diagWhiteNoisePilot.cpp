/* CIP-001C pilot (pre-acceptance): white-noise inducing vs point inducing.
 *
 * Closed-form RBF + Gaussian windows (no CInducing yet). On a 1-D γ×M grid:
 *   - κ(K_uu) and jitChol jitter for point vs white-noise (fixed s)
 *   - DTCVAR-style collapsed bound (whitened C = I + βBB^T) for both
 *   - M_eff = M - tr(C^{-1})
 *   - Equivalence gate: s = ℓ/√2 ⇒ Q_ff and bound match point inducing
 *
 * Build: make diagWhiteNoisePilot && ./diagWhiteNoisePilot
 */
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "CMatrix.h"
#include "ndlexceptions.h"

using namespace std;

namespace {

const double PI = 3.14159265358979323846;

void fillGrid1d(CMatrix& X, CMatrix& y, CMatrix& Z, unsigned int N, unsigned int M)
{
  for(unsigned int i = 0; i < N; i++)
  {
    const double x = -1.0 + 2.0 * (double)i / (double)(N - 1);
    X.setVal(x, i, 0);
    y.setVal(sin(2.0 * x), i, 0);
  }
  for(unsigned int i = 0; i < M; i++)
  {
    const double z = -1.0 + 2.0 * (double)i / (double)(M - 1);
    Z.setVal(z, i, 0);
  }
}

double rbfK(double x, double z, double ell, double var)
{
  const double r = x - z;
  return var * exp(-0.5 * r * r / (ell * ell));
}

// L²-normalised Gaussian windows: Kuu[m,n] = exp(-||zm-zn||^2 / (4 s^2))
void fillKuuWn(CMatrix& Kuu, const CMatrix& Z, double s)
{
  const unsigned int M = Z.getRows();
  const double s2 = s * s;
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
    {
      const double dz = Z.getVal(i, 0) - Z.getVal(j, 0);
      Kuu.setVal(exp(-dz * dz / (4.0 * s2)), i, j);
    }
  }
  Kuu.setSymmetric(true);
}

// Kfu[n,m] = σ (2√2 ℓ s / (ℓ²+2s²))^{d/2} exp(-||x-z||²/(ℓ²+2s²)); d=1
void fillKfuWn(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
               double ell, double var, double s)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const double sigma = sqrt(var);
  const double v = ell * ell + 2.0 * s * s;
  const double amp = sigma * sqrt(2.0 * sqrt(2.0) * ell * s / v); // d=1
  for(unsigned int n = 0; n < N; n++)
  {
    const double x = X.getVal(n, 0);
    for(unsigned int m = 0; m < M; m++)
    {
      const double dz = x - Z.getVal(m, 0);
      Kfu.setVal(amp * exp(-dz * dz / v), n, m);
    }
  }
}

void fillKuuPoint(CMatrix& Kuu, const CMatrix& Z, double ell, double var)
{
  const unsigned int M = Z.getRows();
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
      Kuu.setVal(rbfK(Z.getVal(i, 0), Z.getVal(j, 0), ell, var), i, j);
  }
  Kuu.setSymmetric(true);
}

void fillKufPoint(CMatrix& Kuf, const CMatrix& Z, const CMatrix& X,
                  double ell, double var)
{
  // Kuf is M×N (GPc layout): rows inducing, cols data
  const unsigned int M = Z.getRows();
  const unsigned int N = X.getRows();
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < N; j++)
      Kuf.setVal(rbfK(Z.getVal(i, 0), X.getVal(j, 0), ell, var), i, j);
  }
}

double kappaSym(CMatrix spd)
{
  CMatrix evals(1, spd.getRows());
  spd.setSymmetric(true);
  spd.syev(evals, "n", "u");
  double mx = 0.0, mn = numeric_limits<double>::infinity();
  for(unsigned int i = 0; i < evals.getCols(); i++)
  {
    const double d = fabs(evals.getVal(i));
    mx = max(mx, d);
    if(d > 0.0)
      mn = min(mn, d);
  }
  if(!isfinite(mn) || mn <= 0.0)
    return numeric_limits<double>::infinity();
  return mx / mn;
}

// Trace of diagK - diag(Qff) for DTCVAR; Kuf is M×N, Kuu factored as U^T U.
double traceKffMinusQff(const CMatrix& X, const CMatrix& Kuf, const CMatrix& U,
                        double ell, double var)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Kuf.getRows();
  // B = U^{-T} Kuf  (M×N)
  CMatrix B(Kuf);
  B.trsm(U, 1.0, "l", "u", "t", "n");
  double tr = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    const double kff = var; // RBF diag
    double q = 0.0;
    for(unsigned int m = 0; m < M; m++)
    {
      const double b = B.getVal(m, n);
      q += b * b;
    }
    tr += kff - q;
  }
  return tr;
}

struct BoundResult
{
  bool finite;
  bool cholOk;
  double jitter;
  double kappaKuu;
  double kappaC;
  double ll;
  double meff;
  double trGap;
};

// Collapsed DTCVAR / VFE bound for spherical Gaussian noise, whitened form:
//   B = U^{-T} Kuf,  C = I + β B B^T
//   LL = 0.5*( -N log 2π + N log β - log|C| - β||y||² + β y^T B^T C^{-1} B y
//              - β tr(Kff-Qff) )
BoundResult collapsedDtcvar(const CMatrix& KuuIn, const CMatrix& Kuf,
                           const CMatrix& X, const CMatrix& y,
                           double ell, double var, double beta)
{
  BoundResult r{};
  r.finite = false;
  r.cholOk = false;
  r.jitter = 0.0;
  r.kappaKuu = numeric_limits<double>::infinity();
  r.kappaC = numeric_limits<double>::infinity();
  r.ll = nan("");
  r.meff = nan("");
  r.trGap = nan("");

  const unsigned int M = KuuIn.getRows();
  const unsigned int N = y.getRows();

  CMatrix Kuu(KuuIn);
  r.kappaKuu = kappaSym(Kuu);

  // Member jitChol returns the seed (1e-6 mean diag) even when chol succeeds
  // on the first try without adding anything. Detect real adds by probing chol.
  CMatrix U(M, M);
  r.jitter = 0.0;
  try
  {
    U.deepCopy(Kuu);
    U.chol();
    U.setTriangular(true);
    r.cholOk = true;
  }
  catch(ndlexceptions::MatrixNonPosDef&)
  {
    try
    {
      CMatrix KuuJ(Kuu);
      r.jitter = U.jitChol(KuuJ);
      U.setTriangular(true);
      r.cholOk = true;
    }
    catch(...)
    {
      return r;
    }
  }
  catch(...)
  {
    return r;
  }

  CMatrix B(Kuf);
  B.trsm(U, 1.0, "l", "u", "t", "n");

  CMatrix C(M, M);
  C.eye();
  C.syrk(B, beta, 1.0, "u", "n");
  C.setSymmetric(true);
  r.kappaC = kappaSym(C);

  CMatrix Lc(M, M);
  double jitC = 0.0;
  try
  {
    jitC = Lc.jitChol(C);
    Lc.setTriangular(true);
  }
  catch(...)
  {
    return r;
  }
  (void)jitC;

  // c = L_c^{-T} (B (β y)); then quad = ||c||^2
  CMatrix By(M, 1);
  By.gemm(B, y, beta, 0.0, "n", "n");
  By.trsm(Lc, 1.0, "l", "u", "t", "n");
  double quad = 0.0;
  for(unsigned int i = 0; i < M; i++)
  {
    const double v = By.getVal(i, 0);
    quad += v * v;
  }

  double y2 = 0.0;
  for(unsigned int i = 0; i < N; i++)
  {
    const double yi = y.getVal(i, 0);
    y2 += yi * yi;
  }

  // logDet(Lc) already returns log|C| = 2 Σ log L_ii (CMatrix::logDet).
  const double logDetC = logDet(Lc);
  r.trGap = traceKffMinusQff(X, Kuf, U, ell, var);

  r.ll = 0.5 * (-(double)N * log(2.0 * PI) + (double)N * log(beta)
                - logDetC - beta * y2 + quad - beta * r.trGap);

  // M_eff = M - tr(C^{-1}). Solve C^{-1} via two trsm on I.
  CMatrix Inv(M, M);
  Inv.eye();
  Inv.trsm(Lc, 1.0, "l", "u", "t", "n");
  Inv.trsm(Lc, 1.0, "l", "u", "n", "n");
  double trInv = 0.0;
  for(unsigned int i = 0; i < M; i++)
    trInv += Inv.getVal(i, i);
  r.meff = (double)M - trInv;

  r.finite = isfinite(r.ll) && isfinite(r.meff) && isfinite(r.kappaKuu);
  return r;
}

void reportRow(const string& tag, double gamma, unsigned int M, double s,
               const BoundResult& b, double llRef = nan(""))
{
  cout << setw(6) << tag
       << "  g=" << setw(8) << gamma
       << "  M=" << setw(3) << M
       << "  s=" << setw(10) << s
       << "  jit=" << setw(10) << b.jitter
       << "  kKuu=" << scientific << setprecision(3) << b.kappaKuu
       << "  kC=" << b.kappaC
       << "  Meff=" << fixed << setprecision(2) << b.meff
       << "  LL=" << setprecision(6) << b.ll;
  if(isfinite(llRef))
    cout << "  dLL=" << scientific << setprecision(3) << (b.ll - llRef);
  cout << (b.finite && b.cholOk ? "  OK" : "  FAIL") << endl;
  cout << fixed;
}

int runSweep()
{
  cout << "==== CIP-001C pilot: point vs white-noise (fixed s) ====" << endl;
  const unsigned int N = 40;
  const double beta = 100.0;
  const double var = 1.0;
  // Fixed window from inducing spacing on [-1,1] with M=10 → Δz≈0.222; s≈Δz
  const double sFixed = 0.25;
  const double gammas[] = {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
  const unsigned int Ms[] = {5u, 10u, 20u};

  cout << setprecision(6);
  cout << "Note: GPc RBF uses k=σ² exp(-γ/2 r²) with γ=1/ℓ², so ℓ=1/√γ.\n";
  cout << "White-noise s fixed at " << sFixed
       << " (independent of ℓ). Point Kuu = k(Z,Z).\n" << endl;

  int issues = 0;
  for(unsigned int mi = 0; mi < 3; mi++)
  {
    const unsigned int M = Ms[mi];
    CMatrix X(N, 1), y(N, 1), Z(M, 1);
    fillGrid1d(X, y, Z, N, M);

    for(unsigned int gi = 0; gi < 6; gi++)
    {
      const double gamma = gammas[gi];
      const double ell = 1.0 / sqrt(gamma);

      CMatrix KuuPt(M, M), KufPt(M, N);
      fillKuuPoint(KuuPt, Z, ell, var);
      fillKufPoint(KufPt, Z, X, ell, var);
      BoundResult pt = collapsedDtcvar(KuuPt, KufPt, X, y, ell, var, beta);

      CMatrix KuuWn(M, M), KfuWn(N, M), KufWn(M, N);
      fillKuuWn(KuuWn, Z, sFixed);
      fillKfuWn(KfuWn, X, Z, ell, var, sFixed);
      // Kuf = Kfu^T
      for(unsigned int i = 0; i < M; i++)
        for(unsigned int j = 0; j < N; j++)
          KufWn.setVal(KfuWn.getVal(j, i), i, j);
      BoundResult wn = collapsedDtcvar(KuuWn, KufWn, X, y, ell, var, beta);

      reportRow("point", gamma, M, ell / sqrt(2.0), pt);
      reportRow("wn", gamma, M, sFixed, wn, pt.ll);

      // Expectation: as γ→0 (ℓ→∞), point needs jitter / huge κ; wn stays calm.
      if(gamma <= 0.1 && pt.jitter > 0.0 && wn.jitter == 0.0)
        ; // expected contrast
      if(!wn.finite || !wn.cholOk)
        issues++;
      if(gamma <= 0.1 && wn.jitter > 0.0)
        issues++;
    }
    cout << endl;
  }
  return issues;
}

int runEquivalence()
{
  cout << "==== Equivalence gate: s = ℓ/√2 ====" << endl;
  const unsigned int N = 40;
  const unsigned int M = 10;
  const double beta = 100.0;
  const double var = 1.0;
  const double gammas[] = {0.5, 1.0, 2.0, 5.0, 10.0};

  CMatrix X(N, 1), y(N, 1), Z(M, 1);
  fillGrid1d(X, y, Z, N, M);

  int fail = 0;
  for(unsigned int gi = 0; gi < 5; gi++)
  {
    const double gamma = gammas[gi];
    const double ell = 1.0 / sqrt(gamma);
    const double s = ell / sqrt(2.0);

    CMatrix KuuPt(M, M), KufPt(M, N);
    fillKuuPoint(KuuPt, Z, ell, var);
    fillKufPoint(KufPt, Z, X, ell, var);

    // WN at equivalence: Kuu_wn = Kuu_pt / σ², Kfu matches after scale.
    // Qff = Kfu Kuu^{-1} Kuf is invariant to u ↦ αu, so bounds should match
    // when Kuu_wn = Kuu_pt/var and Kuf_wn = Kuf_pt/sqrt(var).
    CMatrix KuuWn(M, M), KfuWn(N, M), KufWn(M, N);
    fillKuuWn(KuuWn, Z, s);
    fillKfuWn(KfuWn, X, Z, ell, var, s);
    for(unsigned int i = 0; i < M; i++)
      for(unsigned int j = 0; j < N; j++)
        KufWn.setVal(KfuWn.getVal(j, i), i, j);

    // Check Kuu_wn ≈ Kuu_pt / var
    double maxKuu = 0.0;
    for(unsigned int i = 0; i < M; i++)
      for(unsigned int j = 0; j < M; j++)
        maxKuu = max(maxKuu, fabs(KuuWn.getVal(i, j) - KuuPt.getVal(i, j) / var));

    double maxKuf = 0.0;
    for(unsigned int i = 0; i < M; i++)
      for(unsigned int j = 0; j < N; j++)
        maxKuf = max(maxKuf, fabs(KufWn.getVal(i, j) - KufPt.getVal(i, j) / sqrt(var)));

    BoundResult pt = collapsedDtcvar(KuuPt, KufPt, X, y, ell, var, beta);
    BoundResult wn = collapsedDtcvar(KuuWn, KufWn, X, y, ell, var, beta);
    const double dLL = fabs(pt.ll - wn.ll);

    cout << "γ=" << gamma << "  |Kuu_wn - Kuu/σ²|_∞=" << scientific << maxKuu
         << "  |Kuf_wn - Kuf/σ|_∞=" << maxKuf
         << "  |ΔLL|=" << dLL << fixed << endl;

    if(maxKuu > 1e-10 || maxKuf > 1e-10 || dLL > 1e-8 || !pt.finite || !wn.finite)
      fail++;
  }
  return fail;
}

} // namespace

int main()
{
  const int eqFail = runEquivalence();
  cout << endl;
  const int sweepIssues = runSweep();
  cout << "Equivalence failures: " << eqFail << endl;
  cout << "Sweep WN failures (non-finite / unexpected jitter at small γ): "
       << sweepIssues << endl;
  return (eqFail == 0 && sweepIssues == 0) ? 0 : 1;
}
