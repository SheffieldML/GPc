/* CIP-001B: unit tests for LQ of [I,A] vs chol(I+AA^T) and the LQ adjoint FD.
 * Maths: Seeger et al., Auto-differentiating linear algebra, arXiv:1710.08717
 * (§3.3 BLR stacked LQ; §4.3 reverse).
 *
 * Real-world GP map (what each case stands in for):
 *   testFactorMatchesChol     — well-conditioned sparse regression; LQ must
 *                               agree with Gram+chol when both are safe.
 *   testAdjointFD             — reverse-mode grads through the LQ node used when
 *                               training inducing GPs (kernel / Z / β updates).
 *   testIllConditionedKufKfu  — large-N data with clustered inputs and a sane
 *                               inducing grid: κ(K_uf K_fu) ≫ κ(K_uu). Common
 *                               when N is thousands and inducing points cover the
 *                               domain; Gram of C = I+βBB^T loses logdet digits.
 */
#include "CLQAdjoint.h"
#include "CMatrix.h"
#include "lapack.h"
#include "ndlexceptions.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

static int fails = 0;

static void expect(bool ok, const string& what)
{
  if(ok)
    cout << "OK: " << what << endl;
  else
  {
    cout << "FAIL: " << what << endl;
    fails++;
  }
}

static CMatrix randomA(unsigned int m, unsigned int n, double scale)
{
  CMatrix A(m, n);
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < n; j++)
      A.setVal(scale * (0.1 + 0.01 * (double)(3 * i + j) - 0.05 * (double)((i + j) % 5)), i, j);
  return A;
}

// Well-conditioned whitened features A ~ √β U^{-T} K_uf: everyday sparse GP on a
// small, tidy dataset where Gram+chol of I+AA^T is still accurate. Checks that LQ
// is algebraically the same path, not a different approximation.
static void testFactorMatchesChol()
{
  const unsigned int m = 4;
  const unsigned int n = 12;
  CMatrix A = randomA(m, n, 1.0);

  CMatrix Lupper;
  double logDetLq = 0.0;
  CMatrix Q;
  lqFactorIA(Lupper, logDetLq, &Q, A);

  CMatrix C(m, m);
  C.eye();
  C.syrk(A, 1.0, 1.0, "u", "n");
  C.setSymmetric(true);
  CMatrix U(m, m);
  U.jitChol(C);
  const double logDetChol = logDet(U);

  expect(fabs(logDetLq - logDetChol) < 1e-10, "LQ log|I+AA^T| matches chol");

  // Lazy Q (skip dorglq in factor, form later) matches eager Q.
  {
    CMatrix Llazy;
    double ldLazy = 0.0;
    CMatrix Whh;
    vector<double> tau;
    lqFactorIA(Llazy, ldLazy, 0, A, &Whh, &tau);
    CMatrix Qlazy;
    lqFormQ(Qlazy, Whh, tau, Llazy);
    double qDiff = 0.0;
    for(unsigned int i = 0; i < Q.getRows(); i++)
      for(unsigned int j = 0; j < Q.getCols(); j++)
        qDiff = max(qDiff, fabs(Q.getVal(i, j) - Qlazy.getVal(i, j)));
    expect(qDiff < 1e-12, "lqFormQ matches eager dorglq Q");
    expect(fabs(ldLazy - logDetLq) < 1e-14, "lazy factor logdet matches eager");
  }

  double diagDiff = 0.0;
  for(unsigned int i = 0; i < m; i++)
    diagDiff = max(diagDiff, fabs(fabs(Lupper.getVal(i, i)) - fabs(U.getVal(i, i))));
  expect(diagDiff < 1e-8, "LQ |diag| matches chol |diag|");

  CMatrix W;
  lqBuildStackedIA(W, A);
  CMatrix Llow(m, m);
  Llow.zeros();
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j <= i; j++)
      Llow.setVal(Lupper.getVal(j, i), i, j);
  CMatrix Wrecon(m, m + n);
  Wrecon.zeros();
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < m + n; j++)
    {
      double s = 0.0;
      for(unsigned int k = 0; k < m; k++)
        s += Llow.getVal(i, k) * Q.getVal(k, j);
      Wrecon.setVal(s, i, j);
    }
  double recon = 0.0;
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < m + n; j++)
      recon = max(recon, fabs(W.getVal(i, j) - Wrecon.getVal(i, j)));
  expect(recon < 1e-8, "L Q reconstructs [I|A]");
}

// Gradient correctness for the LQ factorisation node itself — the piece hit on
// every sparse-GP training step that differentiates through log|C| (and, with
// y-augmentation, the Woodbury data-fit) w.r.t. whitened features A.
static void testAdjointFD()
{
  const unsigned int m = 3;
  const unsigned int n = 8;
  CMatrix A = randomA(m, n, 0.7);
  CMatrix W;
  lqBuildStackedIA(W, A);
  CMatrix Lupper;
  double logDet = 0.0;
  CMatrix Q;
  CMatrix Wcopy(W);
  lqFactorWide(Lupper, logDet, &Q, Wcopy);

  CMatrix Llow(m, m);
  Llow.zeros();
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j <= i; j++)
      Llow.setVal(Lupper.getVal(j, i), i, j);
  Llow.setTriangular(true);

  CMatrix Lbar(m, m);
  Lbar.zeros();
  for(unsigned int i = 0; i < m; i++)
  {
    const double d = Llow.getVal(i, i);
    Lbar.setVal(1.0 / d, i, i);
  }
  CMatrix Qbar(m, m + n);
  Qbar.zeros();
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < m + n; j++)
      Qbar.setVal(0.1, i, j);

  CMatrix Wbar;
  lqAdjoint(Wbar, Llow, Q, Lbar, Qbar, false);

  CMatrix dW(m, m + n);
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < m + n; j++)
      dW.setVal(0.01 * (1.0 + 0.1 * (double)(i + 2 * j)), i, j);

  const double h = 1e-6;
  CMatrix Wp(W), Wm(W);
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < m + n; j++)
    {
      Wp.setVal(W.getVal(i, j) + h * dW.getVal(i, j), i, j);
      Wm.setVal(W.getVal(i, j) - h * dW.getVal(i, j), i, j);
    }
  CMatrix Lup, Lum;
  CMatrix Qp, Qm;
  double ldp = 0.0, ldm = 0.0;
  CMatrix Wpp(Wp), Wmm(Wm);
  lqFactorWide(Lup, ldp, &Qp, Wpp);
  lqFactorWide(Lum, ldm, &Qm, Wmm);

  double phiP = 0.0, phiM = 0.0;
  for(unsigned int i = 0; i < m; i++)
  {
    phiP += log(fabs(Lup.getVal(i, i)));
    phiM += log(fabs(Lum.getVal(i, i)));
  }
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < m + n; j++)
    {
      phiP += 0.1 * Qp.getVal(i, j);
      phiM += 0.1 * Qm.getVal(i, j);
    }
  const double dPhiFD = (phiP - phiM) / (2.0 * h);
  double dPhiAdj = 0.0;
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int j = 0; j < m + n; j++)
      dPhiAdj += Wbar.getVal(i, j) * dW.getVal(i, j);

  expect(fabs(dPhiAdj - dPhiFD) < 5e-4,
         "lqAdjoint matches central FD on wide W");
  cout << "  adj=" << dPhiAdj << " fd=" << dPhiFD
       << " |diff|=" << fabs(dPhiAdj - dPhiFD) << endl;
}

static double logDetIA_svd(const CMatrix& A)
{
  // log|I+AA^T| = Σ log(1+σ_i²) from SVD of A (reference; avoids forming AA^T).
  const int m = (int)A.getRows();
  const int n = (int)A.getCols();
  CMatrix Acopy(A);
  const int k = min(m, n);
  vector<double> s((unsigned)k);
  int info = 0;
  int lwork = -1;
  double workQuery = 0.0;
  dgesvd_("N", "N", m, n, Acopy.getVals(), m, s.data(),
          0, 1, 0, 1, &workQuery, lwork, info);
  if(info != 0)
    throw ndlexceptions::Error("dgesvd workspace query failed");
  lwork = (int)workQuery;
  if(lwork < 1)
    lwork = max(1, 5 * min(m, n));
  vector<double> work((unsigned)lwork);
  dgesvd_("N", "N", m, n, Acopy.getVals(), m, s.data(),
          0, 1, 0, 1, work.data(), lwork, info);
  if(info != 0)
    throw ndlexceptions::Error("dgesvd failed");
  double ld = 0.0;
  for(int i = 0; i < k; i++)
    ld += log1p(s[(unsigned)i] * s[(unsigned)i]);
  return ld;
}

static double rbfVal(double x, double z, double gamma)
{
  const double d = x - z;
  return exp(-0.5 * gamma * d * d);
}

// Real-world: large training set with inputs in a few dense clusters (sensors,
// repeated trials, spatial hot-spots), inducing points spread over the domain.
// K_uu stays manageable; K_uf K_fu (and thus C = I+βBB^T) becomes severely
// ill-conditioned — exactly where Seeger’s LQ of [I,A] beats Gram+chol.
// Contrast crowdZ tests: those collapse K_uu and never stress C.
static void testIllConditionedKufKfu()
{
  cout << "---- ill-conditioned K_uf K_fu (large N, OK K_uu) ----" << endl;
  const unsigned int n = 4000;
  const unsigned int m = 25;
  const double gamma = 20.0;
  const double beta = 1e6;
  const double jitter = 1e-8;

  CMatrix X(n, 1), Z(m, 1);
  const unsigned int nPer = n / 5;
  for(unsigned int c = 0; c < 5; c++)
  {
    const double centre = -3.0 + 6.0 * (double)c / 4.0;
    for(unsigned int i = 0; i < nPer; i++)
    {
      const double u = (double)((i * 1103515245u + 12345u + c * 97u) % 10000) / 10000.0;
      X.setVal(centre + 0.05 * (u - 0.5), c * nPer + i, 0);
    }
  }
  for(unsigned int i = 0; i < m; i++)
    Z.setVal(-3.0 + 6.0 * (double)i / (double)(m - 1), i, 0);

  CMatrix Kmm(m, m), Kmn(m, n);
  for(unsigned int i = 0; i < m; i++)
  {
    for(unsigned int j = 0; j < m; j++)
      Kmm.setVal(rbfVal(Z.getVal(i, 0), Z.getVal(j, 0), gamma), i, j);
    Kmm.setVal(Kmm.getVal(i, i) + jitter, i, i);
    for(unsigned int j = 0; j < n; j++)
      Kmn.setVal(rbfVal(Z.getVal(i, 0), X.getVal(j, 0), gamma), i, j);
  }
  Kmm.setSymmetric(true);

  CMatrix KufKfu(m, m);
  KufKfu.gemm(Kmn, Kmn, 1.0, 0.0, "n", "t");
  KufKfu.setSymmetric(true);

  CMatrix U(m, m);
  U.jitChol(Kmm);
  U.setTriangular(true);
  CMatrix B(Kmn);
  B.trsm(U, 1.0, "l", "u", "t", "n"); // B = U^{-T} Kmn
  CMatrix A(B);
  A.scale(sqrt(beta));

  auto kappaSym = [](CMatrix spd) {
    CMatrix evals(1, spd.getRows());
    spd.setSymmetric(true);
    spd.syev(evals, "n", "u");
    double mx = 0.0, mn = 1e300;
    for(unsigned int i = 0; i < evals.getCols(); i++)
    {
      const double d = fabs(evals.getVal(i));
      mx = max(mx, d);
      if(d > 0.0)
        mn = min(mn, d);
    }
    if(mn >= 1e300 || mn <= 0.0)
      return 1e300;
    return mx / mn;
  };
  const double kappaKmm = kappaSym(Kmm);
  const double kappaKuf = kappaSym(KufKfu);

  CMatrix C(m, m);
  C.eye();
  C.syrk(A, 1.0, 1.0, "u", "n");
  C.setSymmetric(true);
  CMatrix Uchol(m, m);
  Uchol.jitChol(C);
  const double logDetChol = logDet(Uchol);

  CMatrix Lupper;
  double logDetLq = 0.0;
  lqFactorIA(Lupper, logDetLq, 0, A);

  const double logDetRef = logDetIA_svd(A);
  const double errChol = fabs(logDetChol - logDetRef);
  const double errLq = fabs(logDetLq - logDetRef);

  cout << "  κ~(K_uu)=" << kappaKmm << " κ~(K_uf K_fu)=" << kappaKuf << endl;
  cout << "  |ld chol-svd|=" << errChol << " |ld lq-svd|=" << errLq
       << "  ratio chol/lq=" << (errChol / max(errLq, 1e-30)) << endl;

  expect(kappaKmm < 1e4, "K_uu not crowdZ-collapsed (κ~ < 1e4)");
  expect(kappaKuf > 1e6, "K_uf K_fu ill-conditioned (κ~ > 1e6)");
  expect(errLq < errChol, "LQ logdet closer to SVD than Gram+chol");
  expect(errLq < 1e-10, "LQ logdet within 1e-10 of SVD reference");

  // Same geometry at β=1: low noise not amplifying BB^T — both paths stay safe
  // (typical “small dataset / large observation noise” sparse GP).
  CMatrix Amild(B);
  Amild.scale(1.0);
  CMatrix Cmild(m, m);
  Cmild.eye();
  Cmild.syrk(Amild, 1.0, 1.0, "u", "n");
  Cmild.setSymmetric(true);
  CMatrix Umild(m, m);
  Umild.jitChol(Cmild);
  const double ldCholMild = logDet(Umild);
  CMatrix LupMild;
  double ldLqMild = 0.0;
  lqFactorIA(LupMild, ldLqMild, 0, Amild);
  const double ldRefMild = logDetIA_svd(Amild);
  expect(fabs(ldCholMild - ldRefMild) < 1e-10
         && fabs(ldLqMild - ldRefMild) < 1e-10,
         "mild β=1: both chol and LQ match SVD");
}

int main()
{
  cout << "==== testLQAdjoint (CIP-001B) ====" << endl;
  testFactorMatchesChol();
  testAdjointFD();
  testIllConditionedKufKfu();
  if(fails)
    cout << fails << " FAILURES" << endl;
  else
    cout << "All LQ tests passed." << endl;
  return fails ? 1 : 0;
}
