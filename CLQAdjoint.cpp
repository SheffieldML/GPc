/* CIP-001B: LQ forward + reverse-mode adjoint (Seeger et al., arXiv:1710.08717 §4.3). */
#include "CLQAdjoint.h"
#include "lapack.h"
#include "ndlexceptions.h"
#include <cmath>
#include <vector>

using namespace std;

void lqBuildStackedIA(CMatrix& W, const CMatrix& A)
{
  const unsigned int m = A.getRows();
  const unsigned int nA = A.getCols();
  W.resize(m, m + nA);
  W.zeros();
  for(unsigned int i = 0; i < m; i++)
    W.setVal(1.0, i, i);
  for(unsigned int j = 0; j < nA; j++)
    for(unsigned int i = 0; i < m; i++)
      W.setVal(A.getVal(i, j), i, m + j);
}

void lqSignNormaliseLower(CMatrix& Llower, CMatrix* Q, vector<char>* rowFlipped)
{
  const unsigned int m = Llower.getRows();
  DIMENSIONMATCH(Llower.getCols() == m);
  if(rowFlipped != 0)
    rowFlipped->assign(m, 0);
  for(unsigned int i = 0; i < m; i++)
  {
    const double d = Llower.getVal(i, i);
    if(d >= 0.0)
      continue;
    if(rowFlipped != 0)
      (*rowFlipped)[i] = 1;
    for(unsigned int j = 0; j <= i; j++)
      Llower.setVal(-Llower.getVal(i, j), i, j);
    if(Q != 0)
    {
      DIMENSIONMATCH(Q->getRows() == m);
      for(unsigned int j = 0; j < Q->getCols(); j++)
        Q->setVal(-Q->getVal(i, j), i, j);
    }
  }
}

static void gelqfInPlace(CMatrix& W, vector<double>& tau)
{
  const int m = (int)W.getRows();
  const int n = (int)W.getCols();
  if(m > n)
    throw ndlexceptions::Error("lqFactorWide: need m <= n (wide or square)");
  tau.assign((unsigned int)m, 0.0);
  int info = 0;
  int lwork = -1;
  double workQuery = 0.0;
  dgelqf_(m, n, W.getVals(), m, tau.data(), &workQuery, lwork, info);
  if(info != 0)
    throw ndlexceptions::Error("dgelqf workspace query failed");
  lwork = (int)workQuery;
  if(lwork < 1)
    lwork = max(1, m);
  vector<double> work((unsigned int)lwork);
  dgelqf_(m, n, W.getVals(), m, tau.data(), work.data(), lwork, info);
  if(info != 0)
    throw ndlexceptions::Error("dgelqf failed");
}

static void orglqInPlace(CMatrix& W, const vector<double>& tau)
{
  const int m = (int)W.getRows();
  const int n = (int)W.getCols();
  const int k = (int)tau.size();
  int info = 0;
  int lwork = -1;
  double workQuery = 0.0;
  dorglq_(m, n, k, W.getVals(), m, tau.data(), &workQuery, lwork, info);
  if(info != 0)
    throw ndlexceptions::Error("dorglq workspace query failed");
  lwork = (int)workQuery;
  if(lwork < 1)
    lwork = max(1, m);
  vector<double> work((unsigned int)lwork);
  dorglq_(m, n, k, W.getVals(), m, tau.data(), work.data(), lwork, info);
  if(info != 0)
    throw ndlexceptions::Error("dorglq failed");
}

void lqFactorWide(CMatrix& Lupper, double& logAbsDetL,
                  CMatrix* Qout, CMatrix& W, vector<double>* tauOut)
{
  const unsigned int m = W.getRows();
  const unsigned int n = W.getCols();
  if(m == 0 || n < m)
    throw ndlexceptions::Error("lqFactorWide: empty or tall matrix");

  CMatrix Wwork(W);
  vector<double> tau;
  gelqfInPlace(Wwork, tau);

  // Extract lower L from Householder form (lower triangle including diag).
  CMatrix Llower(m, m);
  Llower.zeros();
  for(unsigned int j = 0; j < m; j++)
    for(unsigned int i = j; i < m; i++)
      Llower.setVal(Wwork.getVal(i, j), i, j);

  vector<char> flips;
  if(Qout != 0)
  {
    // Form Q in a copy of the Householder matrix.
    CMatrix Qwork(Wwork);
    orglqInPlace(Qwork, tau);
    lqSignNormaliseLower(Llower, &Qwork, &flips);
    Qout->deepCopy(Qwork);
  }
  else
  {
    lqSignNormaliseLower(Llower, 0, &flips);
  }

  // Upper R = L^T for GPc LcholC consumers.
  Lupper.resize(m, m);
  Lupper.zeros();
  logAbsDetL = 0.0;
  for(unsigned int i = 0; i < m; i++)
  {
    const double d = Llower.getVal(i, i);
    logAbsDetL += 2.0 * log(fabs(d));
    for(unsigned int j = 0; j <= i; j++)
      Lupper.setVal(Llower.getVal(i, j), j, i);
  }
  Lupper.setTriangular(true);

  // Leave W as Householder (pre-dorglq); encode sign flips in tauOut sentinel
  // via a parallel side channel — callers needing lazy Q use lqFormQ with Lupper.
  W.deepCopy(Wwork);
  if(tauOut != 0)
  {
    *tauOut = tau;
    // Pack row flips into tauOut capacity: store as trailing metadata? No —
    // lqFormQ re-derives flips from Lupper vs raw Householder L extract.
  }
}

void lqFormQ(CMatrix& Qout, const CMatrix& Whh, const vector<double>& tau,
             const CMatrix& Lupper)
{
  const unsigned int m = Whh.getRows();
  const unsigned int n = Whh.getCols();
  DIMENSIONMATCH(tau.size() == m);
  DIMENSIONMATCH(Lupper.getRows() == m && Lupper.getCols() == m);

  CMatrix Qwork(Whh);
  orglqInPlace(Qwork, tau);

  // Match lqSignNormaliseLower: flip Q row i when Householder L_ii was negative
  // (Lupper_ii = |L_ii| after normalisation). Lupper unused except dimension check.
  (void)Lupper;
  for(unsigned int i = 0; i < m; i++)
  {
    if(Whh.getVal(i, i) >= 0.0)
      continue;
    for(unsigned int j = 0; j < n; j++)
      Qwork.setVal(-Qwork.getVal(i, j), i, j);
  }
  Qout.deepCopy(Qwork);
}

void lqFactorIA(CMatrix& Lupper, double& logAbsDetL,
                CMatrix* Qout, const CMatrix& A,
                CMatrix* WhhOut, vector<double>* tauOut)
{
  CMatrix W;
  lqBuildStackedIA(W, A);
  lqFactorWide(Lupper, logAbsDetL, Qout, W, tauOut);
  if(WhhOut != 0)
    WhhOut->deepCopy(W);
}

void lqAdjoint(CMatrix& Wbar,
               const CMatrix& L, const CMatrix& Q,
               const CMatrix& Lbar, const CMatrix& Qbar,
               bool LIsUpper)
{
  const unsigned int m = Q.getRows();
  const unsigned int n = Q.getCols();
  DIMENSIONMATCH(L.getRows() == m && L.getCols() == m);
  DIMENSIONMATCH(Lbar.getRows() == m && Lbar.getCols() == m);
  DIMENSIONMATCH(Qbar.getRows() == m && Qbar.getCols() == n);

  // Work with lower L.
  CMatrix Llow(m, m);
  CMatrix LbarLow(m, m);
  if(LIsUpper)
  {
    Llow.zeros();
    LbarLow.zeros();
    for(unsigned int i = 0; i < m; i++)
      for(unsigned int j = i; j < m; j++)
      {
        Llow.setVal(L.getVal(i, j), j, i);
        LbarLow.setVal(Lbar.getVal(i, j), j, i);
      }
  }
  else
  {
    Llow.deepCopy(L);
    LbarLow.deepCopy(Lbar);
  }

  // M = L^T Lbar - Qbar Q^T   (m×m) — CIP-002D QW3: BLAS gemm
  CMatrix M(m, m);
  M.gemm(Llow, LbarLow, 1.0, 0.0, "t", "n");
  M.gemm(Qbar, Q, -1.0, 1.0, "n", "t");

  // copyltu(M): Seeger et al. arXiv:1710.08717 §4.3 — keep lower incl. diag,
  // mirror to upper so the projected tangent lives in the LQ manifold coords.
  CMatrix Msym(m, m);
  for(unsigned int i = 0; i < m; i++)
  {
    Msym.setVal(M.getVal(i, i), i, i);
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = M.getVal(i, j);
      Msym.setVal(v, i, j);
      Msym.setVal(v, j, i);
    }
  }

  // S = Qbar + copyltu(M) Q
  CMatrix S(m, n);
  S.deepCopy(Qbar);
  S.gemm(Msym, Q, 1.0, 1.0, "n", "n");

  // Wbar = L^{-T} S
  Wbar.deepCopy(S);
  // Solve L^T X = S  ⇒  X = L^{-T} S. L lower ⇒ use trsm with L, trans='T'.
  Wbar.setTriangular(false);
  // trsm requires triangular flag on the factor.
  Llow.setTriangular(true);
  Wbar.trsm(Llow, 1.0, "l", "l", "t", "n");
}
