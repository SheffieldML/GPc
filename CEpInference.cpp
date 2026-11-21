#include "CEpInference.h"
#include "ndlutil.h"
#include <cmath>
#include <algorithm>

using namespace std;

void CEpSites::resize(unsigned int n)
{
  tau.resize(n, 1);
  v.resize(n, 1);
}

void CEpSites::zeros()
{
  tau.zeros();
  v.zeros();
}

CEpInference::CEpInference()
  : maxIters(100), eta(1.0), delta(1.0), tol(1e-6),
    itersUsed(0), didConverge(false)
{
}

void CEpInference::recomputePosterior(const CMatrix& K,
                                      const CEpSites& sites,
                                      const CMatrix& m0,
                                      CMatrix& mu,
                                      CMatrix& Sigma,
                                      CMatrix& varDiag)
{
  // GPy posteriorParams._recompute with upper Cholesky of B.
  const unsigned int n = K.getRows();
  DIMENSIONMATCH(K.getCols() == n);
  DIMENSIONMATCH(sites.tau.getRows() == n);
  DIMENSIONMATCH(m0.getRows() == n);

  CMatrix sqrtTau(n, 1);
  for(unsigned int i = 0; i < n; i++)
  {
    const double ti = max(sites.tau.getVal(i, 0), 0.0);
    sqrtTau.setVal(sqrt(ti), i, 0);
  }

  // SrootK[i,j] = sqrt(τ_i) K[i,j]
  CMatrix SrootK(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    const double si = sqrtTau.getVal(i, 0);
    for(unsigned int j = 0; j < n; j++)
      SrootK.setVal(si * K.getVal(i, j), i, j);
  }

  // B = I + SrootK * diag(sqrt(τ))  ⇒ B[i,j] = δ_ij + sqrt(τ_i) K[i,j] sqrt(τ_j)
  CMatrix B(n, n);
  B.eye();
  for(unsigned int i = 0; i < n; i++)
  {
    const double si = sqrtTau.getVal(i, 0);
    for(unsigned int j = 0; j < n; j++)
      B.setVal(B.getVal(i, j) + si * K.getVal(i, j) * sqrtTau.getVal(j, 0), i, j);
  }
  B.setSymmetric(true);
  CMatrix U(n, n);
  U.deepCopy(B);
  U.jitChol(U); // upper

  // V = (U^T)^{-1} SrootK
  CMatrix V(SrootK);
  V.trsm(U, 1.0, "l", "u", "t", "n");

  // Sigma = K - V^T V
  Sigma.deepCopy(K);
  Sigma.syrk(V, -1.0, 1.0, "u", "t");
  Sigma.setSymmetric(true);
  // Fill lower from upper for diag reads / later use.
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j < i; j++)
      Sigma.setVal(Sigma.getVal(j, i), i, j);

  varDiag.resize(n, 1);
  for(unsigned int i = 0; i < n; i++)
  {
    double sii = Sigma.getVal(i, i);
    if(sii < ndlutil::EPS)
      sii = ndlutil::EPS;
    varDiag.setVal(sii, i, 0);
  }

  // rhs = sqrt(τ) ⊙ (K v + m0)
  CMatrix Kv(n, 1);
  Kv.gemvColCol(0, K, sites.v, 0, 1.0, 0.0, "n");
  CMatrix rhs(n, 1);
  for(unsigned int i = 0; i < n; i++)
    rhs.setVal(sqrtTau.getVal(i, 0) * (Kv.getVal(i, 0) + m0.getVal(i, 0)), i, 0);

  // aux = B^{-1} rhs via U^T U
  CMatrix aux(rhs);
  aux.trsm(U, 1.0, "l", "u", "t", "n");
  aux.trsm(U, 1.0, "l", "u", "n", "n");

  CMatrix alpha(n, 1);
  for(unsigned int i = 0; i < n; i++)
    alpha.setVal(sites.v.getVal(i, 0) - sqrtTau.getVal(i, 0) * aux.getVal(i, 0), i, 0);

  mu.resize(n, 1);
  mu.gemvColCol(0, K, alpha, 0, 1.0, 0.0, "n");
  for(unsigned int i = 0; i < n; i++)
    mu.setVal(mu.getVal(i, 0) + m0.getVal(i, 0), i, 0);
}

double CEpInference::run(const CMatrix& K,
                         const CMatrix& Y,
                         CEpSites& sites,
                         CMatrix& mu,
                         CMatrix& varDiag,
                         const CMatrix* m0OrNull)
{
  const unsigned int n = K.getRows();
  DIMENSIONMATCH(K.getCols() == n);
  DIMENSIONMATCH(Y.getRows() == n);
  DIMENSIONMATCH(Y.getCols() == 1);

  CMatrix m0(n, 1);
  m0.zeros();
  if(m0OrNull != 0)
  {
    DIMENSIONMATCH(m0OrNull->getRows() == n);
    m0.deepCopy(*m0OrNull);
  }

  sites.resize(n);
  sites.zeros();

  CMatrix Sigma(n, n);
  recomputePosterior(K, sites, m0, mu, Sigma, varDiag);

  CEpSites sitesOld;
  sitesOld.resize(n);
  didConverge = false;
  itersUsed = 0;
  double logZsum = 0.0;

  for(unsigned int it = 0; it < maxIters; it++)
  {
    sitesOld.tau.deepCopy(sites.tau);
    sitesOld.v.deepCopy(sites.v);
    logZsum = 0.0;

    for(unsigned int i = 0; i < n; i++)
    {
      const double sii = varDiag.getVal(i, 0);
      const double mui = mu.getVal(i, 0);
      const double tauSite = sites.tau.getVal(i, 0);
      const double vSite = sites.v.getVal(i, 0);

      double tauCav = 1.0 / sii - eta * tauSite;
      double vCav = mui / sii - eta * vSite;
      if(!(tauCav > ndlutil::EPS))
        tauCav = ndlutil::EPS;

      double Zhat = 0.0, muHat = 0.0, s2Hat = 0.0;
      ndlnoise::bernoulliProbitEpMoments(Zhat, muHat, s2Hat,
                                         Y.getVal(i, 0), tauCav, vCav);
      logZsum += log(max(Zhat, ndlutil::EPS));

      const double dTau = (delta / eta) * (1.0 / s2Hat - 1.0 / sii);
      const double dV = (delta / eta) * (muHat / s2Hat - mui / sii);
      double tauNew = tauSite + dTau;
      if(tauNew < ndlutil::EPS)
        tauNew = ndlutil::EPS;
      sites.tau.setVal(tauNew, i, 0);
      sites.v.setVal(vSite + dV, i, 0);
    }

    recomputePosterior(K, sites, m0, mu, Sigma, varDiag);

    double maxDiff = 0.0;
    for(unsigned int i = 0; i < n; i++)
    {
      maxDiff = max(maxDiff, fabs(sites.tau.getVal(i, 0) - sitesOld.tau.getVal(i, 0)));
      maxDiff = max(maxDiff, fabs(sites.v.getVal(i, 0) - sitesOld.v.getVal(i, 0)));
    }
    itersUsed = it + 1;
    if(it > 0 && maxDiff < tol)
    {
      didConverge = true;
      break;
    }
  }
  return logZsum;
}
