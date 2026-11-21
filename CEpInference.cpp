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
  : maxIters(100), eta(1.0), delta(1.0), tol(1e-6), jitter(1e-6),
    itersUsed(0), didConverge(false), logMarginal(0.0), logZTilde(0.0),
    haveGradCache(false)
{
}

void CEpInference::recomputePosterior(const CMatrix& K,
                                      const CEpSites& sites,
                                      const CMatrix& m0,
                                      CMatrix& mu,
                                      CMatrix& Sigma,
                                      CMatrix& varDiag,
                                      double& logDetB)
{
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

  CMatrix SrootK(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    const double si = sqrtTau.getVal(i, 0);
    for(unsigned int j = 0; j < n; j++)
      SrootK.setVal(si * K.getVal(i, j), i, j);
  }

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
  U.jitChol(U);

  logDetB = 0.0;
  for(unsigned int i = 0; i < n; i++)
  {
    const double uii = U.getVal(i, i);
    BOUNDCHECK(uii > 0.0);
    logDetB += 2.0 * log(uii);
  }

  CMatrix V(SrootK);
  V.trsm(U, 1.0, "l", "u", "t", "n");

  Sigma.deepCopy(K);
  Sigma.syrk(V, -1.0, 1.0, "u", "t");
  Sigma.setSymmetric(true);
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

  CMatrix Kv(n, 1);
  Kv.gemvColCol(0, K, sites.v, 0, 1.0, 0.0, "n");
  CMatrix rhs(n, 1);
  for(unsigned int i = 0; i < n; i++)
    rhs.setVal(sqrtTau.getVal(i, 0) * (Kv.getVal(i, 0) + m0.getVal(i, 0)), i, 0);

  CMatrix aux(rhs);
  aux.trsm(U, 1.0, "l", "u", "t", "n");
  aux.trsm(U, 1.0, "l", "u", "n", "n");

  alpha.resize(n, 1);
  for(unsigned int i = 0; i < n; i++)
    alpha.setVal(sites.v.getVal(i, 0) - sqrtTau.getVal(i, 0) * aux.getVal(i, 0), i, 0);

  mu.resize(n, 1);
  mu.gemvColCol(0, K, alpha, 0, 1.0, 0.0, "n");
  for(unsigned int i = 0; i < n; i++)
    mu.setVal(mu.getVal(i, 0) + m0.getVal(i, 0), i, 0);

  // W = S^{1/2} B^{-1} S^{1/2}: solve B X = diag(sqrtTau), then scale rows.
  CMatrix Xb(n, n);
  Xb.zeros();
  for(unsigned int i = 0; i < n; i++)
    Xb.setVal(sqrtTau.getVal(i, 0), i, i);
  Xb.trsm(U, 1.0, "l", "u", "t", "n");
  Xb.trsm(U, 1.0, "l", "u", "n", "n");
  Wi.resize(n, n);
  Wi.zeros();
  for(unsigned int i = 0; i < n; i++)
  {
    const double si = sqrtTau.getVal(i, 0);
    for(unsigned int j = 0; j < n; j++)
      Wi.setVal(si * Xb.getVal(i, j), i, j);
  }
  Wi.setSymmetric(true);
  haveGradCache = true;
}

double CEpInference::run(const CMatrix& KIn,
                         const CMatrix& Y,
                         CEpSites& sites,
                         CMatrix& mu,
                         CMatrix& varDiag,
                         const CMatrix* m0OrNull)
{
  const unsigned int n = KIn.getRows();
  DIMENSIONMATCH(KIn.getCols() == n);
  DIMENSIONMATCH(Y.getRows() == n);
  DIMENSIONMATCH(Y.getCols() == 1);

  CMatrix K(KIn);
  for(unsigned int i = 0; i < n; i++)
    K.setVal(K.getVal(i, i) + jitter, i, i);
  K.setSymmetric(true);

  CMatrix m0(n, 1);
  m0.zeros();
  if(m0OrNull != 0)
  {
    DIMENSIONMATCH(m0OrNull->getRows() == n);
    m0.deepCopy(*m0OrNull);
  }

  sites.resize(n);
  sites.zeros();
  haveGradCache = false;

  CMatrix Sigma(n, n);
  double logDetB = 0.0;
  recomputePosterior(K, sites, m0, mu, Sigma, varDiag, logDetB);

  CEpSites sitesOld;
  sitesOld.resize(n);
  CMatrix tauCav(n, 1), vCav(n, 1), Zhat(n, 1);
  didConverge = false;
  itersUsed = 0;

  for(unsigned int it = 0; it < maxIters; it++)
  {
    sitesOld.tau.deepCopy(sites.tau);
    sitesOld.v.deepCopy(sites.v);

    for(unsigned int i = 0; i < n; i++)
    {
      const double sii = varDiag.getVal(i, 0);
      const double mui = mu.getVal(i, 0);
      const double tauSite = sites.tau.getVal(i, 0);
      const double vSite = sites.v.getVal(i, 0);

      double tCav = 1.0 / sii - eta * tauSite;
      double vC = mui / sii - eta * vSite;
      if(!(tCav > ndlutil::EPS))
        tCav = ndlutil::EPS;
      tauCav.setVal(tCav, i, 0);
      vCav.setVal(vC, i, 0);

      double Zh = 0.0, muHat = 0.0, s2Hat = 0.0;
      ndlnoise::bernoulliProbitEpMoments(Zh, muHat, s2Hat,
                                         Y.getVal(i, 0), tCav, vC);
      Zhat.setVal(Zh, i, 0);

      const double dTau = (delta / eta) * (1.0 / s2Hat - 1.0 / sii);
      const double dV = (delta / eta) * (muHat / s2Hat - mui / sii);
      double tauNew = tauSite + dTau;
      if(tauNew < ndlutil::EPS)
        tauNew = ndlutil::EPS;
      sites.tau.setVal(tauNew, i, 0);
      sites.v.setVal(vSite + dV, i, 0);
    }

    recomputePosterior(K, sites, m0, mu, Sigma, varDiag, logDetB);

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

  // GPy _log_Z_tilde using last cavity / site / Z_hat (from final sweep).
  logZTilde = 0.0;
  for(unsigned int i = 0; i < n; i++)
  {
    const double tS = sites.tau.getVal(i, 0);
    const double vS = sites.v.getVal(i, 0);
    const double tC = tauCav.getVal(i, 0);
    const double vC = vCav.getVal(i, 0);
    const double Zh = max(Zhat.getVal(i, 0), ndlutil::EPS);
    const double den = tC + tS;
    logZTilde += log(Zh)
                 + 0.5 * log(2.0 * M_PI)
                 + 0.5 * log(1.0 + tS / tC)
                 - 0.5 * (vS * vS) / den
                 + 0.5 * (vC * ((tS / tC) * vC - 2.0 * vS) / den);
  }

  // GPy _ep_marginal (mean_prior = m0).
  CMatrix SigmaV(n, 1);
  CMatrix vCentered(n, 1);
  for(unsigned int i = 0; i < n; i++)
  {
    const double Smp = sites.tau.getVal(i, 0) * m0.getVal(i, 0);
    vCentered.setVal(sites.v.getVal(i, 0) - Smp, i, 0);
  }
  SigmaV.gemvColCol(0, Sigma, vCentered, 0, 1.0, 0.0, "n");
  double quad = 0.0;
  double meanTerm = 0.0;
  for(unsigned int i = 0; i < n; i++)
  {
    quad += vCentered.getVal(i, 0) * SigmaV.getVal(i, 0);
    const double Smp = sites.tau.getVal(i, 0) * m0.getVal(i, 0);
    meanTerm += m0.getVal(i, 0) * (Smp - 2.0 * sites.v.getVal(i, 0));
  }
  logMarginal = 0.5 * (-(double)n * log(2.0 * M_PI) - logDetB + quad - meanTerm)
                + logZTilde;
  return logMarginal;
}

void CEpInference::fillDLDK(CMatrix& dLdK) const
{
  if(!haveGradCache)
    throw ndlexceptions::Error("CEpInference::fillDLDK requires a prior run().");
  const unsigned int n = alpha.getRows();
  dLdK.resize(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = i; j < n; j++)
    {
      const double val = 0.5 * (alpha.getVal(i, 0) * alpha.getVal(j, 0)
                                - Wi.getVal(i, j));
      dLdK.setVal(val, i, j);
      if(j != i)
        dLdK.setVal(val, j, i);
    }
  }
  dLdK.setSymmetric(true);
}

void CEpInference::predictiveMeanVar(const CMatrix& KtrainTest,
                                     const CMatrix& kss,
                                     CMatrix& mu,
                                     CMatrix& varDiag) const
{
  if(!haveGradCache)
    throw ndlexceptions::Error(
        "CEpInference::predictiveMeanVar requires a prior run().");
  const unsigned int n = alpha.getRows();
  const unsigned int nt = KtrainTest.getCols();
  DIMENSIONMATCH(KtrainTest.getRows() == n);
  DIMENSIONMATCH(kss.getRows() == nt);
  mu.resize(nt, 1);
  varDiag.resize(nt, 1);
  // μ = K_*ᵀ α
  mu.gemvColCol(0, KtrainTest, alpha, 0, 1.0, 0.0, "t");
  // v_i = kss_i − k_iᵀ W k_i
  CMatrix Wk(n, 1);
  for(unsigned int t = 0; t < nt; t++)
  {
    for(unsigned int i = 0; i < n; i++)
    {
      double s = 0.0;
      for(unsigned int j = 0; j < n; j++)
        s += Wi.getVal(i, j) * KtrainTest.getVal(j, t);
      Wk.setVal(s, i, 0);
    }
    double quad = 0.0;
    for(unsigned int i = 0; i < n; i++)
      quad += KtrainTest.getVal(i, t) * Wk.getVal(i, 0);
    double v = kss.getVal(t, 0) - quad;
    if(v < ndlutil::EPS)
      v = ndlutil::EPS;
    varDiag.setVal(v, t, 0);
  }
}
