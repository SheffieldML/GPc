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

void CEpSites::resize(unsigned int n, unsigned int k)
{
  tau.resize(n, k);
  v.resize(n, k);
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

// ---- CIP-003E Phase 2: block EP ----

double CEpInference::runBlock(const std::vector<CMatrix>& Ks,
                              const CMatrix& Y,
                              const CNoise& noise,
                              CEpSites& sites,
                              CMatrix& mu,
                              CMatrix& varDiag,
                              const CMatrix* m0OrNull)
{
  const unsigned int n = Y.getRows();
  const unsigned int K = Y.getCols();
  const unsigned int bs = noise.blockSize();
  if(Ks.empty() || (Ks.size() != 1 && Ks.size() != K))
    throw ndlexceptions::Error(
        "CEpInference::runBlock: need one shared Gram matrix or one per latent");
  if(bs == 0 || K % bs != 0)
    throw ndlexceptions::Error(
        "CEpInference::runBlock: blockSize must divide the number of output columns");
  noise.requireCapability(CNoise::CapCavityMoments, "CEpInference::runBlock");
  for(unsigned int k = 0; k < Ks.size(); k++)
  {
    DIMENSIONMATCH(Ks[k].getRows() == n);
    DIMENSIONMATCH(Ks[k].getCols() == n);
  }
  if(m0OrNull != 0)
  {
    DIMENSIONMATCH(m0OrNull->getRows() == n);
    DIMENSIONMATCH(m0OrNull->getCols() == K);
  }

  // Jittered Gram per distinct matrix.
  std::vector<CMatrix> Kj(Ks.size());
  for(unsigned int a = 0; a < Ks.size(); a++)
  {
    Kj[a].deepCopy(Ks[a]);
    for(unsigned int i = 0; i < n; i++)
      Kj[a].setVal(Kj[a].getVal(i, i) + jitter, i, i);
    Kj[a].setSymmetric(true);
  }
  const bool shared = (Ks.size() == 1);

  sites.resize(n, K);
  sites.zeros();
  mu.resize(n, K);
  varDiag.resize(n, K);
  alphaK.assign(K, CMatrix());
  WiK.assign(K, CMatrix());
  haveBlockCache = false;
  haveGradCache = false;

  std::vector<CMatrix> Sigma(K);
  std::vector<double> logDetB(K, 0.0);
  std::vector<CMatrix> m0(K);
  for(unsigned int k = 0; k < K; k++)
  {
    m0[k].resize(n, 1);
    m0[k].zeros();
    if(m0OrNull != 0)
      for(unsigned int i = 0; i < n; i++)
        m0[k].setVal(m0OrNull->getVal(i, k), i, 0);
    Sigma[k].resize(n, n);
  }

  CEpSites col;  // N x 1 view of one latent's sites
  col.resize(n);
  CMatrix muK, vdK;
  auto recomputeAll = [&]() {
    for(unsigned int k = 0; k < K; k++)
    {
      for(unsigned int i = 0; i < n; i++)
      {
        col.tau.setVal(sites.tau.getVal(i, k), i, 0);
        col.v.setVal(sites.v.getVal(i, k), i, 0);
      }
      recomputePosterior(Kj[shared ? 0 : k], col, m0[k], muK, Sigma[k], vdK, logDetB[k]);
      alphaK[k].deepCopy(alpha);
      WiK[k].deepCopy(Wi);
      for(unsigned int i = 0; i < n; i++)
      {
        mu.setVal(muK.getVal(i, 0), i, k);
        varDiag.setVal(vdK.getVal(i, 0), i, k);
      }
    }
  };
  recomputeAll();

  CMatrix tauCav(n, K), vCav(n, K), logZ(n, 1);
  CMatrix oldTau(n, K), oldV(n, K);
  std::vector<double> yv(bs), mc(bs), vc(bs), dM(bs), dV(bs);
  didConverge = false;
  itersUsed = 0;

  for(unsigned int it = 0; it < maxIters; it++)
  {
    oldTau.deepCopy(sites.tau);
    oldV.deepCopy(sites.v);

    for(unsigned int i = 0; i < n; i++)
    {
      double logZi = 0.0;
      for(unsigned int j0 = 0; j0 < K; j0 += bs)
      {
        for(unsigned int b = 0; b < bs; b++)
        {
          const unsigned int k = j0 + b;
          const double sii = varDiag.getVal(i, k);
          const double mui = mu.getVal(i, k);
          double tCav = 1.0 / sii - eta * sites.tau.getVal(i, k);
          const double vC = mui / sii - eta * sites.v.getVal(i, k);
          if(!(tCav > ndlutil::EPS))
            tCav = ndlutil::EPS;
          tauCav.setVal(tCav, i, k);
          vCav.setVal(vC, i, k);
          yv[b] = Y.getVal(i, k);
          vc[b] = 1.0 / tCav;       // cavity variance
          mc[b] = vC / tCav;        // cavity mean
        }
        noise.cavityMomentsBlock(yv.data(), mc.data(), vc.data(), j0, dM.data(), dV.data());
        logZi += noise.logZBlock(yv.data(), mc.data(), vc.data(), j0);
        for(unsigned int b = 0; b < bs; b++)
        {
          const unsigned int k = j0 + b;
          const double sii = varDiag.getVal(i, k);
          const double mui = mu.getVal(i, k);
          // Tilted marginal moments from derivatives of log Z at the cavity.
          const double muHat = mc[b] + vc[b] * dM[b];
          double s2Hat = vc[b] + vc[b] * vc[b] * (2.0 * dV[b] - dM[b] * dM[b]);
          if(!(s2Hat > ndlutil::EPS))
            s2Hat = ndlutil::EPS;
          const double dTau = (delta / eta) * (1.0 / s2Hat - 1.0 / sii);
          const double dVs = (delta / eta) * (muHat / s2Hat - mui / sii);
          double tauNew = sites.tau.getVal(i, k) + dTau;
          if(tauNew < ndlutil::EPS)
            tauNew = ndlutil::EPS;
          sites.tau.setVal(tauNew, i, k);
          sites.v.setVal(sites.v.getVal(i, k) + dVs, i, k);
        }
      }
      logZ.setVal(logZi, i, 0);
    }

    recomputeAll();

    double maxDiff = 0.0;
    for(unsigned int i = 0; i < n; i++)
      for(unsigned int k = 0; k < K; k++)
      {
        maxDiff = max(maxDiff, fabs(sites.tau.getVal(i, k) - oldTau.getVal(i, k)));
        maxDiff = max(maxDiff, fabs(sites.v.getVal(i, k) - oldV.getVal(i, k)));
      }
    itersUsed = it + 1;
    if(it > 0 && maxDiff < tol)
    {
      didConverge = true;
      break;
    }
  }

  // Normaliser: per-datum block log Z once, site terms per latent.
  logZTilde = 0.0;
  for(unsigned int i = 0; i < n; i++)
  {
    logZTilde += logZ.getVal(i, 0);
    for(unsigned int k = 0; k < K; k++)
    {
      const double tS = sites.tau.getVal(i, k);
      const double vS = sites.v.getVal(i, k);
      const double tC = tauCav.getVal(i, k);
      const double vC = vCav.getVal(i, k);
      const double den = tC + tS;
      logZTilde += 0.5 * log(2.0 * M_PI)
                   + 0.5 * log(1.0 + tS / tC)
                   - 0.5 * (vS * vS) / den
                   + 0.5 * (vC * ((tS / tC) * vC - 2.0 * vS) / den);
    }
  }

  // Gaussian-prior part, per latent (GPy _ep_marginal with mean_prior = m0).
  double prior = 0.0;
  CMatrix vCentered(n, 1), SigmaV(n, 1);
  for(unsigned int k = 0; k < K; k++)
  {
    for(unsigned int i = 0; i < n; i++)
      vCentered.setVal(sites.v.getVal(i, k) - sites.tau.getVal(i, k) * m0[k].getVal(i, 0), i, 0);
    SigmaV.gemvColCol(0, Sigma[k], vCentered, 0, 1.0, 0.0, "n");
    double quad = 0.0, meanTerm = 0.0;
    for(unsigned int i = 0; i < n; i++)
    {
      quad += vCentered.getVal(i, 0) * SigmaV.getVal(i, 0);
      const double Smp = sites.tau.getVal(i, k) * m0[k].getVal(i, 0);
      meanTerm += m0[k].getVal(i, 0) * (Smp - 2.0 * sites.v.getVal(i, k));
    }
    prior += 0.5 * (-(double)n * log(2.0 * M_PI) - logDetB[k] + quad - meanTerm);
  }
  logMarginal = prior + logZTilde;
  haveBlockCache = true;
  return logMarginal;
}

void CEpInference::fillDLDKBlock(unsigned int k, CMatrix& dLdK) const
{
  if(!haveBlockCache || k >= alphaK.size())
    throw ndlexceptions::Error("CEpInference::fillDLDKBlock requires a prior runBlock().");
  const unsigned int n = alphaK[k].getRows();
  dLdK.resize(n, n);
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = i; j < n; j++)
    {
      const double val = 0.5 * (alphaK[k].getVal(i, 0) * alphaK[k].getVal(j, 0)
                                - WiK[k].getVal(i, j));
      dLdK.setVal(val, i, j);
      if(j != i)
        dLdK.setVal(val, j, i);
    }
  dLdK.setSymmetric(true);
}

void CEpInference::predictiveMeanVarBlock(unsigned int k,
                                          const CMatrix& KtrainTest,
                                          const CMatrix& kss,
                                          CMatrix& mu,
                                          CMatrix& varDiag) const
{
  if(!haveBlockCache || k >= alphaK.size())
    throw ndlexceptions::Error(
        "CEpInference::predictiveMeanVarBlock requires a prior runBlock().");
  const unsigned int n = alphaK[k].getRows();
  const unsigned int nt = KtrainTest.getCols();
  DIMENSIONMATCH(KtrainTest.getRows() == n);
  DIMENSIONMATCH(kss.getRows() == nt);
  mu.resize(nt, 1);
  varDiag.resize(nt, 1);
  mu.gemvColCol(0, KtrainTest, alphaK[k], 0, 1.0, 0.0, "t");
  for(unsigned int t = 0; t < nt; t++)
  {
    double quad = 0.0;
    for(unsigned int i = 0; i < n; i++)
    {
      double s = 0.0;
      for(unsigned int j = 0; j < n; j++)
        s += WiK[k].getVal(i, j) * KtrainTest.getVal(j, t);
      quad += KtrainTest.getVal(i, t) * s;
    }
    double v = kss.getVal(t, 0) - quad;
    if(v < ndlutil::EPS)
      v = ndlutil::EPS;
    varDiag.setVal(v, t, 0);
  }
}
