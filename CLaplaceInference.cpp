#include "CLaplaceInference.h"
#include "ndlutil.h"
#include "ndlexceptions.h"
#include <cmath>
#include <algorithm>

using namespace std;

CLaplaceInference::CLaplaceInference()
  : maxIters(30), tol(1e-4), jitter(1e-6),
    itersUsed(0), didConverge(false), logMarginal(0.0),
    haveGradCache(false)
{
}

void CLaplaceInference::fillDerivs(const CMatrix& f,
                                   const CMatrix& Y,
                                   CMatrix& logp,
                                   CMatrix& d1,
                                   CMatrix& Wdiag,
                                   CMatrix& dWdf) const
{
  const unsigned int n = f.getRows();
  logp.resize(n, 1);
  d1.resize(n, 1);
  Wdiag.resize(n, 1);
  dWdf.resize(n, 1);
  for(unsigned int i = 0; i < n; i++)
  {
    double lp = 0.0, g = 0.0, W = 0.0, dW = 0.0;
    ndlnoise::bernoulliProbitLaplaceDerivs(lp, g, W, dW,
                                           Y.getVal(i, 0), f.getVal(i, 0));
    logp.setVal(lp, i, 0);
    d1.setVal(g, i, 0);
    Wdiag.setVal(W, i, 0);
    dWdf.setVal(dW, i, 0);
  }
}

double CLaplaceInference::objective(const CMatrix& KiFvec,
                                    const CMatrix& f,
                                    const CMatrix& logp) const
{
  double sumLp = 0.0;
  for(unsigned int i = 0; i < logp.getRows(); i++)
    sumLp += logp.getVal(i, 0);
  return -0.5 * KiFvec.dotColCol(0, f, 0) + sumLp;
}

void CLaplaceInference::computeBStats(const CMatrix& K,
                                      const CMatrix& Wdiag,
                                      CMatrix& KWiI,
                                      CMatrix& KiWI,
                                      double& logDetB) const
{
  const unsigned int n = K.getRows();
  CMatrix W12(n, 1);
  for(unsigned int i = 0; i < n; i++)
    W12.setVal(sqrt(max(Wdiag.getVal(i, 0), ndlutil::EPS)), i, 0);

  CMatrix B(n, n);
  B.eye();
  for(unsigned int i = 0; i < n; i++)
  {
    const double wi = W12.getVal(i, 0);
    for(unsigned int j = 0; j < n; j++)
      B.setVal(B.getVal(i, j) + wi * K.getVal(i, j) * W12.getVal(j, 0), i, j);
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

  // K_Wi_i = W^{1/2} B^{-1} W^{1/2}: solve B X = diag(W12), scale rows.
  CMatrix Xb(n, n);
  Xb.zeros();
  for(unsigned int i = 0; i < n; i++)
    Xb.setVal(W12.getVal(i, 0), i, i);
  Xb.trsm(U, 1.0, "l", "u", "t", "n");
  Xb.trsm(U, 1.0, "l", "u", "n", "n");
  KWiI.resize(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    const double wi = W12.getVal(i, 0);
    for(unsigned int j = 0; j < n; j++)
      KWiI.setVal(wi * Xb.getVal(i, j), i, j);
  }
  KWiI.setSymmetric(true);

  // Ki_W_i = K - K K_Wi_i K = K - C^T C with C = L^{-1} W12 K
  CMatrix SrootK(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    const double wi = W12.getVal(i, 0);
    for(unsigned int j = 0; j < n; j++)
      SrootK.setVal(wi * K.getVal(i, j), i, j);
  }
  CMatrix V(SrootK);
  V.trsm(U, 1.0, "l", "u", "t", "n");
  KiWI.deepCopy(K);
  KiWI.syrk(V, -1.0, 1.0, "u", "t");
  KiWI.setSymmetric(true);
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j < i; j++)
      KiWI.setVal(KiWI.getVal(j, i), i, j);
}

double CLaplaceInference::run(const CMatrix& KIn, const CMatrix& Y)
{
  const unsigned int n = KIn.getRows();
  DIMENSIONMATCH(KIn.getCols() == n);
  DIMENSIONMATCH(Y.getRows() == n);
  DIMENSIONMATCH(Y.getCols() == 1);

  CMatrix K(KIn);
  for(unsigned int i = 0; i < n; i++)
    K.setVal(K.getVal(i, i) + jitter, i, i);
  K.setSymmetric(true);

  KiF.resize(n, 1);
  KiF.zeros();
  fHat.resize(n, 1);
  fHat.zeros();
  haveGradCache = false;

  CMatrix logp, d1, Wdiag, dWdf;
  CMatrix KWiI, KiWI;
  double logDetB = 0.0;
  didConverge = false;
  itersUsed = 0;

  double prevObj = -1e300;
  for(unsigned int it = 0; it < maxIters; it++)
  {
    fillDerivs(fHat, Y, logp, d1, Wdiag, dWdf);
    computeBStats(K, Wdiag, KWiI, KiWI, logDetB);

    // b = W f + ∇logp; a = b − K_Wi_i K b  (R&W / GPy rasm_mode)
    CMatrix b(n, 1);
    for(unsigned int i = 0; i < n; i++)
      b.setVal(Wdiag.getVal(i, 0) * fHat.getVal(i, 0) + d1.getVal(i, 0), i, 0);
    CMatrix Kb(n, 1);
    Kb.gemvColCol(0, K, b, 0, 1.0, 0.0, "n");
    CMatrix KWiKb(n, 1);
    KWiKb.gemvColCol(0, KWiI, Kb, 0, 1.0, 0.0, "n");
    CMatrix fullStep(n, 1);
    for(unsigned int i = 0; i < n; i++)
      fullStep.setVal(b.getVal(i, 0) - KWiKb.getVal(i, 0), i, 0);

    CMatrix dKi(n, 1);
    for(unsigned int i = 0; i < n; i++)
      dKi.setVal(fullStep.getVal(i, 0) - KiF.getVal(i, 0), i, 0);

    const double oldObj = objective(KiF, fHat, logp);
    double step = 1.0;
    CMatrix KiTrial(n, 1), fTrial(n, 1), logpTrial;
    double newObj = oldObj;
    bool accepted = false;
    for(unsigned int ls = 0; ls < 20; ls++)
    {
      for(unsigned int i = 0; i < n; i++)
        KiTrial.setVal(KiF.getVal(i, 0) + step * dKi.getVal(i, 0), i, 0);
      fTrial.gemvColCol(0, K, KiTrial, 0, 1.0, 0.0, "n");
      fillDerivs(fTrial, Y, logpTrial, d1, Wdiag, dWdf);
      newObj = objective(KiTrial, fTrial, logpTrial);
      if(newObj >= oldObj - 1e-12)
      {
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if(!accepted)
    {
      // Take a tiny step anyway to avoid stalling.
      step = max(step, 1e-4);
      for(unsigned int i = 0; i < n; i++)
        KiTrial.setVal(KiF.getVal(i, 0) + step * dKi.getVal(i, 0), i, 0);
      fTrial.gemvColCol(0, K, KiTrial, 0, 1.0, 0.0, "n");
      fillDerivs(fTrial, Y, logpTrial, d1, Wdiag, dWdf);
      newObj = objective(KiTrial, fTrial, logpTrial);
    }

    KiF.deepCopy(KiTrial);
    fHat.deepCopy(fTrial);
    itersUsed = it + 1;
    if(it > 0 && fabs(newObj - prevObj) < tol)
    {
      didConverge = true;
      prevObj = newObj;
      break;
    }
    prevObj = newObj;
  }

  // Final marginal + gradient cache at mode.
  fillDerivs(fHat, Y, logp, d1, Wdiag, dWdf);
  computeBStats(K, Wdiag, KWiI, KiWI, logDetB);
  logMarginal = objective(KiF, fHat, logp) - 0.5 * logDetB;

  // dL_dfhat = −½ diag(Ki_W_i) ⊙ dW_df  (GPy)
  CMatrix dLdf(n, 1);
  for(unsigned int i = 0; i < n; i++)
    dLdf.setVal(-0.5 * KiWI.getVal(i, i) * dWdf.getVal(i, 0), i, 0);

  // I_KW_i = I − K K_Wi_i
  CMatrix KKWi(n, n);
  KKWi.gemm(K, KWiI, 1.0, 0.0, "n", "n");
  CMatrix IKWi(n, n);
  IKWi.eye();
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j < n; j++)
      IKWi.setVal(IKWi.getVal(i, j) - KKWi.getVal(i, j), i, j);

  // explicit = ½(ααᵀ − K_Wi_i); implicit = α dL_dfᵀ (I − K K_Wi_i)
  dLdKCache.resize(n, n);
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = i; j < n; j++)
    {
      double expl = 0.5 * (KiF.getVal(i, 0) * KiF.getVal(j, 0) - KWiI.getVal(i, j));
      double impl = 0.0;
      for(unsigned int k = 0; k < n; k++)
        impl += KiF.getVal(i, 0) * dLdf.getVal(k, 0) * IKWi.getVal(k, j);
      // GPy: (Ki_f @ dL_dfhat.T) @ I_KW_i  → element (i,j) =
      //   Ki_f[i] * sum_k dL_df[k] * I_KW_i[k,j]
      // But also need the transpose contribution for symmetry of outer product
      // form Ki_f dL^T — that matrix is rank-1 and generally non-symmetric before
      // multiplying I_KW_i. Match GPy elementwise (not force-symmetrize early).
      const double val = expl + impl;
      dLdKCache.setVal(val, i, j);
      if(j != i)
      {
        double implT = 0.0;
        for(unsigned int k = 0; k < n; k++)
          implT += KiF.getVal(j, 0) * dLdf.getVal(k, 0) * IKWi.getVal(k, i);
        dLdKCache.setVal(0.5 * (KiF.getVal(j, 0) * KiF.getVal(i, 0) - KWiI.getVal(j, i))
                         + implT, j, i);
      }
    }
  }
  // Symmetrize for kern.getGradTransParams (same as averaging GPy's dL_dK).
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = i + 1; j < n; j++)
    {
      const double s = 0.5 * (dLdKCache.getVal(i, j) + dLdKCache.getVal(j, i));
      dLdKCache.setVal(s, i, j);
      dLdKCache.setVal(s, j, i);
    }
  }
  dLdKCache.setSymmetric(true);
  KWiICache.deepCopy(KWiI);
  haveGradCache = true;
  return logMarginal;
}

void CLaplaceInference::fillDLDK(CMatrix& dLdK) const
{
  if(!haveGradCache)
    throw ndlexceptions::Error("CLaplaceInference::fillDLDK requires a prior run().");
  dLdK.deepCopy(dLdKCache);
}

void CLaplaceInference::predictiveMeanVar(const CMatrix& KtrainTest,
                                          const CMatrix& kss,
                                          CMatrix& mu,
                                          CMatrix& varDiag) const
{
  if(!haveGradCache)
    throw ndlexceptions::Error(
        "CLaplaceInference::predictiveMeanVar requires a prior run().");
  const unsigned int n = KiF.getRows();
  const unsigned int nt = KtrainTest.getCols();
  DIMENSIONMATCH(KtrainTest.getRows() == n);
  DIMENSIONMATCH(kss.getRows() == nt);
  DIMENSIONMATCH(KWiICache.getRows() == n);
  mu.resize(nt, 1);
  varDiag.resize(nt, 1);
  // μ = K_*ᵀ α  (α = K^{-1} f̂)
  mu.gemvColCol(0, KtrainTest, KiF, 0, 1.0, 0.0, "t");
  // v_i = kss_i − k_iᵀ (K + W^{-1})^{-1} k_i
  CMatrix Wk(n, 1);
  for(unsigned int t = 0; t < nt; t++)
  {
    for(unsigned int i = 0; i < n; i++)
    {
      double s = 0.0;
      for(unsigned int j = 0; j < n; j++)
        s += KWiICache.getVal(i, j) * KtrainTest.getVal(j, t);
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
