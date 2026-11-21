/* CIP-0033 Phase C: dense Laplace inference (GPy Laplace mirror), Bernoulli–probit. */
#ifndef CLAPLACEINFERENCE_H
#define CLAPLACEINFERENCE_H

#include "CMatrix.h"
#include "CNoise.h"

// Dense (FTC) Laplace approximation for Bernoulli–probit, mirroring
// GPy.inference.latent_function_inference.laplace.Laplace.
class CLaplaceInference
{
 public:
  CLaplaceInference();

  void setMaxIters(unsigned int n) { maxIters = n; }
  void setTol(double t) { tol = t; }
  void setJitter(double j) { jitter = j; }

  // Mode + approximate log marginal on prior Gram K (N×N), labels Y (N×1).
  // Returns GPy log_marginal (no kernel prior).
  double run(const CMatrix& K, const CMatrix& Y);

  // After run: dL/dK including GPy implicit mode term.
  void fillDLDK(CMatrix& dLdK) const;

  // Posterior at test (GPy Laplace._raw_predict): μ_* = K_*ᵀ α,
  // v_* = kss − diag(K_*ᵀ (K+W⁻¹)⁻¹ K_*) with α = K⁻¹ f̂.
  // KtrainTest is N×N_*, kss is N_*×1 prior variances. Requires prior run().
  void predictiveMeanVar(const CMatrix& KtrainTest,
                         const CMatrix& kss,
                         CMatrix& mu,
                         CMatrix& varDiag) const;

  double lastLogMarginal() const { return logMarginal; }
  unsigned int lastIters() const { return itersUsed; }
  bool converged() const { return didConverge; }
  const CMatrix& lastFhat() const { return fHat; }
  const CMatrix& lastAlpha() const { return KiF; } // K^{-1} f̂

 private:
  void computeBStats(const CMatrix& K,
                     const CMatrix& Wdiag,
                     CMatrix& KWiI,
                     CMatrix& KiWI,
                     double& logDetB) const;

  void fillDerivs(const CMatrix& f,
                  const CMatrix& Y,
                  CMatrix& logp,
                  CMatrix& d1,
                  CMatrix& Wdiag,
                  CMatrix& dWdf) const;

  double objective(const CMatrix& KiFvec,
                   const CMatrix& f,
                   const CMatrix& logp) const;

  unsigned int maxIters;
  double tol;
  double jitter;
  unsigned int itersUsed;
  bool didConverge;
  double logMarginal;

  mutable CMatrix fHat;
  mutable CMatrix KiF;
  mutable CMatrix KWiICache; // (K + W^{-1})^{-1}; GPy woodbury_inv
  mutable CMatrix dLdKCache;
  mutable bool haveGradCache;
};

#endif /* CLAPLACEINFERENCE_H */
