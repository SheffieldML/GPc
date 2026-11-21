/* CIP-0033: EP inference workspace (GPy EP mirror), sites not on CNoise. */
#ifndef CEPINFERENCE_H
#define CEPINFERENCE_H

#include "CMatrix.h"
#include "CNoise.h"

// Gaussian site approximation in GPy natural parameters (tau, v).
struct CEpSites
{
  CMatrix tau; // N×1 site precision τ̃
  CMatrix v;   // N×1 site ν̃ = τ̃ μ̃

  void resize(unsigned int n);
  void zeros();
};

// Dense (FTC) expectation propagation for Bernoulli–probit, mirroring
// GPy.inference.latent_function_inference.expectation_propagation.EP.
class CEpInference
{
 public:
  CEpInference();

  void setMaxIters(unsigned int n) { maxIters = n; }
  void setEta(double e) { eta = e; }
  void setDelta(double d) { delta = d; }
  void setTol(double t) { tol = t; }
  void setJitter(double j) { jitter = j; }

  // Run EP on prior Gram K (N×N), labels Y (N×1 in {0,1} or {-1,1}).
  // Optional prior mean m0 (N×1); if null, zero mean.
  // Fills sites, posterior mean mu, diag variance varDiag.
  // Returns GPy _ep_marginal log Z (approximate log marginal likelihood).
  double run(const CMatrix& K,
             const CMatrix& Y,
             CEpSites& sites,
             CMatrix& mu,
             CMatrix& varDiag,
             const CMatrix* m0OrNull = 0);

  // After a successful run: dL/dK = ½(ααᵀ − W) with W=(K+Σ̃)^{-1} (GPy).
  // Out must be N×N; written symmetric.
  void fillDLDK(CMatrix& dLdK) const;

  double lastLogMarginal() const { return logMarginal; }
  double lastLogZTilde() const { return logZTilde; }
  unsigned int lastIters() const { return itersUsed; }
  bool converged() const { return didConverge; }
  const CMatrix& lastAlpha() const { return alpha; }

 private:
  void recomputePosterior(const CMatrix& K,
                          const CEpSites& sites,
                          const CMatrix& m0,
                          CMatrix& mu,
                          CMatrix& Sigma,
                          CMatrix& varDiag,
                          double& logDetB);

  unsigned int maxIters;
  double eta;
  double delta;
  double tol;
  double jitter;
  unsigned int itersUsed;
  bool didConverge;
  double logMarginal;
  double logZTilde;

  // Cached for dL/dK after run (mean-zero prior).
  mutable CMatrix alpha;
  mutable CMatrix Wi; // (K + Σ̃)^{-1}
  mutable bool haveGradCache;
};

#endif /* CEPINFERENCE_H */
