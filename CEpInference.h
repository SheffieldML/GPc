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

  // Run EP on Gram K (N×N), labels Y (N×1 in {0,1} or {-1,1}).
  // Optional prior mean m0 (N×1); if null, zero mean.
  // Fills sites, posterior mean mu, posterior diag variance varDiag.
  // Returns approx site log-mass sum (Σ log Ẑ_i) for diagnostics.
  double run(const CMatrix& K,
             const CMatrix& Y,
             CEpSites& sites,
             CMatrix& mu,
             CMatrix& varDiag,
             const CMatrix* m0OrNull = 0);

  unsigned int lastIters() const { return itersUsed; }
  bool converged() const { return didConverge; }

 private:
  void recomputePosterior(const CMatrix& K,
                          const CEpSites& sites,
                          const CMatrix& m0,
                          CMatrix& mu,
                          CMatrix& Sigma,
                          CMatrix& varDiag);

  unsigned int maxIters;
  double eta;
  double delta;
  double tol;
  unsigned int itersUsed;
  bool didConverge;
};

#endif /* CEPINFERENCE_H */
