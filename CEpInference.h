/* CIP-0033: EP inference workspace (GPy EP mirror), sites not on CNoise. */
#ifndef CEPINFERENCE_H
#define CEPINFERENCE_H

#include "CMatrix.h"
#include "CNoise.h"
#include <vector>

// Gaussian site approximation in GPy natural parameters (tau, v).
struct CEpSites
{
  CMatrix tau; // N×1 site precision τ̃
  CMatrix v;   // N×1 site ν̃ = τ̃ μ̃

  void resize(unsigned int n);                    // N x 1
  void resize(unsigned int n, unsigned int k);    // N x K (CIP-003E block EP)
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

  // Posterior at test: μ_* = K_*ᵀ α,  v_* = kss − diag(K_*ᵀ W K_*).
  // KtrainTest is N×N_*, kss is N_*×1 prior variances.
  void predictiveMeanVar(const CMatrix& KtrainTest,
                         const CMatrix& kss,
                         CMatrix& mu,
                         CMatrix& varDiag) const;

  // ---- CIP-003E Phase 2: block EP over a CNoise ----
  // K output columns (one-hot targets N x K for classification), one Gaussian site
  // per (datum, latent) so sites are N x K and diagonal across the K latents.
  // Given the sites the K latent posteriors are independent solves; the only
  // coupling is the moment match, done per datum through
  // CNoise::cavityMomentsBlock / logZBlock (blocks of noise.blockSize() columns).
  //   Ks: one Gram matrix shared by all latents, or K Gram matrices (one per latent).
  //   Y:  N x K targets; m0OrNull: optional N x K prior means.
  // Fills sites (N x K), mu (N x K), varDiag (N x K). Returns the EP log marginal
  // likelihood, summed over latents (GPy's _ep_marginal per latent, with the
  // per-datum log Z entering once per block). For a blockSize-1 likelihood and
  // K = 1 this is the usual EP; for Gaussian noise it equals the exact marginal.
  double runBlock(const std::vector<CMatrix>& Ks,
                  const CMatrix& Y,
                  const CNoise& noise,
                  CEpSites& sites,
                  CMatrix& mu,
                  CMatrix& varDiag,
                  const CMatrix* m0OrNull = 0);
  // After runBlock: dL/dK for latent k (N x N, symmetric); ½(ααᵀ − W) as in fillDLDK.
  void fillDLDKBlock(unsigned int k, CMatrix& dLdK) const;
  // Posterior at test inputs for latent k.
  void predictiveMeanVarBlock(unsigned int k,
                              const CMatrix& KtrainTest,
                              const CMatrix& kss,
                              CMatrix& mu,
                              CMatrix& varDiag) const;
  unsigned int numLatentsBlock() const { return (unsigned int)alphaK.size(); }

  double lastLogMarginal() const { return logMarginal; }
  double lastLogZTilde() const { return logZTilde; }
  unsigned int lastIters() const { return itersUsed; }
  bool converged() const { return didConverge; }
  const CMatrix& lastAlpha() const { return alpha; }
  const CMatrix& lastW() const { return Wi; }

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
  // Per-latent caches after runBlock.
  mutable std::vector<CMatrix> alphaK;
  mutable std::vector<CMatrix> WiK;
  mutable bool haveBlockCache = false;
};

#endif /* CEPINFERENCE_H */
