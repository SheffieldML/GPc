/* CIP-002A: stochastic variational GP (SVGP), Hensman et al. 2013.
   CIP-0030 Phase 0a: K_uu / K_uf (and inducing grads) via owned CInducing*.
   CIP-0030 Phase 0b: explicit-input core (XB, yRows, c, optional gX) — no
   KL(q(X)) and no ownership of (μ, log S); those stay in CSviBayesGplvm.

   Explicit q(u_j) = N(m_j, S_j) per output column j, shared kernel and inducing
   representation, Gaussian likelihood with scalar precision beta.

     ELBO = sum_j [ sum_i E_q[log N(y_ij | f_ij, 1/beta)] - KL(q(u_j) || p(u)) ]
            + kern.priorLogProb()

   q(u) is updated with natural-gradient steps written in the CVI form
   (likelihood gradients w.r.t. the marginal mean/variance of q(f_i)), so the
   same step serves non-Gaussian likelihoods once E_q[log p(y|f)] and its
   derivatives are available (Phase 4, Gauss-Hermite).  With the full batch,
   Gaussian noise and gamma = 1 one step lands on the Titsias optimum and the
   ELBO equals CGp's DTCVAR bound.

   Kernel matrices go through CInducing (points / white-noise / weight-space /
   Green's). The Z constructor owns a CInducingPoints on member centres storage.
   Whitened parameterisation: u = L v with K_uu = L L^T and q(v) = N(m~, S~).

   Phase 3: CAdam + seeded minibatches live in CAdam / CStochasticOptimisable;
   CSvgp implements the stochastic interface. */

#ifndef CSVGP_H
#define CSVGP_H

#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CTransform.h"
#include "CInducing.h"
#include "CStochasticOptimisable.h"
#include <string>
#include <vector>

using namespace std;

class CSvgp : public CStochasticOptimisable {
 public:
  // X: N x q inputs, y: N x d targets, Z: M x q inducing centres (point inducing).
  // Owns a CInducingPoints on member Z. q(u_j) initialised to the prior.
  CSvgp(CKern* kernel, const CMatrix* X, const CMatrix* y, const CMatrix& Z,
        double beta = 1.0, int verbos = 0);
  virtual ~CSvgp();

  // Take ownership of pInd (non-null). Resizes q(v) if M changes.
  // For points / white-noise / Green's the object must point at getCentresStorage()
  // (or an external matrix the caller keeps alive). Weight-space owns W itself.
  void setInducing(CInducing* pInd);
  void setInducingType(const string& typeName);
  CInducing* getInducing() { return pInducing; }
  const CInducing* getInducing() const { return pInducing; }
  // Centres storage for families that need a non-owned CMatrix* (points, WN, greens).
  CMatrix& getCentresStorage() { return Z; }
  const CMatrix& getCentresStorage() const { return Z; }

  // Recompute K_uu, K_uf, diag K after kernel or inducing changes.
  void updateKernelMatrices();

  // Full-batch ELBO.
  double elbo() const;
  // Unbiased minibatch estimate: data term scaled by N/|batch|.
  double elbo(const vector<unsigned int>& batch) const;

  // Natural-gradient step on q(u) with step size gamma in (0, 1].
  void naturalGradientStep(double gamma);
  void naturalGradientStep(double gamma, const vector<unsigned int>& batch);

  // CIP-0031: latent = p(f_*); predictive adds 1/β for Gaussian observation noise.
  void latentMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xs) const;
  void predictiveMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xs) const;
  // Deprecated alias: same as latentMeanVar (pre-0031 SVGP returned latent variance).
  void posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xs) const;

  // KL(q(u_j) || p(u)).
  double klDivergence(unsigned int j) const;

  unsigned int getNumData() const { return nData; }
  unsigned int getNumInducing() const { return numInducing; }
  unsigned int getOutputDim() const { return outputDim; }
  unsigned int getInputDim() const { return pX->getCols(); }
  double getBeta() const { return beta; }
  void setBeta(double val);
  // Inducing centres (point / WN / greens) or weight-space W.
  const CMatrix& getInducingCentres() const { return pInducing->centres(); }
  void setInducingCentres(const CMatrix& Znew);
  CKern* getKern() { return pkern; }
  const CKern* getKern() const { return pkern; }
  // Optional non-Gaussian likelihood (Phase 4). Null => internal Gaussian beta.
  void setLikelihood(CNoise* noise) { pnoise = noise; }
  CNoise* getLikelihood() const { return pnoise; }

  // Unwhitened q(u_j): m = L m~, S = L S~ L^T.
  CMatrix getQMean() const;
  CMatrix getQCov(unsigned int j) const;
  // Set from unwhitened (m, S); maps to whitened internally.
  void setQ(const CMatrix& mean, const vector<CMatrix>& cov);
  // Whitened q(v_j).
  const CMatrix& getWhitenedMean() const { return qMean; }
  const CMatrix& getWhitenedCov(unsigned int j) const { return qCov[j]; }
  void setWhitenedQ(const CMatrix& mean, const vector<CMatrix>& cov);

  // ---- Phase 2: gradients at fixed whitened q ----
  // Fills gKern (1 x nKernParams), gZ (same shape as centres()), gBeta.
  // Optional gXOrNull (B x q, same rows as the batch inputs) via Phase 0b core.
  // Returns ELBO on batch.
  double elboGrad(CMatrix& gKern, CMatrix& gZ, double& gBeta,
                  const vector<unsigned int>& batch,
                  CMatrix* gXOrNull = 0) const;
  double elboGrad(CMatrix& gKern, CMatrix& gZ, double& gBeta) const;

  // ---- CIP-0030 Phase 0b: explicit-input core ----
  // Evaluate / differentiate the data term at caller-supplied inputs XB (B x q).
  // yRows indexes rows of Y (size B); c is the minibatch weight (typically N/B).
  // Does not include KL(q(u)), kern prior, or any q(X) terms.
  double dataTermAt(const CMatrix& XB, const vector<unsigned int>& yRows,
                    double c) const;
  // Gradients of (dataTermAt + KL(q(u)) + kern prior) w.r.t. kern / inducing /
  // beta, and optionally ∂dataTerm/∂XB into gXOrNull (B x q).
  double elboGradAt(CMatrix& gKern, CMatrix& gZ, double& gBeta, CMatrix* gXOrNull,
                    const CMatrix& XB, const vector<unsigned int>& yRows,
                    double c) const;
  // CVI natural-gradient step using sites at XB (same c scaling as dataTermAt).
  void naturalGradientStepAt(double gamma, const CMatrix& XB,
                             const vector<unsigned int>& yRows, double c);

  // Packed unconstrained parameters: kern transforms, then inducing params
  // (CInducing::getParams), then betaTransform(beta).
  unsigned int getOptNumParams() const;
  void getOptParams(CMatrix& param) const;
  void setOptParams(const CMatrix& param);

  // CStochasticOptimisable
  unsigned int numData() const override { return nData; }
  double objectiveGradBatch(const vector<unsigned int>& batch, CMatrix& g) override;
  void getStochasticParams(CMatrix& param) const override { getOptParams(param); }
  void setStochasticParams(const CMatrix& param) override { setOptParams(param); }
  unsigned int getStochasticNumParams() const override { return getOptNumParams(); }

  // Optional interleaved natgrad after each Adam step (Phase 3).
  void setNatGradGamma(double g) { natGradGamma = g; }
  double getNatGradGamma() const { return natGradGamma; }
  void afterBatchStep(const vector<unsigned int>& batch) override;

 private:
  void initVariationalPrior();
  void packInducingGrad(CMatrix& gFlat, unsigned int offset, const CMatrix& gZ) const;
  void gatherXB(const vector<unsigned int>& batch, CMatrix& XB) const;

  // Gaussian likelihood: E_q[log N(y|f,1/beta)] and its derivatives w.r.t.
  // the marginal mean and variance of q(f).  Non-Gaussian noise plugs in here.
  double expectedLogLik(double yv, double mu, double var,
                        double& gMu, double& gVar, unsigned int j = 0) const;
  double expectedLogLikBetaDeriv(double yv, double mu, double var) const;
  // Marginals of q(f_i) for output j at training point i.
  void marginal(unsigned int i, unsigned int j, double& mu, double& var) const;
  // CIP-002D QW6: batch form of A_B and marginals (BLAS-3).
  bool isFullOrderedBatch(const vector<unsigned int>& batch) const;
  void formAB(const vector<unsigned int>& batch, CMatrix& AB,
              CMatrix& dK, CMatrix& dQ) const;
  void formABFromX(const CMatrix& XB, CMatrix& AB, CMatrix& dK, CMatrix& dQ) const;
  void fillMarginals(const vector<unsigned int>& batch,
                     CMatrix& muOut, CMatrix& varOut) const;
  void fillMarginalsFromX(const CMatrix& XB, CMatrix& muOut, CMatrix& varOut) const;
  double dataTerm(const vector<unsigned int>& batch) const;
  double klPlusPrior() const;
  vector<unsigned int> allIndices() const;

  CKern* pkern;
  const CMatrix* pX;
  const CMatrix* py;
  CNoise* pnoise;  // optional; not owned
  CMatrix Z;       // centres storage for points / WN / greens
  CInducing* pInducing; // owned
  double beta;
  int verbosity;
  CTransform* betaTransform;
  double natGradGamma;  // 0 => skip interleaved natgrad

  unsigned int nData;
  unsigned int numInducing;
  unsigned int outputDim;

  // Kernel quantities.
  CMatrix Kuu;      // M x M (jitter-modified if jitChol needed it)
  CMatrix cholKuu;  // upper Cholesky U of Kuu (Kuu = U^T U, L = U^T)
  CMatrix Kuf;      // M x N
  CMatrix A;        // M x N, L^{-1} K_uf (whitened projection)
  CMatrix diagK;    // N x 1
  CMatrix diagQ;    // N x 1, column norms^2 of A = diag(K_fu K_uu^{-1} K_uf)

  // Whitened variational parameters q(v_j) = N(qMean(:,j), qCov[j]).
  CMatrix qMean;         // M x d
  vector<CMatrix> qCov;  // d of M x M

  // CIP-002D QW7: reusable buffers (resized only when shape changes).
  mutable CMatrix wsAB;
  mutable CMatrix wsDK;
  mutable CMatrix wsDQ;
  mutable CMatrix wsMu;
  mutable CMatrix wsVar;
  mutable CMatrix wsSAB;
  mutable CMatrix wsABS;
  mutable CMatrix wsCoeffs;
  mutable CMatrix wsXB;
};

#endif
