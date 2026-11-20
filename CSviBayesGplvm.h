/* CIP-0030 Phase 1–2: uncollapsed Bayesian GP-LVM via sampled q(X) on CSvgp.

   q(X) = prod_n N(μ_n, diag S_n), S = exp(logS). Reparameterised samples
   x̃ = μ + s ⊙ ε feed CSvgp's explicit-input core (Phase 0b). No psi statistics.

   Stochastic params for CAdam are globals (kern, inducing, beta). Local
   (μ, logS) use row-sparse Adam updated only on the current minibatch.

   Phase 2: any CInducing family via setInducingType (points / whitenoise /
   weightspace / greens) — same contract as CSvgp. */

#ifndef CSVIBAYESGPLVM_H
#define CSVIBAYESGPLVM_H

#include "CSvgp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNdlInterfaces.h"
#include "CStochasticOptimisable.h"
#include <cstdint>
#include <random>
#include <string>
#include <vector>

const std::string SVIBAYESGPLVMVERSION = "0.1";

class CSviBayesGplvm : public CStochasticOptimisable, public CStreamInterface {
 public:
  // Owns a copy of Y. Borrows kernel unless readParamsFromStream replaces it.
  CSviBayesGplvm(CKern* kernel, const CMatrix& Yin, unsigned int latDim,
                 unsigned int numInducing, double beta = 1.0, int verbosity = 0);
  virtual ~CSviBayesGplvm();

  void initXpca();
  void initZFromMu();
  // Owns a new CInducing of the named type (see CSvgp::setInducingType).
  void setInducingType(const std::string& typeName);
  std::string getInducingType() const;

  void setOwnKern(bool val) { ownKern = val; }
  bool isOwnKern() const { return ownKern; }

  void setNumSamples(unsigned int K);
  unsigned int getNumSamples() const { return numSamples; }
  void setAntithetic(bool val);
  bool isAntithetic() const { return antithetic; }
  void setSeed(uint64_t s);
  void setNatGradGamma(double g);
  double getNatGradGamma() const { return natGradGamma; }
  // Freeze ε for FD / partition checks (uses last drawn or setFrozenEps).
  void setFreezeNoise(bool val) { freezeNoise = val; }
  bool isFreezeNoise() const { return freezeNoise; }
  void setFrozenEps(const CMatrix& eps); // rows = |batch|*K (or N*K for full)

  void setLocalLearningRate(double lr);
  double getLocalLearningRate() const { return localAlpha; }

  CKern* getKern() { return pkern; }
  const CKern* getKern() const { return pkern; }
  CSvgp& getSvgp() { return *psvgp; }
  const CSvgp& getSvgp() const { return *psvgp; }
  // Optional non-Gaussian likelihood (not owned); null => Gaussian beta.
  void setLikelihood(CNoise* noise);
  CNoise* getLikelihood() const;

  void display(std::ostream& os) const;
  void writeParamsToStream(std::ostream& os) const;
  void readParamsFromStream(std::istream& is);

  const CMatrix& getY() const { return Y; }
  const CMatrix& getMu() const { return mu; }
  const CMatrix& getLogS() const { return logS; }
  CMatrix getS() const; // exp(logS)
  void setMu(const CMatrix& muIn);
  void setLogS(const CMatrix& logSIn);
  void setS(const CMatrix& SIn);

  unsigned int getNumData() const { return nData; }
  unsigned int getLatentDim() const { return latentDim; }
  unsigned int getOutputDim() const { return outputDim; }
  unsigned int getNumInducing() const { return numInducing; }

  // KL(q(X) || N(0,I)).
  double klX() const;
  // Unbiased ELBO estimate on batch (includes -KL(q(X)) scaled).
  double elboEstimate(const std::vector<unsigned int>& batch);
  double elboEstimate();

  // CStochasticOptimisable — global params only.
  unsigned int numData() const override { return nData; }
  unsigned int getStochasticNumParams() const override;
  void getStochasticParams(CMatrix& param) const override;
  void setStochasticParams(const CMatrix& param) override;
  double objectiveGradBatch(const std::vector<unsigned int>& batch,
                            CMatrix& g) override;
  void afterBatchStep(const std::vector<unsigned int>& batch) override;

 private:
  void drawEps(CMatrix& eps, unsigned int nRows);
  void formSamples(const std::vector<unsigned int>& batch, CMatrix& XB,
                   std::vector<unsigned int>& yRows, CMatrix& epsOut);
  void mapLocalGrads(const std::vector<unsigned int>& batch, const CMatrix& gX,
                     const CMatrix& eps, CMatrix& gMu, CMatrix& gLogS) const;
  void addKlXGrad(const std::vector<unsigned int>& batch, double scale,
                  CMatrix& gMu, CMatrix& gLogS) const;
  void localAdamStep(const std::vector<unsigned int>& batch,
                     const CMatrix& gMu, const CMatrix& gLogS);

  CKern* pkern;
  bool ownKern;
  CMatrix Y;          // N x D
  CMatrix mu;         // N x Q
  CMatrix logS;       // N x Q
  CMatrix Z;          // M x Q (centres storage for point inducing)
  CSvgp* psvgp;       // owned; pX points at mu, py at Y
  int verbosity;

  unsigned int nData;
  unsigned int outputDim;
  unsigned int latentDim;
  unsigned int numInducing;

  unsigned int numSamples;
  bool antithetic;
  bool freezeNoise;
  CMatrix frozenEps;  // used when freezeNoise
  std::mt19937_64 rng;
  double natGradGamma;

  // Row-sparse Adam for (μ, logS).
  double localAlpha;
  double localBeta1;
  double localBeta2;
  double localEps;
  unsigned int localT;
  CMatrix mMu;
  CMatrix vMu;
  CMatrix mLogS;
  CMatrix vLogS;
};

#endif
