/* CIP-0035 Phase 1: deep GP by doubly stochastic VI on CSvgp layer cores.

   Stack of L SVGP layers with fixed mean maps P_ℓ. Hidden layers use
   Cholesky-mode q(v) and marginalsAt / marginalsGradAt; the final layer
   reuses elboGradAt. Training is minibatch Adam via CStochasticOptimisable. */

#ifndef CDEEPGP_H
#define CDEEPGP_H

#include "CSvgp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CStochasticOptimisable.h"
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

class CDeepGp : public CStochasticOptimisable {
 public:
  // Owns copies of X, Y. Borrows kernels (one per layer). Inducing points:
  // random subset of X pushed through mean maps. Hidden widths = dims[1..L-1];
  // dims[0] must equal X.cols, dims[L] must equal Y.cols. chi scales hidden
  // kernel variance as chi/(L-1). All layers start in Cholesky q mode.
  CDeepGp(const CMatrix& Xin, const CMatrix& Yin,
          const std::vector<CKern*>& kerns,
          const std::vector<unsigned int>& dims,
          const std::vector<unsigned int>& numInducing,
          double beta = 1.0, double chi = 1.0, int verbosity = 0);
  virtual ~CDeepGp();

  unsigned int getNumLayers() const { return numLayers; }
  unsigned int getNumData() const { return nData; }
  unsigned int getInputDim() const { return inputDim; }
  unsigned int getOutputDim() const { return outputDim; }
  unsigned int getHiddenDim(unsigned int ell) const; // D_ell
  CSvgp& layer(unsigned int ell);
  const CSvgp& layer(unsigned int ell) const;
  const CMatrix& meanMap(unsigned int ell) const { return meanP[ell]; }

  void setNumSamples(unsigned int K);
  unsigned int getNumSamples() const { return numSamples; }
  void setSeed(uint64_t s);
  void setFreezeNoise(bool val) { freezeNoise = val; }
  bool isFreezeNoise() const { return freezeNoise; }
  // epsRows = |batch| * K * sum_{ℓ=0}^{L-2} D_{ℓ+1}  (hidden-layer noise only)
  void setFrozenEps(const CMatrix& eps);

  void setNatGradGamma(double g) { natGradGamma = g; }
  double getNatGradGamma() const { return natGradGamma; }

  // Unbiased ELBO estimate on a batch (K samples).
  double elboEstimate(const std::vector<unsigned int>& batch);
  double elboEstimate();

  // CStochasticOptimisable
  unsigned int numData() const override { return nData; }
  unsigned int getStochasticNumParams() const override;
  void getStochasticParams(CMatrix& param) const override;
  void setStochasticParams(const CMatrix& param) override;
  double objectiveGradBatch(const std::vector<unsigned int>& batch,
                            CMatrix& g) override;
  void afterBatchStep(const std::vector<unsigned int>& batch) override;

 private:
  void initMeanMaps();
  void initInducing();
  void initKernVariances(double chi);
  void drawEps(CMatrix& eps, unsigned int nRows);
  void packLayerOffsets();

  CMatrix X;
  CMatrix Y;
  unsigned int nData;
  unsigned int inputDim;
  unsigned int outputDim;
  unsigned int numLayers;
  int verbosity;

  std::vector<CKern*> pkerns;          // not owned
  std::vector<unsigned int> layerDims; // L+1 sizes: D_0..D_L
  std::vector<unsigned int> M;         // inducing per layer
  std::vector<std::unique_ptr<CSvgp> > layers;
  std::vector<CMatrix> meanP;          // P_ℓ : D_{ℓ-1} x D_ℓ (index 0 unused)
  std::vector<CMatrix> Zstore;         // centres storage owned per layer
  std::vector<CMatrix> hiddenY;        // dummy targets for hidden CSvgp cores

  unsigned int numSamples;
  uint64_t seed;
  std::mt19937_64 rng;
  bool freezeNoise;
  CMatrix frozenEps;
  double natGradGamma;

  // Stochastic param layout: concat of each layer's getStochasticNumParams().
  std::vector<unsigned int> layerParamOffset; // size L+1; last = total
};

#endif
