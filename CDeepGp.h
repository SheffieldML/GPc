/* CIP-0035: deep GP by doubly stochastic VI on CSvgp layer cores.

   Stack of L SVGP layers with fixed mean maps P_ℓ. Hidden layers use
   Cholesky-mode q(v) and marginalsAt / marginalsGradAt; the final layer
   reuses elboGradAt. Training is minibatch Adam via CStochasticOptimisable.
   Phase 2: per-layer inducing family. Phase 3: non-Gaussian final likelihood.
   Phase 4: stream I/O, residual-off, Jacobian diagnostic. */

#ifndef CDEEPGP_H
#define CDEEPGP_H

#include "CSvgp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CNdlInterfaces.h"
#include "CStochasticOptimisable.h"
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

const std::string DEEPGPVERSION = "0.1";

class CDeepGp : public CStochasticOptimisable, public CStreamInterface {
 public:
  // Owns copies of X, Y. Borrows kernels (one per layer) unless stream load
  // replaces them (then owned). Inducing points: random subset of X pushed
  // through mean maps. Hidden widths = dims[1..L-1]; dims[0]/dims[L] match
  // X/Y columns. chi scales hidden kernel variance as chi/(L-1).
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
  double getChi() const { return chiScale; }
  CSvgp& layer(unsigned int ell);
  const CSvgp& layer(unsigned int ell) const;
  const CMatrix& meanMap(unsigned int ell) const { return meanP[ell]; }
  const CMatrix& getX() const { return X; }
  const CMatrix& getY() const { return Y; }

  void setNumSamples(unsigned int K);
  unsigned int getNumSamples() const { return numSamples; }
  void setSeed(uint64_t s);
  void setFreezeNoise(bool val) { freezeNoise = val; }
  bool isFreezeNoise() const { return freezeNoise; }
  // epsRows = |batch| * K * sum_{ℓ=0}^{L-2} D_{ℓ+1}  (hidden-layer noise only)
  void setFrozenEps(const CMatrix& eps);

  void setNatGradGamma(double g) { natGradGamma = g; }
  double getNatGradGamma() const { return natGradGamma; }

  // Per-layer inducing family (default "points"). "weightspace" uses
  // makeRidgeInit (CIP-0037). Rebuilds stochastic param offsets.
  void setLayerInducingType(unsigned int ell, const std::string& typeName);
  std::string getLayerInducingType(unsigned int ell) const;
  // Enable learnable ridge window width(s); sigmoid-mapped into [0.05, 1].
  void setLayerLearnWindowWidth(unsigned int ell, bool learn,
                                bool untied = false);

  // Phase 3: non-Gaussian final-layer likelihood (borrowed; not owned).
  void setLikelihood(CNoise* noise);
  CNoise* getLikelihood() const;

  // Phase 4: drop Nyström residual on every layer (finite-network bound).
  void setDropResidual(bool val);
  bool isDropResidual() const { return dropResidual; }

  // Phase 4 V7: prior Jacobian log-singular-value spread (max−min). Each probe
  // draws whitened v~N(0,I) per layer and finite-differences the map
  // h ↦ m(h)+A(h)^T v (residual off). Independent of the trained q(v).
  // Requires square Din==Dout. Returns median over nProbe inputs.
  double jacobianSpread(unsigned int nProbe = 32, uint64_t probeSeed = 0) const;

  // Predictive mean/variance at Xstar (N* x Din).
  // samples==0: mean path (hidden means only; final-layer marginal + optional
  //   Gaussian observation noise). Deterministic.
  // samples>=1: S-sample mixture through hidden layers (CIP-0035); mixture
  //   moments of the final-layer Gaussians, plus 1/β when addNoise.
  void predictMeanVar(const CMatrix& Xstar, CMatrix& mu, CMatrix& var,
                      unsigned int samples = 0, uint64_t sampleSeed = 0,
                      bool addNoise = true) const;

  // Unbiased ELBO estimate on a batch (K samples).
  double elboEstimate(const std::vector<unsigned int>& batch);
  double elboEstimate();

  void display(std::ostream& os) const;

  // CStreamInterface (type deepgp).
  void writeParamsToStream(std::ostream& os) const override;
  void readParamsFromStream(std::istream& is) override;

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
  void rebuildLayers(double beta);
  void forwardSample(const CMatrix& Xin, const CMatrix& epsFlat,
                     CMatrix& Hout) const;
  // Prior probe: h_ℓ = m_ℓ(h)+A_ℓ(h)^T v_ℓ (no residual, ignores q).
  void forwardPriorDraw(const CMatrix& Xin,
                        const std::vector<CMatrix>& vs,
                        CMatrix& Hout) const;

  CMatrix X;
  CMatrix Y;
  unsigned int nData;
  unsigned int inputDim;
  unsigned int outputDim;
  unsigned int numLayers;
  int verbosity;
  double chiScale;
  double betaStore;

  std::vector<CKern*> pkerns;          // borrowed unless ownKerns
  std::vector<std::unique_ptr<CKern> > ownedKerns;
  bool ownKerns;
  std::vector<unsigned int> layerDims; // L+1 sizes: D_0..D_L
  std::vector<unsigned int> M;         // inducing per layer
  std::vector<std::unique_ptr<CSvgp> > layers;
  std::vector<CMatrix> meanP;          // P_ℓ : D_{ℓ-1} x D_ℓ
  std::vector<CMatrix> Zstore;         // centres storage owned per layer
  std::vector<CMatrix> hiddenY;        // dummy targets for hidden CSvgp cores
  std::vector<CMatrix> layerX;         // per-layer inputs (X pushed through mean maps), D_{ℓ-1} cols
  std::vector<std::string> inducingTypes;

  unsigned int numSamples;
  uint64_t seed;
  std::mt19937_64 rng;
  bool freezeNoise;
  CMatrix frozenEps;
  double natGradGamma;
  bool dropResidual;

  // Stochastic param layout: concat of each layer's getStochasticNumParams().
  std::vector<unsigned int> layerParamOffset; // size L+1; last = total
};

#endif
