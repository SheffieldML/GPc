/* Bayesian GP-LVM with diagonal q(X) and psi statistics — CIP-0017 Phase 2. */
#ifndef CBAYESGPLVM_H
#define CBAYESGPLVM_H

#include <vector>
#include <algorithm>
#include "CKern.h"
#include "COptimisable.h"
#include "CTransform.h"

const string BAYESGPLVMVERSION = "0.1";

class CBayesGplvm : public CProbabilisticOptimisable, public CStreamInterface
{
public:
  // Owns a copy of Y. Borrows kernel unless readParamsFromStream replaces it
  // (then the model owns the streamed kernel).
  CBayesGplvm(CKern* kernel, const CMatrix& Y, unsigned int latDim,
              unsigned int numInducing, int verbosity = 2);
  virtual ~CBayesGplvm();

  CKern* getKern() const { return pkern; }
  // When true, destructor deletes pkern (default false for borrowed kernels).
  void setOwnKern(bool val) { ownKern = val; }
  bool isOwnKern() const { return ownKern; }

  void initStoreage();
  void initVals();
  void initXpca();
  void initZFromMu();

  double logLikelihood() const;
  double logLikelihoodGradient(CMatrix& g) const;

  // VarDTC bound without KL (CIP-0018 multi-view assembly).
  double varDtcBound() const;
  // Untransformed grads for Z/mu/S; unconstrained (transform-scaled) grads for kern and beta.
  // (Not Fisher-preconditioned natural gradients — see CIP-002A / CSvgp.)
  // If includeKl, subtracts KL grads from gMu/gS. Returns varDtcBound() [- KL if includeKl].
  double computeVarDtcGrads(CMatrix& gKern, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                            double& gBeta, bool includeKl) const;

  // Copy shared latents into this view (MRD).
  void setLatents(const CMatrix& muIn, const CMatrix& SIn, const CMatrix& ZIn);

  // CIP-0031: latent = p(f_*); predictive adds 1/β.
  void latentMeanVar(CMatrix& means, CMatrix& vars, const CMatrix& Xstar) const;
  void predictiveMeanVar(CMatrix& means, CMatrix& vars, const CMatrix& Xstar) const;
  // Deprecated alias: same as predictiveMeanVar (pre-0031 added 1/β).
  void posteriorMeanVar(CMatrix& means, CMatrix& vars, const CMatrix& Xstar) const;

  unsigned int getOptNumParams() const;
  void getOptParams(CMatrix& param) const;
  void setOptParams(const CMatrix& param);

  double klDivergence() const;
  void klGradient(CMatrix& gMu, CMatrix& gS) const;

  void optimise(unsigned int iters = 100);
  void display(ostream& os) const;

  void writeParamsToStream(ostream& os) const;
  void readParamsFromStream(istream& is);

  unsigned int getNumData() const { return numData; }
  unsigned int getOutputDim() const { return outputDim; }
  unsigned int getLatentDim() const { return latentDim; }
  unsigned int getNumActive() const { return numActive; }
  double getBeta() const { return beta; }
  void setBeta(double val) { beta = val; setKupToDate(false); }

  const CMatrix& getY() const { return Y; }
  const CMatrix& getMu() const { return mu; }
  const CMatrix& getS() const { return S; }
  const CMatrix& getZ() const { return Z; }
  CMatrix& getMu() { return mu; }
  CMatrix& getS() { return S; }
  CMatrix& getZ() { return Z; }

  void setLabels(const vector<int>& labs);
  bool isLabels() const { return labelsPresent; }
  int getLabel(unsigned int index) const;
  int getMaxLabelVal() const;
  int getMinLabelVal() const;

  void setKupToDate(bool val) const { KupToDate = val; }
  bool isKupToDate() const { return KupToDate; }

private:
  void updateCaches() const;

  CKern* pkern;
  bool ownKern;
  CMatrix Y;    // N x D
  CMatrix mu;   // N x q
  CMatrix S;    // N x q (variances)
  CMatrix Z;    // M x q
  double beta;

  unsigned int numData;
  unsigned int outputDim;
  unsigned int latentDim;
  unsigned int numActive;

  bool labelsPresent;
  vector<int> labels;

  CTransform* betaTransform;
  CTransform* sTransform;

  // Working caches (mutable).
  mutable bool KupToDate;
  mutable CMatrix Kmm;
  mutable CMatrix LcholKmm;
  mutable CMatrix Psi1;
  mutable CMatrix Psi2;
  mutable double psi0Sum;
  mutable CMatrix A;
  mutable CMatrix B;
  mutable CMatrix LcholB;
  mutable CMatrix LmiPsi1;       // M x N
  mutable CMatrix LBiLmiPsi1;    // M x N
  mutable CMatrix LBiLmiPsi1Vf;  // M x D
  mutable double dataFit;
  mutable double logDetB;
  mutable CMatrix dL_dKmm;
  mutable CMatrix dL_dpsi0; // N x 1
  mutable CMatrix dL_dpsi1; // N x M
  mutable CMatrix dL_dpsi2; // M x M
  mutable double dL_dbeta;

  static const double JITTER;
};

#endif
