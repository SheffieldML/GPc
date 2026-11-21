/* Manifold Relevance Determination — CIP-0018. Shared q(X),Z; per-view kernels. */
#ifndef CMRD_H
#define CMRD_H

#include <vector>
#include <string>
#include "CBayesGplvm.h"

const string MRDVERSION = "0.1";

class CMrd : public CProbabilisticOptimisable, public CStreamInterface
{
public:
  // Takes ownership of kernels in kerns (deleted on destruction).
  // Ys must share N; D_v may differ. Requires V >= 2.
  CMrd(const vector<CMatrix>& Ys, const vector<CKern*>& kerns,
       unsigned int latDim, unsigned int numInducing, int verbosity = 2);
  virtual ~CMrd();

  double logLikelihood() const;
  double logLikelihoodGradient(CMatrix& g) const;

  unsigned int getOptNumParams() const;
  void getOptParams(CMatrix& param) const;
  void setOptParams(const CMatrix& param);

  void optimise(unsigned int iters = 100);
  void display(ostream& os) const;

  void writeParamsToStream(ostream& os) const;
  void readParamsFromStream(istream& is);

  unsigned int getNumViews() const { return (unsigned int)views.size(); }
  unsigned int getNumData() const { return numData; }
  unsigned int getLatentDim() const { return latentDim; }
  unsigned int getNumActive() const { return numActive; }

  const CMatrix& getMu() const { return mu; }
  const CMatrix& getS() const { return S; }
  const CMatrix& getZ() const { return Z; }
  CMatrix& getMu() { return mu; }
  CMatrix& getS() { return S; }
  CMatrix& getZ() { return Z; }

  CBayesGplvm* getView(unsigned int v);
  const CBayesGplvm* getView(unsigned int v) const;

  // Per-view ARD input scales (length q). Throws if view kernel is not RBF-ARD
  // (or a compound whose first component is RBF-ARD).
  void getInputScales(unsigned int view, vector<double>& scales) const;

  // Soft shared/private factorisation (GPy factorize_space spirit, V=2).
  // Scales normalised to max 1; dim is shared if both > threshold.
  void factorizeSpace(vector<unsigned int>& shared, vector<unsigned int>& private0,
                      vector<unsigned int>& private1, double threshold = 0.005) const;

  // CIP-0031: per-view latent / predictive (delegates to CBayesGplvm).
  void latentMeanVar(unsigned int view, CMatrix& means, CMatrix& vars,
                     const CMatrix& Xstar) const;
  void predictiveMeanVar(unsigned int view, CMatrix& means, CMatrix& vars,
                         const CMatrix& Xstar) const;
  [[deprecated("CIP-0031: use predictiveMeanVar (or latentMeanVar for f_*)")]]
  void posteriorMeanVar(unsigned int view, CMatrix& means, CMatrix& vars,
                        const CMatrix& Xstar) const;

private:
  void initStoreage();
  void initVals();
  void initXpcaConcat();
  void initZFromMu();
  void syncViews() const;
  const CRbfardKern* findRbfard(const CKern* k) const;

  unsigned int numData;
  unsigned int latentDim;
  unsigned int numActive;
  CMatrix mu;
  CMatrix S;
  CMatrix Z;
  CTransform* sTransform;

  vector<CBayesGplvm*> views;
  vector<bool> ownKern; // per-view: whether we own the kernel passed in
};

#endif
