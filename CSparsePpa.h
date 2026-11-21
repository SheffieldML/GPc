/* CIP-0026: PPA EM loop backed by CGp.
   Dense CPpa is untouched. Full-data recovery uses DTC with M=N, X_u=X
   because CGp::FTC does not add β^{-1}I to the Gram. Prefer no CWhiteKern
   on this path (white cross-covariance is zero); keep observation noise in β.
   See cip/cip0026/prototype_notes.md. */

#ifndef CSPARSEPPA_H
#define CSPARSEPPA_H

#include "CGp.h"
#include "CKern.h"
#include "CNoise.h"
#include <cassert>

using namespace std;

// Sparse stream payload (dense CPpa remains PPAVERSION "0.2").
const string SPARSE_PPAVERSION = "0.3";

struct CSparsePpaLoadedTag {};

class CSparsePpa {
 public:
  CSparsePpa(const CMatrix& inData, const CMatrix& targetData,
             CKern& kernel, CNoise& outerNoise,
             int approxType, unsigned int actSetSize,
             double betaVal = 1.0, int verbos = 0);
  // Stream load: skip init E-step; caller sets f / fBar / inducing after.
  CSparsePpa(const CMatrix& inData, const CMatrix& targetData,
             CKern& kernel, CNoise& outerNoise,
             int approxType, unsigned int actSetSize,
             double betaVal, int verbos, CSparsePpaLoadedTag);
  ~CSparsePpa();

  void init();
  void eStep();
  void mStep(int kernIters);
  void optimise(int maxIters, int kernIters);

  void setInducingToX();
  void fixInducing(bool val = true);
  void setSiteState(const CMatrix& fBarIn, const CMatrix& fIn);

  // Full q(u) PPA ELBO for DTC/DTCVAR (matches dense approxLogLikelihood at M=N).
  // FITC/PITC: interim CGp LML + outer noise LL.
  double ppaBound() const;
  double monitorLogLikelihood() const { return ppaBound(); }

  // CIP-0031: delegates to inner CGp latent/predictive.
  void latentMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const;
  void predictiveMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const;
  [[deprecated("CIP-0031: use predictiveMeanVar (or latentMeanVar for f_*)")]]
  void posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const;
  void out(CMatrix& yPred, const CMatrix& Xin) const;
  void display(ostream& os) const;

  CGp& gp() { return *pGp; }
  const CGp& gp() const { return *pGp; }
  const CNoise& getOuterNoise() const { return outerNoise; }

  const CMatrix& getF() const { return f; }
  const CMatrix& getFBar() const { return fBar; }
  const CMatrix& getG() const { return g; }
  double getBeta() const { return betaVal; }
  int getApproxType() const { return approxType; }
  unsigned int getNumActive() const { return actSetSize; }

  CMatrix X;
  CMatrix y;
  CMatrix f;
  CMatrix fBar;
  // ⟨f⟩ used for the last fBar / q(u) (dense keeps fBarfBar from that step).
  CMatrix fPost;
  CMatrix g;
  CMatrix nu;
  CMatrix gamma;
  CMatrix ff;

 private:
  void constructGp(bool runInitEstep);
  void syncGpTargets();
  void updateNuG();

  CKern& kern;
  CNoise& outerNoise;
  CGaussianNoise* pInnerNoise;
  CGp* pGp;

  double betaVal;
  int approxType;
  unsigned int actSetSize;
  int verbosity;
  int numData;
  int numTarget;
};

void writeSparsePpaToStream(const CSparsePpa& model, ostream& out);
void writeSparsePpaToFile(const CSparsePpa& model, const string& modelFileName,
                          const string& comment = "");
// Heap-owns kern/noise/X/y so returned model refs stay valid.
CSparsePpa* readSparsePpaFromStream(istream& in, int verbosity = 0);
CSparsePpa* readSparsePpaFromFile(const string& modelFileName, int verbosity = 0);

#endif
