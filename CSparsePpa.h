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

class CSparsePpa {
 public:
  // Owns copies of X/y and an inner CGaussianNoise + CGp. Outer noise and
  // kernel are non-owning (same pattern as CPpa references).
  CSparsePpa(const CMatrix& inData, const CMatrix& targetData,
             CKern& kernel, CNoise& outerNoise,
             int approxType, unsigned int actSetSize,
             double betaVal = 1.0, int verbos = 0);
  ~CSparsePpa();

  void init();
  void eStep();
  void mStep(int kernIters);
  void optimise(int maxIters, int kernIters);

  void setInducingToX();
  void fixInducing(bool val = true);

  // Phase C interim monitor: CGp LML on ⟨f⟩ (not dense approxLogLikelihood).
  double monitorLogLikelihood() const;
  void posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const;
  void out(CMatrix& yPred, const CMatrix& Xin) const;
  void display(ostream& os) const;

  CGp& gp() { return *pGp; }
  const CGp& gp() const { return *pGp; }

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
  CMatrix g;
  CMatrix nu;
  CMatrix gamma;
  CMatrix ff;

 private:
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

#endif
