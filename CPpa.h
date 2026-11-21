/* Gaussian processes via probabilistic point assimilation (PPA).
   CIP-0023: rebased onto CMapModel + CProbabilisticOptimisable + CStreamInterface.
   Algorithm reference: lawrennd/ppa MATLAB (Nat King). */

#ifndef CPPA_H
#define CPPA_H

#include "CMltools.h"
#include "CNoise.h"
#include "CKern.h"
#include <cassert>

using namespace std;

const double NULOW=1e-16;
const string PPAVERSION="0.2";

class CPpa : public CMapModel, public CProbabilisticOptimisable, public CStreamInterface {
 public:
  CPpa(const CMatrix& inData, const CMatrix& targetData,
       CKern& kernel, CNoise& noiseModel, int verbos=2);
  CPpa(const CMatrix& trX, const CMatrix& trY,
       const CMatrix& mmat, const CMatrix& betamat, CKern& kernel,
       CNoise& noiseModel,
       int verbos=2);

#ifdef _NDLMATLAB
  CPpa(const CMatrix& inData,
       const CMatrix& targetData,
       CKern& kernel,
       CNoise& noiseModel,
       const string ppaInfoFile,
       const string ppaInfoVariable,
       int verbos=2);
#endif

  void init();
  void initStoreage();
  void initVals();

  void updateSite(int index);

  void eStep();
  void mStep();

  void updateExpectationf();
  void updateExpectationff();
  void updateExpectationsfBar();

  void test(const CMatrix& ytest, const CMatrix& Xin) const;
  void likelihoods(CMatrix& pout, CMatrix& yTest, const CMatrix& Xin) const;
  double logLikelihood() const;
  double logLikelihood(const CMatrix& yTest, const CMatrix& Xin) const;
  double logLikelihoodGradient(CMatrix& g) const;

  void out(CMatrix& yPred, const CMatrix& inData) const;
  void out(CMatrix& yPred, CMatrix& probPred, const CMatrix& inData) const;
  double outGradParams(CMatrix& g, const CMatrix& Xin, unsigned int pointNo, unsigned int outputNo) const;
  double outGradX(CMatrix& g, const CMatrix& Xin, unsigned int pointNo, unsigned int outputNo) const;

  // CIP-0031: latent = Var[fBar_*]; predictive adds 1/β (Gaussian site noise).
  void latentMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& X) const;
  void predictiveMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& X) const;
  [[deprecated("CIP-0031: use latentMeanVar (or predictiveMeanVar for y_*)")]]
  void posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& X) const;
  string getNoiseName() const
  {
    return noise.getName();
  }

  void updateCovGradient(int index) const;

  inline void setTerminate(bool val) { terminate = val; }
  inline bool isTerminate() const { return terminate; }

  void updateNuG();
  void updateK() const;
  void updateInvK(int index=0) const;
  double approxLogLikelihood() const;
  void approxLogLikelihoodGradient(CMatrix& g) const;
  // MATLAB ppaKernelLogLikelihood / gradient (M-step objective on <f>).
  double kernelLogLikelihood() const;
  void kernelLogLikelihoodGradient(CMatrix& g) const;

  void optimise(int maxIters=15, int kernIters=100, int noiseIters=100);
  bool equals(const CPpa& model, double tol=ndlutil::MATCHTOL) const;
  void display(ostream& os) const;

  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);

  inline unsigned int getOptNumParams() const
  {
    return static_cast<unsigned int>(kern.getNumParams());
  }
  void getOptParams(CMatrix& param) const
  {
    kern.getTransParams(param);
  }
  void setOptParams(const CMatrix& param)
  {
    kern.setTransParams(param);
  }

#ifdef _NDLMATLAB
  mxArray* toMxArray() const;
  void fromMxArray(const mxArray* matlabArray);
#endif

  const CMatrix& X;
  const CMatrix& y;

  int getNumTrainData() const { return numTrainData; }
  int getNumActiveData() const { return numTrainData; }
  double getBetaVal(int /*i*/, int j) const { return beta.getVal(0, j); }
  int getNumProcesses() const { return numTarget; }
  int getNumInputs() const { return activeX.getCols(); }
  double getTrainingX(int i, int j) const { return activeX.getVal(i, j); }
  int getTrainingPoint(int i) const { return i; }

  CMatrix nu;
  CMatrix g;
  CMatrix beta;
  mutable CMatrix Kstore;
  CMatrix wasM;
  CMatrix f;
  CMatrix ff;
  CMatrix fBar;
  CMatrix fBarfBar;
  CMatrix gamma;

  vector<CMatrix*> C;

  mutable CMatrix covGrad;
  mutable CMatrix invK;
  mutable double logDetK;
  mutable CMatrix K;

  mutable CMatrix s;
  mutable CMatrix a;
  mutable CMatrix ainv;

  CMatrix activeX;
  CMatrix trainY;

  CMatrix* M;
  CMatrix* L;
  CMatrix* Linv;

  CKern& kern;
  CNoise& noise;

 private:
  double logLike;
  double oldLogLike;
  double convergenceTol;

  bool terminate;
  bool varUpdate;
  bool loadedModel;
  mutable bool optimisingKernel;

  int numCovStruct;
  int numTrainData;

  int numTarget;
  int numData;
  int numIters;
  int kernOptIters;
};

void writePpaToStream(const CPpa& model, ostream& out);
void writePpaToFile(const CPpa& model, const string modelFileName, const string comment="");
CPpa* readPpaFromStream(istream& in);
CPpa* readPpaFromFile(const string modelfileName, int verbosity=2);

#endif
