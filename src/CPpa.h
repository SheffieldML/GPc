#ifndef CPPA_H
#define CPPA_H
#include "CDataModel.h"
using namespace std;

const double NULOW=1e-16;
const string PPAVERSION="0.1";

class CPpa : public COptimisableKernelModel {
 public:
  // Constructor given a filename.
  CPpa(const string modelFileName, const int verbos=2);
  // Constructor given a kernel and a noise model.
  CPpa(const CMatrix& inData, const CMatrix& targetData, 
       CKern& kernel, CNoise& noiseModel, const int verbos=2);
  CPpa(const CMatrix& trX, const CMatrix& trY, 
       const CMatrix& mmat, const CMatrix& betamat, CKern& kernel, 
       CNoise& noiseModel, 
       const int verbos=2);

#ifdef _NDLMATLAB
  // Constructor using file containing ppaInfo.
  CPpa(const CMatrix& inData, 
       const CMatrix& targetData, 
       CKern& kernel, 
       CNoise& noiseModel, 
       const string ppaInfoFile, 
       const string ppaInfoVariable, 
       const int verbos=2);
#endif
  // Initialise the storeage for the model.
  void initStoreage();
  // Set the initial values for the model.
  void initVals();

  // update the site parameters at index.
  void updateSite(const int index); 

  // Run the expectation step in the E-M algorithm.
  void eStep();
  // Run the maximisation step in the E-M algorithm.
  void mStep();

  // Update expectations of f.
  void updateExpectationf();
  // Update expectations of f^2
  void updateExpectationff();
  // Update expectations of fBar and fBarfBar
  void updateExpectationsfBar();

  void test(const CMatrix& ytest, const CMatrix& Xin) const;
  void likelihoods(CMatrix& pout, CMatrix& yTest, const CMatrix& Xin) const;
  // log likelihood of training set.
  double logLikelihood() const;
  // log likelihood of test set.
  double logLikelihood(const CMatrix& yTest, const CMatrix& Xin) const;
  void out(CMatrix& yPred, const CMatrix& inData) const;
  void out(CMatrix& yPred, CMatrix& probPred, const CMatrix& inData) const;
  void posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& X) const;
  string getNoiseName() const
    {
      return noise.getNoiseName();
    }

  // Gradient routines
  void updateCovGradient(int index) const;
  

  inline void setTerminate(const bool val)
    {
      terminate = val;
    }
  inline bool isTerminate() const
    {
      return terminate;
    }
  void updateNuG();
  // update K with the kernel computed from the training points.
  void updateK() const;
  // update invK with the inverse of the kernel plus beta terms computed from the training points.
  void updateInvK(const int index=0) const;
  // compute the approximation to the log likelihood.
  double approxLogLikelihood() const;
  // compute the gradients of the approximation wrt parameters.
  void approxLogLikelihoodGradient(CMatrix& g) const;
  
  void optimise(const int maxIters=15, const int kernIters=100, const int noiseIters=100);
  bool equals(const CPpa& model, const double tol=ndlutil::MATCHTOL) const;
  void display(ostream& os) const;

  inline int getOptNumParams() const
    {
      return kern.getNumParams();
    }    
  void getOptParams(CMatrix& param) const
    {
      kern.getTransParams(param);
    }
  void setOptParams(const CMatrix& param)
    {
      kern.setTransParams(param);
    }
  string getType() const
    {
      return type;
    }
  void setType(const string name)
    {
      type = name;
    }
  void computeObjectiveGradParams(CMatrix& g) const
    {
      approxLogLikelihoodGradient(g);
      g.negate();
    }
  double computeObjectiveVal() const
    {
      return -approxLogLikelihood();
    }
#ifdef _NDLMATLAB
  mxArray* toMxArray() const;
  void fromMxArray(const mxArray* matlabArray);
#endif
  const CMatrix& X;

  int getNumTrainData() const
    {
      return numTrainData;
    }
  int getNumActiveData() const
    {
      return numTrainData;
    }
  double getBetaVal(const int i, const int j) const
    {
      return beta.getVal(1, j);
    }
  int getNumProcesses() const
    {
      return numTarget;
    }
  int getNumInputs() const
    {
      return activeX.getCols();
    }
  double getTrainingX(const int i, const int j) const
    {
      return activeX.getVal(i, j);
    }
  int getTrainingPoint(const int i) const
    {
      return i;
    }
  // arguably these are noise model associated.
  const CMatrix& y;
  
  CMatrix nu;
  CMatrix g;

  CMatrix beta;
  
  CMatrix Kstore;
  CMatrix wasM;
  CMatrix f;
  CMatrix ff;
  CMatrix fBar;
  CMatrix gamma;
  // Covariance of q distribution over fbar.
  vector<CMatrix*> C;
  
  // these really just provide local storage
  mutable CMatrix covGrad;
  mutable CMatrix invK;
  mutable double logDetK;
  mutable CMatrix K;

  mutable CMatrix s;
  mutable CMatrix a;
  mutable CMatrix ainv;


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

  int numCovStruct;
  int numTrainData;

  int numTarget;
  int numData;
  int numIters;
    
  string type;
};

// Functions which operate on the object
void writePpaToStream(const CPpa& model, ostream& out);
void writePpaToFile(const CPpa& model, const string modelFileName, const string comment="");
CPpa* readPpaFromStream(istream& in);
CPpa* readPpaFromFile(const string modelfileName, const int verbosity=2);

#endif
