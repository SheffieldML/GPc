#ifndef CGP_H
#define CGP_H
#include "CPosteriorTarget.h"
#include "CMltools.h"
#include "CInducing.h"
#include "CEpInference.h"
#include "CLaplaceInference.h"
#include "CModelDescription.h"
#include <vector>

using namespace std;

const string GPVERSION="0.1";

class CGp : public CMapModel, public CProbabilisticOptimisable, public CPosteriorTarget, public CStreamInterface, public CMatInterface
{
public:
  enum 
  {
    FTC, 
    DTC, 
    FITC, 
    PITC,
    DTCVAR
  };

  // Sparse factorisation schedule ([CIP-000E](cip/cip000E.md) / [CIP-0012](cip/cip0012.md) / [CIP-001B](cip/cip001B.md)).
  enum SparseLinalgMode
  {
    SparseLinalgLegacy = 0,
    SparseLinalgStableFactor = 1,
    SparseLinalgLQ = 2
  };

  CGp();

  // Constructor given a kernel
  //  CGp(CKern& kernel, CScaleNoise& nois, CMatrix& Xin, int verbos=2);
  // Constructor given a kernel and sparse approximation
  CGp(CKern* kernel, CNoise* nois, CMatrix* Xin, int approxType=FTC, unsigned int actSetSize=0, int verbos=2);

  CGp(unsigned int q, unsigned int d, CMatrix* Xin, CMatrix* yin, CKern* kernel, CNoise* nois, int approxType=FTC, unsigned int actSetSize=0, int verbos=2);

#ifdef _NDLMATLAB
  // Constructor using file containing gpInfo.
  CGp(CMatrix* inData, CMatrix* targetData, CKern* kernel, CNoise* noiseModel, const string gpInfoFile, const string gpInfoVariable, int verbos=2);
#endif

  // initialise storeage associated with optimizing inputs.
  void initOptimiseXStoreage();
  // initialise storeage associated with active set size.
  void initSparseStoreage();
  // Initialise storeage for anything dependent on number of kernel parameters.
  void initKernStoreage();
  // Initialise the storeage for the model.
  void initStoreage();
  // Set the initial values for the model.
  void initVals();
  
  // For MapModel interface.
  void out(CMatrix& yPred, const CMatrix& inData) const;
  void out(CMatrix& yPred, CMatrix& probPred, const CMatrix& inData) const;
  double outGradParams(CMatrix& g, const CMatrix& Xin, unsigned int pointNo, unsigned int outputNo) const;
  double outGradX(CMatrix& g, const CMatrix& Xin, unsigned int pointNo, unsigned int outputNo) const;

  // update alpha representation.
  void updateAlpha() const;
  // CIP-0031: latent = p(f_*); predictive = p(y_*) for Gaussian (adds obs noise once).
  void latentMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& X) const;
  void predictiveMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& X) const;
  [[deprecated("CIP-0031: use predictiveMeanVar (or latentMeanVar for f_*)")]]
  void posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& X) const;
  void posteriorMean(CMatrix& mu, const CMatrix& X) const;
  // Gradient routines
  void updateCovGradient(unsigned int index, CMatrix &work_invK_Y) const;


  virtual void updateX();
  void updateM() const;
  // update K and dynK and all derived quantities if they are dirty.
  void updateK() const;
  // Update A and D representations
  void updateAD() const;
  // update the gradient matrices.
  void updateG() const;

  // Explicit cache protocol ([CIP-0009](cip/cip0009.md)): mutators call
  // invalidateCache(); const evaluators call ensure* before reading factors.
  // A single CGp instance is single-threaded.
  void invalidateCache();
  void ensureMUpToDate() const { updateM(); }
  void ensureKUpToDate() const { updateK(); }
  void ensureADUpToDate() const { updateAD(); }
  void ensureAlphaUpToDate() const { updateAlpha(); }

  // CIP-001B: SparseLinalgLQ (LQ of [I,A]) is the default.
  // CIP-0012 StableFactor remains available (--sparse-linalg stable).
  // (Seeger et al., arXiv:1710.08717 §3.3 BLR / §4.3 adjoint).
  // Legacy remains for dual-path archaeology via --sparse-linalg legacy.
  // Not persisted in .model streams.
  bool isStableSparseLinalg() const
  {
    return sparseLinalgMode == SparseLinalgStableFactor;
  }
  bool isLQSparseLinalg() const
  {
    return sparseLinalgMode == SparseLinalgLQ;
  }
  // Whitened B / C path shared by StableFactor and LQ (vs Legacy pdinv).
  bool isWhitenedSparseLinalg() const
  {
    return sparseLinalgMode == SparseLinalgStableFactor
      || sparseLinalgMode == SparseLinalgLQ;
  }
  void setStableSparseLinalg(bool val)
  {
    const int mode = val ? SparseLinalgStableFactor : SparseLinalgLegacy;
    if(mode != sparseLinalgMode)
    {
      sparseLinalgMode = mode;
      setADupToDate(false);
    }
  }
  void setSparseLinalgMode(int mode)
  {
    if(mode != SparseLinalgLegacy
       && mode != SparseLinalgStableFactor
       && mode != SparseLinalgLQ)
      throw ndlexceptions::Error("Unknown sparseLinalgMode");
    if(mode != sparseLinalgMode)
    {
      sparseLinalgMode = mode;
      setADupToDate(false);
    }
  }
  int getSparseLinalgMode() const { return sparseLinalgMode; }

  // compute the approximation to the log likelihood.
  virtual double logLikelihood() const;
  // compute the gradients of the approximation wrt parameters.
  virtual double logLikelihoodGradient(CMatrix& g) const;

  // CIP-0033: GPy-style EP / Laplace on FTC (Bernoulli–probit). Prior K =
  // kern(X,X); no likelihood.variance on the Gram. Requires FTC and outputDim==1.
  // EP and Laplace are mutually exclusive.
  void setEpInference(bool on);
  bool isEpInference() const { return epInference; }
  void setLaplaceInference(bool on);
  bool isLaplaceInference() const { return laplaceInference; }
  // Posterior latent mean / var after last EP evaluation (empty if unused).
  const CMatrix& epPosteriorMean() const { return epMu; }
  const CMatrix& epPosteriorVarDiag() const { return epVarDiag; }
  const CMatrix& laplaceMode() const { return laplaceFhat; }
  // CIP-0033 Phase D / CIP-0036: latent predictive at Xin (EP or Laplace).
  void epLatentMeanVar(CMatrix& mu, CMatrix& varDiag, const CMatrix& Xin) const;
  void laplaceLatentMeanVar(CMatrix& mu, CMatrix& varDiag, const CMatrix& Xin) const;
  // CIP-0026: -KL(q(u)||p(u)) - ½β tr(Cov_f̄) [+ DTCVAR residual] + kern prior.
  // Matches dense CPpa approxLogLikelihood parts 1–4 when DTC/DTCVAR and M=N, Xu=X.
  // Throws for FITC/PITC (different A; use interim CGp LML + noise monitor).
  double sparsePpaElboGpTerms() const;
  void gpCovGrads() const;
  virtual void pointLogLikelihood(const CMatrix& y, const CMatrix& X) const;
  void optimise(unsigned int iters=1000);
  bool equals(const CGp& model, double tol=ndlutil::MATCHTOL) const;
  ModelDescription describe() const;
  void display(ostream& os) const;
  virtual std::string progressSnapshot(const OptimiserEvent& event) const;
  
  virtual unsigned int getOptNumParams() const;
  virtual void getOptParams(CMatrix& param) const;
  virtual void setOptParams(const CMatrix& param);

  virtual string getNoiseType() const;

#ifdef _NDLMATLAB
  mxArray* toMxArray() const;
  void fromMxArray(const mxArray* matlabArray);
#endif

  void readParamsFromStream(istream& in);
  void writeParamsToStream(ostream& out) const;

  inline unsigned int getNumActive() const 
  {
    return numActive;
  }
  void setNumActive(unsigned int val) 
  {
    numActive = val;
  }
  
  void setTargetVals(CMatrix& yvals) 
  {
    DIMENSIONMATCH(yvals.getCols()==getOutputDim());
    DIMENSIONMATCH(yvals.getRows()==getNumData());
    py=&yvals;
    pnoise->setTarget(py);
  }
  void setTargetVals(CMatrix* pyvals) 
  {
    DIMENSIONMATCH(pyvals->getCols()==getOutputDim());
    DIMENSIONMATCH(pyvals->getRows()==getNumData());
    py=pyvals;
    pnoise->setTarget(py);
  }
  void setInputVals(CMatrix& Xvals) 
  {
    DIMENSIONMATCH(Xvals.getCols()==getInputDim());
    DIMENSIONMATCH(Xvals.getRows()==getNumData());
    pX=&Xvals;
  }
  void setInputVals(CMatrix* pXvals) 
  {
    DIMENSIONMATCH(pXvals->getCols()==getInputDim());
    DIMENSIONMATCH(pXvals->getRows()==getNumData());
    pX=pXvals;
  }
  void setInducingVals(CMatrix& Xvals) 
  {
    SANITYCHECK(isSparseApproximation());
    DIMENSIONMATCH(Xvals.getCols()==getInputDim());
    DIMENSIONMATCH(Xvals.getRows()==numActive);
    X_u.deepCopy(Xvals);
  }
  // Flag which indicates whether scales are to be learnt.
  bool isOutputScaleLearnt() const 
  {
    return outputScaleLearnt;
  }
  void setOutputScaleLearnt(const bool val) {
    outputScaleLearnt=val;
    if(outputScaleLearnt)
      scaleParamFixed.assign(getOutputDim(), false);
    else
      scaleParamFixed.clear();
  }
  bool isOutputBiasLearnt() const 
  {
    return outputBiasLearnt;
  }
  void setOutputBiasLearnt(const bool val) {
    outputBiasLearnt=val;
  }
  double getScaleVal(unsigned int index) const {
    BOUNDCHECK(index<getOutputDim());
    return scale.getVal(0, index);
  }
  void setScaleVal(double val, unsigned int index) {
    BOUNDCHECK(index<getOutputDim());
    scale.setVal(val, 0, index);
    setMupToDate(false);
  }
  void setScale(const CMatrix& scal) {
    DIMENSIONMATCH(scal.getRows()==1);
    DIMENSIONMATCH(scal.getCols()==getOutputDim());
    scale.deepCopy(scal);
    setMupToDate(false);
  } 
  void setActiveSet(const CMatrix& Xu) {
    DIMENSIONMATCH(Xu.getRows()==numActive);
    DIMENSIONMATCH(Xu.getCols()==getInputDim());
    X_u.deepCopy(Xu);
    setKupToDate(false);
  }
  double getBiasVal(unsigned int index) const {
    BOUNDCHECK(index<getOutputDim());
    return bias.getVal(0, index);
  }
  void setBiasVal(double val, unsigned int index) {
    BOUNDCHECK(index<getOutputDim());
    bias.setVal(val, 0, index);
    setMupToDate(false);
  }
  void setBias(const CMatrix& bia) {
    DIMENSIONMATCH(bia.getRows()==1);
    DIMENSIONMATCH(bia.getCols()==getOutputDim());
    bias.deepCopy(bia);
    setMupToDate(false);
  }
  double getBetaVal(unsigned int i=0, unsigned int j=0) const {
    BOUNDCHECK(i<beta.getRows());
    BOUNDCHECK(j<beta.getCols());
    return beta.getVal(i, j);
  }
  void setBetaVal(double val, unsigned int i=0, unsigned int j=0) {
    BOUNDCHECK(i<beta.getRows());
    BOUNDCHECK(j<beta.getCols());
    beta.setVal(val, i, j);
    setADupToDate(false);
    if(isGaussianNoise())
      syncLikelihoodVarianceFromSparseBeta();
  }
  void setBetaVals(double val) {
    beta.setVals(val);
    setADupToDate(false);
    if(isGaussianNoise())
      syncLikelihoodVarianceFromSparseBeta();
  }
  void setBeta(const CMatrix& bet) {
    DIMENSIONMATCH(bet.dimensionsMatch(beta));
    beta.deepCopy(bet);
    setADupToDate(false);
    if(isGaussianNoise())
      syncLikelihoodVarianceFromSparseBeta();
  }
  int getApproximationType() const {
    return approximationType;
  }
  string getApproximationStr() const {
    switch(approximationType) {
    case FTC:
      return "ftc";
    case DTC:
      return "dtc";
    case DTCVAR:
      return "dtcvar";
    case FITC:
      return "fitc";
    case PITC:
      return "pitc";
    default:
      throw ndlexceptions::Error("Unknown approximation type");
    }
  }
  void setApproximationStr(const string val) {
    if(val=="ftc") 
      setApproximationType(FTC);
    else if(val=="dtc")
      setApproximationType(DTC);
    else if(val=="dtcvar")
      setApproximationType(DTCVAR);
    else if(val=="fitc")
      setApproximationType(FITC);
    else if(val=="pitc")
      setApproximationType(PITC);
    else
      throw ndlexceptions::Error("Unknown approximation type");
  }

  void setApproximationType(unsigned int val) {
    approximationType=val;
    if(approximationType == FTC)
      setSparseApproximation(false);
    else if(approximationType == DTC)
      setSparseApproximation(true);
    else if(approximationType == DTCVAR)
      setSparseApproximation(true);
    else if(approximationType == FITC)
      setSparseApproximation(true);
    else if (approximationType == PITC)
      setSparseApproximation(true);
    else
      throw ndlexceptions::Error("Unknown approximation type");
    invalidateCache();
  }

  // CIP-000F: GPmat-compatible contiguous PITC blocks (0-based inclusive ends).
  void initDefaultBlockEnd();
  unsigned int getNumBlocks() const { return (unsigned int)blockEnd.size(); }
  void getBlockRange(unsigned int block, unsigned int& start, unsigned int& endInclusive) const;

  // Flag which indicates if a sparse approximation is used.
  bool isSparseApproximation() const 
  {
    return sparseApproximation;
  }
  void setSparseApproximation(const bool val) 
  {
    sparseApproximation=val;
  }
  // Whether noise model on outputs is spherical.
  bool isSpherical() const 
  {
    return spherical;
  }
  void setSpherical(const bool val) {
    spherical = val;
  }
  // Whether inducing variables are fixed.
  bool isInducingFixed() const 
  {
    return inducingFixed;
  }
  void setInducingFixed(const bool val) 
  {
    inducingFixed = val;
  }

  // CIP-001C: inducing-variable family ("points" | "whitenoise").
  void setInducingType(const string& typeName);
  void setInducing(CInducing* pInd);
  string getInducingType() const;
  CInducing* getInducing() { return pInducing; }
  const CInducing* getInducing() const { return pInducing; }
  // Meff = M - tr(C^{-1}) after ensureAD (whitened path); legacy uses A.
  double getEffectiveInducingCount() const;

  // Model-level GPy-style paths (CIP-0014 Phase C): kern.*, beta, scale.j,
  // likelihood.variance (CIP-0027; alias Gaussian_noise.variance).
  vector<string> paramPaths() const;
  double getByPath(const string& path) const;
  void setByPath(const string& path, double val);

  // Per-parameter fix masks (CIP-0014 Phase B). Paths use Phase A kernel
  // names (e.g. rbf.variance); also beta, scale.j, likelihood.variance.
  // Not stream-persisted yet.
  void fix(const string& path);
  void unfix(const string& path);
  bool isFixed(const string& path) const;
  void fixKernParam(unsigned int index);
  void unfixKernParam(unsigned int index);
  bool isKernParamFixed(unsigned int index) const;
  void fixBeta(bool val=true);
  bool isBetaFixed() const { return betaFixed; }
  void fixLikelihoodVariance(bool val=true);
  bool isLikelihoodVarianceFixed() const { return likelihoodVarianceFixed; }
  void clearFixes();

  // CIP-0027: Gaussian observation noise on the likelihood.
  bool isGaussianNoise() const;
  double getLikelihoodVariance() const;
  // CIP-003B: attach a prior to a parameter named as in fix()/setByPath
  // (kernel paths such as rbf.lengthscale, or likelihood.variance). The model
  // does not take ownership of dist; it must outlive the model.
  void addPriorByPath(const string& path, CDist* dist);
  // CIP-003B: log prior and its gradient (log-likelihood convention) for likelihood.variance.
  // CIP-003B: sampling target (exact FTC, Gaussian likelihood only for now).
  void sampledParams(std::vector<SampledParam>& params) const;
  const COptimisable& optimisable() const { return *this; }
  double noisePriorLogProb() const;
  double noisePriorGradVariance() const;
  void setLikelihoodVariance(double val);
  bool addsLikelihoodVarianceToGram() const;
  // Phase C: permit free White and free likelihood.variance together (FTC).
  void setAllowWhiteAndLikelihoodNoise(bool val) { allowWhiteAndLikelihoodNoise = val; }
  bool isAllowWhiteAndLikelihoodNoise() const { return allowWhiteAndLikelihoodNoise; }
  // Throws if FTC Gaussian has trainable White and free likelihood.variance
  // without allowWhiteAndLikelihoodNoise (CIP-0027 Phase C harden).
  void enforceObservationNoisePolicy() const;
  // Flag which indicates if K/Kinv/DynK/DynKInv need recomputation.
  bool isKupToDate() const 
  {
    return KupToDate;
  }
  void setKupToDate(const bool val) const
  {
    KupToDate = val;
    if(!KupToDate)
    {
      epUpToDate = false;
      setAlphaUpToDate(false);
      setADupToDate(false);
    }
  }
  // Flag which indicates if A,D etc. need recomputation.
  bool isADupToDate() const 
  {
    return ADupToDate;
  }
  void setADupToDate(const bool val) const 
  {
    ADupToDate = val;
  }
  // Flag which indicates if alphas need recomputation.
  bool isAlphaUpToDate() const 
  {
    return AlphaUpToDate;
  }
  void setAlphaUpToDate(const bool val) const 
  {
    AlphaUpToDate = val;
  }
  // Flag which indicates if M needs recomputation.
  bool isMupToDate() const 
  {
    return MupToDate;
  }
  void setMupToDate(const bool val) const 
  {
    MupToDate = val;
    if(!MupToDate)
    {
      setAlphaUpToDate(false);
      setADupToDate(false);
    }
  }
  bool isOptimiseX() const 
  {
    return optimiseX;
  }
  void setOptimiseX(const bool val) 
  {
    bool change = false;
    if(val!=optimiseX)
      change = true;
    optimiseX = val;
    if(change)
      initOptimiseXStoreage();
  }
  bool isBackConstrained() const
  {
    return backConstrained;
  }
  void setBackConstrained(const bool val)
  {
    backConstrained = val;
  }
  inline const CKern* getKernel() const {
    return pkern;
  }
 
  CMapModel* backConstraintModel; // for mapping constraints on latent variables.
  CMatrix X_u; // for inducing variables if needed (centres storage; inducing object aliases).
  ~CGp();

  CMatrix* pX;
  CMatrix* py;  // target data.

  // if debugging, make lots of these variables available for checking in python.
#ifndef DBG  
 private:
#endif
  bool optimiseX;
  bool backConstrained;

  mutable CMatrix m;  // scaled and biased Y
  mutable CMatrix Alpha; // SVM style 'alphas'.
  

  // Temporary variables for sparse approximations.
  mutable CMatrix Am;  
  mutable CMatrix Lm;  
  mutable CMatrix invLmV;  
  mutable CMatrix bet;  
  mutable CMatrix diagD;  
  mutable CMatrix sqrtDiagD;  
  mutable CMatrix scaledM; // numData*outputDim storage matrix.
  mutable CMatrix V;  // numActive*numData storage matrix
  mutable CMatrix E; // numActive*numData storage matrix
  mutable CMatrix EET; // numActive*numActive storage matrix.
  mutable CMatrix AinvE; // numActive*outputDim storage matrix.
  mutable CMatrix AinvEET; // numActive*numActive storage matrix.
  mutable CMatrix AinvEETAinv; // numActive*numActive storage matrix.
  mutable CMatrix AinvK_uf; // numActive*numData storage matrix.
  mutable CMatrix EMT; // numActive*numData storage matrix.
  mutable CMatrix AinvEMT; // numActive*numData storage matrix.
  mutable CMatrix invK_uuK_uf; // numActive*numData storage matrix.
  mutable CMatrix invK_uuK_ufDinv; // numActive*numData storage matrix. 
  mutable CMatrix invK_uuK_ufDinvQ; // numActive*numData storage matrix. 
  mutable CMatrix diagMMT; // numData*1 storage matrix.
  mutable CMatrix diagQ; // numData*1 storage matrix.
  mutable CMatrix diagK_ufdAinvplusAinvEETAinvK_fu; // numData*1 storage matrix.
  mutable CMatrix K_ufdotTimesAinvEMT; // numActive*numData storage matrix.
  mutable CMatrix diagK_ufAinvEMT; // numData*1 storage matrix.

  // PITC block state ([CIP-000F](cip/cip000F.md)).
  vector<unsigned int> blockEnd; // inclusive 0-based end index per block
  mutable vector<CMatrix> blockK;
  mutable vector<CMatrix> blockD;
  mutable vector<CMatrix> blockDinv;
  mutable vector<CMatrix> blockLcholD;
  mutable vector<double> logDetD;
  mutable vector<CMatrix> gLambdaBlock; // covGrad w.r.t. each block kernel
  mutable CMatrix pitcKufDinvm; // numActive * outputDim
  mutable double logDetDSum;
  
  
  CMatrix beta;
  CMatrix nu;
  CMatrix scale;
  CMatrix bias;
  
  CMatrix g;
  CKern* pkern;
  CNoise* pnoise;
  
  unsigned int maxTries;

  mutable vector<int> inducingIndices;
  mutable CMatrix gDiagX;
  mutable CMatrix gK_uf;
  mutable CMatrix gK_uu;
  mutable CMatrix gK_star;
  mutable CMatrix gLambda;
  mutable CMatrix gBeta;

  mutable CMatrix K;
  mutable CMatrix invK;
  mutable CMatrix diagK;
  mutable CMatrix LcholK;
  mutable double logDetK;

  mutable CMatrix covGrad;
  mutable CMatrix tempgX;

  mutable CMatrix K_uu;
  mutable CMatrix invK_uu;
  mutable double logDetK_uu;

  mutable CMatrix StorekN;

  mutable CMatrix K_uf;

  mutable CMatrix A;
  mutable CMatrix Ainv;
  mutable CMatrix LcholA;
  mutable double logDetA;

  // CIP-0012 / CIP-001B whitened state (spherical sparse).
  mutable CMatrix B;       // U^{-T} K_uf (M × N)
  mutable CMatrix Cmat;    // I + β B B^T (StableFactor); unused for LQ forward
  mutable CMatrix LcholC;  // upper chol(C) or upper R = L^T from LQ of [I,A]
  mutable double logDetC;
  // CIP-001B: Q from LQ of [I, A] (M × (M+N)), rows orthonormal; needed for lqAdjoint.
  // Formed lazily via lqFormQ from Householder (Wlq, tauLq) — skip dorglq in updateAD.
  mutable CMatrix Qlq;
  mutable CMatrix Wlq;
  mutable std::vector<double> tauLq;
  mutable bool QlqReady;

  // Representations for intermediate gradients.
  mutable CMatrix dgKX;
  mutable CMatrix gKX;
  mutable CMatrix gKX_uf;
  mutable CMatrix gKX_uf2;

  // Matrices where gradients are temporarily stored.
  mutable CMatrix g_scaleBias;
  mutable CMatrix g_param; 
  mutable CMatrix gX_u;
  mutable CMatrix gXorW;
  CMapModel* pbackModel;
  // if debugging, still make the remainder available.
#ifdef DBG
 private:
#endif
  void _init();
  void _updateK() const; // update K with the inverse of the kernel plus beta terms computed from the active points.
  void _updateInvK(unsigned int dim=0) const;
  // out := invK_uu * rhs (Legacy: gemm; StableFactor: two trsm against upper LcholK).
  void _multiplyInvKuu(CMatrix& out, const CMatrix& rhs) const;
  // CIP-0012: B := U^{-T} K_uf (one trsm on upper LcholK).
  void _whitenKuf(CMatrix& B, const CMatrix& Kuf) const;
  void _testComputeKx(CMatrix& kX, const CMatrix& Xin) const; /// compute the kernel for some test points.
  
  void _posteriorMean(CMatrix& mu, const CMatrix& kX) const; /// compute the posterior mean given the kernel evaluated at test points.
  // CIP-0031: includeObservationNoise adds 1/β (sparse) or likelihood.variance (FTC Gram path).
  void _posteriorVar(CMatrix& varSigma, CMatrix& kX, const CMatrix& Xin,
                     bool includeObservationNoise) const;
  void _posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin,
                         bool includeObservationNoise) const;
  double _observationNoiseVariance() const;
  double jitter;

  CTransform* betaTransform;

  unsigned int numActive;
  int approximationType; /// FTC, DTC, FITC, PITC, DTCVAR
  int sparseLinalgMode; /// SparseLinalgLegacy, StableFactor, or LQ
  int numCovStruct;

  bool outputScaleLearnt;
  bool outputBiasLearnt;
  bool sparseApproximation;
  bool terminate;
  bool epUpdate; // legacy IVM-era flag (unused on CGp); prefer epInference
  bool epInference; // CIP-0033 FTC EP
  bool laplaceInference; // CIP-0033 FTC Laplace
  mutable bool epUpToDate;
  mutable bool laplaceUpToDate;
  mutable CEpInference epEngine;
  mutable CLaplaceInference laplaceEngine;
  mutable CEpSites epSites;
  mutable CMatrix epMu;
  mutable CMatrix epVarDiag;
  mutable CMatrix laplaceFhat;
  mutable double epLogMarginalCache;
  mutable double laplaceLogMarginalCache;
  double epLogLikelihood() const; // GPy EP marginal + kern prior
  double laplaceLogLikelihood() const;
  bool loadedModel;
  bool spherical;
  bool inducingFixed;
  bool betaFixed;
  bool likelihoodVarianceFixed;
  bool allowWhiteAndLikelihoodNoise;
  mutable bool dualNoiseWarned;
  vector<bool> kernParamFixed;
  vector<bool> scaleParamFixed;
  void ensureKernFixedSize();
  void resolveFixPath(const string& path, int& kind, unsigned int& index) const;
  // kind: 0=kern, 1=beta, 2=scale, 3=likelihood.variance (CIP-0027)
  bool isLikelihoodVariancePath(const string& path) const;
  bool hasWhiteComponent() const;
  bool hasTrainableWhite() const;
  void syncSparseBetaFromLikelihoodVariance();
  void syncLikelihoodVarianceFromSparseBeta();
  void maybeWarnDualObservationNoise() const;
  mutable bool KupToDate;
  mutable bool AlphaUpToDate;
  mutable bool ADupToDate;
  mutable bool MupToDate;

  CInducing* pInducing; // owned; points / white-noise inducing contract
  
  string type;
};

// Functions which operate on the object
void writeGpToStream(const CGp& model, ostream& out);
void writeGpToFile(const CGp& model, const string modelFileName, const string comment="");
CGp* readGpFromStream(istream& in);
CGp* readGpFromFile(const string modelfileName, int verbosity=2);


#endif /* CGP_H */
