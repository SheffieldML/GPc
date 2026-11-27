/* This file and CNoise.cpp contain classes for different noise models including classification, regression and semi-supervised learning. 

10/10/2006 Fixed sign error in likelihood for ordered model. This error was propagated because of a sign error in ndlutil.cpp. In this case two wrongs made a right.

13/11/2026 Ordered missing targets are NaN. Test isnan before casting to int: on ARM
(int)NaN is 0, so missing points were treated as category 0 (macOS OpenBLAS failures).

1/3/2005 While looking for the bug that caused Gregg Keller problems, I found that when noise models where read from disk they were allocating storeage at the rong point. Moved initStoreage() out of setNumProcesses and set the numData to a dummy value in readParamsFromStream() to fix it.

This file now also contains the code for ordered categorical noise models.

21/10/2005 More changes from Bill, mostly removing redundant const modifiers.

 20/10/2005 Incorporated William V. Baxter's code modifications for running the code with MSVC. Mostly this involved removing redundant const statements.*/
#ifndef CNOISE_H
#define CNOISE_H
#include <iostream>
#include <cmath>
#include <climits>
#include <vector>
#include "ndlexceptions.h"
#include "ndlutil.h"
#include "ndlstrutil.h"
#include "CMatrix.h"
#include "CTransform.h"
#include "CDist.h"
#include "COptimisable.h"
#include "CDist.h"
#include "CKern.h"
#include "CNoiseSiteWorkspace.h"
using namespace std;

const string NOISEVERSION="0.1";

/* CIP-0027 Phase 0 — likelihood surface (docs/target-architecture.md rule 2).
 *
 * Canonical / GPy2 names for new work (existing C++ methods stay as oracles/wrappers):
 *   log_density              — log p(y|f) at a latent sample (exact f)
 *   variational_expectations — E_q[log p(y|f)]  (maps to expectedLogLikelihood)
 *   log_predictive_density   — log E_q[p(y|f)]  (maps to logLikelihood(mu,var,y))
 *   predictive_moments       — out() mean / variance of the observation
 *   ep_moments               — CIP-0033: Bernoulli–probit (GPy moments_match_ep)
 *
 * Batched helpers take contiguous (Y, mu, var) blocks and write into
 * caller-provided buffers. They take observation variance as a parameter so
 * they add no fields to CNoise. Per-element virtuals remain for tests only;
 * do not add new hot-path per-element methods here.
 *
 * Parameter naming (rule 7): likelihood.variance is canonical;
 * Gaussian_noise.variance is the GPy1 alias (wired in Phase A via CIP-0014).
 */
class CNoise;  // forward for ndlnoise::laplacePointDerivs
namespace ndlnoise {

/** Add spherical observation noise σ² to the diagonal of a Gram matrix (FTC Phase A). */
void addObservationNoiseDiag(CMatrix& K, double sigma2);

/** log p(y|f) = N(y; f, σ²) elementwise into logDens (bias offsets optional, length = cols). */
void gaussianLogDensity(CMatrix& logDens,
                        const CMatrix& y,
                        const CMatrix& f,
                        double sigma2,
                        const double* biasOrNull = 0);

/** E_q[log p(y|f)] for q(f)=N(μ,v); also fills gMu, gVar. Matches CGaussianNoise::expectedLogLikelihood. */
void gaussianVariationalExpectations(CMatrix& ell,
                                     CMatrix& gMu,
                                     CMatrix& gVar,
                                     const CMatrix& y,
                                     const CMatrix& mu,
                                     const CMatrix& var,
                                     double sigma2,
                                     const double* biasOrNull = 0);

/** log E_q[p(y|f)] for Gaussian noise; matches CGaussianNoise::logLikelihood(mu,var,y) up to bias. */
double gaussianLogPredictiveDensity(const CMatrix& y,
                                    const CMatrix& mu,
                                    const CMatrix& var,
                                    double sigma2,
                                    const double* biasOrNull = 0);

/** GPy Bernoulli+Probit moments_match_ep for one observation.
 *  Cavity natural params (tauCav, vCav); y in {0,1} or {-1,1}.
 *  Writes Zhat, muHat, sigma2Hat (GPy return order). */
void bernoulliProbitEpMoments(double& Zhat,
                              double& muHat,
                              double& sigma2Hat,
                              double y,
                              double tauCav,
                              double vCav);

/** Batched ep_moments: Y, tauCav, vCav are N×1 (or N×d with d=1 for v1). */
void bernoulliProbitEpMoments(CMatrix& Zhat,
                              CMatrix& muHat,
                              CMatrix& sigma2Hat,
                              const CMatrix& y,
                              const CMatrix& tauCav,
                              const CMatrix& vCav);

/** Bernoulli–probit log-density and Laplace Hessian pieces at latent f.
 *  y in {0,1} or {-1,1}. Writes:
 *    logp = log Φ(s f), s=±1
 *    d1   = ∂logp/∂f
 *    W    = −∂²logp/∂f²  (≥0; GPy / R&W)
 *    dWdf = ∂W/∂f        (= −∂³logp/∂f³)
 */
void bernoulliProbitLaplaceDerivs(double& logp,
                                  double& d1,
                                  double& W,
                                  double& dWdf,
                                  double y,
                                  double f);

/** Inverse Mills ratio λ(z)=φ(z)/Φ(z) and derivatives (GPy gaussian_special). */
double invMillsRatio(double z);
double dInvMillsRatio(double z, double lam);
double d2InvMillsRatio(double z, double lam);
inline double dInvMillsRatio(double z) { return dInvMillsRatio(z, invMillsRatio(z)); }
inline double d2InvMillsRatio(double z) { return d2InvMillsRatio(z, invMillsRatio(z)); }

/** SkewGaussian (O'Hagan / Azzalini) log-density and Laplace pieces at latent f.
 *  p(y|f) = (2/σ) φ((y-f)/σ) Φ(α (y-f)/σ). Writes logp, d1=∂logp/∂f,
 *  W=−∂²logp/∂f², dWdf=∂W/∂f. */
void skewGaussianLaplaceDerivs(double& logp,
                               double& d1,
                               double& W,
                               double& dWdf,
                               double y,
                               double f,
                               double variance,
                               double alpha);

/** SkewExponential (probit × Laplace base) at latent f.
 *  p(y|f) = λ exp(−λ |y-f|) Φ(α (y-f)). */
void skewExponentialLaplaceDerivs(double& logp,
                                  double& d1,
                                  double& W,
                                  double& dWdf,
                                  double y,
                                  double f,
                                  double rate,
                                  double alpha);

/** Student-t (identity link): scale² σ², degrees of freedom ν. */
void studentTLaplaceDerivs(double& logp,
                           double& d1,
                           double& W,
                           double& dWdf,
                           double y,
                           double f,
                           double variance,
                           double degFree);

/** Poisson with log link: λ=exp(f). y ∈ {0,1,2,…}. */
void poissonLogLaplaceDerivs(double& logp,
                             double& d1,
                             double& W,
                             double& dWdf,
                             double y,
                             double f);

/** Binomial with probit link: y successes in numTrials trials. */
void binomialProbitLaplaceDerivs(double& logp,
                                 double& d1,
                                 double& W,
                                 double& dWdf,
                                 double y,
                                 double f,
                                 double numTrials);

/** Exponential with log link: λ=exp(f), y>0. */
void exponentialLogLaplaceDerivs(double& logp,
                                 double& d1,
                                 double& W,
                                 double& dWdf,
                                 double y,
                                 double f);

/** Gamma with log link: mean λ=exp(f), shape α=βλ, scale 1/β. */
void gammaLogLaplaceDerivs(double& logp,
                           double& d1,
                           double& W,
                           double& dWdf,
                           double y,
                           double f,
                           double beta);

/** CIP-003A Phase 5: dispatch pointwise Laplace pieces through a typed CNoise.
 *  Supports probit / student_t / poisson / binomial / exponential / gamma /
 *  skew_gaussian / skew_exponential. Throws for unsupported types. */
void laplacePointDerivs(const CNoise& noise,
                        double y,
                        double f,
                        double& logp,
                        double& d1,
                        double& W,
                        double& dWdf);

// CIP-003E Phase 6: multi-class targets are one-hot N x K inside the library; integer
// labels are converted here, at the boundary (loaders, CLI, Python), and nowhere else.
// labels: N x 1 integers in 0..K-1 (numClasses = 0 infers K = max + 1, at least 2), or
// an N x K matrix that is already one-hot (validated and copied). Throws on bad input.
CMatrix oneHotLabels(const CMatrix& labels, unsigned int numClasses = 0);
// Class index (argmax) of each row of a one-hot matrix, as an N x 1 matrix.
CMatrix labelsFromOneHot(const CMatrix& Y);

} // namespace ndlnoise

// The basic noise class. The class allows its parameters to be transformed or optimised. Also noise models can be loaded from matlab files.
// CIP-003B: noise parameters carry priors through CRegularisable, exactly as
// kernel parameters do (and as GPy allows a prior on any parameter, e.g.
// likelihood.variance).
class CNoise : public CTransformable, public CRegularisable, public COptimisable, public CStreamInterface, public CMatInterface {
 public:
  // constructors
  CNoise() {  }
  CNoise(CMatrix* pyin) : py(pyin)
  {
    setNumData(py->getRows());
    setOutputDim(py->getCols());
  }
  virtual ~CNoise(){}
  
  // pure virtual functions
  virtual void initStoreage()=0;
  virtual void initNames()=0;
  virtual void initVals()=0;
  virtual void initParams()=0;
  virtual ostream& display(ostream& os)=0;
  // Re-declared so that name lookup is unambiguous between CTransformable and
  // CRegularisable (CIP-003B). Concrete noise models override these once.
  virtual double getParam(unsigned int paramNo) const=0;
  virtual void setParam(double val, unsigned int paramNo)=0;
  virtual void setParams(const CMatrix& X)=0;
  virtual void getParams(CMatrix& X) const=0;
  virtual void getGradParams(CMatrix& g) const=0;
  virtual void getGradInputs(double& dlnZ_dmu, double& dlnZ_dvs, unsigned int i, unsigned int j) const=0;
  virtual void getGradInputs(CMatrix& dlnZ_dmu, CMatrix& dlnZ_dvs) const
  {
    // gradient with respect to mu and varSigma of log likelihood.  
    DIMENSIONMATCH(dlnZ_dmu.dimensionsMatch(dlnZ_dvs));
    DIMENSIONMATCH(dlnZ_dmu.getRows()==nData);
    DIMENSIONMATCH(dlnZ_dmu.getCols()==nProcesses);
    
    double gmu;
    double gvs;
    for(unsigned int i=0; i<dlnZ_dmu.getRows(); i++)
    {
      for(unsigned int j=0; j<dlnZ_dmu.getCols(); j++)
      {
	getGradInputs(gmu, gvs, i, j);
	dlnZ_dmu.setVal(gmu, i, j);
	dlnZ_dvs.setVal(gvs, i, j);
      }
    }
  }
  // Nu and G are combinations of gradients with respect to the mean and variance of the noise model.
  virtual void getNuG(CMatrix& g, CMatrix& nu, unsigned int index) const;
  virtual void updateSites(CMatrix& m, CMatrix& beta, unsigned int actIndex, const CMatrix& g, const CMatrix& nu, unsigned int index) const;
  virtual void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const=0;
  virtual void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const=0;
  virtual void out(CMatrix& yPred, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const=0;
  virtual void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const=0;
  virtual double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const=0;
  virtual double logLikelihood() const=0;
  // CIP-002A: E_q[log p(y|f)] and grads w.r.t. marginal mean/variance of q(f).
  // Distinct from ADF logLikelihood(mu, var, y) = log E_q[p(y|f)].
  virtual double expectedLogLikelihood(double y, double mu, double var,
                                       double& gMu, double& gVar) const;

  // ---- CIP-003E Phase 1: block oracles and capabilities ----
  // A likelihood term is a function of a block of blockSize() consecutive output
  // columns. Independent likelihoods have blockSize 1 and the defaults below are
  // the existing element-wise code, so behaviour is unchanged. y, mu, var, g* are
  // arrays of length blockSize(); j0 is the first output column of the block
  // (per-output biases are keyed on the column).
  virtual unsigned int blockSize() const { return 1; }
  enum Capability { CapExpectedLogLik = 1, CapPointDerivs = 2,
                    CapCavityMoments = 4, CapPredictive = 8 };
  virtual unsigned int capabilities() const;
  // Throw NotImplementedError naming `consumer` and this noise type.
  void requireCapability(unsigned int cap, const string& consumer) const;
  void requireScalarBlocks(const string& consumer) const;

  // VI: sum_k E_q[log p(y|f)] over the block; gMu/gVar are per output.
  virtual double expectedLogLikelihoodBlock(const double* y, const double* mu,
                                            const double* var, unsigned int j0,
                                            double* gMu, double* gVar) const;
  // Laplace: log p, gradient g (length blockSize), negative Hessian W
  // (blockSize x blockSize) and optionally dW/df: for blockSize 1 a 1x1 matrix, for
  // blockSize b a (b*b) x b matrix with entry (a*b+c, d) = dW_ac/df_d.
  virtual void pointDerivsBlock(const double* y, const double* f, unsigned int j0,
                                double* logp, double* g, CMatrix& negHessian,
                                CMatrix* dNegHessian = 0) const;
  // EP / ADF: derivatives of log Z (cavity N(mu, var)) in the sense of
  // getGradInputs: dMu = dlogZ/dmu, dVar = dlogZ/dvar. (log Z itself is added with
  // block EP in Phase 2.)
  virtual void cavityMomentsBlock(const double* y, const double* mu, const double* var,
                                  unsigned int j0, double* dMu, double* dVar) const;
  // Explicit-argument element form behind getGradInputs; implemented by the
  // likelihoods that advertise CapCavityMoments.
  virtual void cavityMomentsElement(double y, double mu, double var, unsigned int j,
                                    double& gmu, double& gvs) const;
  // log Z = log E_{N(mu,var)}[p(y|f)] over the block (the normaliser behind
  // cavityMomentsBlock). Default: sum of logZElement over the block's outputs.
  virtual double logZBlock(const double* y, const double* mu, const double* var,
                           unsigned int j0) const;
  virtual double logZElement(double y, double mu, double var, unsigned int j) const;
  // Class probabilities for one block of marginals (block likelihoods).
  virtual void predictBlock(const double* mu, const double* var, unsigned int j0,
                            double* prob) const;
  virtual void writeParamsToStream(ostream& out) const;
  virtual void readParamsFromStream(istream& in);
  // CIP-003B: reads an optional trailing numPriors block; leaves the stream untouched if absent.
  void readOptionalPriorsFromStream(istream& in);
#ifdef _NDLMATLAB
  virtual mxArray* toMxArray() const;
  virtual void fromMxArray(const mxArray* matlabArray);
  // Adds parameters to the mxArray.
  virtual void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  virtual void extractParamFromMxArray(const mxArray* matlabArray);
#endif
  // non virtual functions
  inline void setType(const string name)
  {
    type = name;
  }
  inline string getType() const
  {
    return type;
  }
  string getBaseType() const
  {
    return "noise";
  }
  inline string getName() const
  {
    return noiseName;
  }
  inline void setName(const string name)
  {
    noiseName = name;
  }
  
  //void getGradX(CMatrix& gX, const CMatrix& dmu, const CMatrix& cvs);
  
  // Functions arising from optimisable.
  unsigned int getOptNumParams() const
  {
    return getNumParams();
  }
  void getOptParams(CMatrix& params) const
  {
    getTransParams(params);
  }
  void setOptParams(const CMatrix& params)
  {
    setTransParams(params);
  }
  double computeObjectiveGradParams(CMatrix& g) const
  {
    getGradTransParams(g);
    g.negate();
    return -logLikelihood();
  }
  double computeObjectiveVal() const
  {
    return -logLikelihood();
  }
  
  
  inline bool isLogConcave() const
  {
    return logConcave;
  }
  inline bool isSpherical() const
  {
    return spherical;
  }
  inline bool isMissing() const
  {
    return missing;
  }
  inline unsigned int getNumParams() const
  {
    return nParams;
  }
  inline unsigned int getOutputDim() const
  {
    return nProcesses;
  }
  inline unsigned int getNumData() const
  {
    return nData;
  }
  virtual string getParamName(unsigned int index) const
  {
    BOUNDCHECK(index>=0);
    BOUNDCHECK(index<paramNames.size());
    return paramNames[index];
  }
  void setParamName(const string paramName, unsigned int index)
  {
    BOUNDCHECK(index>=0);
    BOUNDCHECK(index<getNumParams());
    if(paramNames.size() == index)
      paramNames.push_back(paramName);
    else 
    {
      if(paramNames.size()<index)
	paramNames.resize(index+1, "no name");
      paramNames[index] = paramName;
    }
  }
  
  // CIP-0034: site / ADF caches live in CNoiseSiteWorkspace (owned or attached).
  // These are inference adapter fields, not likelihood parameters.
  void setMu(double val, unsigned int i, unsigned int j);
  double getMu(unsigned int i, unsigned int j) const;
  void setVarSigma(double val, unsigned int i, unsigned int j);
  double getVarSigma(unsigned int i, unsigned int j) const;
  virtual double getTarget(unsigned int i, unsigned int j) const
  {
    return py->getVal(i, j);
  }
  virtual void setTarget(CMatrix* vals)
  {
    // Non-owning: model / dataset owns Y (CIP-0027 / CIP-0034).
    py=vals;
    setNumData(vals->getRows());
    setOutputDim(vals->getCols());
    initStoreage();
    initNames();      
  }
  
  // Null = use ownedSites; non-null = inference-owned workspace (IVM / PPA).
  void attachSiteWorkspace(CNoiseSiteWorkspace* ws) { pSiteWs = ws; }
  CNoiseSiteWorkspace* siteWorkspace() const { return pSiteWs; }
  bool usingExternalSiteWorkspace() const { return pSiteWs != 0; }

  void setMus(double val)
  {
    for(unsigned int i=0; i<getNumData(); i++)
      for(unsigned int j=0; j<getOutputDim(); j++)
	setMu(val, i, j);
  }
  void setVarSigmas(double val)
  {
    for(unsigned int i=0; i<getNumData(); i++)
      for(unsigned int j=0; j<getOutputDim(); j++)
	setVarSigma(val, i, j);
  }
  
  void setMus(const CMatrix& vals)
  {
    DIMENSIONMATCH(vals.getRows()==getNumData());
    DIMENSIONMATCH(vals.getCols()==getOutputDim());
    for(unsigned int i=0; i<getNumData(); i++)
      for(unsigned int j=0; j<getOutputDim(); j++)
	setMu(vals.getVal(i, j), i, j);
  }
  void setVarSigmas(const CMatrix& vals)
  {
    DIMENSIONMATCH(vals.getRows()==getNumData());
    DIMENSIONMATCH(vals.getCols()==getOutputDim());
    for(unsigned int i=0; i<getNumData(); i++)
      for(unsigned int j=0; j<getOutputDim(); j++)
	setVarSigma(vals.getVal(i, j), i, j);
  }
  bool equals(const CNoise& noise, double tol=ndlutil::MATCHTOL) const;
#ifdef _NDLMATLAB
  void setVarSigmas(const mxArray* matlabArray)
  {
    CMatrix varsig;
    varsig.fromMxArray(matlabArray);
    setVarSigmas(varsig);
  }
  void setMus(const mxArray* matlabArray)
  {
    CMatrix mus;
    mus.fromMxArray(matlabArray);
    setMus(mus);
  }
  mxArray* varSigmaToMxArray() const
  {
    return activeSites().varSigma.toMxArray();
  }
  mxArray* muToMxArray() const
  {
    return activeSites().mu.toMxArray();
  }
  
  mxArray* targetToMxArray() const
  {
    return py->toMxArray();
  }
#endif

  // Non-owning pointer to targets (model owns the matrix).
  CMatrix* py;
  
 protected:
  CNoiseSiteWorkspace ownedSites;
  CNoiseSiteWorkspace* pSiteWs = 0;
  CNoiseSiteWorkspace& activeSites()
  {
    return pSiteWs != 0 ? *pSiteWs : ownedSites;
  }
  const CNoiseSiteWorkspace& activeSites() const
  {
    return pSiteWs != 0 ? *pSiteWs : ownedSites;
  }
  void initSiteStoreage();
  

  
  inline void setLogConcave(const bool val)
  {
    logConcave = val;
  }
  inline void setMissing(const bool val) 
  {
    missing = val;
  }
  inline void setSpherical(const bool val)
  {
    spherical = val;
  }
  inline void setNumParams(unsigned int num) 
  {
    nParams = num;
  }
  inline void setOutputDim(unsigned int num) 
  {
    nProcesses = num;
  }
  
 protected:
  inline void setNumData(unsigned int num) 
  {
    nData = num;
  }

 private:
   void _init();
  bool spherical;
  bool logConcave;
  bool missing;
  vector<string> paramNames;
  unsigned int nParams;
  unsigned int nProcesses;
  unsigned int nData;
  string type;
  string noiseName;
};
// The Gaussian noise model as commonly used in regression.
class CGaussianNoise : public CNoise {
 public:  
  // constructors
  CGaussianNoise()
  {
    _init();
  }
  CGaussianNoise(CMatrix* pyin) 
  {
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CGaussianNoise();
  
  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void cavityMomentsElement(double y, double mu, double var, unsigned int j,
                            double& gmu, double& gvs) const;
  double logZElement(double y, double mu, double var, unsigned int j) const;
  void getNuG(CMatrix& g, CMatrix& nu, unsigned int index) const;
  void updateSites(CMatrix& m, CMatrix& beta, unsigned int actIndex, const CMatrix& g, const CMatrix& nu, unsigned int index) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const 
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;
  //void  getGradX(CMatrix& gX, const CMatrix& dmu, const CMatrix& cvs);
  
  double getBiasVal(unsigned int index) const 
  {
    BOUNDCHECK(index<getOutputDim());
    BOUNDCHECK(index>=0);
    return bias.getVal(0, index);
  }
  void setBiasVal(double val, unsigned int index) 
  {
    BOUNDCHECK(index<getOutputDim());
    BOUNDCHECK(index>=0);
    bias.setVal(val, 0, index);
  }
  void setBias(const CMatrix& bia) 
  {
    DIMENSIONMATCH(bia.getRows()==1);
    DIMENSIONMATCH(bia.getCols()==getOutputDim());
    bias.deepCopy(bia);
  }
  double getSigma2() const { return sigma2; }
  void setSigma2(double val)
  {
    BOUNDCHECK(val > 0.0);
    sigma2 = val;
  }
  // CIP-0027: canonical name is variance (path likelihood.variance); sigma2 is legacy.
  double getVariance() const { return getSigma2(); }
  void setVariance(double val) { setSigma2(val); }
  
#ifdef _NDLMATLAB
  // Adds parameters to the mxArray.
  void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  void extractParamFromMxArray(const mxArray* matlabArray);
#endif
 private:

  void _init();
  
  double sigma2;
  CMatrix bias;
};

// A scaled Gaussian noise model.
class CScaleNoise : public CNoise {
 public:  
  // constructors
  CScaleNoise()
  {
    _init();
  }
  CScaleNoise(CMatrix* pyin)
  {
    _init();
    setTarget(pyin);
    initVals();
    initParams(); 
  }
  ~CScaleNoise();
  
  double getScale(unsigned int index) const
  {
    BOUNDCHECK(index<getOutputDim()&&index>=0);
    return scale.getVal(index);
  }
  void setScale(double val, unsigned int index)
  {
    BOUNDCHECK(index<getOutputDim()&&index>=0);
    scale.setVal(val, index);
  }
  double getBias(unsigned int index) const
  {
    BOUNDCHECK(index<getOutputDim()&&index>=0);
    return bias.getVal(index);
  }
  void setBias(double val, unsigned int index)
  {
    BOUNDCHECK(index<getOutputDim()&&index>=0);
    bias.setVal(val, index);
  }
  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void getNuG(CMatrix& g, CMatrix& nu, unsigned int index) const;
  void updateSites(CMatrix& m, CMatrix& beta, unsigned int actIndex, const CMatrix& g, const CMatrix& nu, unsigned int index) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  //void  getGradX(CMatrix& gX, const CMatrix& dmu, const CMatrix& cvs);
  
 
#ifdef _NDLMATLAB
  // Adds parameters to the mxArray.
  void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  void extractParamFromMxArray(const mxArray* matlabArray);
#endif
 private:
  void _init();
  double sigma2;
  CMatrix bias;
  CMatrix scale;
};

// The probit noise model often used for classification.
class CProbitNoise : public CNoise 
{
 public:  
  // constructors
  CProbitNoise()
  {
    _init();
  }
  CProbitNoise(CMatrix* pyin) 
  {
    _init();
    setTarget(pyin);
    initVals();
    initParams(); 
  }
  ~CProbitNoise();
  
  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void cavityMomentsElement(double y, double mu, double var, unsigned int j,
                            double& gmu, double& gvs) const;
  double logZElement(double y, double mu, double var, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;
  // Select which output's bias enters expectedLogLikelihood (CIP-002A).
  void setExpectedLogLikBias(double b) const { ellBias = b; }
  //void  getGradX(CMatrix& gX, const CMatrix& dmu, const CMatrix& cvs);
  
  double getBiasVal(unsigned int index) const
  {
    return bias.getVal(0, index);
  }
  // CIP-003E: GPy-convention probit. The likelihood is Phi(y f) with the unit
  // softener inside the likelihood (sigma2 = 1), so the cavity moments are
  // Phi(mu/sqrt(1+v)) and the latent is f, as in GPy's Bernoulli. Bias is
  // fixed at zero. The default (false) keeps the CIP-0027 convention: a step
  // function on f + eps with the softener as White in K, which is what IVM/ADF
  // expect. Runtime state: not written to .model streams.
  void setGpyMode(bool on)
  {
    gpyMode = on;
    sigma2 = on ? 1.0 : 1e-6;
    if(on && bias.getCols()>0)
      bias.zeros();
  }
  bool isGpyMode() const { return gpyMode; }
  // CIP-0027: probit σ² is fixed (softener is White-in-K); expose for tests.
  double getSigma2() const { return sigma2; }
  
#ifdef _NDLMATLAB
  // Adds parameters to the mxArray.
  void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  void extractParamFromMxArray(const mxArray* matlabArray);
#endif
  
 private:
  void _init();
  double sigma2;
  CMatrix bias;
  mutable double ellBias;
  bool gpyMode = false;
};

// The null category noise model for semi-supervised learning (see Lawrence and Jordan in NIPS 2004).
class CNcnmNoise : public CNoise {
 public:  
  // constructors
  CNcnmNoise()
  {
    _init();
  }
  CNcnmNoise(CMatrix* pyin) 
  {
    _init();
    setTarget(pyin);
    initVals();
    initParams(); 
  }
  ~CNcnmNoise();
  
  void initStoreage();
  void initNames();
  void initVals();
  void initParams();

  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void cavityMomentsElement(double y, double mu, double var, unsigned int j,
                            double& gmu, double& gvs) const;
  double logZElement(double y, double mu, double var, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  //void  getGradX(CMatrix& gX, const CMatrix& dmu, const CMatrix& cvs);
  
  inline void setSplitGamma(const bool val)
  {
    splitGamma=val;
  }
  bool isSplitGamma() const
  {
    return splitGamma;
  }
  void setTarget(CMatrix* vals)
  {
    setSplitGamma(false);
    CNoise::setTarget(vals);
  }
#ifdef _NDLMATLAB
  // Adds parameters to the mxArray.
  void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  void extractParamFromMxArray(const mxArray* matlabArray);
#endif
  
 private:
  void _init();
  double gamman;
  double gammap;
  bool splitGamma;
  double width;
  double sigma2;
  CMatrix bias;
};
class COrderedNoise : public CNoise {
 public:  
  // constructors
  COrderedNoise()
  {
    _init();
  }
  COrderedNoise(CMatrix* pyin) : numCats(1)
  {
    _init();
    setTarget(pyin);
    initVals();
    initParams(); 
  }
  COrderedNoise(CMatrix* pyin, unsigned int numCts) : numCats(numCts)
  {
    _init();
    setTarget(pyin);
    initVals();
    initParams(); 
  }
  ~COrderedNoise();
  
  unsigned int getNumCategories() const
  {
    return numCats;
  }
  void setNumCategories(unsigned int val)
  {
    SANITYCHECK(val>1);
    numCats=val;
    initStoreage();
  }
  
  void initStoreage();
  void initNames();
  void initVals();
  void initParams();

  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void cavityMomentsElement(double y, double mu, double var, unsigned int j,
                            double& gmu, double& gvs) const;
  double logZElement(double y, double mu, double var, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  // CIP-0030 Phase 3 / CIP-002A: E_q[log p(y|f)] via Gauss–Hermite.
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;
  void setExpectedLogLikBias(double b) const { ellBias = b; }
  double getBiasVal(unsigned int index) const { return bias.getVal(0, index); }
  //void  getGradX(CMatrix& gX, const CMatrix& dmu, const CMatrix& cvs);
  
#ifdef _NDLMATLAB
  // Adds parameters to the mxArray.
  void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  void extractParamFromMxArray(const mxArray* matlabArray);
#endif
  
 private:
   void _init();
  unsigned int numCats;
  CMatrix widths;
  double sigma2;
  CMatrix bias;
  mutable double ellBias;
  
  mutable CMatrix gwidth;
};

/** CIP-003A: skew-normal observation noise (GPy SkewGaussian). Params: variance, alpha. */
class CSkewGaussianNoise : public CNoise {
 public:
  CSkewGaussianNoise() { _init(); }
  CSkewGaussianNoise(CMatrix* pyin)
  {
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CSkewGaussianNoise();

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;

  double getVariance() const { return variance; }
  void setVariance(double val) { BOUNDCHECK(val > 0.0); variance = val; }
  double getAlpha() const { return alpha; }
  void setAlpha(double val) { alpha = val; }

 private:
  void _init();
  double variance;
  double alpha;
};

/** CIP-003A: skew-Laplace observation noise (GPy SkewExponential). Params: rate, alpha. */
class CSkewExponentialNoise : public CNoise {
 public:
  CSkewExponentialNoise() { _init(); }
  CSkewExponentialNoise(CMatrix* pyin)
  {
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CSkewExponentialNoise();

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;

  double getRate() const { return rate; }
  void setRate(double val) { BOUNDCHECK(val > 0.0); rate = val; }
  double getAlpha() const { return alpha; }
  void setAlpha(double val) { alpha = val; }

 private:
  void _init();
  double rate;
  double alpha;
};

/** CIP-003A: Student-t observation noise (GPy StudentT). Params: variance, deg_free. */
class CStudentTNoise : public CNoise {
 public:
  CStudentTNoise() { _init(); }
  CStudentTNoise(CMatrix* pyin)
  {
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CStudentTNoise();

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;

  double getVariance() const { return variance; }
  void setVariance(double val) { BOUNDCHECK(val > 0.0); variance = val; }
  double getDegFree() const { return degFree; }
  void setDegFree(double val) { BOUNDCHECK(val > 0.0); degFree = val; }

 private:
  void _init();
  double variance;
  double degFree;
};

/** CIP-003A: Poisson observation noise with log link (GPy Poisson). No free params. */
class CPoissonNoise : public CNoise {
 public:
  CPoissonNoise() { _init(); }
  CPoissonNoise(CMatrix* pyin)
  {
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CPoissonNoise();

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;

 private:
  void _init();
};

/** CIP-003A: Binomial + probit link (GPy Binomial). Fixed numTrials; no free params. */
class CBinomialNoise : public CNoise {
 public:
  CBinomialNoise() : numTrials(1) { _init(); }
  CBinomialNoise(CMatrix* pyin, unsigned int trials = 1) : numTrials(trials)
  {
    BOUNDCHECK(trials >= 1);
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CBinomialNoise();

  unsigned int getNumTrials() const { return numTrials; }
  void setNumTrials(unsigned int t)
  {
    BOUNDCHECK(t >= 1);
    numTrials = t;
  }

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);

 private:
  void _init();
  unsigned int numTrials;
};

/** CIP-003A: Exponential observation noise with log link (GPy Exponential). */
class CExponentialNoise : public CNoise {
 public:
  CExponentialNoise() { _init(); }
  CExponentialNoise(CMatrix* pyin)
  {
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CExponentialNoise();

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;

 private:
  void _init();
};

/** CIP-003A: Gamma observation noise with log link (GPy Gamma). Param: beta. */
class CGammaNoise : public CNoise {
 public:
  CGammaNoise() { _init(); }
  CGammaNoise(CMatrix* pyin)
  {
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CGammaNoise();

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  double expectedLogLikelihood(double y, double muVal, double var,
                               double& gMu, double& gVar) const;

  double getBeta() const { return beta; }
  void setBeta(double val) { BOUNDCHECK(val > 0.0); beta = val; }

 private:
  void _init();
  double beta;
};


/** CIP-003E Phase 4: robust-max multi-class likelihood (Hernandez-Lobato et al. 2011).
 *  One block of K latents per datum, one-hot targets. p(y=c|f) = 1-eps if c is the
 *  argmax of f and eps/(K-1) otherwise, so the expectation under independent
 *  Gaussian marginals needs only P(argmax = c), a one-dimensional integral
 *  int N(f_c) prod_{k!=c} Phi((f_c-mu_k)/s_k) df_c done by composite
 *  Gauss-Legendre with breakpoints at the Phi transitions (robust to small
 *  variances, unlike fixed Gauss-Hermite). eps is a fixed hyperparameter. The
 *  likelihood is piecewise constant in f, so there is no Laplace capability. */
class CRobustMaxNoise : public CNoise {
 public:
  CRobustMaxNoise() : numClasses(2), eps(1e-3) { _init(); }
  CRobustMaxNoise(CMatrix* pyin, unsigned int classes, double epsilon = 1e-3)
    : numClasses(classes), eps(epsilon)
  {
    BOUNDCHECK(classes >= 2);
    BOUNDCHECK(epsilon > 0.0 && epsilon < 1.0 - 1.0/classes);
    _init();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CRobustMaxNoise();

  unsigned int blockSize() const { return numClasses; }
  unsigned int capabilities() const;
  unsigned int getNumClasses() const { return numClasses; }
  double getEps() const { return eps; }
  void setEps(double e) { BOUNDCHECK(e > 0.0 && e < 1.0 - 1.0/numClasses); eps = e; }

  /** P(argmax f = c) for f_k ~ N(mu_k, var_k) independent, with derivatives in
   *  every mu_k and var_k (either derivative pointer may be null). */
  static double classProbability(const double* mu, const double* var,
                                 unsigned int K, unsigned int c,
                                 double* dMu, double* dVar);
  /** d/d eps of the block expected log-likelihood (for tests / future learning of eps). */
  double expectedLogLikelihoodDEps(const double* y, const double* mu, const double* var) const;

  double expectedLogLikelihoodBlock(const double* y, const double* mu, const double* var,
                                    unsigned int j0, double* gMu, double* gVar) const;
  void cavityMomentsBlock(const double* y, const double* mu, const double* var,
                          unsigned int j0, double* dMu, double* dVar) const;
  double logZBlock(const double* y, const double* mu, const double* var,
                   unsigned int j0) const;
  void predictBlock(const double* mu, const double* var, unsigned int j0,
                    double* prob) const;

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);

 private:
  void _init();
  unsigned int observedClass(const double* y) const;
  unsigned int numClasses;
  double eps;
};

/** CIP-003E Phase 5: softmax multi-class likelihood, p(y=c|f) = exp(f_c)/sum_k exp(f_k).
 *  One block of K latents per datum, one-hot targets. Expectations and cavity
 *  moments are reparameterised Monte Carlo over a fixed set of S standard normal
 *  draws (antithetic pairs; seed fixed per object), so every estimate is a
 *  deterministic, differentiable function of (mu, var): gradients are exact
 *  derivatives of the sampled estimate and finite differences are meaningful.
 *  The estimator is stochastic in the sense that it carries Monte Carlo error
 *  ~ 1/sqrt(S); EP with it is not claimed monotone. Laplace uses the exact
 *  point derivatives (gradient delta - pi, negative Hessian diag(pi) - pi pi^T). */
class CSoftmaxNoise : public CNoise {
 public:
  CSoftmaxNoise() : numClasses(2), numSamples(256), seed(12345) { _init(); resample(); }
  CSoftmaxNoise(CMatrix* pyin, unsigned int classes, unsigned int samples = 256,
                unsigned int seedVal = 12345)
    : numClasses(classes), numSamples(samples), seed(seedVal)
  {
    BOUNDCHECK(classes >= 2);
    BOUNDCHECK(samples >= 2);
    _init();
    resample();
    setTarget(pyin);
    initNames();
    initVals();
    initParams();
  }
  ~CSoftmaxNoise();

  unsigned int blockSize() const { return numClasses; }
  unsigned int capabilities() const;
  unsigned int getNumClasses() const { return numClasses; }
  unsigned int getNumSamples() const { return numSamples; }
  unsigned int getSeed() const { return seed; }
  void setNumSamples(unsigned int s) { BOUNDCHECK(s >= 2); numSamples = s; resample(); }
  void setSeed(unsigned int s) { seed = s; resample(); }

  double expectedLogLikelihoodBlock(const double* y, const double* mu, const double* var,
                                    unsigned int j0, double* gMu, double* gVar) const;
  void pointDerivsBlock(const double* y, const double* f, unsigned int j0,
                        double* logp, double* g, CMatrix& negHessian,
                        CMatrix* dNegHessian = 0) const;
  void cavityMomentsBlock(const double* y, const double* mu, const double* var,
                          unsigned int j0, double* dMu, double* dVar) const;
  double logZBlock(const double* y, const double* mu, const double* var,
                   unsigned int j0) const;
  void predictBlock(const double* mu, const double* var, unsigned int j0,
                    double* prob) const;

  void initStoreage();
  void initNames();
  void initVals();
  void initParams();
  ostream& display(ostream& os);
  void setParams(const CMatrix& params);
  void setParam(double val, unsigned int index);
  void getParams(CMatrix& params) const;
  double getParam(unsigned int index) const;
  void getGradParams(CMatrix& g) const;
  void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
  void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
  void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void out(CMatrix& yPred, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const;
  void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
  double logLikelihood() const
  {
    return logLikelihood(activeSites().mu, activeSites().varSigma, *py);
  }
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);

 private:
  void _init();
  void resample();
  unsigned int observedClass(const double* y) const;
  // softmax probabilities of one sample f (length K), returns log-sum-exp.
  static double softmaxOf(const double* f, unsigned int K, double* pi);
  unsigned int numClasses;
  unsigned int numSamples;
  unsigned int seed;
  std::vector<double> draws;  // numSamples x numClasses standard normals
};

void writeNoiseToStream(const CNoise& noise, ostream& out);
CNoise* readNoiseFromStream(istream& in);


#endif
