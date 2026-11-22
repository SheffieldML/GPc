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

} // namespace ndlnoise

// The basic noise class. The class allows its parameters to be transformed or optimised. Also noise models can be loaded from matlab files.
class CNoise : public CTransformable, public COptimisable, public CStreamInterface, public CMatInterface {
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
  virtual void writeParamsToStream(ostream& out) const;
  virtual void readParamsFromStream(istream& in);
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


void writeNoiseToStream(const CNoise& noise, ostream& out);
CNoise* readNoiseFromStream(istream& in);


#endif
