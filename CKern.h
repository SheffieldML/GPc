/* This file and CKern.cpp contains classes for representing kernels. These classes do not store the kernel matrix, they merely store the kernels characteristics. There are several types of kernel provided, and it should be easy to add others. CCmpndKern provides a class which forms an additive kernel from individual kernels (i.e. a linear plus an RBF kernel).

16/03/2007 Added code to compute parameter gradients when two X matrices are provided. This enables use of the toolbox with DTC, FITC, PITC approximations. Also changed the transformation of kernel parameters to be by exponential rather than negLogLogit, in line with the default in the MATLAB kern toolbox.

22/10/2005 Added updateX method to class, from William V. Baxter's changes to make computation more efficient. The changes involve precomputing the distance matrix and a portion of the kernel matrix and storing it for later use. UpdateX is called when X has changed so that these stored values are recomputed.

21/10/2005 More changes from William V. Baxter. Most are removal of redundant const modifiers, but also vector<CMatrix*> is now being passed as a reference, rather than by value.

20/10/2005 Incorporated changes from William V. Baxter to allow compilation under MSVC. */

#ifndef CKERN_H
#define CKERN_H
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "ndlassert.h"
#include "CTransform.h"
#include "CDataModel.h"
#include "CMatrix.h"
#include "CDist.h"
#include "ndlstrutil.h"
#include "ndlutil.h"
using namespace std;


const string KERNVERSION="0.1";

// The base class for all kernels. It implements: CMatinterface, for
// saving and loading from MATLAB; CRegularisable, for placing priors
// over parameters; CTransformable, for allowing parameters to be
// transformed so that they are only optimised in, for example,
// positive half-spaces

class CKern : public CMatInterface, public CStreamInterface, public CTransformable, public CRegularisable {
 public:
  CKern() : updateXused(false), inputDim(0), stationary(false) {}
  CKern(const CMatrix& X) : updateXused(false), inputDim(0), stationary(false) {}
  CKern(unsigned int inDim) : updateXused(false), inputDim(0), stationary(false) {}
  CKern(const CKern& kern)
    : nParams(kern.nParams),
      kernName(kern.kernName),
      type(kern.type),
      paramNames(kern.paramNames),
      activeDims(kern.activeDims),
      updateXused(kern.updateXused),
      inputDim(kern.inputDim),
      stationary(kern.stationary) {}
  
  virtual ~CKern()  {}
  virtual CKern* clone() const=0;
  // set initial parameters.
  virtual void setInitParam()=0;
  // compute an element of the diagonal.
  virtual double diagComputeElement(const CMatrix& X, unsigned int index) const=0;
  // compute the entire diagonal
  virtual void diagCompute(CMatrix& d, const CMatrix& X) const
  {
    DIMENSIONMATCH(X.rowsMatch(d));
    DIMENSIONMATCH(d.getCols()==1);
    for(unsigned int i=0; i<X.getRows(); i++)
      d.setVal(diagComputeElement(X, i), i);
  }
  // Compute the diagonal at particular indices.
  virtual void diagCompute(CMatrix& d, const CMatrix& X, const vector<unsigned int> indices) const
  {
    DIMENSIONMATCH(d.getRows()==indices.size());
    DIMENSIONMATCH(d.getCols()==1);
    for(unsigned int i=0; i<indices.size(); i++)
      d.setVal(diagComputeElement(X, indices[i]), i);
  }
 
  // Set the parameters of the kernel.
  virtual void setParam(double, unsigned int)=0;
  // Get gradients of the kernel with respect to input values.
  //   g[i].val(k,j) = d kern(X_row_i,X2_row_k)/ d x_component_j
  virtual void getGradX(vector<CMatrix*>& gX, const CMatrix& X, const CMatrix& X2, bool addG=false) const
  {
    for(unsigned int i=0; i<X.getRows(); i++)
      getGradX(*gX[i], X, i, X2, addG);
  }
  virtual void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const=0;
  // Get gradients of the kernel diagonal with respect to input values.
  // WVB: I think this is separate from getGradX just because the diagonal vals
  // (i.e. kern(X_row_i,X_row_i) should always be taken from the same matrix X.
  // Not to imply that I understand why we need X and X2 in getGradX, really.
  virtual void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const=0;
  // return the `signal strength' of the kernel.
  virtual double getVariance() const=0;
  // set the `signal strength' of the kernel.
  virtual void setVariance(double val)=0;
  // Return the white noise component of the kernel.
  virtual double getWhite() const
  {
    return 0.0;
  }
  // Compute an element of the kernel matrix.
  virtual double computeElement(const CMatrix& X1, unsigned int index1,
			 const CMatrix& X2, unsigned int index2) const=0;
  // Compute specified rows and columns of the kernel matrix.
  virtual void compute(CMatrix& K, const CMatrix& X1, const vector<unsigned int> indices1,
			  const CMatrix& X2, const vector<unsigned int> indices2) const
  {
    DIMENSIONMATCH(K.getRows()==indices1.size());
    DIMENSIONMATCH(K.getCols()==indices2.size());
    for(unsigned int i=0; i<indices1.size(); i++)
    {
      for(unsigned int j=0; j<indices2.size(); j++)
      {
	BOUNDCHECK(indices1[i]<X1.getRows());
	BOUNDCHECK(indices2[j]<X2.getRows());
	K.setVal(computeElement(X1, indices1[i], X2, indices2[j]), i, j);
      }
    }
  }
  virtual void compute(CMatrix& K, const CMatrix& X, const vector<unsigned int> indices) const
  {
    DIMENSIONMATCH(K.getRows()==indices.size());
    MATRIXPROPERTIES(K.isSquare());
    double k = 0.0;
    for(unsigned int i=0; i<indices.size(); i++)
    {
      for(unsigned int j=0; j<i; j++)
      {
	BOUNDCHECK(indices[i]<X.getRows());
	BOUNDCHECK(indices[j]<X.getRows());
	k = computeElement(X, indices[i], X, indices[j]);
	K.setVal(k, i, j);
	K.setVal(k, j, i);
      }
      K.setVal(diagComputeElement(X, indices[i]), i, i);
    }
  }
  // Compute the kernel matrix for a data set.
  virtual void compute(CMatrix& K, const CMatrix& X) const
  {
    DIMENSIONMATCH(K.rowsMatch(X));
    MATRIXPROPERTIES(K.isSquare());
    double k = 0.0;
    for(unsigned int i=0; i<K.getRows(); i++)
    {
      for(unsigned int j=0; j<i; j++)
      {
	k = computeElement(X, i, X, j);
	K.setVal(k, i, j);
	K.setVal(k, j, i);
      }
      K.setVal(diagComputeElement(X, i), i, i);
    }	      
    K.setSymmetric(true);
  }
  // Compute portions of the kernel matrix.
  virtual void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const
  {
    DIMENSIONMATCH(K.rowsMatch(X));
    DIMENSIONMATCH(K.getCols()==X2.getRows());
    for(unsigned int i=0; i<K.getRows(); i++)
    {
      for(unsigned int j=0; j<K.getCols(); j++)
      {
	K.setVal(computeElement(X, i, X2, j), i, j);
      }
    }	      
  }
  // Compute portions of the kernel matrix.
  virtual void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const
  {
    DIMENSIONMATCH(K.rowsMatch(X));
    DIMENSIONMATCH(K.getCols()==1);
    for(unsigned int i=0; i<K.getRows(); i++)
    {
      K.setVal(computeElement(X, i, X2, row), i, 0);
    }
  }
  // Dummy function to allow CTransformable to be used.
  virtual void getGradParams(CMatrix& g) const
  {
    // This is a dummy function
    cerr << "getGradParams should not be used in CKern" << endl;
    exit(1);
  }
  // Compute the gradient of the kernel matrix with respect to parameters given an additional gradient matrix.
  virtual void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const
  {
    DIMENSIONMATCH(g.getRows()==1);
    DIMENSIONMATCH(g.getCols()==nParams);
    DIMENSIONMATCH(X.getRows()==cvGrd.getRows());
    DIMENSIONMATCH(X2.getRows()==cvGrd.getCols());
    for(unsigned int i=0; i<nParams; i++)
      g.setVal(getGradParam(i, X, X2, cvGrd), i);
    if(regularise)
      addPriorGrad(g);
  }
  virtual void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const
  {
    DIMENSIONMATCH(g.getRows()==1);
    DIMENSIONMATCH(g.getCols()==nParams);
    DIMENSIONMATCH(X.rowsMatch(cvGrd));
    DIMENSIONMATCH(cvGrd.isSquare());
    for(unsigned int i=0; i<nParams; i++)
      g.setVal(getGradParam(i, X, cvGrd), i);
    if(regularise)
      addPriorGrad(g); /// don't forget to add prior gradient at the end.
  }
  virtual void getDiagGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrad, bool regularise=true) const
  {
    //TODO Code explicitly for things like RBF where it will always be zero.
    CMatrix xi(1, X.getCols());
    CMatrix cvGradi(1, 1);
    CMatrix gtemp(1, g.getCols());
    g.zeros();
    for(unsigned int i=0; i<X.getRows(); i++)
    {
      xi.copyRowRow(0, X, i);
      cvGradi.copyRowRow(0, cvGrad, i);
      cvGradi.setSymmetric(true);
      getGradParams(gtemp, xi, cvGradi, regularise);
      g.axpy(gtemp, 1.0);
    }
  }
  // Get gradient of a particular parameter.
  virtual double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const=0;
  virtual double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const=0;
  // Called to indicate the value of X has changed and kernel should do any
  // precomputation it needs to do per value of X.
  virtual void updateX(const CMatrix& X) {}

  // Psi statistics under diagonal Gaussian q(X)=N(mu, diag(S)) ([CIP-0017]).
  // mu, S: N x Q; Z: M x Q. Defaults throw NotImplementedError.
  // psi0 = sum_n E[k(x_n,x_n)]; psi1 = E[K_fu] (N x M); psi2 = sum_n E[k(x_n,Z)k(Z,x_n)^T] (M x M).
  virtual double psi0(const CMatrix& mu, const CMatrix& S) const;
  virtual void psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  virtual void psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  // Gradients of a scalar that depends on (psi0, psi1, psi2).
  // dL_dpsi0: N x 1; dL_dpsi1: N x M; dL_dpsi2: M x M. Outputs gParams (1 x nParams),
  // gZ (M x Q), gMu (N x Q), gS (N x Q). If add, accumulate into existing outputs.
  virtual void getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                          const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                          const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                          bool add=false) const;

  // White-noise inducing cross-covariance (CIP-001C Phase 2).
  // Kfu is N×M; defaults throw NotImplementedError.
  virtual void wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  virtual void wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                            const CMatrix& gKfu) const;
  virtual void wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                       const CMatrix& gKfu) const;
  virtual void wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                       const CMatrix& gKfu) const;
  virtual void wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                       const CMatrix& gKfu) const;

  // Green's / causal-root inducing cross-covariance (CIP-0029). Kfu is N×M; 1-D.
  virtual void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  virtual void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                                const CMatrix& gKfu) const;
  virtual void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                           const CMatrix& gKfu) const;
  virtual void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                           const CMatrix& gKfu) const;
  virtual void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                           const CMatrix& gKfu) const;

  // Weight-space inducing cross-covariance (CIP-0028). Kfu is N×M; W is M×(d+1).
  // Isotropic windows (low-d oracle). Ridge (shipping): V is M×(d+1) unit directions, a is M×1.
  virtual void wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const;
  virtual void wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                            const CMatrix& gKfu) const;
  virtual void wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
                       const CMatrix& gKfu) const;
  virtual void wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
                       const CMatrix& gKfu) const;
  virtual void wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
                       const CMatrix& gKfu) const;

  virtual void wsRidgeCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& V,
                                   const CMatrix& a, double s) const;
  virtual void wsRidgeGradParams(CMatrix& g, const CMatrix& X, const CMatrix& V,
                                 const CMatrix& a, double s, const CMatrix& gKfu) const;
  virtual void wsRidgeGradX(CMatrix& gX, const CMatrix& X, const CMatrix& V,
                            const CMatrix& a, double s, const CMatrix& gKfu) const;
  virtual void wsRidgeGradS(CMatrix& gS, const CMatrix& X, const CMatrix& V,
                            const CMatrix& a, double s, const CMatrix& gKfu) const;
  // Analytic grads w.r.t. unit directions V (M×(d+1)) and offsets a (M×1).
  virtual void wsRidgeGradVA(CMatrix& gV, CMatrix& ga, const CMatrix& X, const CMatrix& V,
                             const CMatrix& a, double s, const CMatrix& gKfu) const;

  virtual bool isUpdateXused() const
  {
    return updateXused;
  }
  virtual void setUpdateXused(bool val) 
  {
    updateXused = val;
  }
  // For compound kernels when a new kernel is added.
  virtual unsigned int addKern(const CKern* kern)
  {
    cerr << "You cannot add a kernel to this kernel." << endl;
    return 0;
  }
  // Get a particular parameter.
  virtual double getParam(unsigned int) const=0;
  // Test if kernel leads to stationary functions.
  bool isStationary() const
  {
    return stationary;
  }
  void setStationary(bool val)
  {
    stationary = val;
  }
  // Set the parameters from a vector of parameters.
  void setParams(const CMatrix& paramVec)
    {
      for(unsigned int i=0; i<nParams; i++)
	setParam(paramVec.getVal(i), i);
    }
  // Place the parameters in a vector.
  void getParams(CMatrix& paramVec) const
    {
      for(unsigned int i=0; i<nParams; i++)
	paramVec.setVal(getParam(i), i);
    }
  // Return a string representing the kernel type.
  inline string getType() const
  {
      return type;
  }
  // Set a string representing the kernel type.
  inline void setType(const string name)
  {
    type = name;
  }
  string getBaseType() const
  {
    return "kern";
  }
  // Get the long name of the kernel.
  inline string getName() const
    {
      return kernName;
    }
  // Set the long name of the kernel.
  inline void setName(const string name)
    {
      kernName = name;
    }
  // Set the input dimension (also resets activeDims to 0..dim-1).
  inline void setInputDim(unsigned int dim)
    {
      inputDim = dim;
      initDefaultActiveDims();
      setInitParam();
    }
  // Get the input dimension (number of columns this kernel reads).
  inline unsigned getInputDim() const
    {
      return inputDim;
    }
  // Columns of a possibly wider X that this kernel reads (GPy active_dims).
  // Size always equals getInputDim(). Default is 0..inputDim-1.
  void setActiveDims(const vector<unsigned int>& dims);
  const vector<unsigned int>& getActiveDims() const { return activeDims; }
  bool isDefaultActiveDims() const;
  // Squared Euclidean distance on activeDims only. Falls back to dist2Row
  // when activeDims is the default and both matrices have inputDim columns.
  double dist2Active(const CMatrix& X1, unsigned int i1,
                     const CMatrix& X2, unsigned int i2) const;
  // Cast a stored double index (CIP-000D): llround + near-integer + range.
  static unsigned int indexFromDouble(double x, unsigned int outputDim);
  // How many kernel parameters are there?
  inline unsigned int getNumParams() const
  {
    return nParams;
  }
  // Assign a name to the kernel parameters.
  void setParamName(const string name, unsigned int index)
    {
      BOUNDCHECK(index<nParams);
      if(paramNames.size() == index)
	paramNames.push_back(name);
      else 
	{
	  if(paramNames.size()<index)
	    paramNames.resize(index+1, "no name");
	  paramNames[index] = name;
	}
    }
  // Get the name of a kernel parameter.
  virtual string getParamName(unsigned int index) const
    {
      BOUNDCHECK(index<paramNames.size());
      return paramNames[index];
    }

  // GPy-aligned path façade (CIP-0014 / REQ-0008). Indexed setParam remains
  // native storage; these methods use GPy names/units at the boundary.
  virtual vector<string> paramPaths() const;
  virtual unsigned int paramIndex(const string& path) const;
  virtual double getByPath(const string& path) const;
  virtual void setByPath(const string& path, double val);
  virtual void constrainPositive(const string& path);
  virtual void constrainZeroOne(const string& path);
  virtual void constrainNone(const string& path);
  // GPy default component name for this kernel type (e.g. Mat32, rbf).
  virtual string gpyComponentName() const;

  // Write out the kernel parameters to a stream.
  virtual void writeParamsToStream(ostream& out) const;
  // Read in the kernel parameters from a stream.
  virtual void readParamsFromStream(istream& in);
  // Get the gradients of the parameters associated with the priors.
  void getGradPrior(CMatrix& g) const;
  // Get the log probabilities associated with the priors.
  void getPriorLogProb(CMatrix& L) const;
  // Display the kernel on an ostream.
  virtual ostream& display(ostream& os) const;
  
#ifdef _NDLMATLAB
  // Create a kernel from an mxArray* object
  CKern(mxArray* kern){}
  // returns an mxArray of the kern for use with matlab.
  virtual mxArray* toMxArray() const;
  virtual void fromMxArray(const mxArray* matlabArray);
  // Adds parameters to the mxArray.
  virtual void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  virtual void extractParamFromMxArray(const mxArray* matlabArray);
#endif /* _NDLMATLAB*/
  // Get the gradient of the transformed parameters.
  void getGradTransParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradTransParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  void getDiagGradTransParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  // specify tests for equality between kernels.
  bool equals(const CKern& kern, double tol=ndlutil::MATCHTOL) const;

 protected:
  unsigned int nParams;
  string kernName;
  string type;
  vector<string> paramNames;
  void initDefaultActiveDims();
  vector<unsigned int> activeDims;
 private:
  bool updateXused;
  unsigned int inputDim;
  bool stationary;
};

// CArdKern is the base class for any kernel that uses multiple input parameters.
class CArdKern : public CKern {
 public:
  CArdKern() : CKern() {}
  CArdKern(const CMatrix& X) : CKern(X) {}
  CArdKern(unsigned int inDim) : CKern(inDim) {}
  CArdKern(const CKern& kern) : CKern(kern) {}
#ifdef _NDLMATLAB
  virtual void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  virtual void extractParamFromMxArray(const mxArray* matlabArray);
  // returns sum(sum(cvGrd.*dK/dparam)) 
#endif
 protected: 
  CMatrix scales;
};

// CIP-0013 P4: GPy Stationary-shaped base. Owns bulk compute via distances + K_of_r.
// Concrete isotropic kernels reparent here and supply radial maps only.
class CStationaryKern : public CKern {
 public:
  CStationaryKern()
    : CKern(), m_wsEpoch(0), m_symEpoch(0), m_rectEpoch(0),
      m_symX(0), m_rectX(0), m_rectX2(0), m_symValid(false), m_rectValid(false)
  { setStationary(true); }
  CStationaryKern(const CMatrix& X)
    : CKern(X), m_wsEpoch(0), m_symEpoch(0), m_rectEpoch(0),
      m_symX(0), m_rectX(0), m_rectX2(0), m_symValid(false), m_rectValid(false)
  { setStationary(true); }
  CStationaryKern(unsigned int inDim)
    : CKern(inDim), m_wsEpoch(0), m_symEpoch(0), m_rectEpoch(0),
      m_symX(0), m_rectX(0), m_rectX2(0), m_symValid(false), m_rectValid(false)
  { setStationary(true); }
  CStationaryKern(const CKern& kern)
    : CKern(kern), m_wsEpoch(0), m_symEpoch(0), m_rectEpoch(0),
      m_symX(0), m_rectX(0), m_rectX2(0), m_symValid(false), m_rectValid(false)
  { setStationary(true); }
  virtual ~CStationaryKern() {}

  // Multiplier turning unscaled Euclidean distance into GPy-style scaled r
  // (RBF: sqrt(inverseWidth)).
  virtual double scaleFactor() const = 0;
  // k(r) with r already lengthscale-scaled (GPy K_of_r).
  virtual double K_of_r(double r) const = 0;

  // CIP-001C Phase 3: Bochner spectral density S(||ω||). Default throws.
  virtual double spectralDensity(double rho) const;
  // Lengthscale for spectral formulas. Default 1/scaleFactor(); Matérn overrides to lengthScale.
  virtual double spectralLengthScale() const;
  // Hankel / cosine path for white-noise cross-cov (N×M Kfu).
  void wnCrossComputeHankel(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  // Default wnCross* for stationary: Hankel (+ FD grads). RBF keeps closed-form overrides.
  void wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                    const CMatrix& gKfu) const;
  void wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;

  double computeElement(const CMatrix& X1, unsigned int index1,
                        const CMatrix& X2, unsigned int index2) const;
  void compute(CMatrix& K, const CMatrix& X) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const;
  // CIP-0013 P5: fill the symmetric (X,X) workspace (GPLVM still calls this
  // after element-wise K). Parameter changes bump the epoch so a stale
  // workspace cannot survive into the next evaluation. Validity is also
  // bound to the X / X2 addresses used when filling (PITC block grads must
  // not reuse an Xu-sized workspace).
  void updateX(const CMatrix& X);
  void invalidateWorkspace();

 protected:
  bool hasSymWorkspace(const CMatrix& X) const;
  bool hasRectWorkspace(const CMatrix& X, const CMatrix& X2) const;
  double symR2(unsigned int i, unsigned int j) const { return m_workR2Sym.getVal(i, j); }
  double symK(unsigned int i, unsigned int j) const { return m_workKSym.getVal(i, j); }
  double rectR2(unsigned int i, unsigned int j) const { return m_workR2Rect.getVal(i, j); }
  double rectK(unsigned int i, unsigned int j) const { return m_workKRect.getVal(i, j); }

 private:
  void fillSymWorkspace(const CMatrix& X) const;
  void fillRectWorkspace(const CMatrix& X, const CMatrix& X2) const;

  // Per-evaluation workspace: unscaled r^2 and K for (X,X) and (X,X2).
  mutable CMatrix m_workR2Sym;
  mutable CMatrix m_workKSym;
  mutable CMatrix m_workR2Rect;
  mutable CMatrix m_workKRect;
  unsigned int m_wsEpoch;
  mutable unsigned int m_symEpoch;
  mutable unsigned int m_rectEpoch;
  mutable const CMatrix* m_symX;
  mutable const CMatrix* m_rectX;
  mutable const CMatrix* m_rectX2;
  mutable bool m_symValid;
  mutable bool m_rectValid;
};

// Component kernel (such as cmpnd or tensor)
class CComponentKern : public CKern 
{
 public:
  CComponentKern() : CKern() {}
  CComponentKern(unsigned int inDim) : CKern(inDim) {}
  CComponentKern(const CMatrix& X) : CKern(X) {}
  CComponentKern(const CComponentKern& kern) : CKern(kern), components(kern.components) {}
  virtual unsigned int addKern(const CKern* kern)
  {
    components.push_back(kern->clone());
    unsigned int oldNParams = nParams;
    nParams+=kern->getNumParams();
    for(size_t i=0; i<kern->getNumTransforms(); i++)
      addTransform(kern->getTransform(i), kern->getTransformIndex(i)+oldNParams);      
    setStationary(isStationary() && kern->isStationary());
    return components.size()-1;
  }
  virtual void setParam(double val, unsigned int paramNo)
  {
    unsigned int start = 0;
    unsigned int end = 0;
    for(size_t i=0; i<components.size(); i++)
    {
      end = start+components[i]->getNumParams()-1;
      if(paramNo <= end)
      {
	components[i]->setParam(val, paramNo-start);
	return;
      }      
      start = end + 1;
    }
  }
  // Parameters are kernel parameters
  virtual double getParam(unsigned int paramNo) const
  {
    unsigned int start = 0;
    unsigned int end = 0;
    for(size_t i=0; i<components.size(); i++)
    {
      end = start+components[i]->getNumParams()-1;
      if(paramNo <= end)
	return components[i]->getParam(paramNo-start);
      start = end + 1;
    }
    return -1;
  }
  virtual string getParamName(unsigned int paramNo) const;
  virtual vector<string> paramPaths() const;
  virtual unsigned int paramIndex(const string& path) const;
  virtual double getByPath(const string& path) const;
  virtual void setByPath(const string& path, double val);
  virtual void constrainPositive(const string& path);
  virtual void constrainZeroOne(const string& path);
  virtual void constrainNone(const string& path);
  // Rebuild parent transform list from children (shared pointers, no delete).
  void syncTransformsFromComponents();
  string componentPathLabel(unsigned int componentIndex) const;
  unsigned int findComponent(const string& comp) const;
  void splitComponentPath(const string& path, unsigned int& componentIndex, string& leaf) const;
  virtual void updateX(const CMatrix& X)
  {
    for(size_t i=0; i<components.size(); i++)
      components[i]->updateX(X);
  }
  virtual void addPrior(CDist* prior, unsigned int index) 
  {
    throw ndlexceptions::Error("Error cannot add priors to component kernels directly, please add to the components.");
  }

  virtual double priorLogProb() const
  {
    double L = 0.0;
    for(unsigned int i=0; i<components.size(); i++)
    {
      L+=components[i]->priorLogProb();
    }
    return L;
  }
  
  virtual void readParamsFromStream(istream& in); 
  virtual void writeParamsToStream(ostream& out) const;
  virtual unsigned int getNumKerns() const
  {
    return components.size();
  }

  
  
#ifdef _NDLMATLAB
  // sets the parameters in the mxArray.
  virtual void addParamToMxArray(mxArray* matlabArray) const;
  // Gets the parameters from the mxArray.
  virtual void extractParamFromMxArray(const mxArray* matlabArray); 
#endif
 protected:
  // this is a heterogeneous container.
  vector<CKern*> components;
  
};
// Compound Kernel --- This kernel combines other kernels additively together.
class CCmpndKern: public CComponentKern {
 public:
  CCmpndKern();
  CCmpndKern(unsigned int inDim);
  CCmpndKern(const CMatrix& X);
  ~CCmpndKern();
  CCmpndKern(const CCmpndKern&);
  CCmpndKern* clone() const
  {
    return new CCmpndKern(*this);
  }
  //CCmpndKern(vector<CKern*> kernels);
  
  double getVariance() const;
  void setVariance(double val)
  {
    double totalVariance = getVariance();
    double factor = val/totalVariance;
    for(size_t i=0; i<components.size(); i++)
    {
      double newVariance = components[i]->getVariance()*factor;
      components[i]->setVariance(newVariance);
    }  
  }
  double getWhite() const;
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index1) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
		 const CMatrix& X2, unsigned int index2) const;
  void compute(CMatrix& K, const CMatrix& X) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double psi0(const CMatrix& mu, const CMatrix& S) const;
  void psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                  const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                  const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                  bool add=false) const;
  void wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                    const CMatrix& gKfu) const;
  void wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                        const CMatrix& gKfu) const;
  void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const;
  void wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                    const CMatrix& gKfu) const;
  void wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsRidgeCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& V,
                           const CMatrix& a, double s) const;
  void wsRidgeGradParams(CMatrix& g, const CMatrix& X, const CMatrix& V,
                         const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradX(CMatrix& gX, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradS(CMatrix& gS, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradVA(CMatrix& gV, CMatrix& ga, const CMatrix& X, const CMatrix& V,
                     const CMatrix& a, double s, const CMatrix& gKfu) const;
private:
  void _init();
};

// Tensor Kernel --- This kernel combines other multiplicitavely together.
class CTensorKern: public CComponentKern {
 public:
  CTensorKern();
  CTensorKern(unsigned int inDim);
  CTensorKern(const CMatrix& X);
  ~CTensorKern();
  CTensorKern(const CTensorKern&);
  CTensorKern* clone() const
  {
    return new CTensorKern(*this);
  }
  CTensorKern(const CTensorKern&, unsigned int i);
  //CTensorKern(vector<CKern*> kernels);
  
  double getVariance() const;
  void setVariance(double val)
  {
    double totalVariance = getVariance();
    double factor = val/totalVariance;
    for(size_t i=0; i<components.size(); i++)
    {
      double newVariance = components[i]->getVariance()*factor;
      components[i]->setVariance(newVariance);
    } 
  }
  double getWhite() const;
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index1) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double computeElement(const CMatrix& X1, unsigned int index1, const CMatrix& X2, unsigned int index2) const;
/*   void compute(CMatrix& K, const CMatrix& X) const; */
/*   void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const; */
/*   void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const; */
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  unsigned int addKern(const CKern* kern);
private:
  void _init();
};

// White Noise Kernel.
class CWhiteKern: public CKern {
 public:
  CWhiteKern();
  CWhiteKern(unsigned int inDim);
  CWhiteKern(const CMatrix& X);
  ~CWhiteKern();
  CWhiteKern(const CWhiteKern&);
  CWhiteKern* clone() const
  {
    return new CWhiteKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void compute(CMatrix& K, const CMatrix& X) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double psi0(const CMatrix& mu, const CMatrix& S) const;
  void psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                  const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                  const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                  bool add=false) const;
  void wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                    const CMatrix& gKfu) const;
  void wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                        const CMatrix& gKfu) const;
  void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const;
  void wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                    const CMatrix& gKfu) const;
  void wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsRidgeCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& V,
                           const CMatrix& a, double s) const;
  void wsRidgeGradParams(CMatrix& g, const CMatrix& X, const CMatrix& V,
                         const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradX(CMatrix& gX, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradS(CMatrix& gS, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradVA(CMatrix& gV, CMatrix& ga, const CMatrix& X, const CMatrix& V,
                     const CMatrix& a, double s, const CMatrix& gKfu) const;

 private:
  void _init();
  double variance;
};
// Whitefixed Noise Kernel.
class CWhitefixedKern: public CKern {
 public:
  CWhitefixedKern();
  CWhitefixedKern(unsigned int inDim);
  CWhitefixedKern(const CMatrix& X);
  ~CWhitefixedKern();
  CWhitefixedKern(const CWhitefixedKern&);
  CWhitefixedKern* clone() const
  {
    return new CWhitefixedKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void compute(CMatrix& K, const CMatrix& X) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);
  ostream& display(ostream& os) const;


 private:
  void _init();
  double variance;
};

// Bias Kernel.
class CBiasKern: public CKern {
 public:
  CBiasKern();
  CBiasKern(unsigned int inDim);
  CBiasKern(const CMatrix& X);
  ~CBiasKern();
  CBiasKern(const CBiasKern&);
  CBiasKern* clone() const
  {
    return new CBiasKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1,
		 const CMatrix& X2, unsigned int index2) const;
  void compute(CMatrix& K, const CMatrix& X) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double psi0(const CMatrix& mu, const CMatrix& S) const;
  void psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                  const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                  const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                  bool add=false) const;
  void wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;

 private:
  void _init();
  double variance;

};

// RBF Kernel, also known as the Gaussian or squared exponential kernel.
class CRbfKern: public CStationaryKern {
 public:
  CRbfKern();
  CRbfKern(unsigned int inDim);
  CRbfKern(const CMatrix& X);
  ~CRbfKern();
  CRbfKern(const CRbfKern&);
  CRbfKern* clone() const
  {
    return new CRbfKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
    invalidateWorkspace();
  }
  void setInverseWidth(double val)
  {
    inverseWidth = val;
    invalidateWorkspace();
  }
  double getInverseWidth() const
  {
    return inverseWidth;
  }
  void setLengthScale(double val)
  {
    inverseWidth = 1/(val*val);
    invalidateWorkspace();
  }
  double getLengthScale() const
  {
    return 1/sqrt(inverseWidth);
  }
  double scaleFactor() const { return sqrt(inverseWidth); }
  double K_of_r(double r) const { return variance * exp(-0.5 * r * r); }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double psi0(const CMatrix& mu, const CMatrix& S) const;
  void psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                  const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                  const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                  bool add=false) const;
  double spectralDensity(double rho) const;
  void wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                    const CMatrix& gKfu) const;
  void wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;

 private:
  void _init();
  double variance;
  double inverseWidth;

};

// Exponentiated quadratic (GPy ExpQuad): same maths as CRbfKern, lengthscale-first
// GPy surface (CIP-0024 / REQ-0012). Storage remains inverseWidth via the RBF engine.
class CEqKern: public CRbfKern {
 public:
  CEqKern();
  CEqKern(unsigned int inDim);
  CEqKern(const CMatrix& X);
  CEqKern(const CEqKern& kern);
  ~CEqKern();
  CEqKern* clone() const
  {
    return new CEqKern(*this);
  }
  string gpyComponentName() const;

 private:
  void _markEq();
};

// Exponential Kernel, written by Karel Lebeda with no guarantees whatsoever...
class CExpKern: public CStationaryKern {
 public:
  CExpKern();
  CExpKern(unsigned int inDim);
  CExpKern(const CMatrix& X);
  ~CExpKern();
  CExpKern(const CExpKern&);
  CExpKern* clone() const
  {
    return new CExpKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
    invalidateWorkspace();
  }
  void setInverseWidth(double val)
  {
    inverseWidth = val;
    invalidateWorkspace();
  }
  double getInverseWidth() const
  {
    return inverseWidth;
  }
  void setLengthScale(double val)
  {
    inverseWidth = 1/(val*val);
    invalidateWorkspace();
  }
  double getLengthScale() const
  {
    return 1/sqrt(inverseWidth);
  }
  double scaleFactor() const { return inverseWidth; }
  double K_of_r(double r) const { return variance * exp(-r); }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double spectralDensity(double rho) const;
  void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                        const CMatrix& gKfu) const;
  void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;

 private:
  void _init();
  double variance;
  double inverseWidth;

};

// Rational Quadratic Kernel
class CRatQuadKern: public CStationaryKern {
 public:
  CRatQuadKern();
  CRatQuadKern(unsigned int inDim);
  CRatQuadKern(const CMatrix& X);
  ~CRatQuadKern();
  CRatQuadKern(const CRatQuadKern&);
  CRatQuadKern* clone() const
  {
    return new CRatQuadKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
    invalidateWorkspace();
  }
  double scaleFactor() const { return 1.0 / lengthScale; }
  double K_of_r(double r) const
  {
    return variance * pow(1.0 + 0.5 * r * r / alpha, -alpha);
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  
 private:
  void _init();
  double variance;
  double alpha;
  double lengthScale;
  
};

// Matern kernel with dof=3/2.
class CMatern32Kern: public CStationaryKern {
 public:
  CMatern32Kern();
  CMatern32Kern(unsigned int inDim);
  CMatern32Kern(const CMatrix& X);
  ~CMatern32Kern();
  CMatern32Kern(const CMatern32Kern&);
  CMatern32Kern* clone() const
  {
    return new CMatern32Kern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
    invalidateWorkspace();
  }
  double scaleFactor() const { return sqrt(3.0) / lengthScale; }
  double K_of_r(double r) const { return variance * (1.0 + r) * exp(-r); }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double spectralDensity(double rho) const;
  double spectralLengthScale() const;
  void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                        const CMatrix& gKfu) const;
  void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  
 private:
  void _init();
  double variance;
  double lengthScale;
  
};

// Matern kernel with dof=5/2.
class CMatern52Kern: public CStationaryKern {
 public:
  CMatern52Kern();
  CMatern52Kern(unsigned int inDim);
  CMatern52Kern(const CMatrix& X);
  ~CMatern52Kern();
  CMatern52Kern(const CMatern52Kern&);
  CMatern52Kern* clone() const
  {
    return new CMatern52Kern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
    invalidateWorkspace();
  }
  double scaleFactor() const { return sqrt(5.0) / lengthScale; }
  double K_of_r(double r) const
  {
    return variance * (1.0 + r + r * r / 3.0) * exp(-r);
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double spectralDensity(double rho) const;
  double spectralLengthScale() const;
  void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                        const CMatrix& gKfu) const;
  void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  
 private:
  void _init();
  double variance;
  double lengthScale;
  
};

// Linear Kernel, also known as the inner product kernel.
class CLinKern: public CKern {
 public:
  CLinKern();
  CLinKern(unsigned int inDim);
  CLinKern(const CMatrix& X);
  ~CLinKern();
  CLinKern(const CLinKern&);
  CLinKern* clone() const
  {
    return new CLinKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void compute(CMatrix& K, const CMatrix& X) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2) const;
  void compute(CMatrix& K, const CMatrix& X, const CMatrix& X2, unsigned int row) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double psi0(const CMatrix& mu, const CMatrix& S) const;
  void psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                  const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                  const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                  bool add=false) const;

 private:
  void _init();
  double variance;
};

// MLP Kernel or arcsin kernel. Based on a multi-layer perceptron with infinite hidden nodes. See Williams (1996) "Computing with Infinite Networks" in NIPS 9.
class CMlpKern: public CKern {
 public:
  CMlpKern();
  CMlpKern(unsigned int inDim);
  CMlpKern(const CMatrix& X);
  ~CMlpKern();
  CMlpKern(const CMlpKern&);
  CMlpKern* clone() const
  {
    return new CMlpKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;

  // CIP-0028: isotropic (oracle) and ridge erf weight-space features.
  void wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const;
  void wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                    const CMatrix& gKfu) const;
  void wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsRidgeCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& V,
                           const CMatrix& a, double s) const;
  void wsRidgeGradParams(CMatrix& g, const CMatrix& X, const CMatrix& V,
                         const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradX(CMatrix& gX, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradS(CMatrix& gS, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradVA(CMatrix& gV, CMatrix& ga, const CMatrix& X, const CMatrix& V,
                     const CMatrix& a, double s, const CMatrix& gKfu) const;

 private:
  void _init();
  double weightVariance;
  double biasVariance;
  double variance;
  mutable CMatrix innerProdi;
  mutable CMatrix innerProdj;
};

// Arc-cosine kernel (Cho & Saul). Order 0 ≈ Williams MLP at infinite weight/bias variance
// (CIP-0028); order 1 is the ReLU / ridge form. Inputs are augmented as x̃=(x,1).
class CArcCosineKern: public CKern {
 public:
  CArcCosineKern();
  CArcCosineKern(unsigned int inDim);
  CArcCosineKern(const CMatrix& X);
  ~CArcCosineKern();
  CArcCosineKern(const CArcCosineKern&);
  CArcCosineKern* clone() const
  {
    return new CArcCosineKern(*this);
  }
  double getVariance() const;
  void setVariance(double val) { variance = val; }
  void setOrder(unsigned int ord);
  unsigned int getOrder() const { return order; }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1,
                        const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);

  // CIP-0028 Phase 2: weight-space (order 0 = no √p; order 1 = smoothed ReLU).
  void wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const;
  void wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                    const CMatrix& gKfu) const;
  void wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;

  // CIP-0028 Phase 2 revised: ridge windows (order 0 erf; order 1 soft-ReLU).
  void wsRidgeCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& V,
                           const CMatrix& a, double s) const;
  void wsRidgeGradParams(CMatrix& g, const CMatrix& X, const CMatrix& V,
                         const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradX(CMatrix& gX, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradS(CMatrix& gS, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradVA(CMatrix& gV, CMatrix& ga, const CMatrix& X, const CMatrix& V,
                     const CMatrix& a, double s, const CMatrix& gKfu) const;

 private:
  void _init();
  double variance;
  unsigned int order; // 0 or 1
};

// Polynomial Kernel, not generally recommended as it `extreme behaviour' outside the region where the argument's absolute value is less than 1.
class CPolyKern: public CKern {
 public:
  CPolyKern();
  CPolyKern(unsigned int inDim);
  CPolyKern(const CMatrix& X);
  ~CPolyKern();
  CPolyKern(const CPolyKern&);
  CPolyKern* clone() const
  {
    return new CPolyKern(*this);
  }
  void setDegree(double val)
  {
    degree = val;
  }
  double getDegree() const
  {
    return degree;
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);
#ifdef _NDLMATLAB
  void addParamToMxArray(mxArray* matlabArray) const;
  void extractParamFromMxArray(const mxArray* matlabArray);
#endif
 private:
  void _init();
  double weightVariance;
  double biasVariance;
  double variance;
  double degree;
  mutable CMatrix innerProdi;
};

// Linear ARD Kernel --- automatic relevance determination version of the linear kernel.
class CLinardKern: public CArdKern {
 public:
  CLinardKern();
  CLinardKern(unsigned int inDim);
  CLinardKern(const CMatrix& X);
  ~CLinardKern();
  CLinardKern(const CLinardKern&);
  CLinardKern* clone() const
  {
    return new CLinardKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addGrad=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise = true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise = true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  
  
 private:
  void _init();
  double variance;
};

// RBF ARD Kernel --- automatic relevance determination of the RBF kernel.
class CRbfardKern: public CArdKern {
 public:
  CRbfardKern();
  CRbfardKern(unsigned int inDim);
  CRbfardKern(const CMatrix& X);
  ~CRbfardKern();
  CRbfardKern(const CRbfardKern&);
  CRbfardKern* clone() const
  {
    return new CRbfardKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addGrad=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  double psi0(const CMatrix& mu, const CMatrix& S) const;
  void psi1(CMatrix& Psi1, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void psi2(CMatrix& Psi2, const CMatrix& mu, const CMatrix& S, const CMatrix& Z) const;
  void getGradPsi(CMatrix& gParams, CMatrix& gZ, CMatrix& gMu, CMatrix& gS,
                  const CMatrix& dL_dpsi0, const CMatrix& dL_dpsi1, const CMatrix& dL_dpsi2,
                  const CMatrix& mu, const CMatrix& S, const CMatrix& Z,
                  bool add=false) const;
  void wnCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void wnGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                    const CMatrix& gKfu) const;
  void wnGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;
  void wnGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
               const CMatrix& gKfu) const;

 private:
  void _init();
  double variance;
  double inverseWidth;
  mutable CMatrix gscales;
};

// MLP ARD Kernel --- automatic relevance determination version of the MLP kernel.
class CMlpardKern: public CArdKern {
 public:
  CMlpardKern();
  CMlpardKern(unsigned int inDim);
  CMlpardKern(const CMatrix& X);
  ~CMlpardKern();
  CMlpardKern(const CMlpardKern&);
  CMlpardKern* clone() const
  {
    return new CMlpardKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addGrad=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;

  // CIP-0028 Phase 1b: isotropic oracle + ridge via whitened x̂ (diagonal Σ).
  void wsCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& W, double s) const;
  void wsGradParams(CMatrix& g, const CMatrix& X, const CMatrix& W, double s,
                    const CMatrix& gKfu) const;
  void wsGradW(CMatrix& gW, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradX(CMatrix& gX, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsGradS(CMatrix& gS, const CMatrix& X, const CMatrix& W, double s,
               const CMatrix& gKfu) const;
  void wsRidgeCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& V,
                           const CMatrix& a, double s) const;
  void wsRidgeGradParams(CMatrix& g, const CMatrix& X, const CMatrix& V,
                         const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradX(CMatrix& gX, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradS(CMatrix& gS, const CMatrix& X, const CMatrix& V,
                    const CMatrix& a, double s, const CMatrix& gKfu) const;
  void wsRidgeGradVA(CMatrix& gV, CMatrix& ga, const CMatrix& X, const CMatrix& V,
                     const CMatrix& a, double s, const CMatrix& gKfu) const;

 private:
  void _init();
  double weightVariance;
  double biasVariance;
  double variance;
  mutable CMatrix innerProdi;
  mutable CMatrix innerProdj;
  mutable CMatrix gscales;
};

// Polynomial ARD Kernel --- automatic relevance determination version of the polynomial kernel.
class CPolyardKern: public CArdKern {
 public:
  CPolyardKern();
  CPolyardKern(unsigned int inDim);
  CPolyardKern(const CMatrix& X);
  ~CPolyardKern();
  CPolyardKern(const CPolyardKern&);
  CPolyardKern* clone() const
  {
    return new CPolyardKern(*this);
  }
  void setDegree(double val)
  {
    degree = val;
  }
  double getDegree() const
  {
    return degree;
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addGrad=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);
  
  
 private:
  void _init();
  double weightVariance;
  double biasVariance;
  double variance;
  double degree;
  mutable CMatrix innerProdi;
  mutable CMatrix innerProdj;
  mutable CMatrix gscales;
};

// Matern 3/2 ARD Kernel --- automatic relevance determination version of the Matern dof=3/2 kernel.
class CMatern32ardKern: public CArdKern {
 public:
  CMatern32ardKern();
  CMatern32ardKern(unsigned int inDim);
  CMatern32ardKern(const CMatrix& X);
  ~CMatern32ardKern();
  CMatern32ardKern(const CMatern32ardKern&);
  CMatern32ardKern* clone() const
  {
    return new CMatern32ardKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addGrad=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  
  
 private:
  void _init();
  double variance;
  double lengthScale;
  mutable CMatrix gscales;
};

// Matern 5/2 ARD Kernel --- automatic relevance determination version of the Matern dof=5/2 kernel.
class CMatern52ardKern: public CArdKern {
 public:
  CMatern52ardKern();
  CMatern52ardKern(unsigned int inDim);
  CMatern52ardKern(const CMatrix& X);
  ~CMatern52ardKern();
  CMatern52ardKern(const CMatern52ardKern&);
  CMatern52ardKern* clone() const
  {
    return new CMatern52ardKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addGrad=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  
  
 private:
  void _init();
  double variance;
  double lengthScale;
  mutable CMatrix gscales;
};

// Rational Quadratic ARD Kernel --- automatic relevance determination version of the rational quadratic kernel.
class CRatQuadardKern: public CArdKern {
 public:
  CRatQuadardKern();
  CRatQuadardKern(unsigned int inDim);
  CRatQuadardKern(const CMatrix& X);
  ~CRatQuadardKern();
  CRatQuadardKern(const CRatQuadardKern&);
  CRatQuadardKern* clone() const
  {
    return new CRatQuadardKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addGrad=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  
  
 private:
  void _init();
  double variance;
  double alpha;
  double lengthScale;
  mutable CMatrix gscales;
};

// Standard Periodic Kernel --- GPy-aligned StdPeriodic kernel (isotropic period/lengthScale, CIP-000C v1).
class CStdPeriodicKern: public CKern {
 public:
  CStdPeriodicKern();
  CStdPeriodicKern(unsigned int inDim);
  CStdPeriodicKern(const CMatrix& X);
  ~CStdPeriodicKern();
  CStdPeriodicKern(const CStdPeriodicKern&);
  CStdPeriodicKern* clone() const
  {
    return new CStdPeriodicKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;

 private:
  void _init();
  double variance;
  double lengthScale;
  double period;
};

// Cosine Kernel --- GPy-aligned stationary Cosine kernel: k(r)=variance*cos(r).
class CCosineKern: public CStationaryKern {
 public:
  CCosineKern();
  CCosineKern(unsigned int inDim);
  CCosineKern(const CMatrix& X);
  ~CCosineKern();
  CCosineKern(const CCosineKern&);
  CCosineKern* clone() const
  {
    return new CCosineKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
    invalidateWorkspace();
  }
  double scaleFactor() const { return 1.0 / lengthScale; }
  double K_of_r(double r) const { return variance * cos(r); }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void diagCompute(CMatrix& d, const CMatrix& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;

 private:
  void _init();
  double variance;
  double lengthScale;
};

// Brownian Motion Kernel --- GPy-aligned Brownian kernel, k(x,x')=variance*min(x,x') for a 1-D, same-signed index.
class CBrownianKern: public CKern {
 public:
  CBrownianKern();
  CBrownianKern(unsigned int inDim);
  CBrownianKern(const CMatrix& X);
  ~CBrownianKern();
  CBrownianKern(const CBrownianKern&);
  CBrownianKern* clone() const
  {
    return new CBrownianKern(*this);
  }
  double getVariance() const;
  void setVariance(double val)
  {
    variance = val;
  }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1, 
			const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                        const CMatrix& gKfu) const;
  void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;

 private:
  void _init();
  double variance;
};

// Integrated Brownian --- I(t)=∫_0^t W(s) ds on a same-signed 1-D index (CIP-0029 Phase 3).
// For 0≤s≤t: k(s,t)=variance*(s² t / 2 − s³ / 6). Causal root g(u)=σ u 1_{u≥0}
// gives smoothed-ramp (soft-ReLU) features under Gaussian windows.
class CIntegratedBrownianKern: public CKern {
 public:
  CIntegratedBrownianKern();
  CIntegratedBrownianKern(unsigned int inDim);
  CIntegratedBrownianKern(const CMatrix& X);
  ~CIntegratedBrownianKern();
  CIntegratedBrownianKern(const CIntegratedBrownianKern&);
  CIntegratedBrownianKern* clone() const
  {
    return new CIntegratedBrownianKern(*this);
  }
  double getVariance() const;
  void setVariance(double val) { variance = val; }
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const;
  double computeElement(const CMatrix& X1, unsigned int index1,
                        const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void greensCrossCompute(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z, double s) const;
  void greensGradParams(CMatrix& g, const CMatrix& X, const CMatrix& Z, double s,
                        const CMatrix& gKfu) const;
  void greensGradZ(CMatrix& gZ, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradX(CMatrix& gX, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;
  void greensGradS(CMatrix& gS, const CMatrix& X, const CMatrix& Z, double s,
                   const CMatrix& gKfu) const;

 private:
  void _init();
  double variance;
};

// Coregionalize Kernel --- GPy Coregionalize: B = W W' + diag(kappa), K_nm = B_{i_n,i_m}.
// inputDim is 1; setActiveDims({q}) when X has the index in column q (CIP-000D).
class CCoregionalizeKern: public CKern {
 public:
  CCoregionalizeKern();
  CCoregionalizeKern(unsigned int outputDim, unsigned int rank=1);
  CCoregionalizeKern(unsigned int inDim, unsigned int outputDim, unsigned int rank);
  CCoregionalizeKern(const CMatrix& X, unsigned int outputDim, unsigned int rank=1);
  ~CCoregionalizeKern();
  CCoregionalizeKern(const CCoregionalizeKern&);
  CCoregionalizeKern* clone() const
  {
    return new CCoregionalizeKern(*this);
  }
  unsigned int getOutputDim() const { return outputDim; }
  unsigned int getRank() const { return rank; }
  void setW(const CMatrix& Win);
  void setKappa(const CMatrix& kappaIn);
  const CMatrix& getB() const { return B; }
  double getVariance() const;
  void setVariance(double val);
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const { return 0.0; }
  double computeElement(const CMatrix& X1, unsigned int index1,
                        const CMatrix& X2, unsigned int index2) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd, bool regularise=true) const;
  void getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& cvGrd, bool regularise=true) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);

 private:
  void _init();
  void _rebuildB();
  unsigned int _indexOf(const CMatrix& X, unsigned int row) const;
  unsigned int outputDim;
  unsigned int rank;
  CMatrix W;      // outputDim x rank
  CMatrix kappa;  // 1 x outputDim
  CMatrix B;      // outputDim x outputDim
};

// Index-white Kernel --- per-output Gaussian noise diagonal: K_nn = sigma^2_{i_n}, off-diag 0.
// GPy MixedNoise marginal form (CIP-000D). Sum outside a tensor product, never inside CTensorKern.
class CIndexWhiteKern: public CKern {
 public:
  CIndexWhiteKern();
  CIndexWhiteKern(unsigned int outputDim);
  CIndexWhiteKern(unsigned int inDim, unsigned int outputDim);
  CIndexWhiteKern(const CMatrix& X, unsigned int outputDim);
  ~CIndexWhiteKern();
  CIndexWhiteKern(const CIndexWhiteKern&);
  CIndexWhiteKern* clone() const
  {
    return new CIndexWhiteKern(*this);
  }
  unsigned int getOutputDim() const { return outputDim; }
  double getVariance() const;
  void setVariance(double val);
  void setInitParam();
  double diagComputeElement(const CMatrix& X, unsigned int index) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  void getGradX(CMatrix& g, const CMatrix& X, unsigned int pointNo, const CMatrix& X2, bool addG=false) const;
  void getDiagGradX(CMatrix& g, const CMatrix& X, bool addG=false) const;
  double getWhite() const { return 0.0; }
  double computeElement(const CMatrix& X1, unsigned int index1,
                        const CMatrix& X2, unsigned int index2) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& cvGrd) const;
  double getGradParam(unsigned int index, const CMatrix& X, const CMatrix& cvGrd) const;
  void writeParamsToStream(ostream& out) const;
  void readParamsFromStream(istream& in);

 private:
  void _init();
  unsigned int _indexOf(const CMatrix& X, unsigned int row) const;
  unsigned int outputDim;
  CMatrix variances; // 1 x outputDim
};

// ICM: spatial.prod(Coregionalize) with GPy active_dims (CIP-000D).
// Takes ownership of clones of spatial and returns a new CTensorKern with inputDim = spatialDim+1.
CTensorKern* createIcmKern(const CKern& spatial, unsigned int numOutputs, unsigned int rank=1);
// LCM: sum of ICMs; takes ownership of nothing (clones each spatial).
CCmpndKern* createLcmKern(const vector<const CKern*>& spatials, unsigned int numOutputs, unsigned int rank=1);

ostream& operator<<(ostream& os, const CKern& A);
void writeKernToStream(const CKern& kern, ostream& out);
CKern* readKernFromStream(istream& in);




#endif
