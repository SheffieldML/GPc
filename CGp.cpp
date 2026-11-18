#include "CGp.h"
#include "CCholAdjoint.h"
#include "CLQAdjoint.h"
#include <cmath>
CGp::CGp() 
  : CMapModel(), CProbabilisticOptimisable()
{
  _init();
}

CGp::CGp(unsigned int q, unsigned int d, 
	 CMatrix* pXin, CMatrix* pyin, 
	 CKern* pkernel, CNoise* pnois, 
	 int approxType, 
	 unsigned int actSetSize, int verbos)
  : CMapModel(), CProbabilisticOptimisable(), pkern(pkernel), pnoise(pnois), pX(pXin), py(pyin)
{
  DIMENSIONMATCH(pXin->getCols()==q);
  DIMENSIONMATCH(pyin->getCols()==d);
  DIMENSIONMATCH(pyin->getRows()==pXin->getRows());

  pnoise->setTarget(pyin);

  _init();
  numActive = actSetSize;
  setApproximationType(approxType);
  if(isSparseApproximation() && numActive<1)
    throw ndlexceptions::Error("Error sparse approximations need active sets of non-zero size.");

  setVerbosity(verbos);
  initKernStoreage();

  setOutputDim(pnoise->getOutputDim());
  setInputDim(pXin->getCols());
  setNumData(pnoise->getNumData());
  initStoreage();
  // set scale and bias.
  scale.setVals(1.0);
  bias.zeros();
  initVals();
  
  switch(approxType) 
  {
  case FTC:
  case DTC:
  case DTCVAR:
  case FITC:
  case PITC:
    // positive constraint on beta
    betaTransform = CTransform::defaultPositive();
    break;
  default:
    throw ndlexceptions::Error("Unrecognised sparse approximation type.");
  }
}
CGp::CGp(CKern* pkernel, CNoise* pnois,
	 CMatrix* pXin, int approxType, 
	 unsigned int actSetSize, int verbos)
  :
  CMapModel(), CProbabilisticOptimisable(),
  pkern(pkernel), pnoise(pnois), pX(pXin)
{
  DIMENSIONMATCH(pXin->getRows()==pnoise->getNumData());
  _init();
  numActive = actSetSize;
  setApproximationType(approxType);
  if(isSparseApproximation() && numActive<1)
    throw ndlexceptions::Error("Error sparse approximations need active sets of non-zero size.");
  setVerbosity(verbos);
  initKernStoreage();

  setOutputDim(pnoise->getOutputDim());
  setInputDim(pXin->getCols());
  setNumData(pnoise->getNumData());
  py = pnoise->py;
  initStoreage();
  // set scale and bias.
  scale.setVals(1.0);
  bias.zeros();
  initVals();
  
  switch(approxType) 
  {
  case FTC:
  case DTC:
  case DTCVAR:
  case FITC:
  case PITC:
    // positive constraint on beta
    betaTransform = CTransform::defaultPositive();
    break;
  default:
    throw ndlexceptions::Error("Unrecognised sparse approximation type.");
  }
}
void CGp::_init()
{
  setBaseType("dataModel");
  setType("gp");
  setName("Gaussian process");
  setOutputScaleLearnt(false);
  setOutputBiasLearnt(false);
  jitter = 1e-6;
  inducingFixed=false;
  betaFixed=false;
  spherical=true;
  optimiseX = false;
  backConstrained = false;
  KupToDate = false;
  sparseApproximation = false;
  sparseLinalgMode = SparseLinalgLQ;
  setApproximationType(FTC);
  numActive = 0;
}

void CGp::initKernStoreage() 
{
  g_param.resize(1, pkern->getNumParams());
  kernParamFixed.assign(pkern->getNumParams(), false);
}

void CGp::initOptimiseXStoreage()
{
  if(isOptimiseX())
  {
    if(!isBackConstrained())
    {
      if(!isSparseApproximation())
      {
	dgKX.resize(getNumData(), getInputDim());
	gKX.resize(getNumData(), getInputDim());
      }
      else
      {
	gKX_uf2.resize(numActive, getInputDim());
      }
      gXorW.resize(getNumData(), getInputDim());
    }
    else
    {
      gXorW.resize(1, pbackModel->getOptNumParams());
    }
  }
}
void CGp::initSparseStoreage()
{
  if (isSparseApproximation()) 
  {
    X_u.resize(numActive, getInputDim());
    if (!isInducingFixed()) 
    {
      gX_u.resize(numActive, getInputDim());
    }
    E.resize(numActive, getOutputDim());
    EET.resize(numActive, numActive);
    AinvEET.resize(numActive, numActive);
    AinvEETAinv.resize(numActive, numActive);
    Ainv.resize(numActive, numActive);
    AinvK_uf.resize(numActive, getNumData());
    EMT.resize(numActive, getNumData());
    AinvEMT.resize(numActive, getNumData());
    A.resize(numActive, numActive);
    K_uu.resize(numActive, numActive);
    invK_uu.resize(numActive, numActive);
    K_uf.resize(numActive, getNumData());
    LcholK.resize(numActive, numActive);
    gK_uf.resize(numActive, getNumData());
    gK_uu.resize(numActive, numActive);
    gKX.resize(numActive, getInputDim());
    dgKX.resize(numActive, getInputDim());
    gKX_uf.resize(getNumData(), getInputDim());
    gBeta.resize(1, 1);
    Alpha.resize(numActive, getOutputDim());
    // CIP-0012 / CIP-001B whitened factors (allocated for all sparse approx).
    B.resize(numActive, getNumData());
    Cmat.resize(numActive, numActive);
    LcholC.resize(numActive, numActive);
    Qlq.resize(numActive, numActive + getNumData());
    logDetC = 0.0;
  }
  else 
  {
    K.resize(getNumData(), getNumData());
    invK.resize(getNumData(), getNumData());
    LcholK.resize(getNumData(), getNumData());    
    covGrad.resize(getNumData(), getNumData());  
    Alpha.resize(getNumData(), getOutputDim());
  }
}
void CGp::initStoreage()
{
  //py->resize(getNumData(), getOutputDim());
  m.resize(getNumData(), getOutputDim());
  setMupToDate(false);

  // Spherical GPs use a shared beta; storage stays full-sized for noise site init.
  if(isSpherical()) 
  {
    beta.resize(getNumData(), getOutputDim());
  }
  else 
  {
    beta.resize(getNumData(), getOutputDim());
  }
  
  scale.resize(1, getOutputDim());
  bias.resize(1, getOutputDim());
  
  // gradient matrices
  if (isOutputScaleLearnt()) 
  {
    g_scaleBias.resize(1, getOutputDim());
  }
  nu.resize(getNumData(), getOutputDim());
  g.resize(getNumData(), getOutputDim());

  initOptimiseXStoreage();
  initSparseStoreage();
   
  // approximation specific allocations
  switch(getApproximationType()) 
  {
  case FTC:
    break;
  case DTC:
    break;
  case DTCVAR:
    gLambda.resize(getNumData(), 1);
    diagK.resize(getNumData(), 1);
    invK_uuK_uf.resize(numActive, getNumData());
    V.resize(numActive, getNumData());  
    break;
  case FITC:
    // Temporary variables for FITC approximation.
    gLambda.resize(getNumData(), 1);
    diagK.resize(getNumData(), 1);
    scaledM.resize(getNumData(), getOutputDim());
    V.resize(numActive, getNumData());  
    Am.resize(numActive, numActive);  
    Lm.resize(numActive, numActive);  
    invLmV.resize(numActive, getNumData());  
    bet.resize(numActive, getOutputDim());
    AinvE.resize(numActive, getOutputDim());
    diagMMT.resize(getNumData(), 1);
    diagQ.resize(getNumData(), 1);
    diagK_ufdAinvplusAinvEETAinvK_fu.resize(getNumData(), 1); 
    invK_uuK_uf.resize(numActive, getNumData());
    invK_uuK_ufDinv.resize(numActive, getNumData());
    invK_uuK_ufDinvQ.resize(numActive, getNumData());
    K_ufdotTimesAinvEMT.resize(numActive, getNumData());
    diagK_ufAinvEMT.resize(getNumData(), 1);
    break;
  case PITC:
    initDefaultBlockEnd();
    {
      const unsigned int nBlocks = getNumBlocks();
      blockK.resize(nBlocks);
      blockD.resize(nBlocks);
      blockDinv.resize(nBlocks);
      blockLcholD.resize(nBlocks);
      logDetD.resize(nBlocks);
      gLambdaBlock.resize(nBlocks);
      pitcKufDinvm.resize(numActive, getOutputDim());
      AinvE.resize(numActive, getOutputDim());
      logDetDSum = 0.0;
      unsigned int start = 0;
      for(unsigned int b = 0; b < nBlocks; b++)
      {
        unsigned int endInc = 0;
        getBlockRange(b, start, endInc);
        const unsigned int bs = endInc - start + 1;
        blockK[b].resize(bs, bs);
        blockD[b].resize(bs, bs);
        blockDinv[b].resize(bs, bs);
        blockLcholD[b].resize(bs, bs);
        gLambdaBlock[b].resize(bs, bs);
      }
    }
    break;
  }    
   
}

void CGp::initDefaultBlockEnd()
{
  // Match GPmat gpCreate for pitc: first block ends at k, then ceil(N/numBlocks).
  const unsigned int N = getNumData();
  unsigned int k = numActive;
  if(k == 0)
    k = 1;
  if(k > N)
    k = N;
  const unsigned int numBlocks = (N + k - 1) / k;
  const unsigned int numPerBlock = (N + numBlocks - 1) / numBlocks;
  blockEnd.resize(numBlocks);
  unsigned int endVal1 = k; // 1-based end
  for(unsigned int i = 0; i < numBlocks; i++)
  {
    blockEnd[i] = endVal1 - 1;
    endVal1 = numPerBlock + endVal1;
    if(endVal1 > N)
      endVal1 = N;
  }
}

void CGp::getBlockRange(unsigned int block, unsigned int& start, unsigned int& endInclusive) const
{
  BOUNDCHECK(block < blockEnd.size());
  start = (block == 0) ? 0 : blockEnd[block - 1] + 1;
  endInclusive = blockEnd[block];
  BOUNDCHECK(endInclusive >= start);
  BOUNDCHECK(endInclusive < getNumData());
}

void CGp::updateM() const 
{
  if(!isMupToDate()) 
  {
    for(unsigned int i=0; i<getOutputDim(); i++) 
    {
      m.copyColCol(i, *py, i);
      m.addCol(i, -bias.getVal(i));
      m.scaleCol(i, 1/scale.getVal(i));
    }
    setMupToDate(true);
  }
}
void CGp::initVals() 
{
  setSpherical(true); // non-spherical paths throw NotImplementedError.
  // maxTries is retained for API symmetry with CMatrix::jitChol; site init uses default tries.
  maxTries = 10;
  for(size_t i=0; i<getNumData(); i++) 
  {
    pnoise->updateSites(m, beta, i, nu, g, i);
  }
  setBetaVals(1e3);
  updateM();
  
  if(isSparseApproximation()) 
  {
    // Randomise selection of active points.
    if(numActive>getNumData())
      throw ndlexceptions::CommandLineError("Number of active points has to be less than number of data.");
    vector<unsigned long> ind = ndlutil::randpermTrunc(getNumData(), numActive);
    sort(ind.begin(), ind.end());
    for(unsigned int i=0; i<ind.size(); i++) 
    {
      X_u.copyRowRow(i, *pX, ind[i]);	
    }
  }
}


void CGp::invalidateCache()
{
  // Clears K and M; each cascades Alpha and AD via the flag setters.
  setKupToDate(false);
  setMupToDate(false);
}

void CGp::updateX()
{
  invalidateCache();
}
string CGp::getNoiseType() const
{
  return pnoise->getType();
}

void CGp::ensureKernFixedSize()
{
  const unsigned int n = pkern->getNumParams();
  if(kernParamFixed.size()!=n)
    kernParamFixed.assign(n, false);
}

void CGp::resolveFixPath(const string& path, int& kind, unsigned int& index) const
{
  string p=path;
  if(p.compare(0, 5, "kern.")==0)
    p=p.substr(5);
  if(p=="beta")
  {
    if(!isSparseApproximation())
      throw ndlexceptions::Error("beta fix path only valid for sparse approximations.");
    kind=1;
    index=0;
    return;
  }
  if(p.compare(0, 6, "scale.")==0)
  {
    if(!isOutputScaleLearnt())
      throw ndlexceptions::Error("scale fix path only valid when output scales are learnt.");
    index=static_cast<unsigned int>(atoi(p.substr(6).c_str()));
    if(index>=getOutputDim())
      throw ndlexceptions::Error("scale index out of range in path '"+path+"'.");
    kind=2;
    return;
  }
  kind=0;
  index=pkern->paramIndex(p);
}

vector<string> CGp::paramPaths() const
{
  vector<string> paths;
  const vector<string> kpaths=pkern->paramPaths();
  paths.reserve(kpaths.size()+getOutputDim()+1);
  for(size_t i=0; i<kpaths.size(); i++)
    paths.push_back(string("kern.")+kpaths[i]);
  if(isSparseApproximation())
    paths.push_back("beta");
  if(isOutputScaleLearnt())
  {
    for(unsigned int j=0; j<getOutputDim(); j++)
      paths.push_back(string("scale.")+ndlstrutil::itoa(j));
  }
  return paths;
}

double CGp::getByPath(const string& path) const
{
  string p=path;
  if(p.compare(0, 5, "kern.")==0)
    p=p.substr(5);
  if(p=="beta")
  {
    if(!isSparseApproximation())
      throw ndlexceptions::Error("beta path only valid for sparse approximations.");
    return getBetaVal();
  }
  if(p.compare(0, 6, "scale.")==0)
  {
    if(!isOutputScaleLearnt())
      throw ndlexceptions::Error("scale path only valid when output scales are learnt.");
    const unsigned int index=static_cast<unsigned int>(atoi(p.substr(6).c_str()));
    if(index>=getOutputDim())
      throw ndlexceptions::Error("scale index out of range in path '"+path+"'.");
    return getScaleVal(index);
  }
  return pkern->getByPath(p);
}

void CGp::setByPath(const string& path, double val)
{
  string p=path;
  if(p.compare(0, 5, "kern.")==0)
    p=p.substr(5);
  if(p=="beta")
  {
    if(!isSparseApproximation())
      throw ndlexceptions::Error("beta path only valid for sparse approximations.");
    setBetaVal(val);
    return;
  }
  if(p.compare(0, 6, "scale.")==0)
  {
    if(!isOutputScaleLearnt())
      throw ndlexceptions::Error("scale path only valid when output scales are learnt.");
    const unsigned int index=static_cast<unsigned int>(atoi(p.substr(6).c_str()));
    if(index>=getOutputDim())
      throw ndlexceptions::Error("scale index out of range in path '"+path+"'.");
    setScaleVal(val, index);
    return;
  }
  pkern->setByPath(p, val);
  setKupToDate(false);
}

void CGp::fixKernParam(unsigned int index)
{
  ensureKernFixedSize();
  BOUNDCHECK(index<kernParamFixed.size());
  kernParamFixed[index]=true;
}

void CGp::unfixKernParam(unsigned int index)
{
  ensureKernFixedSize();
  BOUNDCHECK(index<kernParamFixed.size());
  kernParamFixed[index]=false;
}

bool CGp::isKernParamFixed(unsigned int index) const
{
  if(index>=kernParamFixed.size())
    return false;
  return kernParamFixed[index];
}

void CGp::fixBeta(bool val)
{
  if(val && !isSparseApproximation())
    throw ndlexceptions::Error("Cannot fix beta on a non-sparse GP.");
  betaFixed=val;
}

void CGp::clearFixes()
{
  ensureKernFixedSize();
  for(size_t i=0; i<kernParamFixed.size(); i++)
    kernParamFixed[i]=false;
  betaFixed=false;
  for(size_t i=0; i<scaleParamFixed.size(); i++)
    scaleParamFixed[i]=false;
}

void CGp::fix(const string& path)
{
  int kind=0;
  unsigned int index=0;
  resolveFixPath(path, kind, index);
  if(kind==0)
    fixKernParam(index);
  else if(kind==1)
    fixBeta(true);
  else
  {
    if(scaleParamFixed.size()!=getOutputDim())
      scaleParamFixed.assign(getOutputDim(), false);
    scaleParamFixed[index]=true;
  }
}

void CGp::unfix(const string& path)
{
  int kind=0;
  unsigned int index=0;
  resolveFixPath(path, kind, index);
  if(kind==0)
    unfixKernParam(index);
  else if(kind==1)
    fixBeta(false);
  else
  {
    if(scaleParamFixed.size()!=getOutputDim())
      scaleParamFixed.assign(getOutputDim(), false);
    scaleParamFixed[index]=false;
  }
}

bool CGp::isFixed(const string& path) const
{
  int kind=0;
  unsigned int index=0;
  resolveFixPath(path, kind, index);
  if(kind==0)
    return isKernParamFixed(index);
  if(kind==1)
    return betaFixed;
  if(index>=scaleParamFixed.size())
    return false;
  return scaleParamFixed[index];
}

unsigned int CGp::getOptNumParams() const
{
  int tot = 0;
  for(unsigned int i=0; i<pkern->getNumParams(); i++)
  {
    if(!isKernParamFixed(i))
      tot++;
  }
  
  // for GP-LVM
  if(isOptimiseX())
  {
    if(!isBackConstrained())
    {
      tot += getInputDim()*getNumData();
    }
    else
    {
      throw ndlexceptions::NotImplementedError("Back constraints not yet implemented.");
    }
  }
  // for FITC, DTC, DTCVAR and PITC
  if(isSparseApproximation()) 
  {
    if(!isInducingFixed()) 
    {
      tot += numActive*getInputDim();
    }
    if(!betaFixed)
      tot++; // for beta value.
  }
  // for learning scales.
  if(isOutputScaleLearnt()) 
  {
    for(unsigned int j=0; j<getOutputDim(); j++)
    {
      if(j>=scaleParamFixed.size() || !scaleParamFixed[j])
        tot++;
    }
  }
  return tot;
}

void CGp::getOptParams(CMatrix& param) const
{
  int counter = 0;
  // for GP-LVM
  if(isOptimiseX())
  {
    if(!isBackConstrained())
    {
      for(unsigned int j=0; j<getInputDim(); j++)
      {
	for(unsigned int i=0; i<getNumData(); i++)
	{
	  param.setVal(pX->getVal(i, j), counter);
	  counter++;
	}
      }
    }
    else
    {
      throw ndlexceptions::NotImplementedError("Back constraints not yet implemented.");
    }
  }
  // for FITC, DTC, DTCVAR and PITC
  if(isSparseApproximation()) 
  {
    if(!isInducingFixed()) 
    {
      for(unsigned int j=0; j<getInputDim(); j++) 
      { 
	for(unsigned int i=0; i<numActive; i++) 
	{
	  param.setVal(X_u.getVal(i, j), counter);
	  counter++;
	}
      }
    }
  }
  for(unsigned int i=0; i<pkern->getNumParams(); i++)
  {
    if(isKernParamFixed(i))
      continue;
    param.setVal(pkern->getTransParam(i), counter);
    counter++;
  }
  if(isOutputScaleLearnt()) 
  {
    for(unsigned int j=0; j<getOutputDim(); j++) 
    {
      if(j<scaleParamFixed.size() && scaleParamFixed[j])
        continue;
      param.setVal(getScaleVal(j), counter);
      counter++;
    }
  }
  if(isSparseApproximation()) 
  {
    if(!betaFixed)
    {
      param.setVal(betaTransform->xtoa(beta.getVal(0, 0)), counter);
      counter++;
    }
  }
}

void CGp::setOptParams(const CMatrix& param)
{
  invalidateCache();
  int counter=0;
  if(isOptimiseX())
  {
    if(!isBackConstrained())
    {
      for(unsigned int j=0; j<getInputDim(); j++)
      {
	for(unsigned int i=0; i<getNumData(); i++)
	{
	  pX->setVal(param.getVal(counter), i, j);
	  counter++;
	}
      }
    }
    else
    {
      throw ndlexceptions::NotImplementedError("Back constraints not yet implemented.");
    }
  }
  if(isSparseApproximation()) 
  {
    if(!isInducingFixed()) 
    {
      for(unsigned int j=0; j<getInputDim(); j++) 
      {
        for(unsigned int i=0; i<numActive; i++) 
        {
          X_u.setVal(param.getVal(counter), i, j);
          counter++;
        }
      }
    }
  }
  for(unsigned int i=0; i<pkern->getNumParams(); i++) 
  {
    if(isKernParamFixed(i))
      continue;
    pkern->setTransParam(param.getVal(counter), i);
    counter++;
  }
  if(isOutputScaleLearnt())
  {
    for(unsigned int j=0; j<getOutputDim(); j++) 
    {
      if(j<scaleParamFixed.size() && scaleParamFixed[j])
        continue;
      scale.setVal(param.getVal(counter), j);
      counter++;
    }      
    updateM();
  }
  if(isSparseApproximation()) 
  {
    if(!betaFixed)
    {
      beta.setVal(betaTransform->atox(param.getVal(counter)), 0);
      counter++;
    }
  }
  
}

void CGp::out(CMatrix& yPred, const CMatrix& Xin) const
{
  DIMENSIONMATCH(yPred.getRows()==Xin.getRows());
  CMatrix muTest(yPred.getRows(), yPred.getCols());
  CMatrix varSigmaTest(yPred.getRows(), yPred.getCols());
  posteriorMeanVar(muTest, varSigmaTest, Xin);
  pnoise->out(yPred, muTest, varSigmaTest);
}

void CGp::out(CMatrix& yPred, CMatrix& probPred, const CMatrix& Xin) const
{
  CMatrix muTest(yPred.getRows(), yPred.getCols());
  CMatrix varSigmaTest(yPred.getRows(), yPred.getCols());
  posteriorMeanVar(muTest, varSigmaTest, Xin);
  pnoise->out(yPred, probPred, muTest, varSigmaTest);
}
double CGp::outGradParams(CMatrix& g, const CMatrix &Xin, unsigned int pointNo, unsigned int outputNo) const
{
  throw ndlexceptions::NotImplementedError("outGradParams not yet implemented for CGp.");
}
double CGp::outGradX(CMatrix& g, const CMatrix &Xin, unsigned int pointNo, unsigned int outputNo) const
{
  throw ndlexceptions::NotImplementedError("outGradX not yet implemented for CGp.");
}
void CGp::updateAlpha() const
{ 
  if(!isAlphaUpToDate())
  { 
    updateM();
    updateK();
    updateAD();  
    switch(getApproximationType())
    {
    case FTC:
      if(isSpherical())
      {
	Alpha.deepCopy(m);
	// LcholK stored upper: solve U^T then U (equiv. lower L then L^T).
	Alpha.trsm(LcholK, 1.0, "l", "u", "t", "n");
	Alpha.trsm(LcholK, 1.0, "l", "u", "n", "n");
      }
      else
      {
	throw ndlexceptions::Error("Non-spherical implementations not yet in place for FTC");
      }
      break;
    case DTC:
    case DTCVAR:
      if(isSpherical())
      {
        if(isWhitenedSparseLinalg())
        {
          // Alpha = β U^{-1} C^{-1} B m  (CIP-0012; equiv. A^{-1} K_uf m).
          CMatrix u(numActive, getOutputDim());
          u.gemm(B, m, 1.0, 0.0, "n", "n");
          Alpha.deepCopy(u);
          Alpha.trsm(LcholC, 1.0, "l", "u", "t", "n");
          Alpha.trsm(LcholC, 1.0, "l", "u", "n", "n");
          Alpha.trsm(LcholK, 1.0, "l", "u", "n", "n");
          Alpha.scale(getBetaVal());
        }
        else
        {
	Alpha.gemm(K_uf, m, 1.0, 0.0, "n", "n");
	Alpha.trsm(LcholA, 1.0, "l", "u", "t", "n");
	Alpha.trsm(LcholA, 1.0, "l", "u", "n", "n");
        }
      }
      else
      {
	throw ndlexceptions::Error("Non-spherical implementations not yet in place for DTC");

      }
      break;
    case FITC:
      if(isSpherical())
      {
	//////////////
	scaledM.deepCopy(m);
	for(unsigned int i=0; i<getNumData(); i++)
	  scaledM.scaleRow(i, 1/diagD.getVal(i));
	/////////////
        if(isWhitenedSparseLinalg())
        {
          // Alpha = β U^{-1} C^{-1} B (m./D) with B, LcholC from updateAD().
          CMatrix u(numActive, getOutputDim());
          u.gemm(B, scaledM, 1.0, 0.0, "n", "n");
          Alpha.deepCopy(u);
          Alpha.trsm(LcholC, 1.0, "l", "u", "t", "n");
          Alpha.trsm(LcholC, 1.0, "l", "u", "n", "n");
          Alpha.trsm(LcholK, 1.0, "l", "u", "n", "n");
          Alpha.scale(getBetaVal());
        }
        else
        {
	setADupToDate(false);
	Alpha.gemm(K_uf, scaledM, 1.0, 0.0, "n", "n");
	Alpha.trsm(LcholA, 1.0, "l", "u", "t", "n");
	Alpha.trsm(LcholA, 1.0, "l", "u", "n", "n");
        }
      }
      else
      {
        throw ndlexceptions::Error("Non-spherical implementations not yet in place for FITC");
      }
      break;
    case PITC:
      if (isSpherical()) 
      {
        if(isWhitenedSparseLinalg())
        {
          // Alpha = β U^{-1} C^{-1} u with u = U^{-T} pitcKufDinvm = Σ B_b Dinv m_b.
          Alpha.deepCopy(pitcKufDinvm);
          Alpha.trsm(LcholK, 1.0, "l", "u", "t", "n");
          Alpha.trsm(LcholC, 1.0, "l", "u", "t", "n");
          Alpha.trsm(LcholC, 1.0, "l", "u", "n", "n");
          Alpha.trsm(LcholK, 1.0, "l", "u", "n", "n");
          Alpha.scale(getBetaVal());
        }
        else
        {
        // Alpha := A^{-1} sum_i K_uf(:,ind) Dinv{i} m(ind,:)
        Alpha.deepCopy(pitcKufDinvm);
        Alpha.trsm(LcholA, 1.0, "l", "u", "t", "n");
        Alpha.trsm(LcholA, 1.0, "l", "u", "n", "n");
        }
      }
      else 
      {
        throw ndlexceptions::Error("Non-spherical implementations not yet in place for PITC");
      }
      break;
    }
    setAlphaUpToDate(true);
  }
}
void CGp::_testComputeKx(CMatrix &kX, const CMatrix& Xin) const
{
  if(isSparseApproximation())
  {
    DIMENSIONMATCH(kX.getRows()==X_u.getRows() && kX.getCols()==Xin.getRows());  
    pkern->compute(kX, X_u, Xin);
  }
  else
  {
    DIMENSIONMATCH(kX.getRows()==pX->getRows() && kX.getCols()==Xin.getRows());  
    pkern->compute(kX, *pX, Xin);
  }
}
void CGp::_posteriorMean(CMatrix& mu, const CMatrix& kX) const
{
  updateAlpha();
  DIMENSIONMATCH(mu.getRows()==kX.getCols());
  DIMENSIONMATCH(mu.getCols()==getOutputDim());
  for(unsigned int i=0; i<kX.getCols(); i++)
  {
    for(unsigned int j=0; j<getOutputDim(); j++)
    {
      mu.setVal(Alpha.dotColCol(j, kX, i), i, j);
    }
  }
  // apply output scale and bias.
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    double scaleVal = scale.getVal(j);
    if(scaleVal!=1.0)
    {
      mu.scaleCol(j, scaleVal);
    }
    double biasVal = bias.getVal(j);
    if(biasVal!=0.0)
    {
      mu.addCol(j, biasVal);
    }
  }
}
void CGp::_posteriorVar(CMatrix& varSigma, CMatrix& kX, const CMatrix& Xin) const
{
  updateK();
  updateAD();
  DIMENSIONMATCH(varSigma.getRows()==kX.getCols());
  DIMENSIONMATCH(varSigma.getCols()==getOutputDim());
  // WARNING: destroys kX through in place operations.
  if(isSparseApproximation())
  {
    if(isWhitenedSparseLinalg()
       && isSpherical()
       && (getApproximationType() == DTC
           || getApproximationType() == DTCVAR
           || getApproximationType() == FITC
           || getApproximationType() == PITC))
    {
      // vs = kxx - ||b||^2 + b^T C^{-1} b + β^{-1}, b = U^{-T} k_xu (CIP-0012).
      CMatrix bStar(kX);
      bStar.trsm(LcholK, 1.0, "l", "u", "t", "n");
      CMatrix CinvB(bStar);
      CinvB.trsm(LcholC, 1.0, "l", "u", "t", "n");
      CinvB.trsm(LcholC, 1.0, "l", "u", "n", "n");
      for(unsigned int i = 0; i < kX.getCols(); i++)
      {
        double vsVal = pkern->diagComputeElement(Xin, i)
          - bStar.dotColCol(i, bStar, i)
          + bStar.dotColCol(i, CinvB, i);
        CHECKZEROORPOSITIVE(vsVal >= 0);
        vsVal += 1.0 / getBetaVal();
        for(unsigned int j = 0; j < getOutputDim(); j++)
          varSigma.setVal(vsVal, i, j);
      }
    }
    else
    {
    CMatrix store(kX.getRows(), kX.getCols());
    EET.deepCopy(invK_uu);
    EET.axpy(Ainv, -1.0/getBetaVal());
    EET.setSymmetric(true);
    store.symm(EET, kX, 1.0, 0.0, "l", "l"); 
    for(unsigned int i=0; i<kX.getCols(); i++)
    {
      double vsVal = pkern->diagComputeElement(Xin, i) - kX.dotColCol(i, store, i);
      CHECKZEROORPOSITIVE(vsVal>=0);
      vsVal += 1.0/getBetaVal();
      for(unsigned int j=0; j<getOutputDim(); j++)
      {
	varSigma.setVal(vsVal, i, j);
      }
    }
    }
  }
  else 
  {
    // solve U^T * tmp = kX with upper Cholesky U (equiv. L * tmp = kX)
    kX.trsm(LcholK, 1.0, "L", "U", "T", "N"); // now it is LcholKinv*K
    for(unsigned int i=0; i<kX.getCols(); i++)
    {
      double vsVal = pkern->diagComputeElement(Xin, i) - kX.norm2Col(i);
      CHECKZEROORPOSITIVE(vsVal>=0);
      for(unsigned int j=0; j<getOutputDim(); j++)
      {
        varSigma.setVal(vsVal, i, j);
      }
    }
  }
  
  
  // apply output scale and bias.
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    double scaleVal = scale.getVal(j);
    if(scaleVal!=1.0)
    {
      varSigma.scaleCol(j, scaleVal*scaleVal);
    }
  }
}
void CGp::posteriorMean(CMatrix& mu, const CMatrix& Xin) const
{
  updateAlpha();
  int alRows = 0;
  if(isSparseApproximation())
  {
    alRows = numActive;
  }
  else
  { 
    alRows = getNumData();
  }
  CMatrix kX(alRows, Xin.getRows());
  _testComputeKx(kX, Xin);
  _posteriorMean(mu, kX);
}
void CGp::posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const
{
  DIMENSIONMATCH(mu.getCols()==getOutputDim());
  DIMENSIONMATCH(varSigma.getCols()==getOutputDim());
  DIMENSIONMATCH(mu.getRows()==Xin.getRows());
  DIMENSIONMATCH(varSigma.getRows()==Xin.getRows());
  
  int alRows = 0;
  if(isSparseApproximation())
  {
    alRows = numActive;
  }
  else
  { 
    alRows = getNumData();
  }
  CMatrix kX(alRows, Xin.getRows());
  _testComputeKx(kX, Xin);
  _posteriorMean(mu, kX);
  _posteriorVar(varSigma, kX, Xin); // destroys kX through in place operations.
  
}

// Gradient routines
void CGp::updateCovGradient(unsigned int j, CMatrix& invKm) const
{
  // covGrad := 0.5*(invK * Y(:,j) * Y^t(:,j) * invK - invK)
  
  // (The 'D' factor doesn't need to be on the last term because we add 
  //  D of these covGrad expressions together in a loop in the caller.  
  //  In other words, this is covGrad for a single process only.)
  
  invKm.resize(invK.getRows(), 1);
  invKm.symvColCol(0, invK, m, j, 1.0, 0.0, "u"); // invKm = invK*m(:,index)
  covGrad.deepCopy(invK);                             // covgrad = invK
  covGrad.syr(invKm, -1.0, "u");                 // covGrad -= invKm * invKm^t
  covGrad.scale(-0.5);                           // covGrad *= -0.5
}


void CGp::updateK() const
{
  if(!isKupToDate())
  {
    _updateK();
    _updateInvK();
    setADupToDate(false);
    setKupToDate(true);
  }
}

void CGp::_updateK() const
{
  switch(getApproximationType()) 
  {
  case FTC:
    // CIP-0013 P4: bulk kernel path (CStationaryKern / CCmpndKern::compute).
    pkern->compute(K, *pX);
    break;
  case DTC:
  case DTCVAR:
  case FITC:
  case PITC:
    pkern->compute(K_uu, X_u);
    pkern->compute(K_uf, X_u, *pX);
    break;
  }
  switch(getApproximationType()) 
  {
  case FTC:
    break;
  case DTC:
    break;
  case FITC:
  case DTCVAR:
    for(unsigned int i=0; i< getNumData(); i++) 
    {
      diagK.setVal(pkern->diagComputeElement(*pX, i), i);
    }
    break;
  case PITC:
    for(unsigned int b = 0; b < getNumBlocks(); b++)
    {
      unsigned int start = 0, endInc = 0;
      getBlockRange(b, start, endInc);
      const unsigned int bs = endInc - start + 1;
      for(unsigned int i = 0; i < bs; i++)
      {
        const unsigned int ii = start + i;
        blockK[b].setVal(pkern->diagComputeElement(*pX, ii), i, i);
        for(unsigned int j = 0; j < i; j++)
        {
          const unsigned int jj = start + j;
          const double kValb = pkern->computeElement(*pX, ii, *pX, jj);
          blockK[b].setVal(kValb, i, j);
          blockK[b].setVal(kValb, j, i);
        }
      }
      blockK[b].setSymmetric(true);
    }
    break;
  default:
    break;
  }
  
}

void CGp::_multiplyInvKuu(CMatrix& out, const CMatrix& rhs) const
{
  // out = K_uu^{-1} rhs. Upper Cholesky LcholK = U with K_uu = U^T U.
  if(isWhitenedSparseLinalg())
  {
    out.deepCopy(rhs);
    out.trsm(LcholK, 1.0, "l", "u", "t", "n"); // U^{-T} rhs
    out.trsm(LcholK, 1.0, "l", "u", "n", "n"); // U^{-1} U^{-T} rhs
  }
  else
  {
    out.gemm(invK_uu, rhs, 1.0, 0.0, "n", "n");
  }
}

void CGp::_whitenKuf(CMatrix& Bout, const CMatrix& Kuf) const
{
  // B = U^{-T} Kuf (CIP-0012). LcholK must already be upper triangular.
  Bout.deepCopy(Kuf);
  Bout.trsm(LcholK, 1.0, "l", "u", "t", "n");
}

void CGp::updateAD() const {
  if(!isADupToDate())
  {
    double betaVal = beta.getVal(0);
    switch(getApproximationType()) 
    {
    case FTC:
      break;
    case DTC:
    case DTCVAR:
      if (isSpherical()) 
      {
        if(isWhitenedSparseLinalg())
        {
          // CIP-0012 whitened forward: B = U^{-T} K_uf, C = I + β B B^T
          // (Λ = β^{-1} I ⇒ Λ^{-1} = β). Legacy A = K_uf K_uf^T + β^{-1} K_uu
          // satisfies |A| = |C| |K_uu| / β^M, so log|A|-log|K_uu| = log|C| - M log β.
          // CIP-001B / Seeger et al. arXiv:1710.08717 §3.3: SparseLinalgLQ forms
          // C via LQ of [I, √β B] (BLR trick; no Gram+chol of I+AA^T).
          _whitenKuf(B, K_uf);
          if(isLQSparseLinalg())
          {
            CMatrix Afeat(B);
            Afeat.scale(sqrt(betaVal));
            lqFactorIA(LcholC, logDetC, &Qlq, Afeat);
          }
          else
          {
            Cmat.eye();
            Cmat.syrk(B, betaVal, 1.0, "u", "n");
            Cmat.setSymmetric(true);
            double jitC = LcholC.jitChol(Cmat);
            if(jitC > 1e-2 && getVerbosity() > 2)
              cout << "Warning: jitter of " << jitC << " added to C in whitened updateAD()." << endl;
            logDetC = logDet(LcholC);
          }
          // Legacy A/Ainv retained for dual-path compares and residual callers.
          A.deepCopy(K_uu);
          A.gemm(K_uf, K_uf, 1.0, 1.0 / betaVal, "n", "t");
          A.setSymmetric(true);
          double jit = LcholA.jitChol(A);
          if(jit > 1e-2 && getVerbosity() > 2)
            cout << "Warning: jitter of " << jit << " added to A in updateAD()." << endl;
          logDetA = logDet(LcholA);
          Ainv.setSymmetric(true);
          Ainv.pdinv(LcholA);
          if(getApproximationType() == DTCVAR)
          {
            // q_i = ||B_{:,i}||^2
            diagD.resize(getNumData(), 1);
            for(unsigned int j = 0; j < getNumData(); j++)
            {
              double q = 0.0;
              for(unsigned int i = 0; i < numActive; i++)
              {
                const double bij = B.getVal(i, j);
                q += bij * bij;
              }
              diagD.setVal(getBetaVal() * (diagK.getVal(j) - q), j);
            }
          }
        }
        else
        {
          A.deepCopy(K_uu);
          A.gemm(K_uf, K_uf, 1.0, 1.0/betaVal, "n", "t");
          A.setSymmetric(true);
          double jit = LcholA.jitChol(A);
          if(jit>1e-2)
            if(getVerbosity()>2)
              cout << "Warning: jitter of " << jit << " added to A in updateAD()." << endl;

          logDetA = logDet(LcholA);
          Ainv.setSymmetric(true);
          Ainv.pdinv(LcholA);
          // Keep LcholA upper; consumers use uplo/trans flags (CIP-0005).
          if(getApproximationType()==DTCVAR)
          {
            _multiplyInvKuu(V, K_uf);
            V.dotMultiply(K_uf);

            diagD.deepCopy(diagK);
            diagD.sumCol(V, -1.0, 1.0);
            diagD.scale(getBetaVal());
          }
        }
      }
      else 
      {
        throw ndlexceptions::NotImplementedError("Non-spherical implementations not yet in place for DTC or DTCVAR");
      }
      break;
    case FITC:  
      if (isSpherical()) 
      {
        const double betaValF = getBetaVal();
        if(isWhitenedSparseLinalg())
        {
          // CIP-0012: q_i = ||B_{:,i}||^2 with B = U^{-T} K_uf; Am = β^{-1}I + V V^T
          // with V = B D^{-1/2} (≡ β^{-1} C when C = I + B Λ^{-1} B^T, Λ_i = diagD_i/β).
          _whitenKuf(B, K_uf);
          diagD.resize(getNumData(), 1);
          for(unsigned int j = 0; j < getNumData(); j++)
          {
            double q = 0.0;
            for(unsigned int i = 0; i < numActive; i++)
            {
              const double bij = B.getVal(i, j);
              q += bij * bij;
            }
            diagD.setVal(1.0 + betaValF * (diagK.getVal(j) - q), j);
          }
          V.deepCopy(K_uf);
          scaledM.deepCopy(m);
          for(unsigned int j = 0; j < getNumData(); j++)
          {
            const double d = 1.0 / diagD.getVal(j);
            V.scaleCol(j, d);
            scaledM.scaleRow(j, sqrt(d));
          }
          // Legacy A/Ainv retained for dual-path compares and residual callers.
          A.deepCopy(K_uu);
          A.gemm(K_uf, V, 1.0, 1.0 / betaValF, "n", "t");
          A.setSymmetric(true);
          double jit = LcholA.jitChol(A);
          if(jit > 1e-2 && getVerbosity() > 2)
            cout << "Warning: jitter of " << jit << " added to A in updateAD()." << endl;
          logDetA = logDet(LcholA);
          Ainv.setSymmetric(true);
          Ainv.pdinv(LcholA);

          V.deepCopy(B);
          for(unsigned int j = 0; j < getNumData(); j++)
            V.scaleCol(j, 1.0 / sqrt(diagD.getVal(j)));
          Am.diag(1.0 / betaValF);
          Am.gemm(V, V, 1.0, 1.0, "n", "t");
          Am.setSymmetric(true);
          // C = β Am = I + β V V^T. CIP-001B / Seeger arXiv:1710.08717 §3.3:
          // LQ of [I, √β V] instead of syrk+chol. FITC LL still uses Lm = chol(Am);
          // LcholC/logDetC feed prediction / dual-path.
          if(isLQSparseLinalg())
          {
            CMatrix Afeat(V);
            Afeat.scale(sqrt(betaValF));
            lqFactorIA(LcholC, logDetC, &Qlq, Afeat);
          }
          else
          {
            Cmat.eye();
            Cmat.syrk(V, betaValF, 1.0, "u", "n");
            Cmat.setSymmetric(true);
            double jitC = LcholC.jitChol(Cmat);
            if(jitC > 1e-2 && getVerbosity() > 2)
              cout << "Warning: jitter of " << jitC << " added to C in FITC updateAD()." << endl;
            logDetC = logDet(LcholC);
          }

          jit = Lm.jitChol(Am);
          if(jit > 1e-2 && getVerbosity() > 2)
            cout << "Warning: jitter of " << jit << " added to Am in updateAD()." << endl;
          invLmV.deepCopy(V);
          invLmV.trsm(Lm, 1.0, "l", "u", "t", "n");
          bet.gemm(invLmV, scaledM, 1.0, 0.0, "n", "n");
        }
        else
        {
	_multiplyInvKuu(V, K_uf);
	V.dotMultiply(K_uf);
	diagD.deepCopy(diagK);
	diagD.negate();
	diagD.sumCol(V, 1.0, 1.0);
	diagD.scale(getBetaVal());
	diagD.negate();
	diagD.add(1.0);
	
	V.deepCopy(K_uf);
	double d = 0.0;
	scaledM.deepCopy(m);
	for(unsigned int j=0; j<getNumData(); j++) 
	{
	  d = 1/diagD.getVal(j);
	  V.scaleCol(j, d);
	  scaledM.scaleRow(j, sqrt(d));
	}
	A.deepCopy(K_uu);
	A.gemm(K_uf, V, 1.0, 1.0/getBetaVal(), "n", "t");
	A.setSymmetric(true);
	// This is initially an upper Cholesky.
	double jit = LcholA.jitChol(A);
	if(jit>1e-2)
	  if(getVerbosity()>2)
	    cout << "Warning: jitter of " << jit << " added to A in updateAD()." << endl;

	logDetA = logDet(LcholA);
	Ainv.setSymmetric(true);
	Ainv.pdinv(LcholA);
	// Keep LcholA upper (CIP-0005).
	
	V.deepCopy(K_uf);
	V.trsm(LcholK, 1.0, "l", "u", "t", "n");
	for(unsigned int j=0; j<getNumData(); j++)
	  V.scaleCol(j, 1/sqrt(diagD.getVal(j)));
	Am.diag(1/getBetaVal());
	Am.gemm(V, V, 1.0, 1.0, "n", "t");
	Am.setSymmetric(true);

	jit = Lm.jitChol(Am); // upper triangular Cholesky.
	if(jit>1e-2)
	  if(getVerbosity()>2)
	    cout << "Warning: jitter of " << jit << " added to Am in updateAD()." << endl;

	invLmV.deepCopy(V);
	invLmV.trsm(Lm, 1.0, "l", "u", "t", "n");
	bet.gemm(invLmV, scaledM, 1.0, 0.0, "n", "n");
        }
      }
      else 
      {
	throw ndlexceptions::NotImplementedError("Non-spherical implementations not yet in place for FITC");
      }
      break;
    case PITC:
      if (isSpherical()) 
      {
        const double betaValP = getBetaVal();
        A.deepCopy(K_uu);
        A.scale(1.0 / betaValP);
        A.setSymmetric(true);
        pitcKufDinvm.zeros();
        logDetDSum = 0.0;
        if(isWhitenedSparseLinalg())
          _whitenKuf(B, K_uf);
        for(unsigned int b = 0; b < getNumBlocks(); b++)
        {
          unsigned int start = 0, endInc = 0;
          getBlockRange(b, start, endInc);
          const unsigned int bs = endInc - start + 1;
          CMatrix Kufb(numActive, bs);
          K_uf.getMatrix(Kufb, 0, numActive - 1, start, endInc);
          blockD[b].deepCopy(blockK[b]);
          blockD[b].scale(betaValP);
          if(isWhitenedSparseLinalg())
          {
            // Q_b = B_b^T B_b with B = U^{-T} K_uf (CIP-0012).
            CMatrix Bb(numActive, bs);
            B.getMatrix(Bb, 0, numActive - 1, start, endInc);
            blockD[b].gemm(Bb, Bb, -betaValP, 1.0, "t", "n");
          }
          else
          {
            CMatrix Qu(numActive, bs);
            Qu.gemm(invK_uu, Kufb, 1.0, 0.0, "n", "n");
            blockD[b].gemm(Kufb, Qu, -betaValP, 1.0, "t", "n");
          }
          blockD[b].addDiag(1.0);
          blockD[b].setSymmetric(true);
          double jitD = blockLcholD[b].jitChol(blockD[b]);
          if(jitD > 1e-2 && getVerbosity() > 2)
            cout << "Warning: jitter of " << jitD << " added to PITC D block." << endl;
          logDetD[b] = logDet(blockLcholD[b]);
          logDetDSum += logDetD[b];
          blockDinv[b].setSymmetric(true);
          blockDinv[b].pdinv(blockLcholD[b]);
          CMatrix KufDinv(numActive, bs);
          KufDinv.gemm(Kufb, blockDinv[b], 1.0, 0.0, "n", "n");
          A.gemm(KufDinv, Kufb, 1.0, 1.0, "n", "t");
          CMatrix mb(bs, getOutputDim());
          m.getMatrix(mb, start, endInc, 0, getOutputDim() - 1);
          CMatrix Dinvm(bs, getOutputDim());
          Dinvm.symm(blockDinv[b], mb, 1.0, 0.0, "l", "l");
          pitcKufDinvm.gemm(Kufb, Dinvm, 1.0, 1.0, "n", "n");
        }
        A.setSymmetric(true);
        double jitA = LcholA.jitChol(A);
        if(jitA > 1e-2 && getVerbosity() > 2)
          cout << "Warning: jitter of " << jitA << " added to A in PITC updateAD()." << endl;
        logDetA = logDet(LcholA);
        Ainv.setSymmetric(true);
        Ainv.pdinv(LcholA);
        if(isWhitenedSparseLinalg())
        {
          // C = I + β Σ_b B_b D_b^{-1} B_b^T; |A| = |C| |K_uu| / β^M.
          Cmat.eye();
          for(unsigned int b = 0; b < getNumBlocks(); b++)
          {
            unsigned int start = 0, endInc = 0;
            getBlockRange(b, start, endInc);
            const unsigned int bs = endInc - start + 1;
            CMatrix Bb(numActive, bs);
            B.getMatrix(Bb, 0, numActive - 1, start, endInc);
            CMatrix W(numActive, bs);
            W.gemm(Bb, blockDinv[b], 1.0, 0.0, "n", "n");
            Cmat.gemm(W, Bb, betaValP, 1.0, "n", "t");
          }
          Cmat.setSymmetric(true);
          double jitC = LcholC.jitChol(Cmat);
          if(jitC > 1e-2 && getVerbosity() > 2)
            cout << "Warning: jitter of " << jitC << " added to C in PITC updateAD()." << endl;
          logDetC = logDet(LcholC);
        }
      }
      else 
      {
	throw ndlexceptions::NotImplementedError("Non-spherical implementations not yet in place for PITC");
      }
      break;
    }
    setADupToDate(true);
  }
}
// update invK with the inverse of the kernel plus beta terms computed from the active points.
void CGp::_updateInvK(unsigned int dim) const
{
  double jit = 0.0;
  switch(getApproximationType()) {
  case FTC:
    jit = LcholK.jitChol(K); // this will initially be upper triangular.
    if(jit>1e-2)
      if(getVerbosity()>2)
	cout << "Warning: jitter of " << jit << " added to K in _updateInvK()." << endl;

    logDetK = logDet(LcholK);
    invK.setSymmetric(true);
    invK.pdinv(LcholK);
    // Keep LcholK upper (CIP-0005).
    break;
  case DTC:
  case DTCVAR:
  case FITC:
  case PITC:
    jit = LcholK.jitChol(K_uu); // this will initially be upper triangular.
    if(jit>1e-2)
      if(getVerbosity()>2)
	cout << "Warning: jitter of " << jit << " added to K_uu in _updateInvK()." << endl;


    logDetK_uu = logDet(LcholK);
    invK_uu.setSymmetric(true);
    invK_uu.pdinv(LcholK);
    // Keep LcholK upper (CIP-0005).
    break;
  }
}


// compute the approximation to the log likelihood.
double CGp::logLikelihood() const
{
  ensureMUpToDate();
  ensureKUpToDate();
  ensureADUpToDate();
  double L=0.0;
  switch(getApproximationType()) 
  {
  case FTC:
    if(isSpherical()) 
    {
      CMatrix invKm(invK.getRows(), 1);
      for(unsigned int j=0; j<getOutputDim(); j++) 
      {
	// This computes trace(invK*M*M'), column by column of M
	// invKm := invK * m(:,j)
	invKm.symvColCol(0, invK, m, j, 1.0, 0.0, "u");
	// L += invKm' * m(:,j)
	L += invKm.dotColCol(0, m, j);
	L += logDetK; 
      }
    }
    else 
    {
      throw ndlexceptions::Error("Non-spherical implementations not yet in place");
    }
    break;
  case DTC:    
  case DTCVAR:
    if(isSpherical()) 
    {
      CMatrix e(numActive, 1);
      CMatrix invAe(numActive, 1);
      // CIP-0012: StableFactor uses C = I+βBB^T; log|A|-log|K_uu| = log|C|-M log β.
      const double logDetTerm = isWhitenedSparseLinalg()
        ? (logDetC - (double)numActive * log(getBetaVal()))
        : (-logDetK_uu + logDetA);
      L+= (double)getOutputDim()*
      (((double)numActive-(double)getNumData())*log(getBetaVal())
       + logDetTerm);
      for(unsigned int j=0; j<getOutputDim(); j++) 
      {
	e.gemvColCol(0, K_uf, m, j, 1.0, 0.0, "n");
	invAe.symvColCol(0, Ainv, e, 0, 1.0, 0.0, "l");
	L -= getBetaVal()*(invAe.dotColCol(0, e, 0)-m.dotColCol(j, m, j));
      }	  
      if(getApproximationType()==DTCVAR)
	L += (double)getOutputDim()*diagD.sum();
    }
    // Non-spherical: updateAD() throws before this switch (CIP-000B / CIP-0011).
    break;
    
  case FITC:
    if(isSpherical()) 
    {
      // 2π term applied once below with other approximations (do not add
      // N*LOGTWOPI here — that double-counted after L*=-0.5 and the common
      // d*N*HALFLOGTWOPI tail; backlog 2026-11-15_fitc-logtwopi-double-count).
      L+=((double)numActive - (double)getNumData())*log(getBetaVal());
      for(unsigned int i=0; i<getNumData(); i++) 
      {
	L += log(diagD.getVal(i));
      }
      double temp = 0.0;
      for(unsigned int i=0; i<numActive; i++) 
      {
	temp += log(Lm.getVal(i, i));
      }
      L += temp*2.0;
      L *= (double)getOutputDim();
      for(unsigned int j=0; j<getOutputDim(); j++) 
      {
	L+= getBetaVal()*(scaledM.dotColCol(j, scaledM, j)-bet.dotColCol(j, bet, j));
	
      }
    }
    break;

  case PITC:
    if(isSpherical()) 
    {
      // GPmat gpLogLikelihood pitc (spherical, scalar beta).
      const double dOut = (double)getOutputDim();
      const double betaValP = getBetaVal();
      // StableFactor: log|A|-log|K_uu| = log|C|-M log β ⇒ first term is d log|C|.
      if(isWhitenedSparseLinalg())
        L = dOut * logDetC;
      else
        L = dOut * (logDetA - logDetK_uu + (double)numActive * log(betaValP));
      CMatrix AinvKufDinvm(numActive, getOutputDim());
      AinvKufDinvm.symm(Ainv, pitcKufDinvm, 1.0, 0.0, "l", "l");
      for(unsigned int j = 0; j < getOutputDim(); j++)
        L -= betaValP * AinvKufDinvm.dotColCol(j, pitcKufDinvm, j);
      for(unsigned int b = 0; b < getNumBlocks(); b++)
      {
        unsigned int start = 0, endInc = 0;
        getBlockRange(b, start, endInc);
        const unsigned int bs = endInc - start + 1;
        L += dOut * (logDetD[b] - (double)bs * log(betaValP));
        CMatrix mb(bs, getOutputDim());
        m.getMatrix(mb, start, endInc, 0, getOutputDim() - 1);
        CMatrix Dinvm(bs, getOutputDim());
        Dinvm.symm(blockDinv[b], mb, 1.0, 0.0, "l", "l");
        for(unsigned int j = 0; j < getOutputDim(); j++)
          L += betaValP * Dinvm.dotColCol(j, mb, j);
      }
      // 2π term applied once below with other approximations.
    }
    break;
    
  }
  if(isOutputScaleLearnt()) 
  {
    // scales lead to terms of the form log w_j to be added.
    for(unsigned int j=0; j<getOutputDim(); j++) 
    {
      L+=2*log(fabs(scale.getVal(j)));
    }
  }
  L*=-0.5;
  L+=pkern->priorLogProb();
  L-=(double)getOutputDim()*(double)getNumData()*ndlutil::HALFLOGTWOPI;
  return L;
}
// compute the gradients wrt parameters and latent variables.
double CGp::logLikelihoodGradient(CMatrix& g) const 
{
  ensureMUpToDate();
  updateG();
  g.zeros();
  int counter = 0;
  if(isOptimiseX())
  {
    if(!isBackConstrained())
    {
      for(unsigned int j=0; j<getInputDim(); j++)
      {
	for(unsigned int i=0; i<getNumData(); i++)
      	{
          g.setVal(gXorW.getVal(i, j), 0, counter);
          counter++;
        }
      }
    }
    else
    {
      for(unsigned int i=0; i<pbackModel->getOptNumParams(); i++)
      {
        g.setVal(gXorW.getVal(0, i), 0, counter);
      }
      throw ndlexceptions::NotImplementedError("Back constraints not yet implemented.");
    }
  }
  if(isSparseApproximation()) 
  {
    if(!isInducingFixed()) 
    {
      for(unsigned int j=0; j<getInputDim(); j++) 
      {
        for(unsigned int i=0; i<numActive; i++) 
        {
          g.setVal(gX_u.getVal(i, j), 0, counter);
          counter++;
        }
      }
    }
  }
  for(unsigned int i=0; i<pkern->getNumParams(); i++) 
  {
    if(isKernParamFixed(i))
      continue;
    g.setVal(g_param.getVal(0, i), 0, counter);
    counter++;
  }
  if(isOutputScaleLearnt()) 
  {
    for(unsigned int i=0; i<getOutputDim(); i++) 
    {
      if(i<scaleParamFixed.size() && scaleParamFixed[i])
        continue;
      g.setVal(g_scaleBias.getVal(0, i), 0, counter);
      counter++;
    }
  }
  if(isSparseApproximation()) 
  {
    if(!betaFixed)
    {
      double gBetaVal = gBeta.getVal(0);
      g.setVal(gBetaVal*betaTransform->gradfact(getBetaVal(0,0)), 0, counter);
      counter++;
    }
  }
  //setADupToDate(false);
  return logLikelihood();

}
void CGp::updateG() const 
{
  if(!isMupToDate())
    throw ndlexceptions::Error("updateG() called when M is not updated.");
  unsigned int numKernParams = pkern->getNumParams();
  unsigned int numParams = numKernParams;
  
  ensureKUpToDate();
  ensureADUpToDate();
  CMatrix tempG(1, numKernParams);
  CMatrix tempG2(1, numKernParams);
  CMatrix tmpV(getOutputDim(), 1);
  g_param.zeros();

  switch(getApproximationType()) 
  {
  case FTC:
    if(isOptimiseX()) // test if GP-LVM is used.
    {
      gXorW.zeros();
      pkern->getDiagGradX(dgKX, *pX);
    }
    for(unsigned int j=0; j<getOutputDim(); j++) 
    {
      updateCovGradient(j, tmpV); //covGrad = -(invK Y(:,j) Y(:,j)^t invK - invK)/2
      if(j==0) 
      {
        pkern->getGradTransParams(tempG, *pX, covGrad, true);
      }
      else 
      {
        pkern->getGradTransParams(tempG, *pX, covGrad, false);
      }
      for(unsigned int i=0; i<numKernParams; i++) 
      {
        g_param.addVal(tempG.getVal(i), i);
      }

      if(isOptimiseX())
      {
	for(unsigned int i=0; i<getNumData(); i++)
	{
	  pkern->getGradX(gKX, *pX, i, *pX); // gX[1...ndata].val(1:ndata,1:latentdim)
	  gKX.scale(2.0); // accounts for symmetric covariance
	  for(unsigned int l=0; l<getInputDim(); l++)
	  {
	    // deal with diagonal 
	    gKX.setVal(dgKX.getVal(i, l), i, l);
	  }
	  if(!isBackConstrained())
	  {
	    for(unsigned int k=0; k<getInputDim(); k++)
	    {
	      int ind = i + getNumData()*k;
	      gXorW.addVal(gKX.dotColCol(k, covGrad, i), ind);
	    }
	  }
	  else
	  {
	    throw ndlexceptions::NotImplementedError("Back constraints not yet implemented.");
	  }
	}
      }
    }
    break;
    
  case DTC:
  case DTCVAR:
  case FITC:
  case PITC:
    gpCovGrads();
    pkern->getGradTransParams(tempG, X_u, gK_uu, true);
    // should we regularize here? I.e. should the true at the end be false?
    pkern->getGradTransParams(tempG2, X_u, *pX, gK_uf, true);
    for(unsigned int i=0; i<numKernParams; i++) 
    {
      g_param.addVal(tempG.getVal(i), i);
      g_param.addVal(tempG2.getVal(i), i);
    }
    if(!isInducingFixed()) 
    {
      // Compute gradients with respect to X_u
      pkern->getDiagGradX(dgKX, X_u);
      // compute portion associated with gK_uf
      for(unsigned int i=0; i<numActive; i++) 
      {
        pkern->getGradX(gKX_uf, X_u, i, *pX);
        pkern->getGradX(gKX, X_u, i, X_u);
        gKX.scale(2.0); // accounts for fact that covGrad is symmetric.
        gKX.copyRowRow(i, dgKX, i);
        gX_u.resize(numActive, getInputDim());
        for(unsigned int j=0; j<getInputDim(); j++) 
        {
          gX_u.setVal(gKX.dotColCol(j, gK_uu, i), i, j);
        }
        for(unsigned int j=0; j<getInputDim(); j++) 
        {
          gX_u.addVal(gKX_uf.dotColRow(j, gK_uf, i), i, j); 
        }
      }
      if(isOptimiseX())
      {
	// for FITC and PITC need extra terms (see matlab code). This
	// is sufficient for DTC and DTCVAR.
	for(unsigned int i=0; i<getNumData(); i++)
	{
	  pkern->getGradX(gKX_uf2, *pX, i, X_u);
	  for(unsigned int j=0; j<getInputDim(); j++)
	  {
	    gXorW.setVal(gKX_uf2.dotColCol(j, gK_uf, i), i, j);
	  }
	}
      }
    }
    break;
  }
  switch(getApproximationType()) 
  {
  case FTC:
    break;    
  case DTC:
    break;
  case FITC:
  case DTCVAR:
    // deal with diagonal term's affect on kernel parameters.
    // false here stops gradients of priors being added ...
    pkern->getDiagGradTransParams(tempG, *pX, gLambda, false);
    for(unsigned int i=0; i<numKernParams; i++) 
    {
      g_param.addVal(tempG.getVal(i), i);
    }	
    if(isOptimiseX())
    {
      // using gKX_uf here as it is correct storeage size, really computing gKXdiag though!!
      pkern->getDiagGradX(gKX_uf, *pX);
      for(unsigned int i=0; i<getNumData(); i++)
      {
	for(unsigned int j=0; j<getInputDim(); j++)
	{
	  gXorW.addVal(gLambda.getVal(i)*gKX_uf.getVal(i, j), i, j);
	}
      }
    }
    break;
  case PITC:
    // Block-diagonal Lambda terms contribute to kernel parameter gradients.
    for(unsigned int b = 0; b < getNumBlocks(); b++)
    {
      unsigned int start = 0, endInc = 0;
      getBlockRange(b, start, endInc);
      const unsigned int bs = endInc - start + 1;
      CMatrix Xb(bs, getInputDim());
      pX->getMatrix(Xb, start, endInc, 0, getInputDim() - 1);
      pkern->getGradTransParams(tempG, Xb, gLambdaBlock[b], false);
      for(unsigned int i = 0; i < numKernParams; i++)
        g_param.addVal(tempG.getVal(i), i);
    }
    if(isOptimiseX())
      throw ndlexceptions::NotImplementedError("Optimisation of X not yet implemented for PITC sparse approximations");
    break;
  }
  
  if(isOutputScaleLearnt()) 
  {
    CMatrix &invKm = tmpV;  tmpV.resize(m.getRows(), 1);
    //  Need to fix this for FITC DTC etc.
    for(unsigned int j=0; j<getOutputDim(); j++) 
    {
      // recomputing this again is inefficient (already done in the likelihood).
      invKm.symvColCol(0, invK, m, j, 1.0, 0.0, "u");      
      g_scaleBias.setVal(1/scale.getVal(j)*(invKm.dotColCol(0, m, j)-1), j);
    }
  }
}

void CGp::gpCovGrads() const
{
  switch(getApproximationType()) 
  {
  case FTC:
    throw ndlexceptions::Error("gpCovGrads should not be called for FTC models.");
    break;
    
  case DTC:
  case DTCVAR:
    if(isSpherical())
    {
      if(isLQSparseLinalg())
      {
        // CIP-001B: reverse through LQ of W=[I,A], A=√β B (Seeger arXiv:1710.08717 §4.3).
        // Seeds: log|C| on L; data-fit via c=L^{-1} A m; DTCVAR on A; then lqAdjoint.
        const double betaVal = getBetaVal();
        const double sqrtBeta = sqrt(betaVal);
        const double dOut = (double)getOutputDim();
        const unsigned int Mact = numActive;
        const unsigned int Ndat = getNumData();
        const bool dtcvar = (getApproximationType() == DTCVAR);

        CMatrix Afeat(B);
        Afeat.scale(sqrtBeta);

        CMatrix Llow(Mact, Mact);
        Llow.zeros();
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j <= i; j++)
            Llow.setVal(LcholC.getVal(j, i), i, j);
        Llow.setTriangular(true);

        CMatrix Lbar(Mact, Mact);
        Lbar.zeros();
        CMatrix Abar(Mact, Ndat);
        Abar.zeros();

        // LL contains -½ dOut logDetC with logDetC = 2 Σ log L_ii.
        for(unsigned int i = 0; i < Mact; i++)
        {
          const double di = Llow.getVal(i, i);
          Lbar.setVal(Lbar.getVal(i, i) - dOut / di, i, i);
        }

        // Data fit: ||c||² = e^T Ainv e with c = L^{-1} A m; final LL += ½ β (||c||² - ||m||²).
        CMatrix z(Mact, 1);
        CMatrix c(Mact, 1);
        CMatrix barz(Mact, 1);
        for(unsigned int j = 0; j < getOutputDim(); j++)
        {
          z.gemvColCol(0, Afeat, m, j, 1.0, 0.0, "n");
          c.deepCopy(z);
          c.trsm(Llow, 1.0, "l", "l", "n", "n"); // c = L^{-1} z
          // bar c = β c; bar z = L^{-T} bar c; Lbar -= barz c^T (lower); Abar += barz m^T
          barz.deepCopy(c);
          barz.scale(betaVal);
          barz.trsm(Llow, 1.0, "l", "l", "t", "n");
          for(unsigned int i = 0; i < Mact; i++)
            for(unsigned int k = 0; k <= i; k++)
              Lbar.setVal(Lbar.getVal(i, k) - barz.getVal(i) * c.getVal(k), i, k);
          Abar.gerColCol(barz, 0, m, j, 1.0);
        }

        if(dtcvar)
        {
          // Final LL += ½ dOut ||A||_F² + β-only pieces from diagD.
          Abar.axpy(Afeat, dOut);
          gLambda.setVals(-0.5 * dOut * betaVal);
        }

        CMatrix Qbar(Mact, Mact + Ndat);
        Qbar.zeros();
        CMatrix Wbar;
        lqAdjoint(Wbar, Llow, Qlq, Lbar, Qbar, false);
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j < Ndat; j++)
            Abar.setVal(Abar.getVal(i, j) + Wbar.getVal(i, Mact + j), i, j);

        // A = √β B ⇒ Bbar = √β Abar.
        CMatrix Bbar(Abar);
        Bbar.scale(sqrtBeta);

        // ∂LL/∂β closed form (same -½ gbetaM2L as CIP-0012 StableFactor; C^{-1} via trsm on LQ R).
        CMatrix CinvB(B);
        CinvB.trsm(LcholC, 1.0, "l", "u", "t", "n");
        CinvB.trsm(LcholC, 1.0, "l", "u", "n", "n");
        double trCinvBB = 0.0;
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j < Ndat; j++)
            trCinvBB += CinvB.getVal(i, j) * B.getVal(i, j);
        double gbetaM2L = dOut * (trCinvBB - (double)Ndat / betaVal);
        CMatrix u(Mact, 1);
        CMatrix a(Mact, 1);
        CMatrix BTa(Ndat, 1);
        for(unsigned int j = 0; j < getOutputDim(); j++)
        {
          u.gemvColCol(0, B, m, j, 1.0, 0.0, "n");
          a.deepCopy(u);
          a.trsm(LcholC, 1.0, "l", "u", "t", "n");
          a.trsm(LcholC, 1.0, "l", "u", "n", "n");
          BTa.gemvColCol(0, B, a, 0, 1.0, 0.0, "t");
          const double uta = u.dotColCol(0, a, 0);
          const double BTaNorm2 = BTa.dotColCol(0, BTa, 0);
          gbetaM2L += m.dotColCol(j, m, j)
            - 2.0 * betaVal * uta
            + betaVal * betaVal * BTaNorm2;
        }
        if(dtcvar)
        {
          double bFro2 = 0.0;
          for(unsigned int i = 0; i < Mact; i++)
            for(unsigned int j = 0; j < Ndat; j++)
              bFro2 += B.getVal(i, j) * B.getVal(i, j);
          gbetaM2L += dOut * (diagK.sum() - bFro2);
        }
        gBeta.setVal(-0.5 * gbetaM2L, 0);

        CMatrix Ubar;
        CMatrix KufBar;
        trsmAdjointUT(KufBar, Ubar, LcholK, B, Bbar);
        gK_uf.deepCopy(KufBar);

        CMatrix KuuBar;
        cholAdjointUpperBlocked(KuuBar, LcholK, Ubar);
        gK_uu.deepCopy(KuuBar);
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j < i; j++)
          {
            const double v = 0.5 * gK_uu.getVal(i, j);
            gK_uu.setVal(v, i, j);
            gK_uu.setVal(v, j, i);
          }
        gK_uu.setSymmetric(true);
      }
      else if(isWhitenedSparseLinalg())
      {
        // CIP-0012 reverse mode on C = I + β B B^T (formulae for -2L, then × -1/2 for L).
        const double betaVal = getBetaVal();
        const double dOut = (double)getOutputDim();
        const unsigned int Mact = numActive;
        const unsigned int Ndat = getNumData();
        const bool dtcvar = (getApproximationType() == DTCVAR);

        CMatrix Cinv(Mact, Mact);
        Cinv.setSymmetric(true);
        Cinv.pdinv(LcholC);

        CMatrix CinvB(Mact, Ndat);
        CinvB.symm(Cinv, B, 1.0, 0.0, "u", "l");

        // barB = ∂(-2L)/∂B
        CMatrix Bbar(Mact, Ndat);
        Bbar.zeros();
        Bbar.axpy(CinvB, 2.0 * betaVal * dOut);
        if(dtcvar)
          Bbar.axpy(B, -2.0 * betaVal * dOut);

        double trCinvBB = 0.0;
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j < Ndat; j++)
            trCinvBB += CinvB.getVal(i, j) * B.getVal(i, j);

        double gbetaM2L = dOut * (trCinvBB - (double)Ndat / betaVal);

        CMatrix u(Mact, 1);
        CMatrix a(Mact, 1);
        CMatrix BTa(Ndat, 1);
        for(unsigned int j = 0; j < getOutputDim(); j++)
        {
          u.gemvColCol(0, B, m, j, 1.0, 0.0, "n");
          a.symvColCol(0, Cinv, u, 0, 1.0, 0.0, "u");
          // barB += -2 β^2 a m_j^T + 2 β^3 a (B^T a)^T
          Bbar.gerColCol(a, 0, m, j, -2.0 * betaVal * betaVal);
          BTa.gemvColCol(0, B, a, 0, 1.0, 0.0, "t");
          Bbar.gerColCol(a, 0, BTa, 0, 2.0 * betaVal * betaVal * betaVal);

          const double uta = u.dotColCol(0, a, 0);
          const double BTaNorm2 = BTa.dotColCol(0, BTa, 0);
          gbetaM2L += m.dotColCol(j, m, j)
            - 2.0 * betaVal * uta
            + betaVal * betaVal * BTaNorm2;
        }

        if(dtcvar)
        {
          double bFro2 = 0.0;
          for(unsigned int i = 0; i < Mact; i++)
            for(unsigned int j = 0; j < Ndat; j++)
            {
              const double bij = B.getVal(i, j);
              bFro2 += bij * bij;
            }
          gbetaM2L += dOut * (diagK.sum() - bFro2);
          gLambda.setVals(-0.5 * dOut * betaVal);
        }

        // Convert to ∂L / ∂B and ∂L / ∂β.
        Bbar.scale(-0.5);
        gBeta.setVal(-0.5 * gbetaM2L, 0);

        CMatrix Ubar;
        CMatrix KufBar;
        trsmAdjointUT(KufBar, Ubar, LcholK, B, Bbar);
        gK_uf.deepCopy(KufBar);

        CMatrix KuuBar;
        cholAdjointUpperBlocked(KuuBar, LcholK, Ubar);
        // cholAdjoint returns tril-pairing Kbar (off-diag = 2 ∂L/∂K_ij).
        // Kernel getGradParams expect Frobenius storage like Legacy (half off-diag).
        gK_uu.deepCopy(KuuBar);
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j < i; j++)
          {
            const double v = 0.5 * gK_uu.getVal(i, j);
            gK_uu.setVal(v, i, j);
            gK_uu.setVal(v, j, i);
          }
        gK_uu.setSymmetric(true);
      }
      else
      {
      E.gemm(K_uf, m, 1.0, 0.0, "n", "n");
      
      EET.gemm(E, E, 1.0, 0.0, "n", "t");
      EET.setSymmetric(true);
      
      AinvEET.symm(Ainv, EET, 1.0, 0.0, "u", "l");
      
      AinvEETAinv.gemm(AinvEET, Ainv, 1.0, 0.0, "n", "n");
      AinvEETAinv.setSymmetric(true);
      
      gK_uu.deepCopy(invK_uu);
      gK_uu.axpy(Ainv,-1.0/getBetaVal());
      gK_uu.scale((double)getOutputDim());
      gK_uu.axpy(AinvEETAinv, -1.0);
      gK_uu.setSymmetric(true);
      if(getApproximationType()==DTCVAR)
      {
	_multiplyInvKuu(invK_uuK_uf, K_uf);
	gK_uu.syrk(invK_uuK_uf, -getBetaVal()*(double)getOutputDim(), 1.0, "l", "n");
      }
      gK_uu.scale(0.5);
      AinvK_uf.symm(Ainv, K_uf, 1.0, 0.0, "l", "l");
      
      EMT.gemm(E, m, 1.0, 0.0, "n", "t");
      
      AinvEMT.symm(Ainv, EMT, 1.0, 0.0, "l", "l");
      
      gK_uf.gemm(AinvEET, AinvK_uf, 1.0, 0.0, "n", "n");
      gK_uf.axpy(AinvEMT, -1.0);
      gK_uf.scale(getBetaVal());
      gK_uf.negate();
      gK_uf.axpy(AinvK_uf, -(double)getOutputDim());
      if(getApproximationType()==DTCVAR)
      {
	gK_uf.axpy(invK_uuK_uf, getBetaVal()*(double)getOutputDim());
      }
      double gbetaVal = ((double)(getNumData()-numActive)/getBetaVal()); 
      double temp = 0.0;
      for(unsigned int i=0; i < Ainv.getRows(); i++)
	temp += Ainv.dotRowRow(i, K_uu, i);
      gbetaVal += temp/(getBetaVal()*getBetaVal());
      gbetaVal *= (double)getOutputDim();
      temp = 0.0;
      for(unsigned int i=0; i < AinvEETAinv.getRows(); i++)
	temp += AinvEETAinv.dotRowRow(i, K_uu, i);
      gbetaVal += temp/getBetaVal();
      //CMatrix StoreNK(getNumData(), getOutputDim());
      for(unsigned int j=0; j<getOutputDim(); j++)
	gbetaVal -= m.dotColCol(j, m, j);
      gbetaVal += AinvEET.trace();
      if(getApproximationType()==DTCVAR)
      {
	gbetaVal -= (double)getOutputDim()*diagD.sum()/getBetaVal();
      }
      gbetaVal *= 0.5;
      gBeta.setVal(gbetaVal, 0);
      if(getApproximationType()==DTCVAR)
      {
	gLambda.setVals(-0.5*(double)getOutputDim()*getBetaVal());
      }
      }
    }
    // Non-spherical: updateAD() throws first (CIP-0011).
    break;

  case FITC:
    if(isSpherical())
    {
      if(isWhitenedSparseLinalg())
      {
        // CIP-0012 FITC reverse on C = I + β V V^T, V = B D^{-1/2}, D = diagD.
        // CIP-001B LQ: replace Vbar = 2β Cbar V with Cbar→Lbar→lqAdjoint→Abar.
        const double betaVal = getBetaVal();
        const double sqrtBeta = sqrt(betaVal);
        const double dOut = (double)getOutputDim();
        const unsigned int Mact = numActive;
        const unsigned int Ndat = getNumData();
        const bool useLq = isLQSparseLinalg();

        CMatrix Cinv(Mact, Mact);
        Cinv.setSymmetric(true);
        Cinv.pdinv(LcholC);

        CMatrix Vfit(Mact, Ndat);
        Vfit.deepCopy(B);
        for(unsigned int j = 0; j < Ndat; j++)
          Vfit.scaleCol(j, 1.0 / sqrt(diagD.getVal(j)));

        CMatrix Cbar(Mact, Mact);
        Cbar.deepCopy(Cinv);
        Cbar.scale(dOut);
        CMatrix Bbar(Mact, Ndat);
        Bbar.zeros();
        CMatrix Dbar(Ndat, 1);
        for(unsigned int j = 0; j < Ndat; j++)
          Dbar.setVal(dOut / diagD.getVal(j), j);

        CMatrix w(Ndat, 1);
        CMatrix u(Mact, 1);
        CMatrix a(Mact, 1);
        double gbetaM2L = -dOut * (double)Ndat / betaVal;
        double mwSum = 0.0;
        double uaSum = 0.0;
        for(unsigned int j = 0; j < getOutputDim(); j++)
        {
          for(unsigned int i = 0; i < Ndat; i++)
            w.setVal(m.getVal(i, j) / diagD.getVal(i), i);
          u.gemvColCol(0, B, w, 0, 1.0, 0.0, "n");
          a.symvColCol(0, Cinv, u, 0, 1.0, 0.0, "u");
          Cbar.gerColCol(a, 0, a, 0, betaVal * betaVal);
          Bbar.gerColCol(a, 0, w, 0, -2.0 * betaVal * betaVal);
          // wbar = B^T ubar + β m_j, ubar = -2 β² a
          CMatrix wbar(Ndat, 1);
          wbar.gemvColCol(0, B, a, 0, -2.0 * betaVal * betaVal, 0.0, "t");
          for(unsigned int i = 0; i < Ndat; i++)
          {
            const double mi = m.getVal(i, j);
            const double Di = diagD.getVal(i);
            wbar.setVal(wbar.getVal(i) + betaVal * mi, i);
            Dbar.setVal(Dbar.getVal(i) - wbar.getVal(i) * mi / (Di * Di), i);
            mwSum += mi * w.getVal(i);
          }
          uaSum += u.dotColCol(0, a, 0);
        }
        // Symmetrise Cbar
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j < i; j++)
          {
            const double v = 0.5 * (Cbar.getVal(i, j) + Cbar.getVal(j, i));
            Cbar.setVal(v, i, j);
            Cbar.setVal(v, j, i);
          }
        Cbar.setSymmetric(true);

        CMatrix Vbar(Mact, Ndat);
        if(useLq)
        {
          // C = L L^T from LQ of [I, √β V]; Lbar from <Cbar, dC>, then Seeger §4.3.
          CMatrix Llow(Mact, Mact);
          Llow.zeros();
          for(unsigned int i = 0; i < Mact; i++)
            for(unsigned int j = 0; j <= i; j++)
              Llow.setVal(LcholC.getVal(j, i), i, j);
          Llow.setTriangular(true);
          CMatrix tmp(Mact, Mact);
          tmp.gemm(Cbar, Llow, 2.0, 0.0, "n", "n"); // (Cbar+Cbar^T)L with Cbar sym
          CMatrix Lbar(Mact, Mact);
          Lbar.zeros();
          for(unsigned int i = 0; i < Mact; i++)
            for(unsigned int j = 0; j <= i; j++)
              Lbar.setVal(tmp.getVal(i, j), i, j);
          CMatrix Qbar(Mact, Mact + Ndat);
          Qbar.zeros();
          CMatrix Wbar;
          lqAdjoint(Wbar, Llow, Qlq, Lbar, Qbar, false);
          for(unsigned int i = 0; i < Mact; i++)
            for(unsigned int j = 0; j < Ndat; j++)
              Vbar.setVal(sqrtBeta * Wbar.getVal(i, Mact + j), i, j);
        }
        else
          Vbar.symm(Cbar, Vfit, 2.0 * betaVal, 0.0, "u", "l");

        for(unsigned int j = 0; j < Ndat; j++)
        {
          const double s = sqrt(diagD.getVal(j));
          double vDot = 0.0;
          for(unsigned int i = 0; i < Mact; i++)
          {
            Bbar.setVal(Bbar.getVal(i, j) + Vbar.getVal(i, j) / s, i, j);
            vDot += Vbar.getVal(i, j) * Vfit.getVal(i, j);
          }
          Dbar.setVal(Dbar.getVal(j) - vDot / (2.0 * diagD.getVal(j)), j);
        }

        // gBeta (-2L) before q→B: explicit β + Dbar·(k-q)
        double vvT_Cbar = 0.0;
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int j = 0; j < Ndat; j++)
            vvT_Cbar += Vbar.getVal(i, j) * Vfit.getVal(i, j) / (2.0 * betaVal);
        // <Cbar, V V^T> = (1/(2β)) <Vbar, V> since Vbar = 2β Cbar V
        gbetaM2L += vvT_Cbar + mwSum - 2.0 * betaVal * uaSum;
        for(unsigned int j = 0; j < Ndat; j++)
        {
          double q = 0.0;
          for(unsigned int i = 0; i < Mact; i++)
          {
            const double bij = B.getVal(i, j);
            q += bij * bij;
          }
          gbetaM2L += Dbar.getVal(j) * (diagK.getVal(j) - q);
        }

        // q → B: Bbar -= 2β Dbar_j B_{:,j}
        for(unsigned int j = 0; j < Ndat; j++)
        {
          const double s = -2.0 * betaVal * Dbar.getVal(j);
          for(unsigned int i = 0; i < Mact; i++)
            Bbar.setVal(Bbar.getVal(i, j) + s * B.getVal(i, j), i, j);
        }

        // gLambda = ∂L/∂k = -0.5 * Dbar * β
        gLambda.resize(Ndat, 1);
        for(unsigned int j = 0; j < Ndat; j++)
          gLambda.setVal(-0.5 * Dbar.getVal(j) * betaVal, j);

        Bbar.scale(-0.5);
        gBeta.setVal(-0.5 * gbetaM2L, 0);

        CMatrix Ubar;
        CMatrix KufBar;
        trsmAdjointUT(KufBar, Ubar, LcholK, B, Bbar);
        gK_uf.deepCopy(KufBar);
        CMatrix KuuBar;
        cholAdjointUpperBlocked(KuuBar, LcholK, Ubar);
        gK_uu.deepCopy(KuuBar);
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int jj = 0; jj < i; jj++)
          {
            const double v = 0.5 * gK_uu.getVal(i, jj);
            gK_uu.setVal(v, i, jj);
            gK_uu.setVal(v, jj, i);
          }
        gK_uu.setSymmetric(true);
      }
      else
      {
      // using V as temp storage here.
      V.deepCopy(K_uf);
      for(unsigned int i=0; i<getNumData(); i++)
      {
	V.scaleCol(i, 1/diagD.getVal(i));
      }
      E.gemm(V, m, 1.0, 0.0, "n", "n");
      
      EET.gemm(E, E, 1.0, 0.0, "n", "t");
      EET.setSymmetric(true);
      AinvE.symm(Ainv, E, 1.0, 0.0, "l", "l");
      EMT.gemm(E, m, 1.0, 0.0, "n", "t");
      AinvEMT.symm(Ainv, EMT, 1.0, 0.0, "l", "l");
      K_ufdotTimesAinvEMT.deepCopy(AinvEMT);
      K_ufdotTimesAinvEMT.dotMultiply(K_uf);
      diagK_ufAinvEMT.sumCol(K_ufdotTimesAinvEMT, 1.0, 0.0);
      
      AinvEETAinv.gemm(AinvE, AinvE, 1.0, 0.0, "n", "t");
      AinvEETAinv.setSymmetric(true);
      // using Am as temp storage here.
      Am.deepCopy(Ainv);
      Am.scale((double)getOutputDim());
      Am.axpy(AinvEETAinv, getBetaVal());
      Am.setSymmetric(true);
      // using V as temp storage here.
      V.symm(Am, K_uf, 1.0, 0.0, "l", "l");
      V.dotMultiply(K_uf);
      diagK_ufdAinvplusAinvEETAinvK_fu.sumCol(V, 1.0, 0.0);
      invK_uuK_uf.symm(invK_uu, K_uf, 1.0, 0.0, "l", "l");
      invK_uuK_ufDinv.deepCopy(invK_uuK_uf);
      for(unsigned int i=0; i<getNumData(); i++)
      {
	invK_uuK_ufDinv.scaleCol(i, 1/diagD.getVal(i));
      }
      for(unsigned int i=0; i<getNumData(); i++)
	diagMMT.setVal(m.dotRowRow(i, m, i), i); 
      diagQ.deepCopy(diagK_ufdAinvplusAinvEETAinvK_fu);
      diagQ.axpy(diagD, -(double)getOutputDim());
      diagQ.axpy(diagMMT, getBetaVal());
      diagQ.axpy(diagK_ufAinvEMT, -2.0*getBetaVal());
      invK_uuK_ufDinvQ.deepCopy(invK_uuK_ufDinv);
      for(unsigned int i=0; i<getNumData(); i++)
      {
	invK_uuK_ufDinvQ.scaleCol(i, diagQ.getVal(i));
      }
      gK_uu.deepCopy(invK_uu);
      
      gK_uu.axpy(Ainv, -1.0/getBetaVal());
      gK_uu.scale((double)getOutputDim());
      gK_uu.axpy(AinvEETAinv, -1.0);
      gK_uu.gemm(invK_uuK_ufDinvQ, invK_uuK_ufDinv, getBetaVal(), 1.0, "n", "t");
      gK_uu.scale(0.5);	 
      gK_uu.setSymmetric(true);
 
      gK_uf.deepCopy(invK_uuK_ufDinvQ);
      gK_uf.symm(Ainv, K_uf, -(double)getOutputDim(), -getBetaVal(), "l", "l");
      gK_uf.symm(AinvEETAinv, K_uf, -getBetaVal(), 1.0, "l", "l");
      gK_uf.axpy(AinvEMT, getBetaVal());
      for(unsigned int j=0; j<getNumData(); j++)
      {
	gK_uf.scaleCol(j, 1/diagD.getVal(j));
      }
      gLambda.deepCopy(diagQ);
      gLambda.dotDivide(diagD);
      gLambda.scale(0.5*getBetaVal());
      gLambda.dotDivide(diagD);
      gBeta.setVal(-gLambda.sum()/(getBetaVal()*getBetaVal()), 0, 0);
      }
    }
    break;
    
  case PITC:
    if(isSpherical())
    {
      const double dOut = (double)getOutputDim();
      const double betaValP = getBetaVal();
      if(isWhitenedSparseLinalg())
      {
        // CIP-0012 reverse on C = I + β Σ B_b D_b^{-1} B_b^T (formulae for -2L, then × -1/2).
        const unsigned int Mact = numActive;
        CMatrix Cinv(Mact, Mact);
        Cinv.setSymmetric(true);
        Cinv.pdinv(LcholC);

        // u = Σ B_b Dinv m_b (whitened); a = C^{-1} u.
        CMatrix u(Mact, getOutputDim());
        u.zeros();
        for(unsigned int b = 0; b < getNumBlocks(); b++)
        {
          unsigned int start = 0, endInc = 0;
          getBlockRange(b, start, endInc);
          const unsigned int bs = endInc - start + 1;
          CMatrix Bb(Mact, bs);
          B.getMatrix(Bb, 0, Mact - 1, start, endInc);
          CMatrix mb(bs, getOutputDim());
          m.getMatrix(mb, start, endInc, 0, getOutputDim() - 1);
          CMatrix Dinvm(bs, getOutputDim());
          Dinvm.symm(blockDinv[b], mb, 1.0, 0.0, "l", "l");
          u.gemm(Bb, Dinvm, 1.0, 1.0, "n", "n");
        }
        CMatrix a(Mact, getOutputDim());
        a.symm(Cinv, u, 1.0, 0.0, "u", "l");

        CMatrix Cbar(Mact, Mact);
        Cbar.deepCopy(Cinv);
        Cbar.scale(dOut);
        for(unsigned int j = 0; j < getOutputDim(); j++)
          Cbar.gerColCol(a, j, a, j, betaValP * betaValP);
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int jj = 0; jj < i; jj++)
          {
            const double v = 0.5 * (Cbar.getVal(i, jj) + Cbar.getVal(jj, i));
            Cbar.setVal(v, i, jj);
            Cbar.setVal(v, jj, i);
          }
        Cbar.setSymmetric(true);

        CMatrix Bbar(Mact, getNumData());
        Bbar.zeros();
        double gBetaAcc = 0.0;
        for(unsigned int b = 0; b < getNumBlocks(); b++)
        {
          unsigned int start = 0, endInc = 0;
          getBlockRange(b, start, endInc);
          const unsigned int bs = endInc - start + 1;
          CMatrix Bb(Mact, bs);
          B.getMatrix(Bb, 0, Mact - 1, start, endInc);
          CMatrix mb(bs, getOutputDim());
          m.getMatrix(mb, start, endInc, 0, getOutputDim() - 1);
          CMatrix W(Mact, bs);
          W.gemm(Bb, blockDinv[b], 1.0, 0.0, "n", "n");

          // Bb_bar = 2β Cbar W + Σ_j outer(ubar_j, w_j)
          CMatrix BbBar(Mact, bs);
          BbBar.symm(Cbar, W, 2.0 * betaValP, 0.0, "u", "l");
          CMatrix Dinvbar(bs, bs);
          Dinvbar.zeros();
          {
            // Mid = B_b^T Cbar (bs×M), Mid2 = β Mid B_b = β B_b^T Cbar B_b.
            CMatrix Mid(bs, Mact);
            Mid.gemm(Bb, Cbar, 1.0, 0.0, "t", "n");
            CMatrix Mid2(bs, bs);
            Mid2.gemm(Mid, Bb, betaValP, 0.0, "n", "n");
            for(unsigned int i = 0; i < bs; i++)
              for(unsigned int jj = 0; jj < bs; jj++)
              {
                const double v = 0.5 * (Mid2.getVal(i, jj) + Mid2.getVal(jj, i));
                Dinvbar.setVal(v, i, jj);
              }
          }
          for(unsigned int j = 0; j < getOutputDim(); j++)
          {
            CMatrix w(bs, 1);
            w.symvColCol(0, blockDinv[b], mb, j, 1.0, 0.0, "l");
            CMatrix ubar(Mact, 1);
            for(unsigned int i = 0; i < Mact; i++)
              ubar.setVal(-2.0 * betaValP * betaValP * a.getVal(i, j), i);
            BbBar.gerColCol(ubar, 0, w, 0, 1.0);
            // Dinvbar += 0.5(v m^T + m v^T) + β m m^T, v = B_b^T ubar
            CMatrix v(bs, 1);
            v.gemvColCol(0, Bb, ubar, 0, 1.0, 0.0, "t");
            for(unsigned int i = 0; i < bs; i++)
              for(unsigned int jj = 0; jj < bs; jj++)
              {
                const double sym = 0.5 * (v.getVal(i) * mb.getVal(jj, j)
                                         + mb.getVal(i, j) * v.getVal(jj));
                const double quad = betaValP * mb.getVal(i, j) * mb.getVal(jj, j);
                Dinvbar.setVal(Dinvbar.getVal(i, jj) + sym + quad, i, jj);
              }
          }
          Dinvbar.setSymmetric(true);

          // Dbar = dOut Dinv - Dinv Dinvbar Dinv
          CMatrix Dbar(bs, bs);
          Dbar.deepCopy(blockDinv[b]);
          Dbar.scale(dOut);
          {
            CMatrix tmp(bs, bs);
            tmp.symm(blockDinv[b], Dinvbar, 1.0, 0.0, "l", "l");
            CMatrix chain(bs, bs);
            chain.gemm(tmp, blockDinv[b], 1.0, 0.0, "n", "n");
            Dbar.axpy(chain, -1.0);
          }
          Dbar.setSymmetric(true);

          // q → B: Bb_bar -= 2β B_b Dbar
          {
            CMatrix tmp(Mact, bs);
            tmp.gemm(Bb, Dbar, 1.0, 0.0, "n", "n");
            BbBar.axpy(tmp, -2.0 * betaValP);
          }
          Bbar.setMatrix(0, start, BbBar);

          // gLambda = ∂L/∂K = -0.5 β Dbar; gBeta via Legacy trace rule.
          gLambdaBlock[b].deepCopy(Dbar);
          gLambdaBlock[b].scale(-0.5 * betaValP);
          gLambdaBlock[b].setSymmetric(true);
          gBetaAcc -= gLambdaBlock[b].trace() / (betaValP * betaValP);
        }

        Bbar.scale(-0.5);
        gBeta.setVal(gBetaAcc, 0);

        CMatrix Ubar;
        CMatrix KufBar;
        trsmAdjointUT(KufBar, Ubar, LcholK, B, Bbar);
        gK_uf.deepCopy(KufBar);
        CMatrix KuuBar;
        cholAdjointUpperBlocked(KuuBar, LcholK, Ubar);
        gK_uu.deepCopy(KuuBar);
        for(unsigned int i = 0; i < Mact; i++)
          for(unsigned int jj = 0; jj < i; jj++)
          {
            const double v = 0.5 * gK_uu.getVal(i, jj);
            gK_uu.setVal(v, i, jj);
            gK_uu.setVal(v, jj, i);
          }
        gK_uu.setSymmetric(true);
      }
      else
      {
      // GPmat gpCovGrads pitc (spherical).
      CMatrix E(numActive, getOutputDim());
      E.zeros();
      for(unsigned int b = 0; b < getNumBlocks(); b++)
      {
        unsigned int start = 0, endInc = 0;
        getBlockRange(b, start, endInc);
        const unsigned int bs = endInc - start + 1;
        CMatrix Kufb(numActive, bs);
        K_uf.getMatrix(Kufb, 0, numActive - 1, start, endInc);
        CMatrix mb(bs, getOutputDim());
        m.getMatrix(mb, start, endInc, 0, getOutputDim() - 1);
        CMatrix Dinvm(bs, getOutputDim());
        Dinvm.symm(blockDinv[b], mb, 1.0, 0.0, "l", "l");
        E.gemm(Kufb, Dinvm, 1.0, 1.0, "n", "n");
      }
      AinvE.symm(Ainv, E, 1.0, 0.0, "l", "l");
      AinvEET.gemm(AinvE, E, 1.0, 0.0, "n", "t");
      AinvEET.setSymmetric(true);
      AinvEETAinv.gemm(AinvEET, Ainv, 1.0, 0.0, "n", "n");
      AinvEETAinv.setSymmetric(true);

      gK_uu.deepCopy(invK_uu);
      gK_uu.scale(dOut);
      gK_uu.axpy(Ainv, -dOut / betaValP);
      gK_uu.axpy(AinvEETAinv, -1.0);

      CMatrix gKufBase(numActive, getNumData());
      CMatrix Aterm(numActive, numActive);
      Aterm.deepCopy(Ainv);
      Aterm.scale(dOut);
      Aterm.axpy(AinvEETAinv, betaValP);
      Aterm.setSymmetric(true);
      gKufBase.symm(Aterm, K_uf, -1.0, 0.0, "l", "l");
      {
        CMatrix EMT(numActive, getNumData());
        EMT.gemm(AinvE, m, betaValP, 0.0, "n", "t");
        gKufBase.axpy(EMT, 1.0);
      }

      double gBetaAcc = 0.0;
      gK_uf.zeros();
      for(unsigned int b = 0; b < getNumBlocks(); b++)
      {
        unsigned int start = 0, endInc = 0;
        getBlockRange(b, start, endInc);
        const unsigned int bs = endInc - start + 1;
        CMatrix Kufb(numActive, bs);
        K_uf.getMatrix(Kufb, 0, numActive - 1, start, endInc);
        CMatrix mb(bs, getOutputDim());
        m.getMatrix(mb, start, endInc, 0, getOutputDim() - 1);

        CMatrix KfuAinvEMT(bs, bs);
        {
          CMatrix tmp(numActive, bs);
          tmp.gemm(AinvE, mb, 1.0, 0.0, "n", "t");
          KfuAinvEMT.gemm(Kufb, tmp, betaValP, 0.0, "t", "n");
        }
        CMatrix blockQ(bs, bs);
        blockQ.deepCopy(blockD[b]);
        blockQ.scale(-dOut);
        {
          CMatrix mmt(bs, bs);
          mmt.gemm(mb, mb, 1.0, 0.0, "n", "t");
          blockQ.axpy(mmt, betaValP);
        }
        {
          CMatrix mid(numActive, numActive);
          mid.deepCopy(Ainv);
          mid.scale(dOut);
          mid.axpy(AinvEETAinv, betaValP);
          mid.setSymmetric(true);
          CMatrix midK(numActive, bs);
          midK.symm(mid, Kufb, 1.0, 0.0, "l", "l");
          CMatrix add(bs, bs);
          add.gemm(Kufb, midK, 1.0, 0.0, "t", "n");
          blockQ.axpy(add, 1.0);
        }
        blockQ.axpy(KfuAinvEMT, -1.0);
        // KfuAinvEMT'
        for(unsigned int i = 0; i < bs; i++)
          for(unsigned int j = 0; j < bs; j++)
            blockQ.addVal(-KfuAinvEMT.getVal(j, i), i, j);
        blockQ.setSymmetric(true);

        CMatrix invKuuKufDinv(numActive, bs);
        {
          CMatrix Qu(numActive, bs);
          Qu.gemm(invK_uu, Kufb, 1.0, 0.0, "n", "n");
          invKuuKufDinv.gemm(Qu, blockDinv[b], 1.0, 0.0, "n", "n");
        }
        {
          CMatrix tmp(numActive, bs);
          tmp.gemm(invKuuKufDinv, blockQ, 1.0, 0.0, "n", "n");
          gK_uu.gemm(tmp, invKuuKufDinv, betaValP, 1.0, "n", "t");
        }

        // gK_uf(:,ind) = (gKufBase(:,ind) - beta * invK_uu K_uf Dinv * blockQ) * Dinv
        {
          CMatrix baseb(numActive, bs);
          gKufBase.getMatrix(baseb, 0, numActive - 1, start, endInc);
          CMatrix tmp(numActive, bs);
          tmp.gemm(invKuuKufDinv, blockQ, betaValP, 0.0, "n", "n");
          baseb.axpy(tmp, -1.0);
          CMatrix outb(numActive, bs);
          outb.gemm(baseb, blockDinv[b], 1.0, 0.0, "n", "n");
          gK_uf.setMatrix(0, start, outb);
        }

        // g_Lambda{i} = 0.5 * beta * Dinv * blockQ * Dinv
        {
          CMatrix tmp(bs, bs);
          tmp.gemm(blockDinv[b], blockQ, 0.5 * betaValP, 0.0, "n", "n");
          gLambdaBlock[b].gemm(tmp, blockDinv[b], 1.0, 0.0, "n", "n");
          gLambdaBlock[b].setSymmetric(true);
        }
        gBetaAcc -= gLambdaBlock[b].trace() / (betaValP * betaValP);
      }
      gK_uu.scale(0.5);
      gK_uu.setSymmetric(true);
      gBeta.setVal(gBetaAcc, 0);
      }
    }
    break;
  }
  
}
void CGp::pointLogLikelihood(const CMatrix& y, const CMatrix& X) const
{
  throw ndlexceptions::NotImplementedError("pointLogLikelihood not yet implemented for CGp.");
}
#ifdef _NDLMATLAB
CGp::CGp(CMatrix* pinData, 
	 CMatrix* ptargetData, 
	 CKern* pkernel, 
	 CNoise* pnoiseModel, 
	 const string gpInfoFile, 
	 const string gpInfoVariable, int verbos) : 
  CMapModel(pinData->getCols(), ptargetData->getCols(), ptargetData->getRows()), CProbabilisticOptimisable(), 
  pX(pinData), py(ptargetData), 
  pkern(pkernel), pnoise(pnoiseModel)
{
  _init();
  initKernStoreage();
  setVerbosity(verbos);
  readMatlabFile(gpInfoFile, gpInfoVariable);
  updateK();
  updateAD();
}
mxArray* CGp::toMxArray() const
{
  size_t dims[1];
  dims[0]=1;
  mxArray* matlabArray;
  if(isSparseApproximation()) 
  {
    if(isInducingFixed()) 
    {
      const char* fieldNames[]={"d", "q", "k", "N", "learnScales", "approx", "fixInducing", "inducingIndices", "scale", "bias", "beta", "betaTransform", "type"};
      matlabArray = mxCreateStructArray(1, dims, 13, fieldNames);
      mxSetField(matlabArray, 0, "learnScales", convertMxArray(isOutputScaleLearnt()));
      mxSetField(matlabArray, 0, "approx", convertMxArray(getApproximationStr()));
      mxSetField(matlabArray, 0, "fixInducing", convertMxArray(true));
      // The inducing indices.
      vector<int> inducingMatlab = inducingIndices;
      for(unsigned int i=0; i<inducingMatlab.size(); i++)
	inducingMatlab[i]++;
      mxSetField(matlabArray, 0, "inducingIndices", convertMxArray(inducingMatlab));
      mxSetField(matlabArray, 0, "scale", scale.toMxArray());
      mxSetField(matlabArray, 0, "bias", bias.toMxArray());
      mxSetField(matlabArray, 0, "beta", beta.toMxArray());
      mxSetField(matlabArray, 0, "betaTransform", convertMxArray(betaTransform->getType()));
    }
    else 
    {
      const char* fieldNames[]={"d", "q", "k", "N", "learnScales", "approx", "fixInducing", "X_u", "scale", "bias", "beta", "betaTransform", "type"};
      matlabArray = mxCreateStructArray(1, dims, 13, fieldNames);
      
      mxSetField(matlabArray, 0, "learnScales", convertMxArray(isOutputScaleLearnt()));
      mxSetField(matlabArray, 0, "approx", convertMxArray(getApproximationStr()));
      mxSetField(matlabArray, 0, "fixInducing", convertMxArray(false));
      mxSetField(matlabArray, 0, "X_u", X_u.toMxArray());
      mxSetField(matlabArray, 0, "scale", scale.toMxArray());
      mxSetField(matlabArray, 0, "bias", bias.toMxArray());
      mxSetField(matlabArray, 0, "beta", beta.toMxArray());
      mxSetField(matlabArray, 0, "betaTransform", convertMxArray(betaTransform->getType()));
    }  
  }
  else {
    const char* fieldNames[]={"d", "q", "k", "N", "learnScales", "approx", "scale", "bias", "type"};
    matlabArray = mxCreateStructArray(1, dims, 9, fieldNames);
    mxSetField(matlabArray, 0, "learnScales", convertMxArray(isOutputScaleLearnt()));
    mxSetField(matlabArray, 0, "approx", convertMxArray(getApproximationStr()));
    mxSetField(matlabArray, 0, "scale", scale.toMxArray());
    mxSetField(matlabArray, 0, "bias", bias.toMxArray());
  }
  mxSetField(matlabArray, 0, "d", convertMxArray((double)getOutputDim()));
  mxSetField(matlabArray, 0, "q", convertMxArray((double)getInputDim()));
  mxSetField(matlabArray, 0, "k", convertMxArray((double)numActive));
  mxSetField(matlabArray, 0, "N", convertMxArray((double)getNumData()));
  mxSetField(matlabArray, 0, "type", convertMxArray(getType()));
  return matlabArray;
}
void CGp::fromMxArray(const mxArray* matlabArray)
{
  string mxType = mxArrayExtractStringField(matlabArray, "type");
  if(mxType!=getType())
  {
    throw ndlexceptions::FileReadError("Error mismatch between saved type, " + mxType + ", and Class type, " + getType() + ".");
  }
  setOutputScaleLearnt(mxArrayExtractBoolField(matlabArray, "learnScales"));
  setApproximationStr(mxArrayExtractStringField(matlabArray, "approx"));
  setOutputDim(mxArrayExtractIntField(matlabArray, "d"));
  setInputDim(mxArrayExtractIntField(matlabArray, "q"));
  setNumData(mxArrayExtractIntField(matlabArray, "N"));
  numActive = mxArrayExtractIntField(matlabArray, "k");
  initStoreage();
  if(isSparseApproximation()) 
  {
    setInducingFixed(mxArrayExtractBoolField(matlabArray, "fixInducing"));
    if(isInducingFixed()) 
    {
      inducingIndices = mxArrayExtractVectorIntField(matlabArray, "inducingIndices");
      numActive = inducingIndices.size();
      for(unsigned int i=0; i<inducingIndices.size(); i++)
	inducingIndices[i]--;
    }
    else 
    {
      X_u.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "X_u"));
      DIMENSIONMATCH(numActive==X_u.getRows());
      DIMENSIONMATCH(getInputDim()==X_u.getCols());
    }
    beta.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "beta"));
    string betaTransformType = mxArrayExtractStringField(matlabArray, "betaTransform");
    if(betaTransformType == "exp")
      betaTransform = new CExpTransform();
    else if(betaTransformType == "negLogLogit")
      betaTransform = new CNegLogLogitTransform();
    else
      throw ndlexceptions::NotImplementedError("fromMxArray: Not yet implemented transform types apart from exp and negLogLogit.");
  }
  
  scale.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "scale"));
  bias.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "bias"));
  updateM();
}
#else /* not _NDLMATLAB */
#endif
// Optimise the GP with respect to latent positions and kernel parameters.
void CGp::optimise(unsigned int iters)
{
  if(getVerbosity()>2)
  {
    cout << "Initial model:" << endl;
    display(cout);
  }
  if(getVerbosity()>2 && getOptNumParams()<40)
    checkGradients();
  setMaxIters(iters);
  runDefaultOptimiser();
  
  if(getVerbosity()>1)
    cout << "... done. " << endl;
  if(getVerbosity()>0)
    display(cout);
}

bool CGp::equals(const CGp& model, const double tol) const 
{
  if(!pnoise->equals(*model.pnoise, tol))
    return false;
  if(!pkern->equals(*model.pkern, tol))
    return false;
  if(!m.equals(model.m, tol))
    return false;
  if(isSparseApproximation())
    if(abs(getBetaVal()-model.getBetaVal())>tol)
      return false;
  if(!bias.equals(model.bias, tol))
    return false;
  if(!scale.equals(model.scale, tol))
    return false;
  if(!pX->equals(*model.pX, tol))
    return false;
  if(approximationType!=model.getApproximationType())
    return false;
  if(isSparseApproximation()) {
    if(numActive!=model.getNumActive())
      return false;
    if(!X_u.equals(model.X_u, tol))
      return false;
  }
  return true;
}

void CGp::display(ostream& os) const 
{
  if(isSparseApproximation())
    cout << "Sparse Approximation GP Model:" << endl << "Approx type: " << getApproximationStr() <<endl;
  else
    cout << "Standard GP Model: " << endl;
  cout << "Optimiser: " << getDefaultOptimiserStr() << endl;
  cout << "Data Set Size: " << getNumData() << endl;
  cout << "Kernel Type: " << endl;
  cout << "Scales learnt: " << isOutputScaleLearnt() << endl;
  cout << "X learnt: " << isOptimiseX() << endl;
  cout << "Bias: " << bias << endl;
  cout << "Scale: " << scale << endl;
  pnoise->display(os);
  pkern->display(os);
  if(isSparseApproximation()) {
    cout << "Inducing fixed: " << isInducingFixed() << endl;
    cout << "Beta Value: " << getBetaVal() << endl;
  }
  if(py && pX)
    cout << "Log likelihood: " << logLikelihood() << endl;
}

void CGp::readParamsFromStream(istream& in)
{
  string tbaseType = getBaseTypeStream(in);
  if(tbaseType != getBaseType())
    throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getBaseType() + ".");
  string ttype = getTypeStream(in);
  if(ttype != getType())
    throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setNumData(readIntFromStream(in, "numData"));
  setOutputDim(readIntFromStream(in, "outputDim"));
  setInputDim(readIntFromStream(in, "inputDim"));
  setApproximationType(readIntFromStream(in, "sparseApproximation"));
  numActive = readIntFromStream(in, "numActive");
  // have all the info for resizing storage.
  initStoreage();

  if(isSparseApproximation()) 
  {
    betaTransform = CTransform::defaultPositive();
    beta.fromStream(in);
  }


  setOutputScaleLearnt(readBoolFromStream(in, "learnScale"));
  setOutputBiasLearnt(readBoolFromStream(in, "learnBias"));

  // next lines are scale and bias
  scale.fromStream(in);
  bias.fromStream(in);

  // load kernel
  pkern = readKernFromStream(in);
  initKernStoreage();
  // load noise
  pnoise = readNoiseFromStream(in);


  if(isSparseApproximation()) 
  {
    setInducingFixed(readBoolFromStream(in, "fixInducing"));
    X_u.fromStream(in);
    if(X_u.getCols()!=getInputDim())
      throw ndlexceptions::StreamFormatError("inputDim", "X_u columns doesn't match input dimension.");
  }
  setMupToDate(false);
  // Don't update M as you need py to be set for that.
  //updateM();
}


void CGp::writeParamsToStream(ostream& out) const 
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "numData", getNumData());
  writeToStream(out, "outputDim", getOutputDim());
  writeToStream(out, "inputDim", getInputDim());

  writeToStream(out, "sparseApproximation", getApproximationType());
  writeToStream(out, "numActive", getNumActive());
  if(isSparseApproximation())
    beta.toStream(out);
  writeToStream(out, "learnScale", isOutputScaleLearnt());
  writeToStream(out, "learnBias", isOutputBiasLearnt());
  scale.toStream(out);
  bias.toStream(out);
  pkern->toStream(out);
  pnoise->toStream(out);
  if(isSparseApproximation()) 
  {
    writeToStream(out, "fixInducing", isInducingFixed());
    X_u.toStream(out);
  }
  //py->toStream(out);
  //pX->toStream(out);

}
// Functions which operate on the object
void writeGpToStream(const CGp& model, ostream& out) 
{
  model.toStream(out);
}

void writeGpToFile(const CGp& model, const string modelFileName, const string comment)
{
  model.toFile(modelFileName, comment);
}

CGp* readGpFromStream(istream& in)
{
  CGp* pmodel = new CGp();
  pmodel->fromStream(in);
  return pmodel;
}
    
CGp* readGpFromFile(const string modelFileName, int verbosity)
{
  // File is m, beta, X
  if(verbosity>0)
    cout << "Loading model file." << endl;
  ifstream in(modelFileName.c_str());
  if(!in.is_open()) throw ndlexceptions::FileReadError(modelFileName);
  CGp* pmodel;
  try 
  {
    pmodel = readGpFromStream(in);
  }
  catch(ndlexceptions::StreamFormatError err) 
  {
    throw ndlexceptions::FileFormatError(modelFileName, err);
  }
  if(verbosity>0)
    cout << "... done." << endl;
  in.close();
  pmodel->setVerbosity(verbosity);
  return pmodel; 
}
