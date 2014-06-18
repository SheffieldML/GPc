#include "CIvm.h"

CIvm::CIvm()
{ 
  _init();
  pkern=NULL;
  pnoise=NULL;
}
void CIvm::_init()
{
  setType("ivm");
  setName("informative vector machine");
  lastEntropyChange = 0.0;
  cumEntropy = 0.0;
  epUpdate = false;
  terminate = false;
}
CIvm::CIvm(CMatrix* inData, CMatrix* targetData, 
	   CKern* pkernel, CNoise* pnoiseModel, int selectCrit,
	   unsigned int dVal, int verbosity) 
  : CMapModel(inData->getCols(), targetData->getCols(), inData->getRows()), CProbabilisticOptimisable(), pX(inData), py(targetData), pkern(pkernel), 
    pnoise(pnoiseModel), selectionCriterion(selectCrit), numActive(dVal)
{
  _init();
  if(dVal>getNumData())
    throw ndlexceptions::Error("Active set is larger than data set.");
  DIMENSIONMATCH(pX->getRows()==py->getRows());
  setVerbosity(verbosity);
  pnoise->setVerbosity(getVerbosity());
  init();
}
CIvm::CIvm(CMatrix& actX, CMatrix& actY, 
	   CMatrix& mmat, CMatrix& betamat, 
	   vector<unsigned int> actSet, CKern* pkernel, 
	   CNoise* pnoiseModel, int selectCrit, 
	   int verbosity) 
  : CMapModel(actX.getCols(), actY.getCols(), actX.getRows()), 
    CProbabilisticOptimisable(), pX(&actX), py(&actY), 
    activeX(actX), activeY(actY), m(mmat), beta(betamat), 
    pkern(pkernel), activeSet(actSet), pnoise(pnoiseModel), 
    selectionCriterion(selectCrit),  numActive(activeSet.size())
{
  DIMENSIONMATCH(activeX.getRows()==activeY.getRows());
  _init();
  setVerbosity(verbosity);
  pnoise->setVerbosity(getVerbosity());
  initStoreage();
  updateK();
  updateInvK();
  for(unsigned int j=0; j<numCovStruct; j++)
  {
    L[j].deepCopy(K);
    for(unsigned int i=0; i<getNumActive(); i++)
    {
      double lval = L[j].getVal(i, i);
      lval += 1/beta.getVal(i, j);
      L[j].setVal(lval, i, i);
    }
    L[j].setSymmetric(true);
    L[j].chol("L");
    Linv[j].deepCopy(L[j]);
    // TODO should not use regular inverse here as matrix is lower triangular.
    Linv[j].inv();
  }
}
void CIvm::init()
{
  initStoreage();
  initVals();
}
void CIvm::test(const CMatrix& ytest, const CMatrix& Xin) const
{
  DIMENSIONMATCH(ytest.getCols()==getOutputDim());
  DIMENSIONMATCH(ytest.getRows()==Xin.getRows());
  CMatrix muout(Xin.getRows(), getOutputDim());
  CMatrix varSigmaOut(Xin.getRows(), getOutputDim());
  posteriorMeanVar(muout, varSigmaOut, Xin);
  pnoise->test(muout, varSigmaOut, ytest);
}
void CIvm::likelihoods(CMatrix& pout, CMatrix& yTest, const CMatrix& Xin) const
{
  DIMENSIONMATCH(pout.getCols()==getOutputDim());
  DIMENSIONMATCH(pout.getRows()==Xin.getRows());
  DIMENSIONMATCH(yTest.dimensionsMatch(pout));
  CMatrix muout(Xin.getRows(), getOutputDim());
  CMatrix varSigmaOut(Xin.getRows(), getOutputDim());
  posteriorMeanVar(muout, varSigmaOut, Xin);
  pnoise->likelihoods(pout, muout, varSigmaOut, yTest);
}
double CIvm::logLikelihood(const CMatrix& yTest, const CMatrix& Xin) const
{
  DIMENSIONMATCH(yTest.getRows()==Xin.getRows());
  DIMENSIONMATCH(getInputDim()==Xin.getCols());
  DIMENSIONMATCH(getOutputDim()==yTest.getCols());
  CMatrix muout(Xin.getRows(), getOutputDim());
  CMatrix varSigmaOut(Xin.getRows(), getOutputDim());
  posteriorMeanVar(muout, varSigmaOut, Xin);
  return pnoise->logLikelihood(muout, varSigmaOut, yTest)+pkern->priorLogProb();
}
void CIvm::out(CMatrix& yout, const CMatrix& Xin) const
{
  DIMENSIONMATCH(yout.getCols()==getOutputDim());
  DIMENSIONMATCH(yout.getRows()==Xin.getRows());
  CMatrix muout(Xin.getRows(), getOutputDim());
  CMatrix varSigmaOut(Xin.getRows(), getOutputDim());
  posteriorMeanVar(muout, varSigmaOut, Xin);
  pnoise->out(yout, muout, varSigmaOut);
}
void CIvm::out(CMatrix& yout, CMatrix& probout, const CMatrix& Xin) const
{
  DIMENSIONMATCH(yout.getCols()==getOutputDim());
  DIMENSIONMATCH(yout.getRows()==Xin.getRows());
  CMatrix muout(Xin.getRows(), getOutputDim());
  CMatrix varSigmaOut(Xin.getRows(), getOutputDim());
  posteriorMeanVar(muout, varSigmaOut, Xin);
  pnoise->out(yout, probout, muout, varSigmaOut);
}
double CIvm::outGradParams(CMatrix& g, const CMatrix &Xin, unsigned int pointNo, unsigned int outputNo) const
{
  throw ndlexceptions::NotImplementedError("outGradParams not yet implemented for CIvm.");
}
double CIvm::outGradX(CMatrix& g, const CMatrix &Xin, unsigned int pointNo, unsigned int outputNo) const
{
  throw ndlexceptions::NotImplementedError("outGradX not yet implemented for CIvm.");
}
void CIvm::posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const
{
  DIMENSIONMATCH(mu.getCols()==getOutputDim());
  DIMENSIONMATCH(varSigma.getCols()==getOutputDim());
  CMatrix kX(getNumActive(), Xin.getRows());
  pkern->compute(kX, activeX, Xin);
  if(numCovStruct==1)
  {
    kX.trsm(L[0], 1.0, "L", "L", "N", "N"); // now it is Linvk
    for(unsigned int i=0; i<Xin.getRows(); i++)
    {
      double vsVal = pkern->diagComputeElement(Xin, i) - kX.norm2Col(i);
      CHECKZEROORPOSITIVE(vsVal>=0);
      for(unsigned int j=0; j<getOutputDim(); j++)
	varSigma.setVal(vsVal, i, j);	    
    }
    kX.trsm(L[0], 1.0, "L", "L", "T", "N"); // now it is Kinvk
    mu.gemm(kX, m, 1.0, 0.0, "T", "N");
  }
  else
  {
    CMatrix Lk(getNumActive(), Xin.getRows());
      
    for(unsigned int k=0; k<numCovStruct; k++)
    {
      Lk.deepCopy(kX);
      Lk.trsm(L[k], 1.0, "L", "L", "N", "N");
      for(unsigned int i=0; i<Xin.getRows(); i++)
      {
	double vsVal=pkern->diagComputeElement(Xin, i) - Lk.norm2Col(i);
	CHECKZEROORPOSITIVE(vsVal>=0);
	varSigma.setVal(vsVal, i, k);
      }
      Lk.trsm(L[k], 1.0, "L", "L", "T", "N"); // now it is Kinvk
      mu.gemvColCol(k, Lk, m, k, 1.0, 0.0, "N");
    }
  }
}
void CIvm::initStoreage()
{
  if(isSpherical())
    numCovStruct = 1;
  else
    numCovStruct = getOutputDim();

  Kstore.resize(getNumData(), getNumActive());
  m.resize(getNumActive(), getOutputDim());
  beta.resize(getNumActive(), getOutputDim());
  nu.resize(getNumData(), getOutputDim());
  g.resize(getNumData(), getOutputDim());

  // set up s, a and ainv
  s.resize(getNumData(), 1); // s is a columnrow vector.
  a.resize(getNumActive(), 1); // a is a column vector.
  ainv.resize(1, getNumActive()); // ainv is a row vector.
  
  // set up L, M and Linv
  M = new CMatrix[numCovStruct];

  L = new CMatrix[numCovStruct];
  Linv = new CMatrix[numCovStruct];
  for(unsigned int c=0; c<numCovStruct; c++)
  {
    M[c].resize(getNumActive(), getNumData());
    L[c].resize(getNumActive(), getNumActive());
    L[c].setTriangular(true);
    Linv[c].resize(getNumActive(), getNumActive());
    Linv[c].setTriangular(true);

  }
  // set up K invK and covGrad
  K.resize(getNumActive(), getNumActive());
  activeX.resize(getNumActive(), getInputDim());
  activeY.resize(getNumActive(), getOutputDim());
  invK.resize(getNumActive(), getNumActive());
  invK.setSymmetric(true);
  covGrad.resize(getNumActive(), getNumActive());
  covGrad.setSymmetric(true);
}

void CIvm::initVals()
{
  // set Kstore to zeros numData, numActive.
  Kstore.setVals(0.0);
  // set m and beta to zeros(size of y) -- could do with sparse representation.
  m.setVals(0.0);
  beta.setVals(0.0);
  // set nu to zeros(size of y)
  nu.setVals(0.0);
  // set g to zeros(size of y)
  g.setVals(0.0);
  // set pnoise->varSigma to diagonal of kernel.
  pnoise->setMus(0.0);
  double dk=0.0;
  for(unsigned int i=0; i<getNumData(); i++)
  {
    dk = pkern->diagComputeElement(*pX, i);
    for(unsigned int j=0; j<getOutputDim(); j++)
    {
      pnoise->setVarSigma(dk, i, j);
    }
  }
  for(unsigned int c=0; c<numCovStruct; c++)
  {
    M[c].setVals(0.0);
    L[c].setVals(0.0);
    Linv[c].setVals(0.0);
  }
  invK.zeros();
  covGrad.zeros();
  
  // fill the inactive set.
  inactiveSet.erase(inactiveSet.begin(), inactiveSet.end());
  for(unsigned int i=0; i<getNumData(); i++)
  {
    inactiveSet.push_back(i);
  }
  
  // empty the active set.
  activeSet.erase(activeSet.begin(), activeSet.end());
  updateNuG();
}
void CIvm::selectPoints()
{
  unsigned int index=0;
  if(getVerbosity()>1)
    cout << "Selecting " << getNumActive() << " points ... " << endl;
  for(unsigned int k=0; k<getNumActive(); k++)
  {
    index = selectPointAdd();
    addPoint(index);
    if(getVerbosity()>2)
      cout << k << "th addition: added point " << index << endl;
      
  }
  if(getVerbosity()>1)
    cout << "... done." << endl;
  if(isEpUpdate())
    throw ndlexceptions::NotImplementedError("EP update not yet implemented.");
}
void CIvm::addPoint(unsigned int index)
{
  // check index is in inactive set
  SANITYCHECK(find(inactiveSet.begin(), inactiveSet.end(), index)!=inactiveSet.end());
  vector<unsigned int>::iterator pos = find(inactiveSet.begin(), inactiveSet.end(), index);
  updateSite(index);
  updateM(index);
  inactiveSet.erase(pos);
  activeX.copyRowRow(activeSet.size(), *pX, index);
  activeY.copyRowRow(activeSet.size(), *py, index);
  activeSet.push_back(index);
  updateNuG();
}
void CIvm::updateSite(unsigned int index)
{
  unsigned int actIndex = activeSet.size();
  pnoise->updateSites(m, beta, actIndex, g, nu, index);
  for(unsigned int j=0; j<beta.getCols(); j++)
  {
    double betVal = beta.getVal(actIndex, j);
    if(betVal<0)
    {
      if(pnoise->isLogConcave())
      {
        cout << "Warning: beta less than zero for log concave model." << endl;
      }
      else
      {
	beta.setVal(1e-6, actIndex, j);
	cout << "Beta less than zero fixing to 1e-6." << endl;
      }
    }
  }
  
}

void CIvm::updateM(unsigned int index)
{
  unsigned int activePoint = activeSet.size();
  for(unsigned int i=0; i<Kstore.getRows(); i++)
  {
    Kstore.setVal(pkern->computeElement(*pX, i, *pX, index), i, activePoint);      
  }
  // add white noise term to relevant index.
  double val = Kstore.getVal(index, activePoint);
  Kstore.setVal(val+pkern->getWhite(), index, activePoint);
  double lValInv = 0.0;
  double vs = 0.0;
  double ms = 0.0;
  double sVal = 0.0;
  for(unsigned int c=0; c<numCovStruct; c++)
  {      
    lValInv = sqrt(nu.getVal(index, c));
    // set s from the kernel -- it is a column vector..
    Kstore.getMatrix(s, 0, getNumData()-1, activePoint, activePoint);
    M[c].getMatrix(a, 0, getNumActive()-1, index, index);
    s.gemv(M[c], a, -1.0, 1.0, "t");
    if(lValInv<NULOW)
      cout << "Warning: square root of nu is " << lValInv << endl;
    // place the vector s at the bottom of M.
    s.trans();
    M[c].setMatrix(activePoint, 0, s);
    s.trans();
    M[c].scaleRow(activePoint, lValInv);
    a.trans(); // turn a into a row vector.
    L[c].setMatrix(activePoint, 0, a);
    L[c].setVal(1/lValInv, activePoint, activePoint);
    a.trans(); // turn a into a column vector.
    // update the varSigma and mu systems.
    double varSig = 0.0;
    for(unsigned int i=0; i<getNumData(); i++)
    {
      sVal = s.getVal(i, 0);
      varSig = pnoise->getVarSigma(i, c)
      -sVal*sVal*nu.getVal(index, c);
      if(isnan(varSig))
	cout << "varSigma is varSig" << endl;
      pnoise->setVarSigma(varSig, i, c);
      pnoise->setMu(pnoise->getMu(i, c) + g.getVal(index, c)*sVal, i, c);
    }
  }
  // this happens for spherical noise models.
  if(numCovStruct==1 && getOutputDim() > 1)
  {
    double varSig = 0.0;
    for(unsigned int c=1; c<getOutputDim(); c++)
    {

      for(unsigned int i=0; i<getNumData(); i++)
      {
	sVal = s.getVal(i, 0);
	varSig = pnoise->getVarSigma(i, c)
	-sVal*sVal*nu.getVal(index, c);
	pnoise->setVarSigma(varSig, i, c);
	pnoise->setMu(pnoise->getMu(i, c) + g.getVal(index, c)*sVal, i, c);
      }
    }
  }

}
unsigned int CIvm::selectPointAdd() 
{
  // returns data index of point to add.
  unsigned int index = 0;
  switch(selectionCriterion)
  {
  case RANDOM:
    index = randomPointAdd();
    break;
  case ENTROPY:
    index = entropyPointAdd();
    break;
  case RENTROPY:
    if(activeSet.size()>0)
      index = entropyPointAdd();
    else
      index = randomPointAdd();
    break;
  default:
    throw ndlexceptions::NotImplementedError("Data point selection type not known");
  }
  return index;
}
unsigned int CIvm::entropyPointAdd()
{
  // choose point from inactive set to add via entropy selection.
  vector<double> delta;
  delta.reserve(inactiveSet.size());
  for(unsigned int i=0; i<inactiveSet.size(); i++)
    delta.push_back(entropyChangeAdd(inactiveSet[i]));
  vector<double>::iterator maxVal = 
  max_element(delta.begin(), delta.end());
  changeEntropy(*maxVal);  // store global entropy change.
  return inactiveSet[maxVal - delta.begin()];
}

unsigned int CIvm::randomPointAdd() 
{
  // choose point from inactive set to add randomly.
  // Fix here 16/7/2007 --- change use of rand.
  double prop = ndlutil::rand();
  unsigned int index = (int)(prop*inactiveSet.size());
  index = inactiveSet[index];
  changeEntropy(entropyChangeAdd(index));
  return index;
}

double CIvm::entropyChangeAdd(unsigned int index) const
{
  // compute the entropy change associated with point addition.
  // make sure that index is in the inactive set.
  SANITYCHECK(find(inactiveSet.begin(), inactiveSet.end(), index)!=inactiveSet.end());
  double entChange=0.0;
  if(isSpherical())
  {
    entChange = -.5*log(1-pnoise->getVarSigma(index, 0)
			*nu.getVal(index, 0)+1e-300)*getOutputDim();
  }
  else
  {
    for(unsigned int j=0; j<getOutputDim(); j++)
      entChange += -.5*log(1-pnoise->getVarSigma(index, j)
			   *nu.getVal(index, j)+1e-300);
  }
  return entChange;
}
unsigned int CIvm::selectPointRemove()
{
  // returns data index of point to remove.
  unsigned int index = 0;
  switch(selectionCriterion)
  {
  case RANDOM:
    index = randomPointRemove();
    break;
  case ENTROPY:
  case RENTROPY:
    index = entropyPointRemove();
    break;
  default:
    throw ndlexceptions::NotImplementedError("Data point selection type not known");
  }
  return index;
}
unsigned int CIvm::entropyPointRemove() 
{
  vector<double> delta;
  delta.reserve(activeSet.size());
  for(unsigned int i=0; i<activeSet.size(); i++)
    delta.push_back(entropyChangeRemove(activeSet[i]));
  vector<double>::iterator maxVal = 
  max_element(delta.begin(), delta.end());
  changeEntropy(*maxVal);
  return inactiveSet[maxVal - delta.begin()];
}

unsigned int CIvm::randomPointRemove() 
{
  unsigned int index = rand();
  index = (index*activeSet.size())/RAND_MAX;
  index = activeSet[index];
  changeEntropy(entropyChangeRemove(index));
  return index;
}

double CIvm::entropyChangeRemove(unsigned int index) const
{
  // compute entropy change associated with point removal.
  // make sure that index is in the active set.
  SANITYCHECK(find(activeSet.begin(), activeSet.end(), index)!=activeSet.end());
  double entChange = 0.0;
  if(isSpherical())
  {
    entChange = -.5*log(1-pnoise->getVarSigma(index, 0)
			*beta.getVal(activeSet[index], 0)+1e-300)*getOutputDim();
  }
  else
  {
    for(unsigned int j=0; j<getOutputDim(); j++)
      entChange += -.5*log(1-pnoise->getVarSigma(index, j)
			   *beta.getVal(activeSet[index], j)+1e-300);
  }
  return entChange;
}
void CIvm::updateNuG()
{
  for(unsigned int i=0; i<getNumData(); i++)
    pnoise->getNuG(g, nu, i);
}
void CIvm::updateK() const
{
  double kVal=0.0;
  for(unsigned int i=0; i<activeSet.size(); i++)
  {
    K.setVal(pkern->diagComputeElement(activeX, i), i, i);
    for(unsigned int j=0; j<i; j++)
    {
      kVal=pkern->computeElement(activeX, i, activeX, j);
      K.setVal(kVal, i, j);
      K.setVal(kVal, j, i);
    }
  }
  K.setSymmetric(true);
}
void CIvm::updateInvK(unsigned int dim) const
{
  invK.deepCopy(K);
  for(unsigned int i=0; i<getNumActive(); i++)
    invK.setVal(invK.getVal(i, i) + 1/beta.getVal(i, dim), i, i);
  invK.setSymmetric(true);
  CMatrix U(chol(invK));
  logDetK = logDet(U); 
  invK.pdinv(U);
}
  
double CIvm::logLikelihood() const
{
  double L=0.0;
  updateK();
  CMatrix invKm(invK.getRows(), 1);
  if(isSpherical())
  {
    updateInvK(0);
  }
  for(unsigned int j=0; j<m.getCols(); j++)
  {
    if(!isSpherical())
      updateInvK(j);
    invK.setSymmetric(true);
    invKm.symvColCol(0, invK, m, j, 1.0, 0.0, "u");
    L -= .5*(logDetK + invKm.dotColCol(0, m, j));
  }
  L+=pkern->priorLogProb();
  return L;
}  
double CIvm::logLikelihoodGradient(CMatrix& g) const
{
  DIMENSIONMATCH(g.getRows()==1);
  DIMENSIONMATCH(g.getCols()==getOptNumParams());
  g.zeros();
  CMatrix tempG(1, getOptNumParams());
  updateK();
  if(isSpherical())
  {
    updateInvK(0);
  }
  for(unsigned int j=0; j<m.getCols(); j++)
  {
    if(!isSpherical())
    {
      updateInvK(j);
    }
    updateCovGradient(j);
    if(j==0)
      pkern->getGradTransParams(tempG, activeX, covGrad, true);
    else
      pkern->getGradTransParams(tempG, activeX, covGrad, false);
      
    g+=tempG;
      
  }
  return logLikelihood();
}


#ifdef _NDLMATLAB
CIvm::CIvm(CMatrix* inData, 
	   CMatrix* targetData, 
	   CKern* pkernel, 
	   CNoise* pnoiseModel, 
	   const string ivmInfoFile, 
	   const string ivmInfoVariable, 
	   int verbos) : 
  CMapModel(inData->getCols(), targetData->getCols(), inData->getRows()), CProbabilisticOptimisable(), pX(inData), py(targetData), pkern(pkernel), pnoise(pnoiseModel)
{
  _init();
  DIMENSIONMATCH(pX->getRows()==py->getRows());
  setVerbosity(verbos);
  readMatlabFile(ivmInfoFile, ivmInfoVariable);
  initStoreage(); // storeage has to be allocated after finding active set size.
  for(unsigned int i=0; i<getNumData(); i++)
    for(unsigned int j=0; j<getNumActive(); j++)
      Kstore.setVal(pkern->computeElement(*pX, i, *pX, activeSet[j]), i, j);
  
  Kstore.getMatrix(K, activeSet, 0, getNumActive()-1);

  for(unsigned int i=0; i<activeSet.size(); i++)
    K.setVal(pkern->diagComputeElement(*pX, activeSet[i]), i, i);
  for(unsigned int j=0; j<numCovStruct; j++)
  {
    L[j].deepCopy(K);
    for(unsigned int i=0; i<getNumActive(); i++)
    {
      double lval = L[j].getVal(i, i);
      lval += 1/beta.getVal(i, j);
      L[j].setVal(lval, i, i);
    }
      
    L[j].setSymmetric(true);
    L[j].chol("L");
    Linv[j].deepCopy(L[j]);
    // TODO should not use regular inverse here as matrix is lower triangular.
    Linv[j].inv();
    M[j].gemm(Linv[j], Kstore, 1.0, 0.0, "n", "t");
  }
  for(unsigned int i=0; i<getNumActive(); i++)
  {
    activeX.copyRowRow(i, *pX, activeSet[i]);
    activeY.copyRowRow(i, *py, activeSet[i]);
  }
  CMatrix mu(getNumData(), getOutputDim());
  CMatrix varSigma(getNumData(), getOutputDim());
  posteriorMeanVar(mu, varSigma, *pX);
  pnoise->setMus(mu);
  pnoise->setVarSigmas(varSigma);
  updateNuG();
}
mxArray* CIvm::toMxArray() const
{
  int dims[1];
  dims[0]=1;
  const char* fieldNames[]={"I", "J", "m", "beta"};
  mxArray* matlabArray = mxCreateStructArray(1, dims, 4, fieldNames);

  // The I and J fields.
  vector<unsigned int> activeMatlab = activeSet;
  for(unsigned int i=0; i<activeMatlab.size(); i++)
    activeMatlab[i]++;
  mxSetField(matlabArray, 0, "I", convertMxArray(activeMatlab));
  vector<unsigned int> inactiveMatlab = inactiveSet;
  for(unsigned int i=0; i<inactiveMatlab.size(); i++)
    inactiveMatlab[i]++;
  mxSetField(matlabArray, 0, "J", convertMxArray(inactiveMatlab));
  
  // Other matrix fields.
  CMatrix tempM(getNumData(), m.getCols());
  CMatrix tempB(getNumData(), beta.getCols());
  for(unsigned int i=0; i<activeSet.size(); i++)
  {
    tempM.copyRowRow(activeSet[i], m, i);
    tempB.copyRowRow(activeSet[i], beta, i);
  }
  
  mxSetField(matlabArray, 0, "m", tempM.toMxArray());
  mxSetField(matlabArray, 0, "beta", tempB.toMxArray());
  return matlabArray;
}
void CIvm::fromMxArray(const mxArray* matlabArray)
{
  string mxType = mxArrayExtractStringField(matlabArray, "type");
  if(mxType!=getType())
  {
    throw ndlexceptions::FileReadError("Error mismatch between saved type, " + mxType + ", and Class type, " + getType() + ".");
  }
  activeSet = mxArrayExtractVectorUintField(matlabArray, "I");
  for(unsigned int i=0; i<activeSet.size(); i++)
    activeSet[i]--;
  inactiveSet = mxArrayExtractVectorUintField(matlabArray, "J");
  for(unsigned int i=0; i<inactiveSet.size(); i++)
    inactiveSet[i]--;
  setNumActive(activeSet.size());
  CMatrix tempM;
  tempM.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "m"));
  CMatrix tempB;
  tempB.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "beta"));
  m.resize(getNumActive(), tempM.getCols());
  beta.resize(getNumActive(), tempB.getCols());
  for(unsigned int i=0; i<activeSet.size(); i++)
  {
    m.copyRowRow(i, tempM, activeSet[i]);
    beta.copyRowRow(i, tempB, activeSet[i]);
  }
  BOUNDCHECK(getNumActive()<getNumData());
  DIMENSIONMATCH(m.getCols()==getOutputDim());
  DIMENSIONMATCH(beta.dimensionsMatch(m));
  // TODO check that I and J cover 1:numData
}
#else /* not _NDLMATLAB */
#endif
void CIvm::optimise(unsigned int maxIters, unsigned int kernIters, unsigned int noiseIters)
{
  if(getVerbosity()>2)
  {
    cout << "Initial model:" << endl;
    display(cout);
  }
  
  if(kernIters>0 || noiseIters>0)
  {
    for(unsigned int iters=0; iters<maxIters; iters++)
    {
      
      if(getVerbosity()>1)
	cout << "IVM External Iteration: " << iters+1 << endl;
      if(kernIters>0)
      {
	init();
	selectPoints();
	if(getVerbosity()>2 && getOptNumParams()<10)
	  checkGradients();
	if(getVerbosity()>1)
	  cout << "Optimising kernel parameters ..." <<endl;
	setMaxIters(kernIters);
	runDefaultOptimiser();
	if(getVerbosity()>1)
	  cout << "... done. " << endl;
	if(getVerbosity()>2)
	  pkern->display(cout);
      }
      if(noiseIters>0)
      {
	init();
	selectPoints();
	if(getVerbosity()>2 && pnoise->getOptNumParams()<10)
	  pnoise->checkGradients();
	if(getVerbosity()>1)
	  cout << "Optimising noise parameters ..." <<endl;
	pnoise->setMaxIters(noiseIters);
	pnoise->runDefaultOptimiser();
	if(getVerbosity()>1)
	  cout << "... done." <<endl;
	if(getVerbosity()>2)
	  pnoise->display(cout);
      }
    }
  }
  init();
  selectPoints();
  if(getVerbosity()>0)
    display(cout);
}
bool CIvm::equals(const CIvm& model, double tol) const
{
  if(!pnoise->equals(*model.pnoise, tol))
    return false;
  if(!pkern->equals(*model.pkern, tol))
    return false;
  if(!m.equals(model.m, tol))
    return false;
  if(!beta.equals(model.beta, tol))
    return false;
  if(activeSet!=model.activeSet)
    return false;
  if(inactiveSet!=model.inactiveSet)
    return false;
  return true;
}
void CIvm::display(ostream& os) const 
{
  cout << "IVM Model: " << endl;
  cout << "Active Set Size: " << getNumActive() << endl;
  cout << "Kernel Type: " << endl;
  pkern->display(os);
  cout << "Noise Type: " << endl;
  pnoise->display(os);
}

void CIvm::updateCovGradient(unsigned int index) const
{
  CMatrix invKm(invK.getRows(), 1);
  invK.setSymmetric(true);
  invKm.symvColCol(0, invK, m, index, 1.0, 0.0, "u");
  covGrad.deepCopy(invK);
  covGrad.syr(invKm, -1.0, "u");
  covGrad.scale(-0.5);
}

void CIvm::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "numData", getNumData());
  writeToStream(out, "outputDim", getOutputDim());
  writeToStream(out, "inputDim", getInputDim());
  writeToStream(out, "numActive", getNumActive());

  pkern->toStream(out);
  pnoise->toStream(out);

  writeToStream(out, "activeSet", activeSet);
  activeY.toStream(out);
  activeX.toStream(out);
  m.toStream(out);
  beta.toStream(out);
}
void CIvm::readParamsFromStream(istream& in)
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
  setNumActive(readIntFromStream(in, "numActive"));


  // read kernel from the stream.
  //delete pkern; --- want to destroy it if it exists ... but not sure how to.
  pkern = readKernFromStream(in);

  // read noise model from the stream.
  //delete pnoise -- same as above.
  pnoise = readNoiseFromStream(in);

  // initialise storage
  initStoreage();
  activeSet = readVectorUintFromStream(in, "activeSet");
  if(activeSet.size() != getNumActive())
    throw ndlexceptions::StreamFormatError("numActive", "Number of active points does not match active set size.");
  activeY.fromStream(in);
  if(activeY.getCols() !=getOutputDim())
    throw ndlexceptions::StreamFormatError("outputDim", "Number of columns of activeY does not match output dimension.");
  if(activeY.getRows() != getNumActive())
    throw ndlexceptions::StreamFormatError("numActive", "Number of rows of activeY does not match active set size.");
    
  activeX.fromStream(in);
  if(activeX.getCols() !=getInputDim())
    throw ndlexceptions::StreamFormatError("inputDim", "Number of columns of activeX does not match input dimension.");
  if(activeX.getRows() != getNumActive())
    throw ndlexceptions::StreamFormatError("numActive", "Number of rowss of activeX does not match active set size.");
    
  m.fromStream(in);
  if(m.getCols() != getOutputDim())
    throw ndlexceptions::StreamFormatError("outputDim", "Number of columns of m does not match output dimension.");
  if(m.getRows() != getNumActive())
    throw ndlexceptions::StreamFormatError("numActive", "Number of rowss of m does not match active set size.");
    
  beta.fromStream(in);
  if(beta.getCols() !=getOutputDim())
    throw ndlexceptions::StreamFormatError("outputDim", "Number of columns of beta does not match output dimension.");
  if(beta.getRows() != getNumActive())
    throw ndlexceptions::StreamFormatError("numActive", "Number of rowss of beta does not match active set size.");

  updateK();
  updateInvK();
  for(unsigned int j=0; j<numCovStruct; j++)
  {
    L[j].deepCopy(K);
    for(unsigned int i=0; i<getNumActive(); i++)
    {
      double lval = L[j].getVal(i, i);
      lval += 1/beta.getVal(i, j);
      L[j].setVal(lval, i, i);
    }
    L[j].setSymmetric(true);
    L[j].chol("L");
    Linv[j].deepCopy(L[j]);
    // TODO should not use regular inverse here as matrix is lower triangular.
    Linv[j].inv();
  }
}
void writeIvmToStream(const CIvm& model, ostream& out)
{
  model.toStream(out);
}
void writeIvmToFile(const CIvm& model, const string modelFileName, const string comment)
{
  if(model.getVerbosity()>0)
    cout << "Saving model file." << endl;
  ofstream out(modelFileName.c_str());
  if(!out) throw ndlexceptions::FileWriteError(modelFileName);
  if(comment.size()>0)
    out << "# " << comment << endl;
  writeIvmToStream(model, out);
  out.close();
}


CIvm* readIvmFromStream(istream& in)
{
  CIvm* pmodel = new CIvm();
  pmodel->fromStream(in);
  return pmodel;
}

CIvm* readIvmFromFile(const string modelFileName, int verbosity)
{
  // File is m, beta, X
  if(verbosity>0)
    cout << "Loading model file." << endl;
  ifstream in(modelFileName.c_str());
  if(!in.is_open()) throw ndlexceptions::FileReadError(modelFileName);
  CIvm* pmodel;
  try
  {
    pmodel = readIvmFromStream(in);
  }
  catch(ndlexceptions::StreamFormatError err)
  {
    throw ndlexceptions::FileFormatError(modelFileName, err);
  }
  if(verbosity>0)
    cout << "... done." << endl;
  in.close();
  return pmodel;
}

