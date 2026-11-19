#include "CPpa.h"

CPpa::CPpa(const CMatrix& inData, const CMatrix& targetData,
           CKern& kernel, CNoise& noiseModel, int verbosity)
  : CMapModel(),
    X(inData), y(targetData), kern(kernel),
    noise(noiseModel), numTarget(y.getCols()), numData(y.getRows())
{
  assert(X.getRows()==y.getRows());
  setInputDim(X.getCols());
  setOutputDim(numTarget);
  setNumData(numData);
  setVerbosity(verbosity);
  terminate = false;
  noise.setVerbosity(getVerbosity());
  varUpdate = false;
  loadedModel = false;
  convergenceTol = 1e-6;
  numTrainData = numData;
  optimisingKernel = false;
  kernOptIters = 100;
  M = nullptr;
  L = nullptr;
  Linv = nullptr;
  init();
}

CPpa::CPpa(const CMatrix& trX, const CMatrix& trY,
           const CMatrix& mmat, const CMatrix& betamat, CKern& kernel,
           CNoise& noiseModel, int verbosity)
  : CMapModel(),
    X(trX), y(trY), activeX(trX), trainY(trY), wasM(mmat), beta(betamat),
    kern(kernel), noise(noiseModel),
    numTarget(trY.getCols()), numData(trY.getRows())
{
  assert(activeX.getRows()==trainY.getRows());
  setInputDim(activeX.getCols());
  setOutputDim(numTarget);
  setNumData(numData);
  setVerbosity(verbosity);
  terminate = false;
  noise.setVerbosity(getVerbosity());
  varUpdate = false;
  loadedModel = true;
  convergenceTol = 1e-6;
  numTrainData = activeX.getRows();
  optimisingKernel = false;
  kernOptIters = 100;
  M = nullptr;
  L = nullptr;
  Linv = nullptr;
  initStoreage();
  fBar.deepCopy(wasM);
  updateK();
  updateInvK();
}
void CPpa::test(const CMatrix& ytest, const CMatrix& Xin) const
{
  assert(ytest.getCols()==numTarget);
  assert(ytest.getRows()==Xin.getRows());
  CMatrix muout(Xin.getRows(), numTarget);
  CMatrix varSigmaOut(Xin.getRows(), numTarget);
  posteriorMeanVar(muout, varSigmaOut, Xin);
  noise.test(muout, varSigmaOut, ytest);
}
void CPpa::likelihoods(CMatrix& pout, CMatrix& yTest, const CMatrix& Xin) const
{
  assert(pout.getCols()==numTarget);
  assert(pout.getRows()==Xin.getRows());
  assert(yTest.dimensionsMatch(pout));
  CMatrix muout(Xin.getRows(), numTarget);
  CMatrix varSigmaOut(Xin.getRows(), numTarget);
  posteriorMeanVar(muout, varSigmaOut, Xin);
  noise.likelihoods(pout, muout, varSigmaOut, yTest);
}
double CPpa::logLikelihood() const
{
  if(optimisingKernel)
    return kernelLogLikelihood();
  return approxLogLikelihood();
}

double CPpa::logLikelihoodGradient(CMatrix& g) const
{
  if(optimisingKernel)
    kernelLogLikelihoodGradient(g);
  else
    approxLogLikelihoodGradient(g);
  return logLikelihood();
}

double CPpa::outGradParams(CMatrix& /*g*/, const CMatrix& /*Xin*/,
                           unsigned int /*pointNo*/, unsigned int /*outputNo*/) const
{
  throw ndlexceptions::NotImplementedError("CPpa::outGradParams not implemented (CIP-0023).");
}

double CPpa::outGradX(CMatrix& /*g*/, const CMatrix& /*Xin*/,
                      unsigned int /*pointNo*/, unsigned int /*outputNo*/) const
{
  throw ndlexceptions::NotImplementedError("CPpa::outGradX not implemented (CIP-0023).");
}

void CPpa::writeParamsToStream(ostream& out) const
{
  writePpaToStream(*this, out);
}

void CPpa::readParamsFromStream(istream& /*in*/)
{
  throw ndlexceptions::NotImplementedError("CPpa::readParamsFromStream: use readPpaFromStream.");
}

void CPpa::init()
{
  numTrainData = numData;
  wasM.resize(numTrainData, numTarget);
  // Scalar B: one precision per output (MATLAB options.scalarB=1).
  beta.resize(1, numTarget);
  beta.setVals(1.0);
  initStoreage();
  activeX.deepCopy(X);
  trainY.deepCopy(y);
  initVals();
  updateK();
  updateInvK();
  // MATLAB ppaInit ends with ppaEStep.
  eStep();
}

void CPpa::updateK() const
{
  // Full-data Gram on activeX (== X for the live path). Matches MATLAB kern.Kstore.
  kern.compute(K, activeX);
  K.setSymmetric(true);
  // Kstore is N x M in the old IVM-shaped layout; for full-data PPA keep N x N in K.
  // Mirror into Kstore when shapes match so legacy readers see a Gram.
  if(Kstore.getRows()==K.getRows() && Kstore.getCols()==K.getCols())
    Kstore.deepCopy(K);
}

void CPpa::updateInvK(int /*index*/) const
{
  // MATLAB: kern.invKstore = pdinv(kern.Kstore). Not K+B (that enters q(fBar)).
  CMatrix U(chol(K));
  logDetK = logDet(U);
  invK.deepCopy(K);
  invK.pdinv(U);
  invK.setSymmetric(true);
}

void CPpa::updateSite(int /*index*/)
{
}

double CPpa::logLikelihood(const CMatrix& yTest, const CMatrix& Xin) const
{
  assert(yTest.getRows()==Xin.getRows());
  assert(getNumInputs()==Xin.getCols());
  assert(getNumProcesses()==yTest.getCols());
  CMatrix muout(Xin.getRows(), numTarget);
  CMatrix varSigmaOut(Xin.getRows(), numTarget);
  posteriorMeanVar(muout, varSigmaOut, Xin);
  return noise.logLikelihood(muout, varSigmaOut, yTest)+kern.priorLogProb();
}
void CPpa::out(CMatrix& yout, const CMatrix& Xin) const
{
  assert(yout.getCols()==numTarget);
  assert(yout.getRows()==Xin.getRows());
  CMatrix muout(Xin.getRows(), numTarget);
  CMatrix varSigmaOut(Xin.getRows(), numTarget);
  posteriorMeanVar(muout, varSigmaOut, Xin);
  noise.out(yout, muout, varSigmaOut);
}
void CPpa::out(CMatrix& yout, CMatrix& probout, const CMatrix& Xin) const
{
  assert(yout.getCols()==numTarget);
  assert(yout.getRows()==Xin.getRows());
  CMatrix muout(Xin.getRows(), numTarget);
  CMatrix varSigmaOut(Xin.getRows(), numTarget);
  posteriorMeanVar(muout, varSigmaOut, Xin);
  noise.out(yout, probout, muout, varSigmaOut);
}
void CPpa::posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xin) const
{
  // MATLAB ppaPosteriorMeanVar: invSigma = inv(diag(1/B)+K), mu = k(X)' invSigma fBar.
  assert(mu.getCols()==numTarget);
  assert(varSigma.getCols()==numTarget);
  DIMENSIONMATCH(mu.getRows()==Xin.getRows());
  DIMENSIONMATCH(varSigma.getRows()==Xin.getRows());
  updateK();
  CMatrix kX(numTrainData, Xin.getRows());
  kern.compute(kX, activeX, Xin); // numTrainData x N*
  const double b0 = beta.getVal(0, 0);
  for(int j=0; j<numTarget; j++)
    {
      const double b = (beta.getCols()==1) ? b0 : beta.getVal(0, j);
      CMatrix Sigma(K);
      for(int i=0; i<numTrainData; i++)
        Sigma.setVal(Sigma.getVal(i, i) + 1.0 / b, i, i);
      Sigma.setSymmetric(true);
      CMatrix U(chol(Sigma));
      CMatrix invSigma(Sigma);
      invSigma.pdinv(U);
      invSigma.setSymmetric(true);

      CMatrix Kinvk(numTrainData, Xin.getRows());
      Kinvk.symm(invSigma, kX, 1.0, 0.0, "l", "l"); // invSigma * kX

      for(int n=0; n<Xin.getRows(); n++)
        {
          double vsVal = kern.diagComputeElement(Xin, n) - Kinvk.dotColCol(n, kX, n);
          if(vsVal < 0.0)
            vsVal = 0.0;
          varSigma.setVal(vsVal, n, j);
        }
      CMatrix fBarCol(numTrainData, 1);
      for(int i=0; i<numTrainData; i++)
        fBarCol.setVal(fBar.getVal(i, j), i, 0);
      CMatrix muCol(Xin.getRows(), 1);
      muCol.gemv(Kinvk, fBarCol, 1.0, 0.0, "T");
      for(int n=0; n<Xin.getRows(); n++)
        mu.setVal(muCol.getVal(n, 0), n, j);
    }
}
void CPpa::initStoreage()
{
  if(noise.isSpherical())
    numCovStruct = 1;
  else
    numCovStruct = numTarget;
  Kstore.resize(numData, numTrainData);
  nu.resize(numData, numTarget);
  g.resize(numData, numTarget);
  gamma.resize(numData, numTarget);

  f.resize(numData, numTarget);
  ff.resize(numData, numTarget);
  fBar.resize(numData, numTarget);
  fBarfBar.resize(numData, numData);
  fBarfBar.setSymmetric(true);
  C.clear();
  for(int i=0; i<numTarget; i++)
    C.push_back(new CMatrix(numData, numData));

  // set up s, a and ainv
  s.resize(numData, 1); // s is a columnrow vector.
  a.resize(numTrainData, 1); // a is a column vector.
  ainv.resize(1, numTrainData); // ainv is a row vector.
  
  // set up L, M and Linv
  M = new CMatrix[numCovStruct];

  L = new CMatrix[numCovStruct];
  Linv = new CMatrix[numCovStruct];
  for(int c=0; c<numCovStruct; c++)
    {
      M[c].resize(numTrainData, numData);
      L[c].resize(numTrainData, numTrainData);
      Linv[c].resize(numTrainData, numTrainData);
    }
  // set up K invK and covGrad
  activeX.resize(numTrainData, X.getCols());
  trainY.resize(numTrainData, numTarget);
  covGrad.resize(numTrainData, numTrainData);
  covGrad.setSymmetric(true);
  
  K.resize(numTrainData, numTrainData);
  K.setSymmetric(true);
  invK.resize(numTrainData, numTrainData);
  invK.setSymmetric(true);

}

void CPpa::initVals()
{
  // set Kstore to zeros numData, numTrainData.
  Kstore.setVals(0.0);
  wasM.setVals(0.0);
  // beta is set in init() to scalar 1 (MATLAB ppaInit); do not clear it here.

  f.setVals(0.0);
  ff.setVals(0.0);
  fBar.setVals(0.0);
  fBarfBar.setVals(0.0);
  gamma.setVals(0.0);
  nu.setVals(0.0);
  g.setVals(0.0);
  // Site variance for the first E-step is 1/B (set in eStep); seed diag(K) only as a placeholder.
  noise.setMus(0.0);
  double dk=0.0;
  for(int i=0; i<numData; i++)
    {
      dk = kern.diagComputeElement(X, i);
      for(int j=0; j<numTarget; j++)
	{
	  noise.setVarSigma(dk, i, j);
	}
    }
  for(int c=0; c<numCovStruct; c++)
    {
      M[c].setVals(0.0);
      L[c].setVals(0.0);
      Linv[c].setVals(0.0);
    }
  invK.zeros();
  covGrad.zeros();
}
void CPpa::updateNuG()
{
  for(int i=0; i<numData; i++)
    noise.getNuG(g, nu, i);
}
  
double CPpa::approxLogLikelihood() const
{
  // MATLAB ppaCalculateLogLike (scalar B path).
  updateK();
  updateInvK();
  double L = 0.0;
  CMatrix vs(numData, numTarget);
  for(int j=0; j<numTarget; j++)
    {
      const double b = (beta.getCols()==1) ? beta.getVal(0, 0) : beta.getVal(0, j);
      CMatrix invC(invK);
      for(int i=0; i<numData; i++)
        invC.setVal(invC.getVal(i, i) + b, i, i);
      invC.setSymmetric(true);
      CMatrix UinvC(chol(invC));
      const double logDetInvC = logDet(UinvC);

      CMatrix invKfBarfBar(numData, numData);
      invKfBarfBar.gemm(fBarfBar, invK, 1.0, 0.0, "n", "n");
      const double part1 = -0.5 * logDetK - 0.5 * logDetInvC;
      const double part2 = -0.5 * invKfBarfBar.trace();
      const double part3 = 0.5 * static_cast<double>(numData);
      double part4 = 0.0;
      for(int i=0; i<numData; i++)
        {
          const double diagFF = fBarfBar.getVal(i, i);
          const double fBarVal = fBar.getVal(i, j);
          part4 += b * (diagFF - fBarVal * fBarVal);
          vs.setVal(1.0 / b, i, j);
        }
      part4 *= -0.5;
      CMatrix fBarCol(numData, 1);
      CMatrix yCol(numData, 1);
      for(int i=0; i<numData; i++)
        {
          fBarCol.setVal(fBar.getVal(i, j), i, 0);
          yCol.setVal(y.getVal(i, j), i, 0);
        }
      CMatrix vsCol(numData, 1);
      vsCol.setVals(1.0 / b);
      const double part5 = noise.logLikelihood(fBarCol, vsCol, yCol);
      L += part1 + part2 + part3 + part4 + part5;
    }
  L += kern.priorLogProb();
  return L;
}

void CPpa::approxLogLikelihoodGradient(CMatrix& g) const
{
  // Outer monitoring gradient: use kernel M-step gradient as a practical stand-in.
  kernelLogLikelihoodGradient(g);
}

double CPpa::kernelLogLikelihood() const
{
  // MATLAB ppaKernelLogLikelihood: m = expectations.f, inv(K + diag(1/B)).
  updateK();
  double L = 0.0;
  CMatrix Kbeta(K);
  const double b0 = beta.getVal(0, 0);
  for(int i=0; i<numData; i++)
    Kbeta.setVal(Kbeta.getVal(i, i) + 1.0 / b0, i, i);
  Kbeta.setSymmetric(true);
  CMatrix U(chol(Kbeta));
  const double logDetTerm = logDet(U);
  CMatrix invKbeta(Kbeta);
  invKbeta.pdinv(U);
  invKbeta.setSymmetric(true);

  CMatrix invKm(numData, 1);
  for(int j=0; j<numTarget; j++)
    {
      invKm.symvColCol(0, invKbeta, f, j, 1.0, 0.0, "u");
      L -= 0.5 * logDetTerm + 0.5 * invKm.dotColCol(0, f, j);
    }
  L += kern.priorLogProb();
  return L;
}

void CPpa::kernelLogLikelihoodGradient(CMatrix& g) const
{
  assert(g.getRows()==1);
  assert(g.getCols()==getOptNumParams());
  g.zeros();
  CMatrix tempG(1, getOptNumParams());
  updateK();
  CMatrix Kbeta(K);
  const double b0 = beta.getVal(0, 0);
  for(int i=0; i<numData; i++)
    Kbeta.setVal(Kbeta.getVal(i, i) + 1.0 / b0, i, i);
  Kbeta.setSymmetric(true);
  CMatrix U(chol(Kbeta));
  CMatrix invKbeta(Kbeta);
  invKbeta.pdinv(U);
  invKbeta.setSymmetric(true);

  for(int j=0; j<numTarget; j++)
    {
      // ppaCovarianceGradient(invK, m(:,j))
      CMatrix invKm(numData, 1);
      invKm.symvColCol(0, invKbeta, f, j, 1.0, 0.0, "u");
      covGrad.deepCopy(invKbeta);
      covGrad.syr(invKm, -1.0, "u");
      covGrad.scale(-0.5);
      covGrad.setSymmetric(true);
      kern.getGradTransParams(tempG, activeX, covGrad);
      g += tempG;
    }
  // kernPriorGradient is included inside getGradTransParams when priors exist.
}


void CPpa::optimise(const int maxIters, const int kernIters, const int /*noiseIters*/)
{
  // noiseIters reserved for a future β / noise M-step (MATLAB updateBeta).
  kernOptIters = kernIters;
  if(getVerbosity()>0)
    {
      cout << "Initial model:" << endl;
      display(cout);
    }
  int counter = 0;
  terminate = false;
  oldLogLike = approxLogLikelihood();
  while(!terminate && counter < maxIters)
    {
      counter++;
      mStep();
      eStep();
      logLike = approxLogLikelihood();
      double logLikeDiff = logLike - oldLogLike;
      if(getVerbosity()>1)
	cout << "PPA Iteration " << counter << ", log-likelihood change " << logLikeDiff << endl;
      if(logLikeDiff<0)
	cout << "Log likelihood went down" << endl;
      else if(logLikeDiff<convergenceTol)
	{
	  terminate=true;
	  cout << "Algorithm has converged" << endl;
	}
      oldLogLike = logLike;
    }
  if(!terminate)
    cout << "Algorithm reached maximum number of iterations." << endl;
  numIters = counter;
  if(getVerbosity()>0)
    {
      cout << "Final model:" << endl;
      display(cout);
    }
}
void CPpa::mStep()
{
  // MATLAB ppaUpdateKernel with varKern=0: optimise kernel then refresh K, invK.
  if(getOptNumParams()==0 || kernOptIters<=0)
    return;
  optimisingKernel = true;
  setMaxIters(static_cast<unsigned int>(kernOptIters));
  runDefaultOptimiser();
  optimisingKernel = false;
  updateK();
  updateInvK();
}

void CPpa::eStep()
{
  // Order matches MATLAB ppaEStep: fBar moments → nu/g/gamma → f, ff.
  updateExpectationsfBar();
  // noiseUpdateNuG(noise, fBar, 1./B, y)
  noise.setMus(fBar);
  for(int i=0; i<numData; i++)
    for(int j=0; j<numTarget; j++)
      {
        const double b = (beta.getCols()==1) ? beta.getVal(0, 0) : beta.getVal(0, j);
        noise.setVarSigma(1.0 / b, i, j);
      }
  updateNuG();
  for(int i=0; i<numData; i++)
    for(int j=0; j<numTarget; j++)
      {
	double gVal = g.getVal(i, j);
	gamma.setVal((gVal*gVal-nu.getVal(i, j))/2, i, j);
      }
  updateExpectationf();
  updateExpectationff();
  // Persist fBar in wasM so stream I/O / prediction after load stay consistent.
  wasM.deepCopy(fBar);
  logLike = logLikelihood();
}
void CPpa::updateExpectationf()
{
  for(int i=0; i<numData; i++)
    for(int j=0; j<numTarget; j++)
      {
        const double b = (beta.getCols()==1) ? beta.getVal(0, 0) : beta.getVal(0, j);
        f.setVal(fBar.getVal(i, j)+g.getVal(i,j)/b, i, j);
      }
}
void CPpa::updateExpectationff()
{
  for(int j=0; j<numTarget; j++)
    {
      const double b = (beta.getCols()==1) ? beta.getVal(0, 0) : beta.getVal(0, j);
      double b2 = b*b;
      for(int i=0; i<numData; i++)
	{
	  double mVal = fBar.getVal(i, j);
	  double ffVal = 2/b2*gamma.getVal(i, j) + 1/b + 2*mVal*f.getVal(i, j) - mVal*mVal;
	  ff.setVal(ffVal, i, j);
	}
    }
}
void CPpa::updateExpectationsfBar()
{
  // MATLAB ppaExpectfBar / ppaExpectfBarfBar (scalar B):
  //   Cov = inv(diag(B) + inv(K));  <fBar> = Cov * B * <f>;
  //   <fBar fBar'> = Cov + <fBar><fBar>'.
  updateK();
  updateInvK();
  DIMENSIONMATCH(f.getRows()==numData);
  DIMENSIONMATCH(fBar.getRows()==numData);
  DIMENSIONMATCH(fBarfBar.getRows()==numData && fBarfBar.getCols()==numData);

  for(int j=0; j<numTarget; j++)
    {
      const double b = (beta.getCols()==1) ? beta.getVal(0, 0) : beta.getVal(0, j);
      CMatrix invC(invK);
      for(int i=0; i<numData; i++)
        invC.setVal(invC.getVal(i, i) + b, i, i);
      invC.setSymmetric(true);
      CMatrix U(chol(invC));
      CMatrix Cov(invC);
      Cov.pdinv(U);
      Cov.setSymmetric(true);

      CMatrix bf(numData, 1);
      for(int i=0; i<numData; i++)
        bf.setVal(b * f.getVal(i, j), i, 0);

      CMatrix fBarCol(numData, 1);
      fBarCol.symvColCol(0, Cov, bf, 0, 1.0, 0.0, "u");
      for(int i=0; i<numData; i++)
        fBar.setVal(fBarCol.getVal(i, 0), i, j);

      // Second moment for this output into fBarfBar (single-output path uses the matrix).
      fBarfBar.deepCopy(Cov);
      fBarfBar.syr(fBarCol, 1.0, "u");
      fBarfBar.setSymmetric(true);
      if(j < static_cast<int>(C.size()) && C[j]!=nullptr)
        {
          C[j]->deepCopy(fBarfBar);
          C[j]->setSymmetric(true);
        }
    }
}
bool CPpa::equals(const CPpa& model, const double tol) const
{
  if(!noise.equals(model.noise, tol))
    return false;
  if(!kern.equals(model.kern, tol))
    return false;
  if(!wasM.equals(model.wasM, tol))
    return false;
  if(!beta.equals(model.beta, tol))
    return false;
  return true;
}
void CPpa::display(ostream& os) const 
{
  cout << "PPA Model: " << endl;
  cout << "Number of training data: " << numTrainData << endl;
  cout << "Kernel Type: " << endl;
  kern.display(os);
  cout << "Noise Type: " << endl;
  noise.display(os);
}

void CPpa::updateCovGradient(const int index) const
{
  CMatrix invKm(invK.getRows(), 1);
  invK.setSymmetric(true);
  invKm.symvColCol(0, invK, wasM, index, 1.0, 0.0, "u");
  covGrad.deepCopy(invK);
  covGrad.syr(invKm, -1.0, "u");
  covGrad.scale(-0.5);
}

#ifdef _NDLMATLAB

// TODO This is not in a working state.
CPpa::CPpa(const CMatrix& inData, 
	   const CMatrix& targetData, 
	   CKern& kernel, 
	   CNoise& noiseModel, 
	   const string ppaInfoFile, 
	   const string ppaInfoVariable, 
	   const int verbos) : 
  X(inData), y(targetData), 
  kern(kernel), noise(noiseModel), 
  numTarget(y.getCols()), numData(y.getRows()),  
  lastEntropyChange(0.0), cumEntropy(0.0), 
  varUpdate(false), terminate(false), loadedModel(true)
{
  setVerbosity(verbos);
  readMatlabFile(ppaInfoFile, ppaInfoVariable);
  initStoreage(); // storeage has to be allocated after finding training set size.
  for(int i=0; i<numData; i++)
    for(int j=0; j<numTrainData; j++)
      Kstore.setVal(kern.computeElement(X, i, X, j), i, j);
  
  Kstore.getMatrix(K, activeSet, 0, numTrainData-1);

  for(int i=0; i<activeSet.size(); i++)
    K.setVal(kern.diagComputeElement(X, activeSet[i]), i, i);
  K.writeMatlabFile("crap.mat", "K");
  for(int j=0; j<numCovStruct; j++)
    {
      L[j].deepCopy(K);
      L[j].updateMatlabFile("crap.mat", "KL");
      for(int i=0; i<numTrainData; i++)
	{
	  double lval = L[j].getVal(i, i);
	  lval += 1/beta.getVal(i, j);
	  L[j].setVal(lval, i, i);
	}
      L[j].updateMatlabFile("crap.mat", "KB");
      L[j].setSymmetric(true);
      L[j].chol("L");
      L[j].updateMatlabFile("crap.mat", "L");
      Linv[j].deepCopy(L[j]);
      // TODO should not use regular inverse here as matrix is lower triangular.
      Linv[j].inv();
      M[j].gemm(Linv[j], Kstore, 1.0, 0.0, "n", "t");
    }
  for(int i=0; i<numTrainData; i++)
    {
      activeX.copyRowRow(i, X, activeSet[i]);
      trainY.copyRowRow(i, X, activeSet[i]);
    }
  CMatrix mu(numData, numTarget);
  CMatrix varSigma(numData, numTarget);
  posteriorMeanVar(mu, varSigma, X);
  noise.setMus(mu);
  noise.setVarSigmas(varSigma);
  updateNuG();
}
mxArray* CPpa::toMxArray() const
{
  int dims[1];
  dims[0]=1;
  const char* fieldNames[]={"I", "J", "m", "beta"};
  mxArray* matlabArray = mxCreateStructArray(1, dims, 4, fieldNames);

  // The I and J fields.
  vector<int> activeMatlab = activeSet;
  for(int i=0; i<activeMatlab.size(); i++)
    activeMatlab[i]++;
  mxSetField(matlabArray, 0, "I", convertMxArray(activeMatlab));
  vector<int> inactiveMatlab = inactiveSet;
  for(int i=0; i<inactiveMatlab.size(); i++)
    inactiveMatlab[i]++;
  mxSetField(matlabArray, 0, "J", convertMxArray(inactiveMatlab));
  
  // Other matrix fields.
  CMatrix tempM(numData, m.getCols());
  CMatrix tempB(numData, beta.getCols());
  for(int i=0; i<activeSet.size(); i++)
    {
      tempM.copyRowRow(activeSet[i], m, i);
      tempB.copyRowRow(activeSet[i], beta, i);
    }
  
  mxSetField(matlabArray, 0, "m", tempM.toMxArray());
  mxSetField(matlabArray, 0, "beta", tempB.toMxArray());
  return matlabArray;
}
void CPpa::fromMxArray(const mxArray* matlabArray)
{
  activeSet = mxArrayExtractVectorIntField(matlabArray, "I");
  for(int i=0; i<activeSet.size(); i++)
    activeSet[i]--;
  inactiveSet = mxArrayExtractVectorIntField(matlabArray, "J");
  for(int i=0; i<inactiveSet.size(); i++)
    inactiveSet[i]--;
  numTrainData = activeSet.size();
  CMatrix tempM;
  tempM.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "m"));
  CMatrix tempB;
  tempB.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "beta"));
  m.resize(numTrainData, tempM.getCols());
  beta.resize(numTrainData, tempB.getCols());
  for(int i=0; i<activeSet.size(); i++)
    {
      m.copyRowRow(i, tempM, activeSet[i]);
      beta.copyRowRow(i, tempB, activeSet[i]);
    }
  assert(numTrainData<numData);
  assert(m.getCols()==numTarget);
  assert(beta.dimensionsMatch(m));
  // TODO check that I and J cover 1:numData
}
#else /* not _NDLMATLAB */
#endif

void writePpaToStream(const CPpa& model, ostream& out)
{
  out << "ppaVersion=" << PPAVERSION << endl;
  out << "numTrainData=" << model.getNumTrainData() << endl;
  out << "numProcesses=" << model.getNumProcesses() << endl;
  out << "numFeatures=" << model.getNumInputs() << endl;

  writeKernToStream(model.kern, out);
  writeNoiseToStream(model.noise, out);

  // Write out beta (row 0; scalar B is 1 x outputDim).
  for(int j=0; j<model.getNumProcesses()-1; j++)
    {
      out << model.beta.getVal(0, j) << " ";
    }
  out << model.beta.getVal(0, model.getNumProcesses()-1) << endl;

  // Wright out m, Y and X.
  for(int i=0; i<model.getNumTrainData(); i++)
    {
      out << model.getTrainingPoint(i) << " ";
      for(int j=0; j<model.getNumProcesses(); j++)
	{
	  double yval = model.y.getVal(model.getTrainingPoint(i), j);
	  if((yval - (int)yval)==0.0)
	    out << (int)yval << " ";
	  else
	    out << yval << " ";
	}
      for(int j=0; j<model.getNumProcesses(); j++)
	{
	  out << model.wasM.getVal(i, j) << " ";
	}
      for(int j=0; j<model.getNumInputs()-1; j++)
	{
	  double x = model.getTrainingX(i, j);
	  if(x!=0.0)
	    out << j+1 << ":" << x << " ";
	}
      double x = model.getTrainingX(i, model.getNumInputs()-1);
      if(x!=0.0)
	out << model.getNumInputs() << ":" << x << endl;
    }
}
void writePpaToFile(const CPpa& model, const string modelFileName, const string comment)
{
  if(model.getVerbosity()>0)
    cout << "Saving model file." << endl;
  ofstream out(modelFileName.c_str());
  if(!out) throw ndlexceptions::FileWriteError(modelFileName);
  out << setiosflags(ios::scientific);
  out << setprecision(17);
  if(comment.size()>0)
    out << "# " << comment << endl;
  writePpaToStream(model, out);
  out.close();
}

CPpa* readPpaFromStream(istream& in)
{
  string line;
  vector<string> tokens;
  // first line is version info.
  ndlstrutil::getline(in, line);
  ndlstrutil::tokenise(tokens, line, "=");
  if(tokens.size()>2 || tokens[0]!="ppaVersion")
    throw ndlexceptions::FileFormatError();
  // 0.1: historical; 0.2: CIP-0023 revive (same payload layout for these fields).
  if(tokens[1]!="0.1" && tokens[1]!="0.2")
    throw ndlexceptions::FileFormatError();

  // next line is training set size
  tokens.clear();
  ndlstrutil::getline(in, line);
  ndlstrutil::tokenise(tokens, line, "=");
  if(tokens.size()>2 || tokens[0]!="numTrainData")
    throw ndlexceptions::FileFormatError();
  int numTrainData=atoi(tokens[1].c_str());
  
  // next line is number of processes
  tokens.clear();
  ndlstrutil::getline(in, line);
  ndlstrutil::tokenise(tokens, line, "=");
  if(tokens.size()>2 || tokens[0]!="numProcesses")
    throw ndlexceptions::FileFormatError();
  int numProcesses=atoi(tokens[1].c_str());

  // next line is number of features
  tokens.clear();
  ndlstrutil::getline(in, line);
  ndlstrutil::tokenise(tokens, line, "=");
  if(tokens.size()>2 || tokens[0]!="numFeatures")
    throw ndlexceptions::FileFormatError();
  int numFeatures=atoi(tokens[1].c_str());

  CKern* pkern = readKernFromStream(in);
  CNoise* pnoise = readNoiseFromStream(in);

  // load in values for beta.
  CMatrix beta(1, numProcesses);
  tokens.clear();
  ndlstrutil::getline(in, line);
  ndlstrutil::tokenise(tokens, line, " ");
  for(int j=0; j<numProcesses; j++)
    {
      beta.setVal(atof(tokens[j].c_str()), j);
    }

  // Row layout: pointIndex, y[P], wasM/fBar[P], sparse features.
  // Heap-allocate so CPpa's const refs remain valid after return.
  CMatrix* pm = new CMatrix(numTrainData, numProcesses);
  CMatrix* pactiveX = new CMatrix(numTrainData, numFeatures, 0.0);
  CMatrix* ptrainY = new CMatrix(numTrainData, numProcesses);
  CMatrix* pbeta = new CMatrix(beta);
  for(int i=0; i<numTrainData; i++)
    {
      tokens.clear();
      ndlstrutil::getline(in, line);
      ndlstrutil::tokenise(tokens, line, " ");
      if(static_cast<int>(tokens.size()) < 1 + 2*numProcesses)
        throw ndlexceptions::FileFormatError("Corrupt file format.");
      for(int j=0; j<numProcesses; j++)
	{
	  ptrainY->setVal(atof(tokens[1+j].c_str()), i, j);
	}
      for(int j=0; j<numProcesses; j++)
	{
	  pm->setVal(atof(tokens[1+numProcesses+j].c_str()), i, j);
	}
      for(size_t j=static_cast<size_t>(1+2*numProcesses); j<tokens.size(); j++)
	{
	  int ind = tokens[j].find(':');
	  string featStr=tokens[j].substr(0, ind);
	  string featValStr=tokens[j].substr(ind+1, tokens[j].size()-ind);
	  int featNum = atoi(featStr.c_str());
	  if(featNum<1 || featNum>numFeatures)
	    throw ndlexceptions::FileFormatError("Corrupt file format.");
	  double featVal = atof(featValStr.c_str());
	  pactiveX->setVal(featVal, i, featNum-1);
	}
    }
  CPpa* pmodel= new CPpa(*pactiveX, *ptrainY, *pm, *pbeta, *pkern, *pnoise);
  return pmodel;
}

CPpa* readPpaFromFile(const string modelFileName, const int verbosity)
{
  // File is m, beta, X
  if(verbosity>0)
    cout << "Loading model file." << endl;
  ifstream in(modelFileName.c_str());
  if(!in.is_open()) throw ndlexceptions::FileReadError(modelFileName);
  CPpa* pmodel;
  try
    {
      pmodel = readPpaFromStream(in);
    }
  catch(ndlexceptions::FileFormatError err)
    {
      throw ndlexceptions::FileFormatError(modelFileName);
    }
  if(verbosity>0)
    cout << "... done." << endl;
  in.close();
  return pmodel;
}

