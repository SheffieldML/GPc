#include "CPpa.h"
CPpa::CPpa(const CMatrix& inData, const CMatrix& targetData, 
	   CKern& kernel, CNoise& noiseModel, const int verbosity) 
  : X(inData), y(targetData), kern(kernel), 
    noise(noiseModel), numTarget(y.getCols()), numData(y.getRows())
{
  assert(X.getRows()==y.getRows());
  setVerbosity(verbosity);
  terminate = false;
  noise.setVerbosity(getVerbosity());
  varUpdate = false;
  loadedModel = false;
  init();
}
CPpa::CPpa(const CMatrix& trX, const CMatrix& trY, 
     const CMatrix& mmat, const CMatrix& betamat, CKern& kernel, 
     CNoise& noiseModel,
	   const int verbosity) : X(trX), y(trY), activeX(trX), trainY(trY), wasM(mmat), beta(betamat), kern(kernel), noise(noiseModel), numTarget(trY.getCols()), numData(trY.getRows())
{
  assert(activeX.getRows()==trainY.getRows());
  setVerbosity(verbosity);
  terminate = false;
  noise.setVerbosity(getVerbosity());
  varUpdate = false;
  loadedModel = true;
  initStoreage();
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
  assert(mu.getCols()==numTarget);
  assert(varSigma.getCols()==numTarget);
  CMatrix kX(numTrainData, Xin.getRows());
  kern.compute(kX, activeX, Xin);
  if(numCovStruct==1)
      {
	kX.trsm(L[0], 1.0, "L", "L", "N", "N"); // now it is Linvk
	for(int i=0; i<Xin.getRows(); i++)
	  {
	    double vsVal = kern.diagComputeElement(Xin, i) - kX.norm2Col(i);
	    assert(vsVal>=0);
	    for(int j=0; j<numTarget; j++)
	      varSigma.setVal(vsVal, i, j);	    
	  }
	kX.trsm(L[0], 1.0, "L", "L", "T", "N"); // now it is Kinvk
	mu.gemm(kX, wasM, 1.0, 0.0, "T", "N");
      }
  else
    {
      CMatrix Lk(numTrainData, Xin.getRows());
      
      for(int k=0; k<numCovStruct; k++)
	{
	  Lk.deepCopy(kX);
	  Lk.trsm(L[k], 1.0, "L", "L", "N", "N");
	  for(int i=0; i<Xin.getRows(); i++)
	    {
	      double vsVal=kern.diagComputeElement(Xin, i) - Lk.norm2Col(i);
	      assert(vsVal>=0);
	      varSigma.setVal(vsVal, i, k);
	    }
	  Lk.trsm(L[k], 1.0, "L", "L", "T", "N"); // now it is Kinvk
	  mu.gemvColCol(k, Lk, wasM, k, 1.0, 0.0, "N");
	}
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
  // set m and beta to zeros
  wasM.setVals(0.0);
  beta.setVals(0.0);

  //TODO What about ff and fbar and fbarfbar
 
  // set nu to zeros(size of y)
  nu.setVals(0.0);
  // set g to zeros(size of y)
  g.setVals(0.0);
  // set noise.varSigma to diagonal of kernel.
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
  
  updateNuG();
}
void CPpa::updateNuG()
{
  for(int i=0; i<numData; i++)
    noise.getNuG(g, nu, i);
}
  
double CPpa::approxLogLikelihood() const
{
  double L=0.0;
  updateK();
  CMatrix invKm(invK.getRows(), 1);
  if(noise.isSpherical())
    {
      updateInvK(0);
    }
  for(int j=0; j<numTarget; j++)
    {
      if(!noise.isSpherical())
	updateInvK(j);
      invK.setSymmetric(true);
      invKm.symvColCol(0, invK, wasM, j, 1.0, 0.0, "u");
      L -= .5*(logDetK + invKm.dotColCol(0, wasM, j));
    }
  L+=kern.priorLogProb();
  return L;
}  
void CPpa::approxLogLikelihoodGradient(CMatrix& g) const
{
  assert(g.getRows()==1);
  assert(g.getCols()==getOptNumParams());
  g.zeros();
  CMatrix tempG(1, getOptNumParams());
  updateK();
  if(noise.isSpherical())
    {
      updateInvK(0);
    }
  for(int j=0; j<numTarget; j++)
    {
      if(!noise.isSpherical())
	{
	  updateInvK(j);
	}
      updateCovGradient(j);
      kern.getGradTransParams(tempG, activeX, covGrad);
      g+=tempG;
      
    }
}


void CPpa::optimise(const int maxIters, const int estepIters, const int mstepIters)
{
  if(getVerbosity()>0)
    {
      cout << "Initial model:" << endl;
      display(cout);
    }
  int counter = 0;
  terminate = false;
  oldLogLike = logLikelihood();
  while(!terminate && counter < maxIters)
    {
      counter++;
      mStep();
      eStep();
      logLike = logLikelihood();
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
}
void CPpa::eStep()
{
  updateNuG();
  for(int i=0; i<numData; i++)
    for(int j=0; j<numTarget; j++)
      {
	double gVal = g.getVal(i, j);
	gamma.setVal((gVal*gVal-nu.getVal(i, j))/2, i, j);
      }
  logLike = logLikelihood();
  updateExpectationf();
  updateExpectationff();
  updateExpectationsfBar();
}
void CPpa::updateExpectationf()
{
  for(int i=0; i<numData; i++)
    for(int j=0; j<numTarget; j++)
      f.setVal(fBar.getVal(i, j)+g.getVal(i,j)/beta.getVal(1, j), i, j);
}
void CPpa::updateExpectationff()
{
  for(int j=0; j<numTarget; j++)
    {
      double b = beta.getVal(1, j);
      double b2 = b*b;
      for(int i=0; i<numData; i++)
	{
	  double mVal = fBar.getVal(i, j); // TODO Need to check this!
	  double ffVal = 2/b2*gamma.getVal(i, j) + 1/b + 2*mVal*f.getVal(i, j) - mVal*mVal;
	  ff.setVal(ffVal, i, j);
	}
    }
}
void CPpa::updateExpectationsfBar()
{
  throw ndlexceptions::NotImplementedError("Expectation of fbar not yet implemented.");
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

  // Write out beta.
  for(int j=0; j<model.getNumProcesses()-1; j++)
    {
      out << model.beta.getVal(1, j) << " ";
    }
  out << model.beta.getVal(1, model.getNumProcesses()-1) << endl;

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
  if(tokens[1]!="0.1")
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

  // Get m, X and Y
  CMatrix m(numTrainData, numProcesses);
  CMatrix activeX(numTrainData, numFeatures, 0.0);
  CMatrix trainY(numTrainData, numProcesses);
  for(int i=0; i<numTrainData; i++)
    {
      tokens.clear();
      ndlstrutil::getline(in, line);
      ndlstrutil::tokenise(tokens, line, " ");
      for(int j=0; j<numProcesses; j++)
	{
	  trainY.setVal(atof(tokens[j].c_str()), i, j);
	}
      for(int j=0; j<numProcesses; j++)
	{
	  m.setVal(atof(tokens[j+numProcesses].c_str()), i, j);
	}
      // get activeX and trainY now.
      for(int j=numProcesses*3; j<tokens.size(); j++)
	{
	  int ind = tokens[j].find(':');
	  // TODO Check that : is in the string.
	  string featStr=tokens[j].substr(0, ind);
	  string featValStr=tokens[j].substr(ind+1, tokens[j].size()-ind);
	  int featNum = atoi(featStr.c_str());
	  if(featNum<1 || featNum>numFeatures)
	    throw ndlexceptions::FileFormatError("Corrupt file format.");
	  double featVal = atof(featValStr.c_str());
	  activeX.setVal(featVal, i, featNum-1);	  
	}
    }
  CPpa* pmodel= new CPpa(activeX, trainY, m, beta, *pkern, *pnoise); 
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

