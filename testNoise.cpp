#include "CNoise.h"

int testType(const string noiseType);
int testNoise(CNoise* noise, CNoise* noise2, const string fileName);

int main()
{
  int fail=0;
  try {    
    fail += testType("gaussian");
    fail += testType("probit");
    fail += testType("ordered");
    fail += testType("ncnm");
    //fail += testType("mgaussian");
    //fail += testType("cmpnd");
    cout << "Number of failures: " << fail << "." << endl;
  }
  catch(ndlexceptions::FileFormatError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileReadError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileWriteError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::Error err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(std::bad_alloc err)
  {
    cerr << "Out of memory.";
    exit(1);
  }
  catch(std::exception err)
  {
    cerr << "Unhandled exception.";
    exit(1);
  }

}

int testType(string noiseType)
{
  string fileName = "matfiles" + ndlstrutil::dirSep() + noiseType + "NoiseTest.mat";

  CMatrix y;
  y.readMatlabFile(fileName, "y");

  CNoise* noise;
  CNoise* noise2;
 
  if(noiseType=="gaussian")
  {
    noise = new CGaussianNoise(&y);
    noise2 = new CGaussianNoise(&y);
  }
  else if(noiseType=="probit")
  {
    noise = new CProbitNoise(&y);
    noise2 = new CProbitNoise(&y);
  }
  else if(noiseType=="ncnm")
  {
    noise = new CNcnmNoise(&y);
    noise2 = new CNcnmNoise(&y);
  }
  else if(noiseType=="ordered")
  {
    noise = new COrderedNoise(&y, 10);
    noise2 = new COrderedNoise(&y, 10);
  }
  /*
    else if(noiseType=="mgaussian")
    {
    noise = new CMgaussianNoise(&y);
    noise2 = new CMgaussianNoise(&y);
    }
    else if(noiseType=="ngauss")
    {
    noise = new CNgaussNoise(&y);
    noise2 = new CNgaussNoise(&y);
    }
    else if(noiseType=="cmpnd")
    {
    noise = new CCmpndNoise(&y);
    noise->addNoise(new CProbitNoise(&y));
    noise->addNoise(new CGaussianNoise(&y));
    noise->addNoise(new COrderedNoise(&y));
    noise2 = new CCmpndNoise(&y);

    }*/
  int fail = testNoise(noise, noise2, fileName);
  delete noise;
  delete noise2;
  return fail;
}
int testNoise(CNoise* noise, CNoise* noise2, string fileName)
{
  int fail = 0;
  CMatrix params;
  params.readMatlabFile(fileName, "params");
  CMatrix y;
  y.readMatlabFile(fileName, "y");
  CMatrix mu;
  mu.readMatlabFile(fileName, "mu");
  CMatrix varSigma;
  varSigma.readMatlabFile(fileName, "varsigma");
  noise->setTransParams(params);
  noise->setMus(mu);
  noise->setVarSigmas(varSigma);
  noise2->readMatlabFile(fileName, "noise2");
  if(noise2->equals(*noise))
    cout << noise->getName() << " Initial Noise matches." << endl;
  else
  {
    cout << "FAILURE: " << noise->getName() << " Initial Noise." << endl;
    cout << "Matlab initial noise: " << endl;
    noise2->display(cout);
    cout << "C++ initial noise: " << endl;
    noise->display(cout);
    fail++;
  }
  CMatrix L2;
  L2.readMatlabFile(fileName, "L2");
  double L = noise->logLikelihood();
  
  if(abs(L2.getVal(0)-L)<1e-6)
    cout << noise->getName() << " log likelihood matches." << endl;
  else
  { 
    cout << "FAILURE: " << noise->getName() << " log likelihood." << endl;
    cout << "MATLAB: " << L2.getVal(0) << endl;
    cout << "C++: " << L << endl;
    fail++;
  }
  CMatrix g(1, noise->getNumParams());
  noise->getGradTransParams(g);
  CMatrix g2;
  g2.readMatlabFile(fileName, "g2");
  if(g.equals(g2))
    cout << noise->getName() << " parameter gradient matches." << endl;
  else
  { 
    cout << "FAILURE: " << noise->getName() << " parameter gradient." << endl;
    cout << "MATLAB: " << endl;
    cout << g2;
    cout << "C++: " << endl;
    cout << g;
    cout << "Max absolute difference: " << g.maxAbsDiff(g2) << endl; 
    fail++;
  }
  CMatrix gmu2;
  gmu2.readMatlabFile(fileName, "gmu2");
  CMatrix gvs2;
  gvs2.readMatlabFile(fileName, "gvs2");
  CMatrix gmu(mu.getRows(), mu.getCols());
  CMatrix gvs(varSigma.getRows(), varSigma.getCols());
  noise->getGradInputs(gmu, gvs);
  if(gmu.equals(gmu2))
    cout << noise->getName() << " mu gradient matches." << endl;
  else
  {
    cout << "FAILURE: " << noise->getName() << " mu gradient." << endl;
    fail++;
    cout << "Maximum absolute difference: " << gmu2.maxAbsDiff(gmu)  << endl;
  }
  if(gvs.equals(gvs2))
    cout << noise->getName() << " vs gradient matches." << endl;
  else
  {
    cout << "FAILURE: " << noise->getName() << " vs gradient." << endl;
    fail++;
    cout << "Maximum absolute difference: " << gvs2.maxAbsDiff(gvs)  << endl;

  }
   
  // Write to MATLAB
  noise->writeMatlabFile("crap.mat", "writtenNoise");
  noise2->readMatlabFile("crap.mat", "writtenNoise");
  if(noise->equals(*noise2))
    cout << "Matlab written " << noise->getName() << " matches read in noise. Read and write to matlab passes." << endl;
  else
  {
    cout << "FAILURE: MATLAB read in " << noise->getName() << " does not match written out noise." << endl;
    cout << "Matlab read in noise:" << endl;
    noise2->display(cout);
    cout << "C++ written noise:" << endl;
    noise->display(cout);
    fail++;
  }

  // Write to stream.
  noise->toFile("crap_noise");
  noise2->fromFile("crap_noise");
  if(noise->equals(*noise2))
    cout << "Text written " << noise->getName() << " matches read in noise. Read and write to stream passes." << endl;
  else
  {
    cout << "FAILURE: Stream read in " << noise->getName() << " does not match written noise." << endl;
    fail++;
  }
  return fail;
}
/*
  int main()
  {
  int numProcess = 3;
  int numData = 10;
  CMatrix target(numData, numProcess);
  target.randn();
  CMatrix mu(numData, numProcess);
  mu.randn();
  CMatrix varSigma(numData, numProcess);
  varSigma.randn();
  varSigma*=varSigma;

  CGaussianNoise noise(target);
  noise.setMus(mu);
  noise.setVarSigmas(varSigma);
  CMatrix params(1, noise.getNumParams());
  noise.getTransParams(params);
  cout << "Original parameters: " << endl << params << endl;
  params.randn();
  double varVal = randn();
  // make sure the variance is positive
  params.setVals(varVal*varVal, 0, params.getCols()-1);
  noise.setTransParams(params);
  cout << "New parameters: " << endl << params << endl;

    
  CMatrix y(numData, numProcess);
  y.randn();
  cout << "mu values" << endl << mu << endl;
  cout << "varSigma values" << endl << varSigma << endl;
  cout << "y values " << endl << y << endl;


  // Check mu gradients.
  double epsilon=1e-6;
  double Lminus=0.0;
  double Lplus=0.0;
  CMatrix diffGradMu(mu.getRows(), mu.getCols());
  CMatrix origMu(mu);
  for(int i=0; i<noise.getNumData(); i++)
  {
  for(int j=0; j<noise.getNumProcesses(); j++)
  {
  noise.setMu(origMu.getVal(i, j) + epsilon, i, j);
  Lplus = noise.logLikelihood();
  noise.setMu(origMu.getVal(i, j) - epsilon, i, j);
  Lminus = noise.logLikelihood();
  diffGradMu.setVals(0.5*(Lplus - Lminus)/epsilon, i, j);
  noise.setMu(origMu.getVal(i, j), i, j);
  }
  }
  // Check varSigma gradients.
  CMatrix diffGradVarSigma(varSigma.getRows(), varSigma.getCols());
  CMatrix origVarSigma(varSigma);
  origVarSigma.deepCopy(varSigma);
  for(int i=0; i<noise.getNumData(); i++)
  {
  for(int j=0; j<noise.getNumProcesses(); j++)
  {
  noise.setVarSigma(origVarSigma.getVal(i, j) + epsilon, i, j);
  Lplus=noise.logLikelihood();
  noise.setVarSigma(origVarSigma.getVal(i, j) - epsilon, i, j);
  Lminus=noise.logLikelihood();
  diffGradVarSigma.setVals(0.5*(Lplus - Lminus)/epsilon, i, j);
  noise.setVarSigma(origVarSigma.getVal(i, j), i, j);
  }
  }
  
  CMatrix analyticalGradMu(origMu.getRows(), origMu.getCols());
  CMatrix analyticalGradVarSigma(varSigma.getRows(), varSigma.getCols());
  noise.getGradInputs(analyticalGradMu, analyticalGradVarSigma);
  cout << endl;
  cout << "Mu numerical differences " << endl << diffGradMu << endl;
  cout << "Mu analytical gradient " << endl << analyticalGradMu << endl;

  CMatrix diffMu(origMu.getRows(), origMu.getCols());
  diffMu.deepCopy(analyticalGradMu);
  diffMu-=diffGradMu;
  cout << "Maximum mu difference: " << diffMu.max() << endl;
  cout << endl;
  cout << "VarSigma numerical differences " << endl << diffGradVarSigma << endl;
  cout << "VarSigma analytical gradient " << endl << analyticalGradVarSigma << endl;
  CMatrix diffVarSigma(varSigma.getRows(), varSigma.getCols());
  diffVarSigma.deepCopy(analyticalGradVarSigma);
  diffVarSigma-=diffGradVarSigma;
  cout << "Maximum varSigma difference: " << diffVarSigma.max() << endl;
  
  noise.checkGradients();

  }
*/
