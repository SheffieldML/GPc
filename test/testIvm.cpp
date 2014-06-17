#include "CKern.h"
#include "CMatrix.h"
#include "CIvm.h"

int testGaussian();
int testProbit();
int testNcnm();

int main()
{
  int fail = 0;
  fail += testGaussian();
  //fail += testOrdered();
  fail += testProbit();
  fail += testNcnm();
  cout << "Number of failures: " << fail << endl;
}

int testGaussian()
{
  int fail = 0;
  CMatrix X;
  X.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmGaussian.mat", "X");
  CMatrix y;
  y.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmGaussian.mat", "y");

  CCmpndKern kernInit(X);
  kernInit.addKern(new CRbfKern(X));
  kernInit.addKern(new CLinKern(X));
  kernInit.addKern(new CBiasKern(X));
  kernInit.addKern(new CWhiteKern(X));
  CGaussianNoise noiseInit(&y);
  
  CIvm modelInit(&X, &y, &kernInit, &noiseInit, CIvm::ENTROPY, 50);
  modelInit.selectPoints();
   
  CCmpndKern kern(X);
  kern.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmGaussian.mat", "kernInit");
  CGaussianNoise noise(&y);
  noise.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmGaussian.mat", "noiseInit");

  CIvm model(&X, &y, &kern, &noise, "matfiles" + ndlstrutil::dirSep() + "testIvmGaussian.mat", "ivmInfoInit", 0);
  if(model.equals(modelInit))
    cout << model.getNoiseName() << " Noise IVM passed." << endl;
  else
  {
    cout << "FAILURE: " << model.getNoiseName() << " Noise IVM." << endl;
    fail++;
  }
  return fail;
  model.toFile("crap_ivm");
  modelInit.fromFile("crap_ivm");
  if(model.equals(modelInit))
    cout << "Stream read and write " << model.getNoiseName() << " Noise IVM passed." << endl;
  else
  {
    cout << "FAILURE: " << "Stream read and write " << model.getNoiseName() << " Noise IVM." << endl;
    cout << "Written noise model: " << endl;
    model.display(cout);
    cout << "Read in noise model: " << endl;
    modelInit.display(cout);
    fail++;
  }
}
int testProbit()
{
  int fail = 0;
  CMatrix X;
  X.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmProbit.mat", "X");
  CMatrix y;
  y.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmProbit.mat", "y");

  CCmpndKern kernInit(X);
  kernInit.addKern(new CRbfKern(X));
  kernInit.addKern(new CLinKern(X));
  kernInit.addKern(new CBiasKern(X));
  kernInit.addKern(new CWhiteKern(X));
  CProbitNoise noiseInit(&y);
  
  CIvm modelInit(&X, &y, &kernInit, &noiseInit, CIvm::ENTROPY, 50);
  modelInit.selectPoints();
   
  CCmpndKern kern(X);
  kern.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmProbit.mat", "kernInit");
  CProbitNoise noise(&y);
  noise.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmProbit.mat", "noiseInit");

  CIvm model(&X, &y, &kern, &noise, "matfiles" + ndlstrutil::dirSep() + "testIvmProbit.mat", "ivmInfoInit", 0);
  if(model.equals(modelInit))
    cout << model.getNoiseName() << " Noise IVM passed." << endl;
  else
  {
    cout << "FAILURE: " << model.getNoiseName() << " Noise IVM." << endl;
    fail++;
  }
  model.toFile("crap_ivm");
  modelInit.fromFile("crap_ivm");
  if(model.equals(modelInit))
    cout << "Stream read and write " << model.getNoiseName() << " Noise IVM passed." << endl;
  else
  {
    cout << "FAILURE: " << "Stream read and write " << model.getNoiseName() << " Noise IVM." << endl;
    cout << "Written noise model: " << endl;
    model.display(cout);
    cout << "Read in noise model: " << endl;
    modelInit.display(cout);
    fail++;
  }

  return fail;
}

int testNcnm()
{
  int fail = 0;
  CMatrix X;
  X.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmNcnm.mat", "X");
  CMatrix y;
  y.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmNcnm.mat", "y");


  // Add L1 prior for the kernel.
  CGammaDist* prior = new CGammaDist();
  prior->setParam(1.0, 0);
  prior->setParam(1.0, 1);

  CRbfKern* rbfKern = new CRbfKern(X);
  rbfKern->addPrior(prior, 1);
  CLinKern* linKern = new CLinKern(X);
  linKern->addPrior(prior, 0);
  CBiasKern* biasKern = new CBiasKern(X);
  biasKern->addPrior(prior, 0);
  CWhiteKern* whiteKern = new CWhiteKern(X);
  whiteKern->addPrior(prior, 0);

  CCmpndKern kernInit(X);
  kernInit.addKern(rbfKern);
  kernInit.addKern(linKern);
  kernInit.addKern(biasKern);
  kernInit.addKern(whiteKern);


  CNcnmNoise noiseInit(&y);
  
  CIvm modelInit(&X, &y, &kernInit, &noiseInit, CIvm::ENTROPY, 50);
  modelInit.selectPoints();
  modelInit.checkGradients(); 
  CCmpndKern kern(X);
  kern.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmNcnm.mat", "kernInit");
  CNcnmNoise noise(&y);
  noise.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "testIvmNcnm.mat", "noiseInit");
  
  CIvm model(&X, &y, &kern, &noise, "matfiles" + ndlstrutil::dirSep() + "testIvmNcnm.mat", "ivmInfoInit", 0);
  if(model.equals(modelInit))
    cout << model.getNoiseName() << " Noise IVM passed." << endl;
  else
  {
    cout << "FAILURE: " << model.getNoiseName() << " Noise IVM." << endl;
    fail++;
  }
  model.toFile("crap_ivm");
  modelInit.fromFile("crap_ivm");
  if(model.equals(modelInit))
    cout << "Stream read and write " << model.getNoiseName() << " Noise IVM passed." << endl;
  else
  {
    cout << "FAILURE: " << "Stream read and write " << model.getNoiseName() << " Noise IVM." << endl;
    cout << "Written noise model: " << endl;
    model.display(cout);
    cout << "Read in noise model: " << endl;
    modelInit.display(cout);
    fail++;
  }
  return fail;
}


