#include "CKern.h"
#include "CMatrix.h"
#include "CGp.h"
#include "CClctrl.h"

int testGaussian(string type);
class CClgptest : public CClctrl 
{
 public:
  CClgptest(int arc, char** arv) : CClctrl(arc, arv){}
  void helpInfo(){}
  void helpHeader(){}
};

int main(int argc, char* argv[])
{
  CClgptest command(argc, argv);
  int fail = 0;
  try 
  {
    //fail += testGaussian("ftc");
    fail += testGaussian("dtc");
    //fail += testGaussian("fitc");
    //fail += testGaussian("pitc");
    command.exitNormal();
  }
  catch(ndlexceptions::Error& err) 
  {
   command.exitError(err.getMessage());
  }
  catch(std::bad_alloc&) 
  {
    command.exitError("Out of memory.");
  }
  catch(std::exception& err) 
  {
    std::string what(err.what());
    command.exitError("Unhandled exception: " + what);
  }

}

int testGaussian(string type)
{
  string fileName = "matfiles" + ndlstrutil::dirSep() + "testGp" + type + ".mat";
  int fail = 0;
  CMatrix X;
  X.readMatlabFile(fileName.c_str(), "X");
  CMatrix y;
  y.readMatlabFile(fileName.c_str(), "y");
  CMatrix actSetMat;
  actSetMat.readMatlabFile(fileName.c_str(), "numActive");
  int numActive = (int)actSetMat.getVal(0);
  CMatrix apprxTypeMat;
  apprxTypeMat.readMatlabFile(fileName.c_str(), "approxInt");
  int approxType = (int)apprxTypeMat.getVal(0);
  CMatrix scale;
  scale.readMatlabFile(fileName.c_str(), "scale");
  CMatrix bias;
  bias.readMatlabFile(fileName.c_str(), "bias");

  CCmpndKern kernInit(X);
  kernInit.addKern(new CRbfKern(X));
  kernInit.addKern(new CLinKern(X));
  kernInit.addKern(new CBiasKern(X));
  kernInit.addKern(new CWhiteKern(X));
  CGaussianNoise noiseInit(&y);
  for(int i=0; i<noiseInit.getOutputDim(); i++) 
  {
    noiseInit.setBiasVal(0.0, i);
  }
  
  // Initialise a model from read in MATLAB
  CGp modelInit(&kernInit, &noiseInit, &X, approxType, numActive, 2);
  modelInit.setScale(scale);
  modelInit.setBias(bias);
  modelInit.updateM();

  CCmpndKern kern(X);
  kern.readMatlabFile(fileName.c_str(), "kernInit");
  //CGaussianNoise noise(y);
  //noise.readMatlabFile("testGp.mat", "noiseInit");

  // Initialise a model from gpInfo.
  CGp model(&X, &y, &kern, &noiseInit, fileName.c_str(), "gpInfoInit", 0);
  if(model.equals(modelInit))
  { 
    cout << "GP initial model passed." << endl;
  }
  else 
  {
    cout << "FAILURE: GP." << endl;
    cout << "Matlab loaded model " << endl;
    model.display(cout);
    cout << "C++ Initialised Model " << endl;
    modelInit.display(cout);
    fail++;
  }

  // Compare C++ parameters with MATLAB provided parameters.
  CMatrix fileParams;
  fileParams.readMatlabFile(fileName.c_str(), "params");
  CMatrix params(1, model.getOptNumParams());
  model.getOptParams(params);
  if(params.equals(fileParams)) 
  {
    cout << "Parameter match passed." << endl;
  }
  else 
  {
    cout << "FAILURE: GP parameter match." << endl;
    cout << "Matlab loaded params: " << fileParams << endl;
    cout << "C++ Initialised params: " << params << endl;
    cout << "Maximum difference: " << params.maxAbsDiff(fileParams) << endl;
    fail++;
  }
  
  // Compare C++ gradients with MATLAB provided gradients.
  CMatrix fileGrads;
  fileGrads.readMatlabFile(fileName.c_str(), "grads");
  CMatrix grads(1, model.getOptNumParams());
  model.logLikelihoodGradient(grads);
  if(grads.equals(fileGrads)) 
  {
    cout << "GP Gradient match passed." << endl;
  }
  else {
    cout << "FAILURE: GP gradient match." << endl;
    cout << "Matlab Gradient " << fileGrads << endl;
    cout << "C++ Gradient " << grads << endl;
    cout << "Maximum difference: " << grads.maxAbsDiff(fileGrads) << endl;
    fail++;
  }

  // Compare C++ log likelihood with MATLAB provided log likelihood.
  CMatrix fileLl;
  fileLl.readMatlabFile(fileName.c_str(), "ll");
  CMatrix ll(1, 1);
  ll.setVal(model.logLikelihood(), 0, 0);
  if(ll.equals(fileLl)) 
  {
    cout << "GP Log Likelihood match passed." << endl;
  }
  else 
  {
    cout << "FAILURE: GP gradient match." << endl;
    cout << "Matlab Log Likelihood " << fileLl << endl;
    cout << "C++ Log Likelihood " << ll << endl;
    fail++;
  }
  
  // Read and write tests
  // Matlab read/write
  model.writeMatlabFile("crap.mat", "writtenModel");
  modelInit.readMatlabFile("crap.mat", "writtenModel");
  if(model.equals(modelInit))
    cout << "MATLAB written " << model.getName() << " matches read in model. Read and write to MATLAB passes." << endl;
  else
  {
    cout << "FAILURE: MATLAB read in " << model.getName() << " does not match written out model." << endl;
    fail++;
  }

  // Write to stream.
  model.toFile("crap_gp");
  modelInit.fromFile("crap_gp");
  if(model.equals(modelInit))
  {
    cout << "Text written " << model.getName() << " matches read in model. Read and write to stream passes." << endl;
  }
  else
  {
    cout << "FAILURE: Stream read in " << model.getName() << " does not match written model." << endl;
    cout << "Matlab loaded model " << endl;
    model.display(cout);
    cout << "Text Read in Model " << endl;
    modelInit.display(cout);
    fail++;
  }
  
  //  model.checkGradients();

  return fail;
}


