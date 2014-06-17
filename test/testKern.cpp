#include "CKern.h"
using namespace std;

int testType(const string kernelType);
int testKern(CKern* kern, CKern* kern2, const string fileName);

int main()
{
  int fail=0;
  try
  {
    fail += testType("white");
    fail += testType("bias");

    fail += testType("rbf");
    fail += testType("ratquad");
    fail += testType("matern32");
    fail += testType("matern52");

    fail += testType("lin");
    fail += testType("poly");
    fail += testType("mlp");
    //fail += testType("gibbs");

    //fail += testType("rbfperiodic");
    //fail += testType("gibbsperiodic");
 
    fail += testType("rbfard");
    // fail += testType("ratquadard");
    // fail += testType("matern32ard");
    // fail += testType("matern52ard");

    fail += testType("linard");
    fail += testType("polyard");
    fail += testType("mlpard");
    // fail += testType("gibbsard");
   
    fail += testType("cmpnd");
    fail += testType("tensor");
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

int testType(const string kernelType)
{
  string fileName = "matfiles" + ndlstrutil::dirSep() + kernelType + "KernTest.mat";

  CMatrix X;
  X.readMatlabFile(fileName, "X");

  CKern* kern;
  CKern* kern2;
 
  if(kernelType=="white")
  {
    kern = new CWhiteKern(X);
    kern2 = new CWhiteKern(X);
  }
  else if(kernelType=="bias")
  {
    kern = new CBiasKern(X);
    kern2 = new CBiasKern(X);
  }
  else if(kernelType=="rbf")
  {
    kern = new CRbfKern(X);
    kern2 = new CRbfKern(X);
  }
  else if(kernelType=="ratquad")
  {
    kern = new CRatQuadKern(X);
    kern2 = new CRatQuadKern(X);
  }
  else if(kernelType=="matern32")
  {
    kern = new CMatern32Kern(X);
    kern2 = new CMatern32Kern(X);
  }
  else if(kernelType=="matern52")
  {
    kern = new CMatern52Kern(X);
    kern2 = new CMatern52Kern(X);
  }
  else if(kernelType=="lin")
  {
    kern = new CLinKern(X);
    kern2 = new CLinKern(X);
  }
  else if(kernelType=="poly")
  {
    kern = new CPolyKern(X);
    kern2 = new CPolyKern(X);
  }
  else if(kernelType=="mlp")
  {
    kern = new CMlpKern(X);
    kern2 = new CMlpKern(X);
  }
  // else if(kernelType=="gibbs")
  // { 
  //   kern = new CGibbsKern(X);
  //   kern2 = new CGibbsKern(X);
  // }
  // else if(kernelType=="rbfperiodic")
  // { 
  //   kern = new CRbfPeriodicKern(X);
  //   kern2 = new CRbfPeriodicKern(X);
  // }
  // else if(kernelType=="gibbsperiodic")
  // { 
  //   kern = new CGibbsPeriodicKern(X);
  //   kern2 = new CGibbsPeriodicKern(X);
  // }
  else if(kernelType=="rbfard")
  {
    kern = new CRbfardKern(X);
    kern2 = new CRbfardKern(X);
  }
  // else if(kernelType=="ratquadard")
  // { 
  //   kern = new CRatQuadardKern(X);
  //   kern2 = new CRatQuadardKern(X);
  // }
  // else if(kernelType=="matern32ard")
  // { 
  //   kern = new CMatern32ardKern(X);
  //   kern2 = new CMatern32ardKern(X);
  // }
  // else if(kernelType=="matern52ard")
  // { 
  //   kern = new CMatern52ardKern(X);
  //   kern2 = new CMatern52ardKern(X);
  // }
  else if(kernelType=="linard")
  {
    kern = new CLinardKern(X);
    kern2 = new CLinardKern(X);
  }
  else if(kernelType=="polyard")
  {
    kern = new CPolyardKern(X);
    kern2 = new CPolyardKern(X);
  }
  else if(kernelType=="mlpard")
  {
    kern = new CMlpardKern(X);
    kern2 = new CMlpardKern(X);
  }
  // else if(kernelType=="gibbsard")
  // { 
  //   kern = new CGibbsardKern(X);
  //   kern2 = new CGibbsardKern(X);
  // }

  else if(kernelType=="cmpnd")
  {
    kern = new CCmpndKern(X);
    
    kern->addKern(new CRbfardKern(X));
    kern->addKern(new CMlpardKern(X));
    kern->addKern(new CLinardKern(X));
    kern->addKern(new CPolyardKern(X));
    kern->addKern(new CMlpKern(X));
    kern->addKern(new CRatQuadKern(X));
    kern->addKern(new CRbfKern(X));
    kern->addKern(new CLinKern(X));
    kern->addKern(new CPolyKern(X));
    kern->addKern(new CBiasKern(X));
    kern->addKern(new CWhiteKern(X));
    kern2 = new CCmpndKern(X);
    
  }
  else if(kernelType=="tensor")
  {
    kern = new CTensorKern(X);
    kern->addKern(new CRbfKern(X));
    kern->addKern(new CLinKern(X));
    kern->addKern(new CPolyKern(X));
    kern2 = new CTensorKern(X);
    
  }
  else
  {
    throw ndlexceptions::Error("Unrecognised kernel type requested.");
  }
  int fail = testKern(kern, kern2, fileName);
  delete kern;
  delete kern2;
  return fail;
}
int testKern(CKern* kern, CKern* kern2, const string fileName)
{
  
  int fail = 0;
  CMatrix params;
  params.readMatlabFile(fileName, "params");
  CMatrix X;
  X.readMatlabFile(fileName, "X");
  CMatrix X2;
  X2.readMatlabFile(fileName, "X2");
  kern->setTransParams(params);
  kern2->readMatlabFile(fileName, "kern2");
  if(kern2->equals(*kern))
    cout << kern->getName() << " Initial Kernel matches." << endl;
  else
  {
    cout << "FAILURE: " << kern->getName() << " Initial Kernel." << endl;
    cout << "Matlab kernel" << endl << *kern2 << endl << "C++ Kernel " << endl << *kern << endl;
    fail++;
  }
  CMatrix K1(X.getRows(), X.getRows());
  kern->compute(K1, X);
  CMatrix K2;
  K2.readMatlabFile(fileName, "K2");
  if(K1.equals(K2))
    cout << kern->getName() << " full compute matches." << endl;
  else
  { 
    cout << "FAILURE: " << kern->getName() << " full compute." << endl;
    double diff=K1.maxAbsDiff(K2);
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }
  CMatrix K3(X.getRows(), X2.getRows());
  kern->compute(K3, X, X2);
  CMatrix K4;
  K4.readMatlabFile(fileName, "K4");
  if(K3.equals(K4))
    cout << kern->getName() << " double compute matches." << endl;
  else
  { 
    cout << "FAILURE: " << kern->getName() << " double compute." << endl;
    double diff=K3.maxAbsDiff(K4);
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }
  CMatrix k1(X.getRows(), 1);
  kern->diagCompute(k1, X);
  CMatrix k2;
  k2.readMatlabFile(fileName, "k2");
  if(k1.equals(k2))
    cout << kern->getName() << " diag compute matches." << endl;
  else
  {
    cout << "FAILURE: " << kern->getName() << " diag compute." << endl;
    double diff=k1.maxAbsDiff(k2);
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }
  CMatrix covGrad;
  covGrad.readMatlabFile(fileName, "covGrad");
  covGrad.setSymmetric(true);
  CMatrix g1(1, kern->getNumParams());
  kern->getGradTransParams(g1, X, covGrad);
  CMatrix g2;
  g2.readMatlabFile(fileName, "g2");
  if(g1.equals(g2))
    cout << kern->getName() << " parameter gradient matches." << endl;
  else
  {
    cout << "FAILURE: " << kern->getName() << " parameter gradient." << endl;
    double diff=g1.maxAbsDiff(g2);
    cout << "Matlab gradient: " << endl;
    cout << g2 << endl;
    cout << "C++ gradient: " << endl;
    cout << g1 << endl;
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }
  
  CMatrix covGrad2;
  covGrad2.readMatlabFile(fileName, "covGrad2");
  covGrad.setSymmetric(true);
  CMatrix g3(1, kern->getNumParams());
  kern->getGradTransParams(g3, X, X2, covGrad2);
  CMatrix g4;
  g4.readMatlabFile(fileName, "g4");
  if(g3.equals(g4))
    cout << kern->getName() << " parameter X, X2 gradient matches." << endl;
  else
  {
    cout << "FAILURE: " << kern->getName() << " parameter X, X2 gradient." << endl;
    double diff=g3.maxAbsDiff(g4);
    cout << "Matlab gradient: " << endl;
    cout << g4 << endl;
    cout << "C++ gradient: " << endl;
    cout << g3 << endl;
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }
  
  /* Gradient with respect to X */
  vector<CMatrix*> G2;
  MATFile* matFile=matOpen(fileName.c_str(), "r");
  if(matFile==NULL)
    throw ndlexceptions::FileReadError(fileName);
  mxArray* matlabArray = matGetVariable(matFile, "G2");
  if(matlabArray==NULL)
    throw ndlexceptions::MatlabInterfaceReadError("G2");
  if(!mxIsCell(matlabArray))
    throw ndlexceptions::MatlabInterfaceReadError("G2");
  int elems = mxGetNumberOfElements(matlabArray);
  for(int i=0; i<elems; i++)
  {
    mxArray* matlabMatrix = mxGetCell(matlabArray, i);
    G2.push_back(new CMatrix(1, 1));
    G2[i]->fromMxArray(matlabMatrix);
  }
  vector<CMatrix*> gX;
  for(int i=0; i<X.getRows(); i++)
    gX.push_back(new CMatrix(X2.getRows(), X2.getCols()));
  kern->getGradX(gX, X, X2);
  double maxDiff=0.0;
  for(int i=0; i<gX.size(); i++)
  {
    double diff=gX[i]->maxAbsDiff(*G2[i]);
    if(diff>maxDiff)
      maxDiff=diff;
  }
  if(maxDiff<ndlutil::MATCHTOL)
    cout << kern->getName() << " X gradient matches." << endl;
  else
  {
    cout << "FAILURE: " << kern->getName() << " gradient with respect to X does not match." << endl;
    cout << "Maximum absolute difference: " << maxDiff << endl;
    fail++;
  }
  
  /* Gradient with respect to diagonal X */
  CMatrix GD2(X.getRows(), X.getCols());
  GD2.readMatlabFile(fileName, "GD2");
  CMatrix gDX(X.getRows(), X.getCols());
  kern->getDiagGradX(gDX, X);
  if(GD2.equals(gDX))
    cout << kern->getName() << " diagonal X gradient matches." << endl;
  else
  {
    cout << "FAILURE: " << kern->getName() << " gradient of diagonal with respect to X does not match." << endl;
    cout << "Maximum absolute difference: " << GD2.maxAbsDiff(gDX) << endl;
    fail++;
  }
  
  
  
  // Read and write tests
  // Matlab read/write
  kern->writeMatlabFile("crap.mat", "writtenKern");
  kern2->readMatlabFile("crap.mat", "writtenKern");
  if(kern->equals(*kern2))
    cout << "MATLAB written " << kern->getName() << " matches read in kernel. Read and write to MATLAB passes." << endl;
  else
  {
    cout << "FAILURE: MATLAB read in " << kern->getName() << " does not match written out kernel." << endl;
    cout << "Matlab written:" << endl;
    cout << kern->display(cout) << endl;
    cout << "Read back in:" << endl;
    cout << kern2->display(cout) << endl;
    fail++;
  }

  // Write to stream.
  kern->toFile("crap_kern");
  kern2->fromFile("crap_kern");
  if(kern->equals(*kern2))
    cout << "Text written " << kern->getName() << " matches read in kernel. Read and write to stream passes." << endl;
  else
  {
    cout << "FAILURE: Text read in " << kern->getName() << " does not match written kernel." << endl;
    cout << "Text written:" << endl;
    cout << kern->display(cout) << endl;
    cout << "Read back in:" << endl;
    cout << kern2->display(cout) << endl;
    fail++;
  }


  return fail;


}

