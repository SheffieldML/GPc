/* Generate params/grads/ll for matfiles/testGpdtcvar.mat from numeric inputs
   written by scripts/gen_testGpdtcvar.py (_gen_in.mat). */
#include <iostream>
#include <string>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"

using namespace std;

static void load(CMatrix& M, const char* file, const char* var)
{
  M.readMatlabFile(file, var);
}

int main()
{
  const char* inFile = "matfiles/_gen_in.mat";
  const char* outFile = "matfiles/_gen_out.mat";

  CMatrix X, y, Xu, bias, scale, numActiveMat, betaMat;
  load(X, inFile, "X");
  load(y, inFile, "y");
  load(Xu, inFile, "X_u");
  load(bias, inFile, "bias");
  load(scale, inFile, "scale");
  load(numActiveMat, inFile, "numActive");
  load(betaMat, inFile, "beta");

  const int numActive = (int)numActiveMat.getVal(0);
  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CLinKern(X));
  kern.addKern(new CBiasKern(X));
  kern.addKern(new CWhiteKern(X));

  CGaussianNoise noise(&y);
  for(unsigned int i = 0; i < noise.getOutputDim(); i++)
    noise.setBiasVal(0.0, i);

  CGp model(&kern, &noise, &X, CGp::DTCVAR, (unsigned int)numActive, 0);
  model.setInducingFixed(false);
  model.setActiveSet(Xu);
  model.setScale(scale);
  model.setBias(bias);
  model.setBetaVals(betaMat.getVal(0));
  model.updateM();

  CMatrix params(1, model.getOptNumParams());
  model.getOptParams(params);
  CMatrix grads(1, model.getOptNumParams());
  model.logLikelihoodGradient(grads);
  CMatrix ll(1, 1);
  ll.setVal(model.logLikelihood(), 0);

  // writeMatlabFile creates a fresh file; write each var then merge in Python.
  // Use a simple text dump for robustness across matio write-one-var API.
  params.writeMatlabFile("matfiles/_gen_params.mat", "params");
  grads.writeMatlabFile("matfiles/_gen_grads.mat", "grads");
  ll.writeMatlabFile("matfiles/_gen_ll.mat", "ll");
  Xu.writeMatlabFile("matfiles/_gen_Xu.mat", "X_u");

  cout << "DTCVAR ll=" << ll.getVal(0) << " nParams=" << params.getCols() << endl;
  cout << "Wrote matfiles/_gen_{params,grads,ll,Xu}.mat" << endl;
  (void)outFile;
  return 0;
}
