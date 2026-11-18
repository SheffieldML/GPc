/* Stamp PITC params/grads/ll from DTC companion geometry (CIP-000F). */
#include <iostream>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"

using namespace std;

int main()
{
  const char* src = "matfiles/testGpdtc.mat";

  CMatrix X, y, bias, scale, numActiveMat, fileParams;
  X.readMatlabFile(src, "X");
  y.readMatlabFile(src, "y");
  bias.readMatlabFile(src, "bias");
  scale.readMatlabFile(src, "scale");
  numActiveMat.readMatlabFile(src, "numActive");
  fileParams.readMatlabFile(src, "params");

  const int numActive = (int)numActiveMat.getVal(0);
  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CLinKern(X));
  kern.addKern(new CBiasKern(X));
  kern.addKern(new CWhiteKern(X));
  CGaussianNoise noise(&y);
  for(unsigned int i = 0; i < noise.getOutputDim(); i++)
    noise.setBiasVal(0.0, i);

  CGp model(&kern, &noise, &X, CGp::PITC, (unsigned int)numActive, 0);
  model.setScale(scale);
  model.setBias(bias);
  model.setOptParams(fileParams);
  model.updateM();

  CMatrix params(1, model.getOptNumParams());
  model.getOptParams(params);
  CMatrix grads(1, model.getOptNumParams());
  model.logLikelihoodGradient(grads);
  CMatrix ll(1, 1);
  ll.setVal(model.logLikelihood(), 0);
  CMatrix approxInt(1, 1);
  approxInt.setVal((double)CGp::PITC, 0);

  params.writeMatlabFile("matfiles/_gen_pitc_params.mat", "params");
  grads.writeMatlabFile("matfiles/_gen_pitc_grads.mat", "grads");
  ll.writeMatlabFile("matfiles/_gen_pitc_ll.mat", "ll");
  approxInt.writeMatlabFile("matfiles/_gen_pitc_approxInt.mat", "approxInt");

  cout << "PITC ll=" << ll.getVal(0)
       << " nParams=" << params.getCols()
       << " blocks=" << model.getNumBlocks() << endl;
  return 0;
}
