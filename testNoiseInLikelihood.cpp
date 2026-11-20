/* CIP-0027 Phase A: likelihood-owned observation noise on FTC.
   Equivalence: RBF + likelihood.variance ≡ RBF+White (legacy).
   No MATLAB / matio fixtures. */
#include <cmath>
#include <iostream>
#include <string>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlutil.h"

using namespace std;

namespace {

const unsigned int N = 6;
const unsigned int Q = 1;
const double SIGMA2 = 0.25;

void fillXy(CMatrix& X, CMatrix& y)
{
  for(unsigned int i = 0; i < N; i++)
  {
    const double xv = -1.0 + 2.0 * (double)i / (double)(N - 1);
    X.setVal(xv, i, 0);
    y.setVal(sin(3.0 * xv) + 0.1 * xv, i, 0);
  }
}

int checkClose(const string& label, double a, double b, double tol = ndlutil::MATCHTOL)
{
  const double diff = fabs(a - b);
  if(diff <= tol)
  {
    cout << label << " passes." << endl;
    return 0;
  }
  cout << "FAILURE: " << label << ": |a-b|=" << diff
       << " (a=" << a << ", b=" << b << ")" << endl;
  return 1;
}

int failTrue(bool cond, const string& what)
{
  if(cond)
  {
    cout << what << " passes." << endl;
    return 0;
  }
  cout << "FAILURE: " << what << "." << endl;
  return 1;
}

} // namespace

int testFtcEquivalence()
{
  cout << "==== FTC RBF+variance vs RBF+White ====" << endl;
  int fail = 0;
  CMatrix X(N, Q);
  CMatrix y(N, 1);
  fillXy(X, y);

  CCmpndKern kernLik(X);
  kernLik.addKern(new CRbfKern(X));
  CGaussianNoise noiseLik(&y);
  for(unsigned int j = 0; j < noiseLik.getOutputDim(); j++)
    noiseLik.setBiasVal(0.0, j);
  CGp gpLik(&kernLik, &noiseLik, &X, CGp::FTC, 0, 0);
  gpLik.setLikelihoodVariance(SIGMA2);

  CCmpndKern kernWhite(X);
  kernWhite.addKern(new CRbfKern(X));
  kernWhite.addKern(new CWhiteKern(X));
  kernWhite.setByPath("white.variance", SIGMA2);
  // Match RBF hyperparameters.
  kernWhite.setByPath("rbf.variance", kernLik.getByPath("rbf.variance"));
  kernWhite.setByPath("rbf.inverseWidth", kernLik.getByPath("rbf.inverseWidth"));
  CGaussianNoise noiseWhite(&y);
  for(unsigned int j = 0; j < noiseWhite.getOutputDim(); j++)
    noiseWhite.setBiasVal(0.0, j);
  CGp gpWhite(&kernWhite, &noiseWhite, &X, CGp::FTC, 0, 0);
  // Legacy: White owns noise; tiny free variance must not be added on the Gram.
  fail += failTrue(!gpWhite.addsLikelihoodVarianceToGram(),
                   "legacy White model skips Gram σ²");
  fail += failTrue(gpLik.addsLikelihoodVarianceToGram(),
                   "likelihood-owned model adds Gram σ²");

  const double Llik = gpLik.logLikelihood();
  const double Lwhite = gpWhite.logLikelihood();
  fail += checkClose("FTC LL equivalence", Llik, Lwhite);

  CMatrix muL(2, 1);
  CMatrix varL(2, 1);
  CMatrix muW(2, 1);
  CMatrix varW(2, 1);
  CMatrix Xtest(2, Q);
  Xtest.setVal(0.2, 0, 0);
  Xtest.setVal(-0.7, 1, 0);
  gpLik.posteriorMeanVar(muL, varL, Xtest);
  gpWhite.posteriorMeanVar(muW, varW, Xtest);
  fail += checkClose("pred mu[0]", muL.getVal(0, 0), muW.getVal(0, 0));
  fail += checkClose("pred mu[1]", muL.getVal(1, 0), muW.getVal(1, 0));
  fail += checkClose("pred var[0]", varL.getVal(0, 0), varW.getVal(0, 0));
  fail += checkClose("pred var[1]", varL.getVal(1, 0), varW.getVal(1, 0));
  return fail;
}

int testPathsAndFix()
{
  cout << "==== likelihood.variance paths ====" << endl;
  int fail = 0;
  CMatrix X(N, Q);
  CMatrix y(N, 1);
  fillXy(X, y);

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  CGaussianNoise noise(&y);
  for(unsigned int j = 0; j < noise.getOutputDim(); j++)
    noise.setBiasVal(0.0, j);
  CGp gp(&kern, &noise, &X, CGp::FTC, 0, 0);

  gp.setByPath("likelihood.variance", 0.3);
  fail += checkClose("get likelihood.variance", gp.getByPath("likelihood.variance"), 0.3);
  fail += checkClose("alias Gaussian_noise.variance",
                     gp.getByPath("Gaussian_noise.variance"), 0.3);
  fail += checkClose("getLikelihoodVariance", gp.getLikelihoodVariance(), 0.3);

  gp.setByPath("Gaussian_noise.variance", 0.5);
  fail += checkClose("set via alias", gp.getLikelihoodVariance(), 0.5);

  bool found = false;
  const vector<string> paths = gp.paramPaths();
  for(size_t i = 0; i < paths.size(); i++)
  {
    if(paths[i] == "likelihood.variance")
      found = true;
  }
  fail += failTrue(found, "paramPaths lists likelihood.variance");

  gp.fix("likelihood.variance");
  fail += failTrue(gp.isFixed("likelihood.variance"), "fix likelihood.variance");
  fail += failTrue(gp.isLikelihoodVarianceFixed(), "isLikelihoodVarianceFixed");
  gp.unfix("likelihood.variance");
  fail += failTrue(!gp.isLikelihoodVarianceFixed(), "unfix likelihood.variance");
  return fail;
}

int testSparseBetaSync()
{
  cout << "==== sparse beta ↔ variance sync ====" << endl;
  int fail = 0;
  CMatrix X(N, Q);
  CMatrix y(N, 1);
  fillXy(X, y);
  CMatrix Xu(3, Q);
  for(unsigned int i = 0; i < 3; i++)
    Xu.copyRowRow(i, X, i);

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  CGaussianNoise noise(&y);
  for(unsigned int j = 0; j < noise.getOutputDim(); j++)
    noise.setBiasVal(0.0, j);
  CGp gp(&kern, &noise, &X, CGp::DTC, 3, 0);
  gp.setInducingVals(Xu);
  gp.setInducingFixed(true);

  gp.setByPath("likelihood.variance", 0.2);
  fail += checkClose("beta = 1/variance", gp.getBetaVal(), 1.0 / 0.2);
  fail += checkClose("path variance", gp.getByPath("likelihood.variance"), 0.2);

  gp.setByPath("beta", 5.0);
  fail += checkClose("variance = 1/beta", gp.getLikelihoodVariance(), 0.2);
  // set beta to 5 → variance 0.2; wait 1/5=0.2 yes.
  fail += checkClose("variance after beta set", gp.getByPath("Gaussian_noise.variance"), 1.0 / 5.0);
  return fail;
}

int testGradients()
{
  cout << "==== checkGradients FTC likelihood.variance ====" << endl;
  CMatrix X(N, Q);
  CMatrix y(N, 1);
  fillXy(X, y);
  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  CGaussianNoise noise(&y);
  for(unsigned int j = 0; j < noise.getOutputDim(); j++)
    noise.setBiasVal(0.0, j);
  CGp gp(&kern, &noise, &X, CGp::FTC, 0, 0);
  gp.setLikelihoodVariance(SIGMA2);
  gp.setVerbosity(0);
  try
  {
    gp.checkGradients();
    cout << "checkGradients passes." << endl;
    return 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cout << "FAILURE: checkGradients: " << err.getMessage() << endl;
    return 1;
  }
}

int main()
{
  int fail = 0;
  try
  {
    fail += testFtcEquivalence();
    fail += testPathsAndFix();
    fail += testSparseBetaSync();
    fail += testGradients();
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
  catch(std::exception& err)
  {
    cerr << "Unhandled exception: " << err.what() << endl;
    return 1;
  }
}
