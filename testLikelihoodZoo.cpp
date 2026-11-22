/* CIP-003A Phase 2: Student-t, Poisson, Binomial density / grad / stream tests.
   References from SciPy / GPy (no matio). */
#include "CNoise.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;

static int checkClose(const string& label, double a, double b, double tol = 1e-10)
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

static int fdParamGrads(CNoise& noise, double tol)
{
  const unsigned int n = noise.getNumParams();
  if(n == 0)
  {
    cout << "no-param model: skip FD param grads." << endl;
    return 0;
  }
  CMatrix g(1, n);
  noise.getGradParams(g);
  CMatrix params(1, n);
  noise.getParams(params);
  const double eps = 1e-6;
  int fail = 0;
  for(unsigned int j = 0; j < n; j++)
  {
    const double p0 = params.getVal(j);
    params.setVal(p0 + eps, j);
    noise.setParams(params);
    const double lp = noise.logLikelihood();
    params.setVal(p0 - eps, j);
    noise.setParams(params);
    const double lm = noise.logLikelihood();
    params.setVal(p0, j);
    noise.setParams(params);
    const double fd = 0.5 * (lp - lm) / eps;
    fail += checkClose("paramGrad[" + ndlstrutil::itoa(static_cast<long>(j)) + "]",
                       g.getVal(j), fd, tol);
  }
  return fail;
}

static int testStudentT()
{
  const double y = 1.2, f = 0.3, variance = 0.5, nu = 5.0;
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::studentTLaplaceDerivs(logp, d1, W, dW, y, f, variance, nu);
  int fail = 0;
  fail += checkClose("StudentT logp", logp, -1.4640183713192005);
  fail += checkClose("StudentT d1", d1, 1.6314199395770392);
  fail += checkClose("StudentT W", W, 0.9255118153357493);
  fail += checkClose("StudentT dWdf", dW, 1.9923511825869418);

  CMatrix Y(3, 1);
  Y.setVal(0.5, 0, 0);
  Y.setVal(-0.2, 1, 0);
  Y.setVal(1.1, 2, 0);
  CStudentTNoise noise(&Y);
  noise.setVariance(0.4);
  noise.setDegFree(6.0);
  for(unsigned int i = 0; i < Y.getRows(); i++)
  {
    noise.setMu(0.1 * static_cast<double>(i), i, 0);
    noise.setVarSigma(0.0, i, 0);
  }
  fail += fdParamGrads(noise, 1e-5);

  ostringstream oss;
  writeNoiseToStream(noise, oss);
  istringstream iss(oss.str());
  CNoise* loaded = readNoiseFromStream(iss);
  fail += checkClose("stream type student_t",
                     loaded->getType() == "student_t" ? 1.0 : 0.0, 1.0);
  CStudentTNoise* st = dynamic_cast<CStudentTNoise*>(loaded);
  fail += checkClose("stream variance", st->getVariance(), 0.4);
  fail += checkClose("stream deg_free", st->getDegFree(), 6.0);
  delete loaded;
  return fail;
}

static int testPoisson()
{
  const double y = 3.0, f = 0.5;
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::poissonLogLaplaceDerivs(logp, d1, W, dW, y, f);
  int fail = 0;
  fail += checkClose("Poisson logp", logp, -1.9404807399281832);
  fail += checkClose("Poisson d1", d1, 1.3512787292998718);
  fail += checkClose("Poisson W", W, 1.6487212707001282);
  fail += checkClose("Poisson dWdf", dW, 1.6487212707001282);

  CMatrix Y(3, 1);
  Y.setVal(1.0, 0, 0);
  Y.setVal(2.0, 1, 0);
  Y.setVal(0.0, 2, 0);
  CPoissonNoise noise(&Y);
  for(unsigned int i = 0; i < Y.getRows(); i++)
  {
    noise.setMu(0.2 * static_cast<double>(i), i, 0);
    noise.setVarSigma(0.0, i, 0);
  }
  fail += fdParamGrads(noise, 1e-5);
  const double L = noise.logLikelihood();
  fail += checkClose("Poisson logLik finite", isfinite(L) ? 1.0 : 0.0, 1.0);

  ostringstream oss;
  writeNoiseToStream(noise, oss);
  istringstream iss(oss.str());
  CNoise* loaded = readNoiseFromStream(iss);
  fail += checkClose("stream type poisson",
                     loaded->getType() == "poisson" ? 1.0 : 0.0, 1.0);
  delete loaded;
  return fail;
}

static int testBinomial()
{
  const double y = 7.0, f = 0.2, N = 10.0;
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::binomialProbitLaplaceDerivs(logp, d1, W, dW, y, f, N);
  int fail = 0;
  fail += checkClose("Binomial logp", logp, -1.6317573013221582);
  fail += checkClose("Binomial d1", d1, 1.9372647142751154);
  fail += checkClose("Binomial W", W, 6.168960985378519, 1e-9);

  // N=1, y=1 should match Bernoulli–probit Laplace at f.
  double lpB = 0.0, d1B = 0.0, WB = 0.0, dWB = 0.0;
  ndlnoise::bernoulliProbitLaplaceDerivs(lpB, d1B, WB, dWB, 1.0, f);
  double lp1 = 0.0, d11 = 0.0, W1 = 0.0, dW1 = 0.0;
  ndlnoise::binomialProbitLaplaceDerivs(lp1, d11, W1, dW1, 1.0, f, 1.0);
  fail += checkClose("Binomial N=1 ≡ Bernoulli logp", lp1, lpB);
  fail += checkClose("Binomial N=1 ≡ Bernoulli d1", d11, d1B);
  fail += checkClose("Binomial N=1 ≡ Bernoulli W", W1, WB, 1e-9);

  CMatrix Y(3, 1);
  Y.setVal(2.0, 0, 0);
  Y.setVal(5.0, 1, 0);
  Y.setVal(1.0, 2, 0);
  CBinomialNoise noise(&Y, 8);
  for(unsigned int i = 0; i < Y.getRows(); i++)
  {
    noise.setMu(0.1 * static_cast<double>(i) - 0.2, i, 0);
    noise.setVarSigma(0.0, i, 0);
  }
  fail += fdParamGrads(noise, 1e-5);

  ostringstream oss;
  writeNoiseToStream(noise, oss);
  istringstream iss(oss.str());
  CNoise* loaded = readNoiseFromStream(iss);
  fail += checkClose("stream type binomial",
                     loaded->getType() == "binomial" ? 1.0 : 0.0, 1.0);
  CBinomialNoise* bn = dynamic_cast<CBinomialNoise*>(loaded);
  fail += checkClose("stream numTrials",
                     static_cast<double>(bn->getNumTrials()), 8.0);
  delete loaded;
  return fail;
}

static int testExponential()
{
  const double y = 1.5, f = 0.2;
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::exponentialLogLaplaceDerivs(logp, d1, W, dW, y, f);
  int fail = 0;
  fail += checkClose("Exponential logp", logp, -1.632104137240255);
  fail += checkClose("Exponential d1", d1, -0.8321041372402549);
  fail += checkClose("Exponential W", W, 1.832104137240255);

  CMatrix Y(3, 1);
  Y.setVal(0.5, 0, 0);
  Y.setVal(1.2, 1, 0);
  Y.setVal(0.8, 2, 0);
  CExponentialNoise noise(&Y);
  for(unsigned int i = 0; i < Y.getRows(); i++)
  {
    noise.setMu(-0.1 * static_cast<double>(i), i, 0);
    noise.setVarSigma(0.0, i, 0);
  }
  fail += fdParamGrads(noise, 1e-5);
  ostringstream oss;
  writeNoiseToStream(noise, oss);
  istringstream iss(oss.str());
  CNoise* loaded = readNoiseFromStream(iss);
  fail += checkClose("stream type exponential",
                     loaded->getType() == "exponential" ? 1.0 : 0.0, 1.0);
  delete loaded;
  return fail;
}

static int testGamma()
{
  const double y = 1.5, f = 0.0, beta = 2.0;
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::gammaLogLaplaceDerivs(logp, d1, W, dW, y, f, beta);
  int fail = 0;
  fail += checkClose("Gamma logp", logp, -1.208240530771945, 1e-9);
  fail += checkClose("Gamma d1", d1, 1.3516559071392853, 1e-9);
  fail += checkClose("Gamma W", W, 1.2280803602536212, 1e-6);
  fail += checkClose("Gamma dWdf", dW, 3.1546424444859253, 1e-4);

  CMatrix Y(3, 1);
  Y.setVal(0.7, 0, 0);
  Y.setVal(1.1, 1, 0);
  Y.setVal(0.9, 2, 0);
  CGammaNoise noise(&Y);
  noise.setBeta(1.5);
  for(unsigned int i = 0; i < Y.getRows(); i++)
  {
    noise.setMu(0.05 * static_cast<double>(i), i, 0);
    noise.setVarSigma(0.0, i, 0);
  }
  fail += fdParamGrads(noise, 5e-5);
  ostringstream oss;
  writeNoiseToStream(noise, oss);
  istringstream iss(oss.str());
  CNoise* loaded = readNoiseFromStream(iss);
  fail += checkClose("stream type gamma",
                     loaded->getType() == "gamma" ? 1.0 : 0.0, 1.0);
  CGammaNoise* gn = dynamic_cast<CGammaNoise*>(loaded);
  fail += checkClose("stream beta", gn->getBeta(), 1.5);
  delete loaded;
  return fail;
}

int main()
{
  int fail = 0;
  fail += testStudentT();
  fail += testPoisson();
  fail += testBinomial();
  fail += testExponential();
  fail += testGamma();
  if(fail)
  {
    cout << "testLikelihoodZoo: " << fail << " failure(s)." << endl;
    return 1;
  }
  cout << "All Phase-2/3 likelihood zoo tests passed." << endl;
  return 0;
}
