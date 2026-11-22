/* CIP-003A Phase 0/1: inverse Mills helpers + SkewGaussian / SkewExponential.
   Reference numbers from SciPy / GPy (see cip/cip003A/gen_skew_fixture.py).
   No MATLAB / matio required. */
#include "CNoise.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;

static int checkClose(const string& label, double a, double b, double tol = ndlutil::MATCHTOL)
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

static int testInvMills()
{
  // SciPy / GPy gaussian_special at selected z.
  struct Row { double z, lam, dlam, d2lam; };
  const Row rows[] = {
    {-3.0, 3.2830986549304400e+00, -9.2944081321474414e-01, 3.1470672830889092e-02},
    {-1.0, 1.5251352761609811e+00, -8.0090233442965097e-01, 1.1693119540604828e-01},
    { 0.0, 7.9788456080286541e-01, -6.3661977236758138e-01, 2.1801361414499024e-01},
    { 0.5, 5.0916043383703347e-01, -5.1382456430363288e-01, 2.7099012446870779e-01},
    { 2.0, 5.5247862678989949e-02, -1.1354805168857643e-01, 1.8439481503247757e-01},
  };
  int fail = 0;
  const double tol = 1e-12;
  for(unsigned int k = 0; k < sizeof(rows) / sizeof(rows[0]); k++)
  {
    const double z = rows[k].z;
    const double lam = ndlnoise::invMillsRatio(z);
    ostringstream lab;
    lab << z;
    fail += checkClose("invMills(" + lab.str() + ")", lam, rows[k].lam, tol);
    fail += checkClose("dInvMills(" + lab.str() + ")",
                       ndlnoise::dInvMillsRatio(z, lam), rows[k].dlam, tol);
    fail += checkClose("d2InvMills(" + lab.str() + ")",
                       ndlnoise::d2InvMillsRatio(z, lam), rows[k].d2lam, tol);
  }
  return fail;
}

static int testSkewGaussianDensity()
{
  const double y = 1.2;
  const double f = 0.3;
  const double variance = 0.25;
  const double alpha = 2.0;
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewGaussianLaplaceDerivs(logp, d1, W, dW, y, f, variance, alpha);
  int fail = 0;
  fail += checkClose("SkewGaussian logp vs SciPy skewnorm",
                     logp, -1.1528032933340535, 1e-12);
  fail += checkClose("SkewGaussian d1", d1, 3.5975520027821584, 1e-12);
  fail += checkClose("SkewGaussian W=-d2", W, 4.035257152627291, 1e-12);
  // dW/df = −∂³logp/∂f³; GPy d3logpdf_dlink3 ≈ -0.4687 ⇒ dW ≈ +0.4687
  fail += checkClose("SkewGaussian dWdf", dW, 0.4687076611706116, 1e-12);

  double logp0 = 0.0, d10 = 0.0, W0 = 0.0, dW0 = 0.0;
  ndlnoise::skewGaussianLaplaceDerivs(logp0, d10, W0, dW0, y, f, variance, 0.0);
  const double sigma = sqrt(variance);
  const double e = y - f;
  const double gauss = -ndlutil::HALFLOGTWOPI - log(sigma) - 0.5 * (e * e) / variance;
  fail += checkClose("SkewGaussian alpha=0 vs Gaussian", logp0, gauss, 1e-12);
  fail += checkClose("SkewGaussian alpha=0 vs SciPy", logp0, -1.845791352644727, 1e-12);
  return fail;
}

static int testSkewExponentialDensity()
{
  const double y = 1.2;
  const double f = 0.3;
  const double rate = 1.5;
  const double alpha = 1.0;
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewExponentialLaplaceDerivs(logp, d1, W, dW, y, f, rate, alpha);
  return checkClose("SkewExponential logp", logp, -1.1479495016475825, 1e-12);
}

static int fdParamGrads(CNoise& noise, double tol)
{
  const unsigned int n = noise.getNumParams();
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

static int testSkewGaussianGradients()
{
  CMatrix y(4, 1);
  y.setVal(0.5, 0, 0);
  y.setVal(-0.2, 1, 0);
  y.setVal(1.1, 2, 0);
  y.setVal(0.0, 3, 0);
  CSkewGaussianNoise noise(&y);
  noise.setVariance(0.4);
  noise.setAlpha(1.5);
  for(unsigned int i = 0; i < y.getRows(); i++)
  {
    noise.setMu(0.1 * static_cast<double>(i), i, 0);
    noise.setVarSigma(0.0, i, 0);
  }
  noise.checkGradients(); // prints transform-space comparison
  return fdParamGrads(noise, 1e-5);
}

static int testSkewExponentialGradients()
{
  CMatrix y(4, 1);
  y.setVal(0.5, 0, 0);
  y.setVal(-0.2, 1, 0);
  y.setVal(1.1, 2, 0);
  y.setVal(0.3, 3, 0);
  CSkewExponentialNoise noise(&y);
  noise.setRate(1.2);
  noise.setAlpha(0.8);
  for(unsigned int i = 0; i < y.getRows(); i++)
  {
    noise.setMu(0.05 * static_cast<double>(i), i, 0);
    noise.setVarSigma(0.0, i, 0);
  }
  noise.checkGradients();
  return fdParamGrads(noise, 1e-5);
}

static int testStreamRoundTrip()
{
  CMatrix y(2, 1);
  y.setVal(1.0, 0, 0);
  y.setVal(-1.0, 1, 0);
  CSkewGaussianNoise src(&y);
  src.setVariance(0.3);
  src.setAlpha(2.0);
  ostringstream oss;
  writeNoiseToStream(src, oss);
  istringstream iss(oss.str());
  CNoise* loaded = readNoiseFromStream(iss);
  int fail = 0;
  if(loaded->getType() != "skew_gaussian")
  {
    cout << "FAILURE: stream type " << loaded->getType() << endl;
    fail = 1;
  }
  else
  {
    CSkewGaussianNoise* sk = dynamic_cast<CSkewGaussianNoise*>(loaded);
    fail += checkClose("stream variance", sk->getVariance(), 0.3, 1e-12);
    fail += checkClose("stream alpha", sk->getAlpha(), 2.0, 1e-12);
  }
  delete loaded;
  return fail;
}

int main()
{
  int fail = 0;
  fail += testInvMills();
  fail += testSkewGaussianDensity();
  fail += testSkewExponentialDensity();
  fail += testSkewGaussianGradients();
  fail += testSkewExponentialGradients();
  fail += testStreamRoundTrip();
  if(fail)
  {
    cout << "testSkewNoise: " << fail << " failure(s)." << endl;
    return 1;
  }
  cout << "All skew noise tests passed." << endl;
  return 0;
}
