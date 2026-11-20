/* CIP-0027 Phase 0: batched Gaussian likelihood helpers vs per-element oracles.
   No MATLAB / matio fixtures required. */
#include "CNoise.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

static int checkClose(const string& label, double a, double b)
{
  const double diff = fabs(a - b);
  if(diff <= ndlutil::MATCHTOL)
  {
    cout << label << " passes." << endl;
    return 0;
  }
  cout << "FAILURE: " << label << ": |a-b|=" << diff
       << " (a=" << a << ", b=" << b << ")" << endl;
  return 1;
}

static int testAddObservationNoiseDiag()
{
  CMatrix K(3, 3);
  K.zeros();
  K.setVal(2.0, 0, 0);
  K.setVal(3.0, 1, 1);
  K.setVal(4.0, 2, 2);
  ndlnoise::addObservationNoiseDiag(K, 0.5);
  int fail = 0;
  fail += checkClose("diag(0)+0.5", K.getVal(0, 0), 2.5);
  fail += checkClose("diag(1)+0.5", K.getVal(1, 1), 3.5);
  fail += checkClose("diag(2)+0.5", K.getVal(2, 2), 4.5);
  fail += checkClose("off-diag untouched", K.getVal(0, 1), 0.0);
  return fail;
}

static int testVariationalExpectationsOracle()
{
  const unsigned int n = 4;
  const unsigned int d = 2;
  CMatrix y(n, d);
  CMatrix mu(n, d);
  CMatrix var(n, d);
  CMatrix ell(n, d);
  CMatrix gMu(n, d);
  CMatrix gVar(n, d);
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = 0; j < d; j++)
    {
      y.setVal(0.1 * static_cast<double>(i + 1) + 0.01 * j, i, j);
      mu.setVal(0.05 * static_cast<double>(i) - 0.02 * j, i, j);
      var.setVal(0.2 + 0.01 * static_cast<double>(i + j), i, j);
    }
  }
  const double sigma2 = 0.35;
  vector<double> bias(d, 0.0);
  bias[0] = 0.1;
  bias[1] = -0.05;

  ndlnoise::gaussianVariationalExpectations(ell, gMu, gVar, y, mu, var, sigma2, &bias[0]);

  CGaussianNoise noise(&y);
  noise.setSigma2(sigma2);
  noise.setBiasVal(bias[0], 0);
  noise.setBiasVal(bias[1], 1);

  int fail = 0;
  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = 0; j < d; j++)
    {
      // expectedLogLikelihood does not subtract process bias; pass y - bias.
      double gMuEl = 0.0;
      double gVarEl = 0.0;
      const double yAdj = y.getVal(i, j) - bias[j];
      const double ellEl = noise.expectedLogLikelihood(yAdj, mu.getVal(i, j),
                                                       var.getVal(i, j), gMuEl, gVarEl);
      fail += checkClose("VE ell(" + ndlstrutil::itoa(i) + "," + ndlstrutil::itoa(j) + ")",
                         ell.getVal(i, j), ellEl);
      fail += checkClose("VE gMu(" + ndlstrutil::itoa(i) + "," + ndlstrutil::itoa(j) + ")",
                         gMu.getVal(i, j), gMuEl);
      fail += checkClose("VE gVar(" + ndlstrutil::itoa(i) + "," + ndlstrutil::itoa(j) + ")",
                         gVar.getVal(i, j), gVarEl);
    }
  }
  return fail;
}

static int testLogPredictiveOracle()
{
  const unsigned int n = 3;
  const unsigned int d = 1;
  CMatrix y(n, d);
  CMatrix mu(n, d);
  CMatrix var(n, d);
  for(unsigned int i = 0; i < n; i++)
  {
    y.setVal(1.0 + 0.2 * i, i, 0);
    mu.setVal(0.5 * i, i, 0);
    var.setVal(0.1 * (i + 1), i, 0);
  }
  const double sigma2 = 0.25;
  CGaussianNoise noise(&y);
  noise.setSigma2(sigma2);
  noise.setBiasVal(0.0, 0);

  const double batched = ndlnoise::gaussianLogPredictiveDensity(y, mu, var, sigma2, 0);
  const double oracle = noise.logLikelihood(mu, var, y);
  return checkClose("log_predictive_density vs logLikelihood", batched, oracle);
}

static int testLogDensityMatchesVEAtZeroVar()
{
  CMatrix y(2, 1);
  CMatrix f(2, 1);
  CMatrix var(2, 1);
  CMatrix logDens(2, 1);
  CMatrix ell(2, 1);
  CMatrix gMu(2, 1);
  CMatrix gVar(2, 1);
  y.setVal(1.0, 0, 0);
  y.setVal(-0.5, 1, 0);
  f.setVal(0.8, 0, 0);
  f.setVal(-0.4, 1, 0);
  var.zeros();
  const double sigma2 = 0.5;
  ndlnoise::gaussianLogDensity(logDens, y, f, sigma2, 0);
  ndlnoise::gaussianVariationalExpectations(ell, gMu, gVar, y, f, var, sigma2, 0);
  int fail = 0;
  fail += checkClose("log_density == VE at v=0 (0)", logDens.getVal(0, 0), ell.getVal(0, 0));
  fail += checkClose("log_density == VE at v=0 (1)", logDens.getVal(1, 0), ell.getVal(1, 0));
  return fail;
}

static int testVarianceAccessors()
{
  CMatrix y(1, 1);
  y.setVal(0.0, 0, 0);
  CGaussianNoise noise(&y);
  noise.setVariance(0.42);
  int fail = 0;
  fail += checkClose("getVariance", noise.getVariance(), 0.42);
  fail += checkClose("getSigma2 alias", noise.getSigma2(), 0.42);
  noise.setSigma2(0.17);
  fail += checkClose("setSigma2 -> getVariance", noise.getVariance(), 0.17);
  return fail;
}

int main()
{
  int fail = 0;
  try
  {
    fail += testAddObservationNoiseDiag();
    fail += testVariationalExpectationsOracle();
    fail += testLogPredictiveOracle();
    fail += testLogDensityMatchesVEAtZeroVar();
    fail += testVarianceAccessors();
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error err)
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
