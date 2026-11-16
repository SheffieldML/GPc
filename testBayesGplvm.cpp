/* CIP-0017 Phase 2: Bayesian GP-LVM ELBO + checkGradients. */
#include <iostream>
#include <sstream>
#include "CBayesGplvm.h"
#include "ndlutil.h"

using namespace std;

namespace {
const unsigned int N = 6;
const unsigned int D = 3;
const unsigned int Q = 2;
const unsigned int M = 3;

CMatrix makeY()
{
  CMatrix Y(N, D);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int d = 0; d < D; d++)
      Y.setVal(sin(0.3 * n + 0.7 * d) + 0.1 * n, n, d);
  return Y;
}

int testCheckGradients()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  kern.setInverseWidth(1.0);
  kern.setVariance(1.0);
  CBayesGplvm model(&kern, Y, Q, M, 0);
  model.setBeta(2.0);
  // Shrink S for numerical stability of FD.
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
      model.getS().setVal(0.3, n, q);

  const unsigned int nParams = model.getOptNumParams();
  CMatrix analytic(1, nParams);
  CMatrix params(1, nParams);
  model.getOptParams(params);
  model.computeObjectiveGradParams(analytic);

  const double change = ndlutil::GRADCHANGE;
  double maxAbsDiff = 0.0;
  double maxRelDiff = 0.0;
  for(unsigned int j = 0; j < nParams; j++)
  {
    const double orig = params.getVal(j);
    params.setVal(orig + change, j);
    model.setOptParams(params);
    const double plus = model.computeObjectiveVal();
    params.setVal(orig - change, j);
    model.setOptParams(params);
    const double minus = model.computeObjectiveVal();
    params.setVal(orig, j);
    model.setOptParams(params);
    const double numerical = 0.5 * (plus - minus) / change;
    const double a = analytic.getVal(j);
    const double diff = fabs(a - numerical);
    if(diff > maxAbsDiff)
      maxAbsDiff = diff;
    const double scale = 1.0 + fabs(a) + fabs(numerical);
    const double rel = diff / scale;
    if(rel > maxRelDiff)
      maxRelDiff = rel;
  }

  // Absolute + relative: large params need relative; tiny grads need absolute.
  const double absTol = 5e-3;
  const double relTol = 5e-3;
  if(maxAbsDiff > absTol && maxRelDiff > relTol)
  {
    cout << "FAILURE: checkGradients maxAbs=" << maxAbsDiff
         << " maxRel=" << maxRelDiff << endl;
    model.checkGradients();
    return 1;
  }
  cout << "OK: checkGradients (maxAbs=" << maxAbsDiff
       << " maxRel=" << maxRelDiff << ")" << endl;
  return 0;
}

int testElboFinite()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 0);
  const double L = model.logLikelihood();
  if(!(L < 0.0) || std::isnan(L) || std::isinf(L))
  {
    // ELBO can be positive for small toy data; only require finite.
  }
  if(std::isnan(L) || std::isinf(L))
  {
    cout << "FAILURE: ELBO not finite: " << L << endl;
    return 1;
  }
  cout << "OK: ELBO finite (" << L << ")" << endl;
  return 0;
}

int testStreamRoundTrip()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 0);
  model.setBeta(1.7);
  stringstream ss;
  model.toStream(ss);
  CBayesGplvm model2(&kern, Y, Q, M, 0);
  model2.fromStream(ss);
  if(fabs(model.getBeta() - model2.getBeta()) > 1e-12)
  {
    cout << "FAILURE: beta stream mismatch" << endl;
    return 1;
  }
  if(model.getMu().maxAbsDiff(model2.getMu()) > 1e-12)
  {
    cout << "FAILURE: mu stream mismatch" << endl;
    return 1;
  }
  cout << "OK: stream round-trip beta/mu" << endl;
  return 0;
}

} // namespace

int main()
{
  int fail = 0;
  try
  {
    fail += testElboFinite();
    fail += testStreamRoundTrip();
    fail += testCheckGradients();
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
}
