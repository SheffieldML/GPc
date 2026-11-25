#include "CTransform.h"
#include <cmath>

CParamTransforms* ptransforms;
int testTransform(string transType);

int main()
{
  ptransforms = new CParamTransforms();
  int fail = 0;
  fail+=testTransform("exp");
  fail+=testTransform("negLogLogit");
  fail+=testTransform("sigmoid");

  CParamTransforms* preadTrans = new CParamTransforms();
#ifdef _NDLMATLAB
  ptransforms->writeMatlabFile("crap.mat", "writtenTransforms");
  preadTrans->readMatlabFile("crap.mat", "writtenTransforms");
  if(ptransforms->equals(*preadTrans))
    cout << "MATLAB Read and write transforms to stream passed"<<endl;
  else
  {
    cout << "FAILURE: Read and write to MATLAB" << endl;
    cout << "Written transforms:" << endl;
    ptransforms->display(cout);
    cout << "Read in transforms: " << endl;
    preadTrans->display(cout);
    fail++;
  }
#endif

  ptransforms->toFile("crap_paramtransform");
  preadTrans->fromFile("crap_paramtransform");
  if(ptransforms->equals(*preadTrans))
    cout << "Read and write transforms to stream passed"<<endl;
  else
  {
    cout << "FAILURE: Read and write to stream" << endl;
    cout << "Written transforms:" << endl;
    ptransforms->display(cout);
    cout << "Read in transforms: " << endl;
    preadTrans->display(cout);
    fail++;
  }
  cout << "Number of failures: " << fail << endl;
  return fail > 0 ? 1 : 0;
}

int testTransform(string transType)
{
  int fail = 0;
  CTransform* trans = CTransform::getNewTransformPointer(transType);
  // Probe both deep-negative (clamp) and deep-positive sides, plus near-zero.
  const int n = 19;
  double a[19] = {-1e16, -1e8, -1e4, -100, -1, -0.1, -1e-5, -1e-9, -1e-13, 0, 1e-13, 1e-9, 1e-5, 0.1, 1, 100, 1e4, 1e8, 1e16};
  const double roundTol = ndlutil::MATCHTOL;
  cout << "Transform " << transType << endl;
  for(int i=0; i<n; i++)
  {
    double x = trans->atox(a[i]);
    double newa = trans->xtoa(x);
    // limVal clamp: atox saturates, so xtoa(atox(a)) recovers a boundary, not a.
    const bool clamped = (a[i] < -limVal) || (a[i] > limVal);
    if(clamped)
    {
      // Bound must be finite and stable under a further atox/xtoa round-trip.
      double x2 = trans->atox(newa);
      double a2 = trans->xtoa(x2);
      if(!std::isfinite(newa) || !std::isfinite(x) ||
         fabs(x2 - x) > roundTol * (1.0 + fabs(x)) ||
         fabs(a2 - newa) > roundTol * (1.0 + fabs(newa)))
      {
        cout << "FAIL: clamp round-trip" << endl;
        cout << "a: " << a[i] << " x: " << x << " newa: " << newa
             << " atox(newa): " << x2 << " xtoa(atox(newa)): " << a2 << endl;
        fail++;
      }
      else
        cout << "atox -> xtoa clamp check passed: a: " << a[i] << " -> " << newa << endl;
    }
    else if(fabs(newa - a[i]) > roundTol * (1.0 + fabs(a[i])))
    {
      cout << "FAIL: " << endl;
      cout << "a: " << a[i] << " x: " << x << " newa: " << newa << endl;
      fail++;
    }
    else
      cout << "atox -> xtoa check passed: a: " << a[i] << endl;

    double g = trans->gradfact(x);
    double diff = 1e-6;
    double aplus = a[i]+diff;
    double xplus = trans->atox(aplus);
    double aminus = a[i]-diff;
    double xminus = trans->atox(aminus);
    double gdiff = (xplus - xminus)/(2*diff);
    // Finite-difference is unreliable once atox has saturated.
    if(!clamped && fabs(gdiff-g)>2*diff)
    {
      cout << "FAIL: Gradient: " << "g: " << g << " gdiff: " << gdiff << endl;
      fail++;
    }
    else if(!clamped)
      cout << "Gradient check passed: a: " << a[i] << endl;
    else
      cout << "Gradient check skipped (clamped): a: " << a[i] << endl;

    // CIP-003B: Jacobian terms. Probe a moderate range where the closed forms
    // are well conditioned (deep tails lose digits in the finite difference).
    if(!clamped && fabs(a[i])<=100 && fabs(a[i])>1e-6)
    {
      double lj = trans->logJacobian(x);
      if(fabs(lj - log(g)) > roundTol*(1.0+fabs(lj)))
      {
        cout << "FAIL: logJacobian " << lj << " != log(gradfact) " << log(g) << " at a=" << a[i] << endl;
        fail++;
      }
      double h = 1e-5*(1.0+fabs(a[i]));
      double ljPlus = trans->logJacobian(trans->atox(a[i]+h));
      double ljMinus = trans->logJacobian(trans->atox(a[i]-h));
      double ljGrad = (ljPlus-ljMinus)/(2*h);
      double glj = trans->gradLogJacobian(x);
      if(fabs(ljGrad-glj) > 1e-5*(1.0+fabs(glj)))
      {
        cout << "FAIL: gradLogJacobian " << glj << " vs finite difference " << ljGrad << " at a=" << a[i] << endl;
        fail++;
      }
      else
        cout << "Jacobian check passed: a: " << a[i] << endl;
    }
  }
  ptransforms->addTransform(trans, 1);
  ptransforms->addTransform(trans, 2);
  return fail;
}
