/* CIP-000F: spherical PITC finite-difference gradient check on a tiny synthetic problem. */
#include <cmath>
#include <iomanip>
#include <iostream>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlutil.h"

using namespace std;

int main()
{
  CMatrix X(10, 2);
  CMatrix y(10, 1);
  CMatrix Xu(5, 2);
  const double xdat[10][2] = {
      {-1.2, 0.3}, {-0.8, -0.5}, {-0.4, 0.9}, {-0.1, -0.2}, {0.2, 0.4},
      {0.5, -0.7}, {0.7, 0.1},   {1.0, 0.6},  {1.3, -0.4}, {1.6, 0.2}};
  const double ydat[10] = {0.8, -0.3, 1.1, 0.2, -0.5, 0.4, -0.9, 0.7, -0.1, 0.3};
  for(unsigned int i = 0; i < 10; i++)
  {
    X.setVal(xdat[i][0], i, 0);
    X.setVal(xdat[i][1], i, 1);
    y.setVal(ydat[i], i, 0);
  }
  for(unsigned int i = 0; i < 5; i++)
  {
    Xu.setVal(xdat[i][0], i, 0);
    Xu.setVal(xdat[i][1], i, 1);
  }

  CRbfKern kern(X);
  kern.setParam(1.5, 0);
  kern.setParam(0.8, 1);
  CGaussianNoise noise(&y);
  for(unsigned int i = 0; i < noise.getOutputDim(); i++)
    noise.setBiasVal(0.0, i);

  CGp model(&kern, &noise, &X, CGp::PITC, 5, 0);
  model.setInducingFixed(true);
  model.setActiveSet(Xu);
  model.setBetaVals(100.0);
  model.updateM();

  cout << setprecision(16);
  cout << "blocks: " << model.getNumBlocks() << endl;
  const double ll = model.logLikelihood();
  cout << "pitc logLikelihood: " << ll << endl;

  const int nParams = (int)model.getOptNumParams();
  CMatrix analytic(1, nParams);
  CMatrix params(1, nParams);
  CMatrix orig(1, nParams);
  model.getOptParams(params);
  orig.deepCopy(params);
  model.computeObjectiveGradParams(analytic);

  double maxAbs = 0.0;
  const double change = ndlutil::GRADCHANGE;
  for(int j = 0; j < nParams; j++)
  {
    const double o = orig.getVal(j);
    params.setVal(o + change, j);
    model.setOptParams(params);
    const double plus = model.computeObjectiveVal();
    params.setVal(o - change, j);
    model.setOptParams(params);
    const double minus = model.computeObjectiveVal();
    const double nd = 0.5 * (plus - minus) / change;
    const double diff = fabs(analytic.getVal(j) - nd);
    if(diff > maxAbs)
      maxAbs = diff;
    cout << j << " analytic=" << analytic.getVal(j) << " numerical=" << nd
         << " abs_diff=" << diff << endl;
    params.setVal(o, j);
  }
  model.setOptParams(orig);

  const double tol = 1e-5;
  const bool ok = maxAbs < tol;
  cout << "maxAbsDiff: " << maxAbs << endl;
  cout << (ok ? "PASS" : "FAIL") << " pitc checkGradients (tol=" << tol << ")" << endl;
  return ok ? 0 : 1;
}
