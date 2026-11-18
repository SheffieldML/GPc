/* Diagnostic for DTC / DTCVAR: finite-difference gradient check, state dump,
   and CIP-000E dual-path Legacy vs StableFactor LL/grad compare. */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlutil.h"

using namespace std;

namespace {

void fillSynthetic(CMatrix& X, CMatrix& y, CMatrix& Xu)
{
  // Fixed N=10, q=2, d=1, M=5 — deterministic, no RNG.
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
}

double maxAbsDiffCheck(CGp& model, double change, ostream& os)
{
  const int nParams = static_cast<int>(model.getOptNumParams());
  CMatrix analyticGrad(1, nParams);
  CMatrix numericalDiff(1, nParams);
  CMatrix params(1, nParams);
  CMatrix origParams(1, nParams);

  model.getOptParams(params);
  origParams.deepCopy(params);
  model.computeObjectiveGradParams(analyticGrad);

  double maxAbs = 0.0;
  for(int j = 0; j < nParams; j++)
  {
    const double origParam = origParams.getVal(j);
    params.setVal(origParam + change, j);
    model.setOptParams(params);
    const double objectivePlus = model.computeObjectiveVal();
    params.setVal(origParam - change, j);
    model.setOptParams(params);
    const double objectiveMinus = model.computeObjectiveVal();
    const double nd = 0.5 * (objectivePlus - objectiveMinus) / change;
    numericalDiff.setVal(nd, j);
    const double diff = fabs(analyticGrad.getVal(j) - nd);
    if(diff > maxAbs)
      maxAbs = diff;
    params.setVal(origParams.getVal(j), j);
  }
  model.setOptParams(origParams);

  os << "param_index analytic numerical abs_diff" << endl;
  for(int j = 0; j < nParams; j++)
  {
    const double a = analyticGrad.getVal(j);
    const double n = numericalDiff.getVal(j);
    os << j << " " << setprecision(16) << a << " " << n << " " << fabs(a - n) << endl;
  }
  return maxAbs;
}

void dumpState(const string& path, const CMatrix& X, const CMatrix& y, const CMatrix& Xu,
               CGp& model, const string& approx)
{
  ofstream out(path.c_str());
  out << setprecision(16);
  out << "approx " << approx << endl;
  out << "N " << X.getRows() << endl;
  out << "q " << X.getCols() << endl;
  out << "d " << y.getCols() << endl;
  out << "M " << Xu.getRows() << endl;
  out << "inverseWidth " << model.getKernel()->getParam(0) << endl;
  out << "variance " << model.getKernel()->getParam(1) << endl;
  out << "beta " << model.getBetaVal() << endl;
  out << "ll " << model.logLikelihood() << endl;
  out << "X" << endl;
  for(unsigned int i = 0; i < X.getRows(); i++)
  {
    for(unsigned int j = 0; j < X.getCols(); j++)
      out << X.getVal(i, j) << (j + 1 == X.getCols() ? "" : " ");
    out << endl;
  }
  out << "y" << endl;
  for(unsigned int i = 0; i < y.getRows(); i++)
    out << y.getVal(i, 0) << endl;
  out << "Xu" << endl;
  for(unsigned int i = 0; i < Xu.getRows(); i++)
  {
    for(unsigned int j = 0; j < Xu.getCols(); j++)
      out << Xu.getVal(i, j) << (j + 1 == Xu.getCols() ? "" : " ");
    out << endl;
  }
  CMatrix g(1, model.getOptNumParams());
  model.logLikelihoodGradient(g);
  out << "grads_ll " << g.getCols() << endl;
  for(unsigned int j = 0; j < g.getCols(); j++)
    out << g.getVal(0, j) << endl;
  out.close();
}

int dualPathCompare(CGp& model, const string& name)
{
  cout << "---- dual-path " << name << " (CIP-000E) ----" << endl;
  model.setStableSparseLinalg(false);
  const double llLegacy = model.logLikelihood();
  CMatrix gLegacy(1, model.getOptNumParams());
  model.logLikelihoodGradient(gLegacy);

  model.setStableSparseLinalg(true);
  const double llStable = model.logLikelihood();
  CMatrix gStable(1, model.getOptNumParams());
  model.logLikelihoodGradient(gStable);

  const double llDiff = fabs(llLegacy - llStable);
  double gMax = 0.0;
  for(unsigned int j = 0; j < gLegacy.getCols(); j++)
  {
    const double d = fabs(gLegacy.getVal(0, j) - gStable.getVal(0, j));
    if(d > gMax)
      gMax = d;
  }
  cout << setprecision(16);
  cout << "ll_legacy: " << llLegacy << endl;
  cout << "ll_stable: " << llStable << endl;
  cout << "ll_abs_diff: " << llDiff << endl;
  cout << "grad_max_abs_diff: " << gMax << endl;

  const double tol = ndlutil::MATCHTOL;
  const bool ok = (llDiff <= tol) && (gMax <= tol);
  cout << (ok ? "PASS" : "FAIL") << " dual-path (tol=" << tol << ")" << endl;

  // Restore default for any later Legacy-facing callers.
  model.setStableSparseLinalg(false);
  return ok ? 0 : 1;
}

int runApprox(int approxType, const string& name, const string& dumpPath)
{
  cout << "==== " << name << " ====" << endl;
  CMatrix X(10, 2);
  CMatrix y(10, 1);
  CMatrix Xu(5, 2);
  fillSynthetic(X, y, Xu);

  CRbfKern kern(X);
  kern.setParam(1.5, 0); // inverseWidth
  kern.setParam(0.8, 1); // variance
  CGaussianNoise noise(&y);
  for(unsigned int i = 0; i < noise.getOutputDim(); i++)
    noise.setBiasVal(0.0, i);

  CGp model(&kern, &noise, &X, approxType, 5, 0);
  model.setInducingFixed(true);
  model.setActiveSet(Xu);
  model.setBetaVals(100.0);
  model.updateM();

  const double ll = model.logLikelihood();
  cout << "logLikelihood: " << setprecision(16) << ll << endl;

  const double maxAbs = maxAbsDiffCheck(model, ndlutil::GRADCHANGE, cout);
  cout << "maxAbsDiff(analytic, numerical): " << maxAbs << endl;
  const double tol = 1e-5;
  const bool ok = maxAbs < tol;
  cout << (ok ? "PASS" : "FAIL") << " checkGradients (tol=" << tol << ")" << endl;

  dumpState(dumpPath, X, y, Xu, model, name);
  cout << "Wrote " << dumpPath << endl;

  int fail = ok ? 0 : 1;
  // Dual-path is most informative for DTCVAR (uses invK_uu K_uf); still run DTC.
  fail += dualPathCompare(model, name);

  // StableFactor must also pass FD grads.
  model.setStableSparseLinalg(true);
  const double maxAbsStable = maxAbsDiffCheck(model, ndlutil::GRADCHANGE, cout);
  cout << "stable maxAbsDiff(analytic, numerical): " << maxAbsStable << endl;
  const bool okStable = maxAbsStable < tol;
  cout << (okStable ? "PASS" : "FAIL") << " StableFactor checkGradients (tol=" << tol << ")" << endl;
  if(!okStable)
    fail += 1;
  model.setStableSparseLinalg(false);

  return fail;
}

} // namespace

int main()
{
  int fail = 0;
  fail += runApprox(CGp::DTC, "dtc", "diag_dtc_state.txt");
  fail += runApprox(CGp::DTCVAR, "dtcvar", "diag_dtcvar_state.txt");
  return fail == 0 ? 0 : 1;
}
