/* CIP-0018: CMrd ELBO, grads, factorizeSpace. */
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include "CMrd.h"
#include "ndlutil.h"

using namespace std;

namespace {

const unsigned int N = 8;
const unsigned int Q = 3;
const unsigned int M = 3;
const unsigned int D0 = 2;
const unsigned int D1 = 3;

void fillViewY(CMatrix& Y, unsigned int sharedQ, unsigned int privateQ, double scale)
{
  // Y[:,d] ≈ shared latent + private latent + noise (synthetic).
  for(unsigned int n = 0; n < Y.getRows(); n++)
  {
    const double xs = 0.4 * sin(0.7 * n + sharedQ);
    const double xp = 0.3 * cos(0.5 * n + privateQ);
    for(unsigned int d = 0; d < Y.getCols(); d++)
      Y.setVal(scale * (xs + 0.5 * xp + 0.05 * d), n, d);
  }
}

CMrd* makeToy(int verbosity = 0)
{
  CMatrix Y0(N, D0);
  CMatrix Y1(N, D1);
  fillViewY(Y0, 0, 1, 1.0);
  fillViewY(Y1, 0, 2, 1.2);
  vector<CMatrix> Ys;
  Ys.push_back(Y0);
  Ys.push_back(Y1);
  CRbfardKern* k0 = new CRbfardKern(Q);
  k0->setInitParam();
  CRbfardKern* k1 = new CRbfardKern(Q);
  k1->setInitParam();
  vector<CKern*> ks;
  ks.push_back(k0);
  ks.push_back(k1);
  return new CMrd(Ys, ks, Q, M, verbosity);
}

int testElboFinite()
{
  CMrd* model = makeToy();
  const double L = model->logLikelihood();
  delete model;
  if(std::isnan(L) || std::isinf(L))
  {
    cout << "FAILURE: MRD ELBO not finite: " << L << endl;
    return 1;
  }
  cout << "OK: MRD ELBO finite (" << L << ")" << endl;
  return 0;
}

int testCheckGradients()
{
  // Isotropic RBF views: ARD input-scale FD is soft at the psi layer;
  // joint packing is what we need to gate here.
  CMatrix Y0(N, D0);
  CMatrix Y1(N, D1);
  fillViewY(Y0, 0, 1, 1.0);
  fillViewY(Y1, 0, 2, 1.2);
  vector<CMatrix> Ys;
  Ys.push_back(Y0);
  Ys.push_back(Y1);
  CRbfKern* k0 = new CRbfKern(Q);
  k0->setInitParam();
  CRbfKern* k1 = new CRbfKern(Q);
  k1->setInitParam();
  vector<CKern*> ks;
  ks.push_back(k0);
  ks.push_back(k1);
  CMrd model(Ys, ks, Q, M, 0);

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
  if(maxAbsDiff > 5e-3 && maxRelDiff > 5e-3)
  {
    cout << "FAILURE: MRD checkGradients maxAbs=" << maxAbsDiff
         << " maxRel=" << maxRelDiff << endl;
    return 1;
  }
  cout << "OK: MRD checkGradients (maxAbs=" << maxAbsDiff
       << " maxRel=" << maxRelDiff << ")" << endl;
  return 0;
}

int testSingleKl()
{
  CMrd* model = makeToy();
  const double L = model->logLikelihood();
  double sumVar = 0.0;
  for(unsigned int v = 0; v < model->getNumViews(); v++)
    sumVar += model->getView(v)->varDtcBound();
  const double kl = model->getView(0)->klDivergence();
  delete model;
  if(fabs(L - (sumVar - kl)) > 1e-8)
  {
    cout << "FAILURE: ELBO != sum VarDTC - KL once (" << L
         << " vs " << (sumVar - kl) << ")" << endl;
    return 1;
  }
  cout << "OK: single KL accounting" << endl;
  return 0;
}

int testOptimiseAndFactorize()
{
  CMrd* model = makeToy();
  model->setVerbosity(0);
  model->optimise(25);
  vector<unsigned int> shared, p0, p1;
  model->factorizeSpace(shared, p0, p1, 0.05);
  stringstream ss;
  model->display(ss);
  const bool okDisp = ss.str().find("MRD:") != string::npos;
  delete model;
  if(!okDisp)
  {
    cout << "FAILURE: display missing MRD header" << endl;
    return 1;
  }
  cout << "OK: optimise + factorizeSpace (shared=" << shared.size()
       << " priv0=" << p0.size() << " priv1=" << p1.size() << ")" << endl;
  return 0;
}

int testStreamRoundTrip()
{
  CMrd* model = makeToy();
  model->setVerbosity(0);
  model->optimise(5);
  const double L0 = model->logLikelihood();
  stringstream ss;
  model->toStream(ss);
  // Reload into a fresh shell of the same shape.
  CMatrix Y0(N, D0);
  CMatrix Y1(N, D1);
  fillViewY(Y0, 0, 1, 1.0);
  fillViewY(Y1, 0, 2, 1.2);
  vector<CMatrix> Ys;
  Ys.push_back(Y0);
  Ys.push_back(Y1);
  CRbfardKern* k0 = new CRbfardKern(Q);
  k0->setInitParam();
  CRbfardKern* k1 = new CRbfardKern(Q);
  k1->setInitParam();
  vector<CKern*> ks;
  ks.push_back(k0);
  ks.push_back(k1);
  CMrd loaded(Ys, ks, Q, M, 0);
  loaded.fromStream(ss);
  const double L1 = loaded.logLikelihood();
  delete model;
  if(fabs(L0 - L1) > 1e-6)
  {
    cout << "FAILURE: stream ELBO " << L0 << " vs " << L1 << endl;
    return 1;
  }
  cout << "OK: MRD stream round-trip" << endl;
  return 0;
}

int testRejectOneView()
{
  CMatrix Y0(N, D0);
  fillViewY(Y0, 0, 1, 1.0);
  vector<CMatrix> Ys;
  Ys.push_back(Y0);
  CRbfardKern* k0 = new CRbfardKern(Q);
  k0->setInitParam();
  vector<CKern*> ks;
  ks.push_back(k0);
  try
  {
    CMrd model(Ys, ks, Q, M, 0);
    delete k0;
    cout << "FAILURE: one view should throw" << endl;
    return 1;
  }
  catch(ndlexceptions::Error&)
  {
    delete k0;
    cout << "OK: reject V<2" << endl;
    return 0;
  }
}

} // namespace

int main()
{
  int fail = 0;
  try
  {
    fail += testElboFinite();
    fail += testSingleKl();
    fail += testCheckGradients();
    fail += testOptimiseAndFactorize();
    fail += testStreamRoundTrip();
    fail += testRejectOneView();
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
}
