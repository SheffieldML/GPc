/* CIP-0018: CMrd ELBO, grads, factorizeSpace, GPy fixture compare. */
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include "CMrd.h"
#include "ndlutil.h"

using namespace std;

namespace {

const unsigned int N = 8;
const unsigned int Q = 3;
const unsigned int M = 3;
const unsigned int D0 = 2;
const unsigned int D1 = 3;

// Looser than MATCHTOL: GPy vs C++ Cholesky / BLAS stack differences.
const double FIXTURE_TOL = 1e-6;

string nextToken(istream& in)
{
  string tok;
  while(in >> tok)
  {
    if(!tok.empty() && tok[0] == '#')
    {
      string rest;
      getline(in, rest);
      continue;
    }
    return tok;
  }
  return string();
}

void expectKey(istream& in, const string& key)
{
  const string tok = nextToken(in);
  if(tok != key)
    throw ndlexceptions::Error("fixture: expected key " + key + " got " + tok);
}

double readDouble(istream& in)
{
  const string tok = nextToken(in);
  return atof(tok.c_str());
}

unsigned int readUint(istream& in)
{
  return (unsigned int)readDouble(in);
}

void readMatrix(istream& in, const string& key, CMatrix& A)
{
  expectKey(in, key);
  const unsigned int rows = readUint(in);
  const unsigned int cols = readUint(in);
  A.resize(rows, cols);
  for(unsigned int i = 0; i < rows; i++)
    for(unsigned int j = 0; j < cols; j++)
      A.setVal(readDouble(in), i, j);
}

int failNear(double a, double b, double tol, const string& what)
{
  if(fabs(a - b) > tol)
  {
    cout << "FAILURE: " << what << " a=" << a << " b=" << b
         << " |diff|=" << fabs(a - b) << endl;
    return 1;
  }
  return 0;
}

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

int testGpyFixture(const string& path)
{
  ifstream in(path.c_str());
  if(!in.is_open())
  {
    cout << "FAILURE: cannot open fixture " << path << endl;
    return 1;
  }

  expectKey(in, "N");
  const unsigned int n = readUint(in);
  expectKey(in, "D0");
  const unsigned int d0 = readUint(in);
  expectKey(in, "D1");
  const unsigned int d1 = readUint(in);
  expectKey(in, "Q");
  const unsigned int q = readUint(in);
  expectKey(in, "M");
  const unsigned int m = readUint(in);
  expectKey(in, "beta0");
  const double beta0 = readDouble(in);
  expectKey(in, "beta1");
  const double beta1 = readDouble(in);
  expectKey(in, "inverseWidth0");
  const double iw0 = readDouble(in);
  expectKey(in, "variance0");
  const double var0 = readDouble(in);
  expectKey(in, "inverseWidth1");
  const double iw1 = readDouble(in);
  expectKey(in, "variance1");
  const double var1 = readDouble(in);

  CMatrix Y0, Y1, mu, S, Z;
  readMatrix(in, "Y0", Y0);
  readMatrix(in, "Y1", Y1);
  readMatrix(in, "mu", mu);
  readMatrix(in, "S", S);
  readMatrix(in, "Z", Z);
  expectKey(in, "elbo");
  const double elboRef = readDouble(in);

  if(Y0.getRows() != n || Y0.getCols() != d0 || Y1.getRows() != n || Y1.getCols() != d1
     || mu.getRows() != n || mu.getCols() != q || S.getRows() != n || S.getCols() != q
     || Z.getRows() != m || Z.getCols() != q)
  {
    cout << "FAILURE: fixture dimension mismatch" << endl;
    return 1;
  }

  vector<CMatrix> Ys;
  Ys.push_back(Y0);
  Ys.push_back(Y1);
  CRbfKern* k0 = new CRbfKern(q);
  k0->setInitParam();
  k0->setInverseWidth(iw0);
  k0->setVariance(var0);
  CRbfKern* k1 = new CRbfKern(q);
  k1->setInitParam();
  k1->setInverseWidth(iw1);
  k1->setVariance(var1);
  vector<CKern*> ks;
  ks.push_back(k0);
  ks.push_back(k1);

  CMrd model(Ys, ks, q, m, 0);
  model.getView(0)->setBeta(beta0);
  model.getView(1)->setBeta(beta1);
  model.getMu().deepCopy(mu);
  model.getS().deepCopy(S);
  model.getZ().deepCopy(Z);

  const double elbo = model.logLikelihood();
  const int fail = failNear(elbo, elboRef, FIXTURE_TOL, "MRD ELBO vs GPy");
  if(fail == 0)
    cout << "OK: GPy fixture " << path << " (ELBO=" << elbo << ")" << endl;
  return fail;
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
    fail += testGpyFixture("testdata/cip0018/mrd_rbf_tiny.txt");
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
}
