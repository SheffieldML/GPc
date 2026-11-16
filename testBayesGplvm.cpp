/* CIP-0017: Bayesian GP-LVM ELBO, checkGradients, GPy fixture compare. */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "CBayesGplvm.h"
#include "ndlutil.h"

using namespace std;

namespace {
const unsigned int N = 6;
const unsigned int D = 3;
const unsigned int Q = 2;
const unsigned int M = 3;

// Looser than MATCHTOL: GPy vs C++ Cholesky / BLAS stack differences.
const double FIXTURE_TOL = 1e-6;

CMatrix makeY()
{
  CMatrix Y(N, D);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int d = 0; d < D; d++)
      Y.setVal(sin(0.3 * n + 0.7 * d) + 0.1 * n, n, d);
  return Y;
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

int failNearMat(const CMatrix& A, const CMatrix& B, double tol, const string& what)
{
  if(A.getRows() != B.getRows() || A.getCols() != B.getCols())
  {
    cout << "FAILURE: " << what << " shape mismatch" << endl;
    return 1;
  }
  double maxDiff = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
    {
      const double d = fabs(A.getVal(i, j) - B.getVal(i, j));
      if(d > maxDiff)
        maxDiff = d;
    }
  if(maxDiff > tol)
  {
    cout << "FAILURE: " << what << " maxAbsDiff=" << maxDiff << endl;
    return 1;
  }
  return 0;
}

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

void readVector(istream& in, const string& key, vector<double>& v)
{
  expectKey(in, key);
  const unsigned int n = readUint(in);
  v.resize(n);
  for(unsigned int i = 0; i < n; i++)
    v[i] = readDouble(in);
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
  expectKey(in, "D");
  const unsigned int d = readUint(in);
  expectKey(in, "Q");
  const unsigned int q = readUint(in);
  expectKey(in, "M");
  const unsigned int m = readUint(in);
  expectKey(in, "beta");
  const double beta = readDouble(in);
  expectKey(in, "inverseWidth");
  const double iw = readDouble(in);
  expectKey(in, "variance");
  const double var = readDouble(in);

  CMatrix Y, mu, S, Z, psi1Ref, psi2Ref;
  vector<double> psi0Ref, gNatRef;
  readMatrix(in, "Y", Y);
  readMatrix(in, "mu", mu);
  readMatrix(in, "S", S);
  readMatrix(in, "Z", Z);
  readVector(in, "psi0", psi0Ref);
  readMatrix(in, "psi1", psi1Ref);
  readMatrix(in, "psi2", psi2Ref);
  expectKey(in, "elbo");
  const double elboRef = readDouble(in);
  readVector(in, "g_natural", gNatRef);

  if(Y.getRows() != n || Y.getCols() != d || mu.getRows() != n || mu.getCols() != q
     || S.getRows() != n || S.getCols() != q || Z.getRows() != m || Z.getCols() != q)
  {
    cout << "FAILURE: fixture dimension mismatch" << endl;
    return 1;
  }

  CRbfKern kern(q);
  kern.setInitParam();
  kern.setInverseWidth(iw);
  kern.setVariance(var);

  CMatrix Psi1(n, m);
  CMatrix Psi2(m, m);
  const double psi0Sum = kern.psi0(mu, S);
  kern.psi1(Psi1, mu, S, Z);
  kern.psi2(Psi2, mu, S, Z);

  double psi0RefSum = 0.0;
  for(size_t i = 0; i < psi0Ref.size(); i++)
    psi0RefSum += psi0Ref[i];

  int fail = 0;
  fail += failNear(psi0Sum, psi0RefSum, FIXTURE_TOL, "psi0_sum vs GPy");
  fail += failNearMat(Psi1, psi1Ref, FIXTURE_TOL, "psi1 vs GPy");
  fail += failNearMat(Psi2, psi2Ref, FIXTURE_TOL, "psi2 vs GPy");

  CBayesGplvm model(&kern, Y, q, m, 0);
  model.setBeta(beta);
  model.getMu().deepCopy(mu);
  model.getS().deepCopy(S);
  model.getZ().deepCopy(Z);
  model.setKupToDate(false);

  const double elbo = model.logLikelihood();
  fail += failNear(elbo, elboRef, FIXTURE_TOL, "ELBO vs GPy");

  CMatrix gUnc(1, model.getOptNumParams());
  model.logLikelihoodGradient(gUnc);
  // Convert unconstrained grads back to natural (exp transform: gradfact = x).
  vector<double> gNat(model.getOptNumParams());
  unsigned int c = 0;
  gNat[c] = gUnc.getVal(c) / kern.getInverseWidth();
  c++;
  gNat[c] = gUnc.getVal(c) / kern.getVariance();
  c++;
  gNat[c] = gUnc.getVal(c) / model.getBeta();
  c++;
  for(; c < 3 + m * q + n * q; c++)
    gNat[c] = gUnc.getVal(c); // Z, mu
  for(unsigned int i = 0; i < n * q; i++, c++)
  {
    const unsigned int nn = i / q;
    const unsigned int qq = i % q;
    gNat[c] = gUnc.getVal(c) / model.getS().getVal(nn, qq);
  }

  if(gNatRef.size() != gNat.size())
  {
    cout << "FAILURE: g_natural length " << gNat.size()
         << " vs fixture " << gNatRef.size() << endl;
    return fail + 1;
  }
  double maxG = 0.0;
  unsigned int worst = 0;
  for(size_t i = 0; i < gNat.size(); i++)
  {
    const double diff = fabs(gNat[i] - gNatRef[i]);
    if(diff > maxG)
    {
      maxG = diff;
      worst = (unsigned int)i;
    }
  }
  if(maxG > FIXTURE_TOL)
  {
    cout << "FAILURE: g_natural vs GPy maxAbs=" << maxG
         << " at index " << worst
         << " (cpp=" << gNat[worst] << " gpy=" << gNatRef[worst] << ")" << endl;
    fail++;
  }

  if(fail == 0)
    cout << "OK: GPy fixture " << path
         << " (ELBO=" << elbo << " maxGradDiff=" << maxG << ")" << endl;
  return fail;
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
    fail += testGpyFixture("testdata/cip0017/bayes_gplvm_rbf_tiny.txt");
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
}
