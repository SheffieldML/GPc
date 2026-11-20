/* CIP-002D follow-up: batched inducingGradZ vs per-inducing getGradX oracle. */
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include "CKern.h"
#include "CInducing.h"
#include "ndlutil.h"

using namespace std;

namespace {

int fails = 0;

string str(double x)
{
  ostringstream os;
  os << x;
  return os.str();
}

void expect(bool ok, const string& msg)
{
  if(!ok)
  {
    cerr << "FAIL: " << msg << endl;
    ++fails;
  }
  else
    cout << "ok: " << msg << endl;
}

void expectNear(double a, double b, double tol, const string& msg)
{
  expect(fabs(a - b) <= tol, msg + " (got " + str(a) + " vs " + str(b) + ")");
}

void fillRand(CMatrix& A, double scale)
{
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      A.setVal(scale * (0.1 * (i + 1) - 0.07 * j + 0.02 * (i * j % 5)), i, j);
}

void symmetrise(CMatrix& A)
{
  DIMENSIONMATCH(A.isSquare());
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = 0.5 * (A.getVal(i, j) + A.getVal(j, i));
      A.setVal(v, i, j);
      A.setVal(v, j, i);
    }
  A.setSymmetric(true);
}

// Oracle matching the historical CInducingPoints M-loop (per-inducing getGradX).
void oracleInducingGradZ(CMatrix& gZ, const CKern& kern, const CMatrix& Z, const CMatrix& X,
                         const CMatrix& gKuu, const CMatrix& gKuf)
{
  const unsigned int M = Z.getRows();
  const unsigned int d = Z.getCols();
  const unsigned int N = X.getRows();
  gZ.resize(M, d);
  gZ.zeros();
  CMatrix dgKX(M, d);
  CMatrix gKX(M, d);
  CMatrix gKX_uf(N, d);
  kern.getDiagGradX(dgKX, Z);
  for(unsigned int i = 0; i < M; i++)
  {
    kern.getGradX(gKX_uf, Z, i, X);
    kern.getGradX(gKX, Z, i, Z);
    gKX.scale(2.0);
    gKX.copyRowRow(i, dgKX, i);
    for(unsigned int j = 0; j < d; j++)
      gZ.setVal(gKX.dotColCol(j, gKuu, i), i, j);
    for(unsigned int j = 0; j < d; j++)
      gZ.addVal(gKX_uf.dotColRow(j, gKuf, i), i, j);
  }
}

void compareGradZ(const CKern& kern, const string& label)
{
  const unsigned int M = 5;
  const unsigned int N = 11;
  const unsigned int Q = 3;
  CMatrix Z(M, Q);
  CMatrix X(N, Q);
  fillRand(Z, 1.0);
  fillRand(X, 0.8);
  CMatrix gKuu(M, M);
  CMatrix gKuf(M, N);
  fillRand(gKuu, 0.3);
  fillRand(gKuf, 0.2);
  symmetrise(gKuu);

  CMatrix gBulk(M, Q);
  CMatrix gOracle(M, Q);
  kern.inducingGradZ(gBulk, Z, X, gKuu, gKuf);
  oracleInducingGradZ(gOracle, kern, Z, X, gKuu, gKuf);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int q = 0; q < Q; q++)
      expectNear(gBulk.getVal(i, q), gOracle.getVal(i, q), 1e-9,
                 label + " gZ(" + str(i) + "," + str(q) + ")");
}

void testMlpBulkCompute()
{
  const unsigned int N = 7;
  const unsigned int M = 5;
  const unsigned int Q = 2;
  CMatrix X(N, Q);
  CMatrix X2(M, Q);
  fillRand(X, 1.0);
  fillRand(X2, 0.9);
  CMlpKern mlp(Q);
  mlp.setParam(2.5, 0);
  mlp.setParam(1.5, 1);
  mlp.setParam(0.8, 2);

  CMatrix Kbulk(N, M);
  CMatrix Koracle(N, M);
  mlp.compute(Kbulk, X, X2);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < M; j++)
      Koracle.setVal(mlp.computeElement(X, i, X2, j), i, j);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < M; j++)
      expectNear(Kbulk.getVal(i, j), Koracle.getVal(i, j), 1e-12,
                 "MLP rect compute vs element");

  CMatrix Ksym(N, N);
  CMatrix KsymEl(N, N);
  mlp.compute(Ksym, X);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < N; j++)
      KsymEl.setVal(mlp.computeElement(X, i, X, j), i, j);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < N; j++)
      expectNear(Ksym.getVal(i, j), KsymEl.getVal(i, j), 1e-12,
                 "MLP sym compute vs element");
}

void testInducingWire()
{
  const unsigned int M = 4;
  const unsigned int N = 9;
  const unsigned int Q = 2;
  CMatrix Z(M, Q);
  CMatrix X(N, Q);
  fillRand(Z, 1.0);
  fillRand(X, 0.7);
  CMatrix gKuu(M, M);
  CMatrix gKuf(M, N);
  fillRand(gKuu, 0.25);
  fillRand(gKuf, 0.15);
  symmetrise(gKuu);

  CRbfKern rbf(Q);
  rbf.setParam(1.3, 0);
  rbf.setParam(0.9, 1);
  CInducingPoints inducing(&Z);
  CMatrix gKern(1, rbf.getNumParams());
  gKern.zeros();
  CMatrix gZ(M, Q);
  inducing.gradFromKuuKuf(gKern, gZ, 0, rbf, X, gKuu, gKuf, false, false);

  CMatrix gZdirect(M, Q);
  rbf.inducingGradZ(gZdirect, Z, X, gKuu, gKuf);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int q = 0; q < Q; q++)
      expectNear(gZ.getVal(i, q), gZdirect.getVal(i, q), 1e-12,
                 "CInducingPoints wires inducingGradZ");
}

} // namespace

int main()
{
  try
  {
    {
      CRbfKern rbf(3);
      rbf.setParam(1.4, 0);
      rbf.setParam(1.1, 1);
      compareGradZ(rbf, "RBF");
    }
    {
      CLinKern lin(3);
      lin.setParam(0.7, 0);
      compareGradZ(lin, "Lin");
    }
    {
      CCmpndKern cmpnd(3);
      CRbfKern rbf(3);
      CBiasKern bias(3);
      CWhiteKern white(3);
      rbf.setParam(0.8, 0);
      rbf.setParam(1.2, 1);
      bias.setParam(0.2, 0);
      white.setParam(0.05, 0);
      cmpnd.addKern(&rbf);
      cmpnd.addKern(&bias);
      cmpnd.addKern(&white);
      compareGradZ(cmpnd, "Cmpnd");
    }
    testMlpBulkCompute();
    testInducingWire();
    cout << "Number of failures: " << fails << "." << endl;
    return fails > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
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
