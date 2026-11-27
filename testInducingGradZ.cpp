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
  expect(fabs(a - b) <= tol, msg + " (got " + str(a) + " vs " + str(b) + ", diff " + str(a - b) + ")");
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

// CIP-003F A2: batched inducingGradX equals the default per-datum getGradX loop,
// to 1e-12 relative, with and without a cached K_uf; and inducingGradZ for
// kernels that gained an override (MLP family) equals the per-inducing oracle.
void compareGradX(const CKern& kern, const string& label, bool useCache)
{
  const unsigned int M = 6;
  const unsigned int N = 9;
  const unsigned int Q = 3;
  CMatrix Z(M, Q);
  CMatrix X(N, Q);
  fillRand(Z, 1.0);
  fillRand(X, 0.6);
  CMatrix gKuf(M, N);
  fillRand(gKuf, 0.25);
  CMatrix gBulk(N, Q), gOracle(N, Q);
  gBulk.zeros();
  gOracle.zeros();
  CMatrix Kuf(M, N);
  kern.compute(Kuf, Z, X);
  kern.inducingGradX(gBulk, Z, X, gKuf, useCache ? &Kuf : 0);
  kern.CKern::inducingGradX(gOracle, Z, X, gKuf, 0);
  double scale = 1.0;
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int q = 0; q < Q; q++)
      scale = max(scale, fabs(gOracle.getVal(i, q)));
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int q = 0; q < Q; q++)
      expectNear(gBulk.getVal(i, q), gOracle.getVal(i, q), 1e-12 * scale,
                 label + (useCache ? " (cached Kuf)" : "") + " gX(" + str(i) + "," + str(q) + ")");
  // accumulates rather than overwrites
  kern.inducingGradX(gBulk, Z, X, gKuf, 0);
  expectNear(gBulk.getVal(1, 1), 2.0 * gOracle.getVal(1, 1), 2e-12 * scale,
             label + " gX accumulates");
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

// CIP-003F A3: bulk ridge windows vs the per-element reference.
void compareRidge(CKern& kern, const string& label)
{
  const unsigned int N = 17, M = 6, d = kern.getInputDim();
  const double sc = 0.7;
  CMatrix X(N, d), V(M, d + 1), a(M, 1), gKfu(N, M);
  fillRand(X, 1.0);
  fillRand(V, 1.0);
  fillRand(a, 1.0);
  fillRand(gKfu, 1.0);
  struct Out { CMatrix K, g, gX, gS, gV, ga; } o[2];
  for(int leg = 0; leg < 2; leg++)
  {
    CKern::setLegacyRidge(leg == 1);
    Out& r = o[leg];
    kern.wsRidgeCrossCompute(r.K, X, V, a, sc);
    r.g.resize(1, kern.getNumParams());
    kern.wsRidgeGradParams(r.g, X, V, a, sc, gKfu);
    kern.wsRidgeGradX(r.gX, X, V, a, sc, gKfu);
    kern.wsRidgeGradS(r.gS, X, V, a, sc, gKfu);
    kern.wsRidgeGradVA(r.gV, r.ga, X, V, a, sc, gKfu);
  }
  CKern::setLegacyRidge(false);
  auto cmp = [&](const CMatrix& A, const CMatrix& B, const string& what) {
    expect(A.getRows() == B.getRows() && A.getCols() == B.getCols(), label + " " + what + " shape");
    for(unsigned int i = 0; i < A.getRows(); i++)
      for(unsigned int j = 0; j < A.getCols(); j++)
        expectNear(A.getVal(i, j), B.getVal(i, j), 1e-12 * (1.0 + fabs(B.getVal(i, j))),
                   label + " ridge " + what);
  };
  cmp(o[0].K, o[1].K, "Kfu");
  cmp(o[0].g, o[1].g, "gParams");
  cmp(o[0].gX, o[1].gX, "gX");
  cmp(o[0].gS, o[1].gS, "gS");
  cmp(o[0].gV, o[1].gV, "gV");
  cmp(o[0].ga, o[1].ga, "ga");
}

int main()
{
  try
  {
    {
      CMlpKern mlp(3);
      mlp.setParam(2.5, 0);
      mlp.setParam(1.5, 1);
      mlp.setParam(0.8, 2);
      compareRidge(mlp, "MLP");
    }
    {
      CMlpardKern mlpard(3);
      mlpard.setParam(1.7, 0);
      mlpard.setParam(0.6, 1);
      mlpard.setParam(1.3, 2);
      mlpard.setParam(0.3, 3);
      mlpard.setParam(1.1, 4);
      mlpard.setParam(0.7, 5);
      compareRidge(mlpard, "MLP-ARD");
    }
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
    {
      CMlpKern mlp(3);
      mlp.setParam(2.5, 0);
      mlp.setParam(1.5, 1);
      mlp.setParam(0.8, 2);
      compareGradZ(mlp, "MLP");
      compareGradX(mlp, "MLP", false);
    }
    {
      CMlpardKern mlpard(3);
      mlpard.setParam(1.7, 0);
      mlpard.setParam(0.6, 1);
      mlpard.setParam(1.3, 2);
      mlpard.setParam(0.3, 3);
      mlpard.setParam(1.1, 4);
      mlpard.setParam(0.7, 5);
      compareGradZ(mlpard, "MLP-ARD");
      compareGradX(mlpard, "MLP-ARD", false);
    }
    {
      CRbfKern rbf(3);
      rbf.setParam(1.4, 0);
      rbf.setParam(1.1, 1);
      compareGradX(rbf, "RBF", false);
      compareGradX(rbf, "RBF", true);
    }
    {
      CRbfardKern rbfard(3);
      rbfard.setParam(0.9, 0);
      rbfard.setParam(1.2, 1);
      rbfard.setParam(0.4, 2);
      rbfard.setParam(1.5, 3);
      rbfard.setParam(0.8, 4);
      compareGradX(rbfard, "RBF-ARD", false);
      compareGradX(rbfard, "RBF-ARD", true);
    }
    {
      CLinKern lin(3);
      lin.setParam(0.7, 0);
      compareGradX(lin, "Lin", false);
    }
    {
      CCmpndKern cmpnd(3);
      CRbfKern rbf(3);
      CMlpKern mlp(3);
      CLinKern lin(3);
      CBiasKern bias(3);
      CWhiteKern white(3);
      rbf.setParam(0.8, 0);
      rbf.setParam(1.2, 1);
      mlp.setParam(2.0, 0);
      lin.setParam(0.5, 0);
      cmpnd.addKern(&rbf);
      cmpnd.addKern(&mlp);
      cmpnd.addKern(&lin);
      cmpnd.addKern(&bias);
      cmpnd.addKern(&white);
      compareGradX(cmpnd, "Cmpnd(rbf+mlp+lin+bias+white)", false);
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
