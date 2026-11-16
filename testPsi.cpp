/* CIP-0017 Phase 1: psi statistics — Monte Carlo and finite-difference gates. */
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "CKern.h"
#include "ndlutil.h"
#include "ndlstrutil.h"

using namespace std;

namespace {
const unsigned int N = 4;
const unsigned int M = 3;
const unsigned int Q = 2;
const unsigned int MC = 8000;
const double MCTOL = 0.08;
const double FDTOL = 5e-4;
const double EPS = 1e-6;

void fillQS(CMatrix& mu, CMatrix& S, CMatrix& Z)
{
  mu.resize(N, Q);
  S.resize(N, Q);
  Z.resize(M, Q);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
    {
      mu.setVal(0.3 * (n + 1) - 0.5 * q, n, q);
      S.setVal(0.2 + 0.05 * n + 0.03 * q, n, q);
    }
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int q = 0; q < Q; q++)
      Z.setVal(-0.4 + 0.5 * m + 0.1 * q, m, q);
}

double mcPsi0(const CKern& kern, const CMatrix& mu, const CMatrix& S)
{
  double acc = 0.0;
  CMatrix x(1, Q);
  for(unsigned int n = 0; n < N; n++)
  {
    double s = 0.0;
    for(unsigned int t = 0; t < MC; t++)
    {
      for(unsigned int q = 0; q < Q; q++)
        x.setVal(mu.getVal(n, q) + sqrt(S.getVal(n, q)) * ndlutil::randn(), q);
      s += kern.diagComputeElement(x, 0);
    }
    acc += s / MC;
  }
  return acc;
}

void mcPsi1(CMatrix& out, const CKern& kern, const CMatrix& mu, const CMatrix& S, const CMatrix& Z)
{
  out.resize(N, M);
  out.zeros();
  CMatrix x(1, Q);
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int t = 0; t < MC; t++)
    {
      for(unsigned int q = 0; q < Q; q++)
        x.setVal(mu.getVal(n, q) + sqrt(S.getVal(n, q)) * ndlutil::randn(), q);
      for(unsigned int m = 0; m < M; m++)
        out.addVal(kern.computeElement(x, 0, Z, m), n, m);
    }
  }
  out.scale(1.0 / MC);
}

void mcPsi2(CMatrix& out, const CKern& kern, const CMatrix& mu, const CMatrix& S, const CMatrix& Z)
{
  out.resize(M, M);
  out.zeros();
  CMatrix x(1, Q);
  for(unsigned int n = 0; n < N; n++)
  {
    for(unsigned int t = 0; t < MC; t++)
    {
      for(unsigned int q = 0; q < Q; q++)
        x.setVal(mu.getVal(n, q) + sqrt(S.getVal(n, q)) * ndlutil::randn(), q);
      for(unsigned int m1 = 0; m1 < M; m1++)
      {
        const double k1 = kern.computeElement(x, 0, Z, m1);
        for(unsigned int m2 = 0; m2 < M; m2++)
          out.addVal(k1 * kern.computeElement(x, 0, Z, m2), m1, m2);
      }
    }
  }
  out.scale(1.0 / MC);
}

int failNear(double got, double want, double tol, const string& label)
{
  if(fabs(got - want) > tol)
  {
    cout << "FAILURE: " << label << " got=" << got << " want=" << want
         << " diff=" << fabs(got - want) << endl;
    return 1;
  }
  cout << "OK: " << label << endl;
  return 0;
}

int failNearMat(const CMatrix& got, const CMatrix& want, double tol, const string& label)
{
  const double d = got.maxAbsDiff(want);
  if(d > tol)
  {
    cout << "FAILURE: " << label << " maxAbsDiff=" << d << endl;
    return 1;
  }
  cout << "OK: " << label << endl;
  return 0;
}

int testForward(CKern& kern, const string& name)
{
  CMatrix mu, S, Z;
  fillQS(mu, S, Z);
  int fail = 0;
  const double p0 = kern.psi0(mu, S);
  fail += failNear(p0, mcPsi0(kern, mu, S), MCTOL * (1.0 + fabs(p0)), name + " psi0 vs MC");

  CMatrix P1, P1mc;
  kern.psi1(P1, mu, S, Z);
  mcPsi1(P1mc, kern, mu, S, Z);
  fail += failNearMat(P1, P1mc, MCTOL * (1.0 + P1.max()), name + " psi1 vs MC");

  CMatrix P2, P2mc;
  kern.psi2(P2, mu, S, Z);
  mcPsi2(P2mc, kern, mu, S, Z);
  fail += failNearMat(P2, P2mc, MCTOL * (1.0 + P2.max()), name + " psi2 vs MC");
  return fail;
}

double objective(CKern& kern, const CMatrix& d0, const CMatrix& d1, const CMatrix& d2,
                 const CMatrix& mu, const CMatrix& S, const CMatrix& Z)
{
  // L = sum_n d0_n * psi0_n + <d1, psi1> + <d2, psi2>, with psi0_n via N=1 slices.
  CMatrix P1, P2;
  kern.psi1(P1, mu, S, Z);
  kern.psi2(P2, mu, S, Z);
  double L = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    CMatrix mu1(1, Q), S1(1, Q);
    for(unsigned int q = 0; q < Q; q++)
    {
      mu1.setVal(mu.getVal(n, q), 0, q);
      S1.setVal(S.getVal(n, q), 0, q);
    }
    L += d0.getVal(n) * kern.psi0(mu1, S1);
  }
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      L += d1.getVal(n, m) * P1.getVal(n, m);
  for(unsigned int m1 = 0; m1 < M; m1++)
    for(unsigned int m2 = 0; m2 < M; m2++)
      L += d2.getVal(m1, m2) * P2.getVal(m1, m2);
  return L;
}

int testGradParams(CKern& kern, const string& name)
{
  CMatrix mu, S, Z;
  fillQS(mu, S, Z);
  CMatrix d0(N, 1);
  d0.setVals(1.0);
  CMatrix d1(N, M);
  d1.setVals(0.5);
  CMatrix d2(M, M);
  d2.setVals(0.25);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < i; j++)
      d2.setVal(d2.getVal(j, i), i, j);

  CMatrix gP, gZ, gMu, gS;
  kern.getGradPsi(gP, gZ, gMu, gS, d0, d1, d2, mu, S, Z, false);

  int fail = 0;
  CMatrix params(1, kern.getNumParams());
  kern.getParams(params);
  for(unsigned int p = 0; p < kern.getNumParams(); p++)
  {
    const double base = params.getVal(p);
    params.setVal(base + EPS, p);
    kern.setParams(params);
    const double Lp = objective(kern, d0, d1, d2, mu, S, Z);
    params.setVal(base - EPS, p);
    kern.setParams(params);
    const double Lm = objective(kern, d0, d1, d2, mu, S, Z);
    params.setVal(base, p);
    kern.setParams(params);
    const double fd = (Lp - Lm) / (2.0 * EPS);
    fail += failNear(gP.getVal(p), fd, FDTOL * (1.0 + fabs(fd)), name + " dL/dparam" + ndlstrutil::itoa(p));
  }

  // Spot-check a few mu / S / Z entries
  for(unsigned int q = 0; q < Q; q++)
  {
    const double base = mu.getVal(0, q);
    mu.setVal(base + EPS, 0, q);
    const double Lp = objective(kern, d0, d1, d2, mu, S, Z);
    mu.setVal(base - EPS, 0, q);
    const double Lm = objective(kern, d0, d1, d2, mu, S, Z);
    mu.setVal(base, 0, q);
    const double fd = (Lp - Lm) / (2.0 * EPS);
    fail += failNear(gMu.getVal(0, q), fd, FDTOL * (1.0 + fabs(fd)), name + " dL/dmu0" + ndlstrutil::itoa(q));
  }
  for(unsigned int q = 0; q < Q; q++)
  {
    const double base = S.getVal(1, q);
    S.setVal(base + EPS, 1, q);
    const double Lp = objective(kern, d0, d1, d2, mu, S, Z);
    S.setVal(base - EPS, 1, q);
    const double Lm = objective(kern, d0, d1, d2, mu, S, Z);
    S.setVal(base, 1, q);
    const double fd = (Lp - Lm) / (2.0 * EPS);
    fail += failNear(gS.getVal(1, q), fd, FDTOL * (1.0 + fabs(fd)), name + " dL/dS1" + ndlstrutil::itoa(q));
  }
  {
    const double base = Z.getVal(0, 0);
    Z.setVal(base + EPS, 0, 0);
    const double Lp = objective(kern, d0, d1, d2, mu, S, Z);
    Z.setVal(base - EPS, 0, 0);
    const double Lm = objective(kern, d0, d1, d2, mu, S, Z);
    Z.setVal(base, 0, 0);
    const double fd = (Lp - Lm) / (2.0 * EPS);
    fail += failNear(gZ.getVal(0, 0), fd, FDTOL * (1.0 + fabs(fd)), name + " dL/dZ00");
  }
  return fail;
}

int testDefaultThrows()
{
  CMatern32Kern kern(Q);
  kern.setInitParam();
  CMatrix mu, S, Z;
  fillQS(mu, S, Z);
  try
  {
    kern.psi0(mu, S);
    cout << "FAILURE: expected NotImplementedError for matern32 psi0" << endl;
    return 1;
  }
  catch(ndlexceptions::NotImplementedError&)
  {
    cout << "OK: matern32 psi0 throws" << endl;
    return 0;
  }
}

} // namespace

int main()
{
  int fail = 0;
  ndlutil::init_genrand(42UL);
  try
  {
    fail += testDefaultThrows();

    {
      CRbfKern kern(Q);
      kern.setInitParam();
      kern.setInverseWidth(1.5);
      kern.setVariance(0.8);
      fail += testForward(kern, "rbf");
      fail += testGradParams(kern, "rbf");
    }
    {
      CRbfardKern kern(Q);
      kern.setInitParam();
      kern.setParam(1.2, 0);
      kern.setParam(0.9, 1);
      kern.setParam(0.6, 2);
      kern.setParam(0.4, 3);
      fail += testForward(kern, "rbfard");
      fail += testGradParams(kern, "rbfard");
    }
    {
      CLinKern kern(Q);
      kern.setInitParam();
      kern.setVariance(0.7);
      fail += testForward(kern, "lin");
      fail += testGradParams(kern, "lin");
    }
    {
      CBiasKern kern(Q);
      kern.setInitParam();
      kern.setVariance(0.5);
      fail += testForward(kern, "bias");
      fail += testGradParams(kern, "bias");
    }
    {
      CWhiteKern kern(Q);
      kern.setInitParam();
      kern.setVariance(0.3);
      fail += testForward(kern, "white");
      fail += testGradParams(kern, "white");
    }
    {
      CCmpndKern kern(Q);
      kern.setInitParam();
      CRbfKern* rbf = new CRbfKern(Q);
      rbf->setInitParam();
      rbf->setInverseWidth(1.0);
      rbf->setVariance(0.6);
      CBiasKern* bias = new CBiasKern(Q);
      bias->setInitParam();
      bias->setVariance(0.2);
      CWhiteKern* white = new CWhiteKern(Q);
      white->setInitParam();
      white->setVariance(0.1);
      kern.addKern(rbf);
      kern.addKern(bias);
      kern.addKern(white);
      delete rbf;
      delete bias;
      delete white;
      fail += testForward(kern, "cmpnd");
      fail += testGradParams(kern, "cmpnd");
    }

    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
  catch(std::exception&)
  {
    cerr << "Unhandled exception." << endl;
    return 1;
  }
}
