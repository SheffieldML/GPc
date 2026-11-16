/* CIP-0012: unit tests for upper Cholesky / trsm adjoints. */
#include "CCholAdjoint.h"
#include "CMatrix.h"
#include <cmath>
#include <iostream>

using namespace std;

static int fails = 0;

static void expect(bool ok, const string& what)
{
  if(ok)
    cout << "OK: " << what << endl;
  else
  {
    cout << "FAIL: " << what << endl;
    fails++;
  }
}

static CMatrix makeSpd(unsigned int n, double condScale)
{
  CMatrix A(n, n);
  A.eye();
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = 0.1 / (1.0 + (double)(i - j));
      A.setVal(v, i, j);
      A.setVal(v, j, i);
    }
  A.setVal(condScale, 0, 0);
  A.setSymmetric(true);
  return A;
}

static double maxAbsDiffSym(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j <= i; j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

static void testUnblockedMatchesFD()
{
  const unsigned int n = 5;
  CMatrix K = makeSpd(n, 3.0);
  CMatrix U(n, n);
  U.jitChol(K); // upper
  U.setTriangular(true);

  CMatrix Ubar(n, n);
  Ubar.zeros();
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = i; j < n; j++)
      Ubar.setVal(0.01 * (1.0 + (double)(i + 2 * j)), i, j);
  Ubar.setTriangular(true);

  CMatrix Kbar;
  cholAdjointUpperUnblocked(Kbar, U, Ubar);

  // Central difference on K via U=chol(K): <Ubar, dU> ≈ <Kbar, dK>
  const double h = 1e-6;
  double lhs = 0.0;
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = i; j < n; j++)
      lhs += Ubar.getVal(i, j) * 0.0; // filled below

  // Direction dK symmetric
  CMatrix dK(n, n);
  dK.zeros();
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j <= i; j++)
    {
      const double v = 0.02 * (1.0 + (double)(3 * i + j));
      dK.setVal(v, i, j);
      dK.setVal(v, j, i);
    }
  dK.setSymmetric(true);

  CMatrix Kp(K), Km(K);
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j < n; j++)
    {
      Kp.setVal(K.getVal(i, j) + h * dK.getVal(i, j), i, j);
      Km.setVal(K.getVal(i, j) - h * dK.getVal(i, j), i, j);
    }
  Kp.setSymmetric(true);
  Km.setSymmetric(true);
  CMatrix Up(n, n), Um(n, n);
  Up.jitChol(Kp);
  Um.jitChol(Km);
  // Pairing on the upper triangle of U (storage convention) and the lower
  // triangle of Kbar as returned by the Murray/GPy tril VJP (symmetrised copy).
  double dObj = 0.0;
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = i; j < n; j++)
      dObj += Ubar.getVal(i, j) * (Up.getVal(i, j) - Um.getVal(i, j)) / (2.0 * h);

  double pred = 0.0;
  for(unsigned int i = 0; i < n; i++)
  {
    pred += Kbar.getVal(i, i) * dK.getVal(i, i);
    for(unsigned int j = 0; j < i; j++)
      pred += Kbar.getVal(i, j) * dK.getVal(i, j);
  }

  expect(fabs(dObj - pred) < 5e-4 * max(1.0, fabs(dObj)),
         "unblocked chol adjoint matches FD (tril pairing)");
  cout << "  FD=" << dObj << " pred=" << pred << " |diff|=" << fabs(dObj - pred) << endl;
}

static void testBlockedMatchesUnblocked()
{
  const unsigned int n = 17;
  CMatrix K = makeSpd(n, 5.0);
  CMatrix U(n, n);
  U.jitChol(K);
  U.setTriangular(true);
  CMatrix Ubar(n, n);
  Ubar.zeros();
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = i; j < n; j++)
      Ubar.setVal(0.001 * (double)(i + j + 1), i, j);
  Ubar.setTriangular(true);

  CMatrix Kb1, Kb2;
  cholAdjointUpperUnblocked(Kb1, U, Ubar);
  cholAdjointUpperBlocked(Kb2, U, Ubar, 4);
  const double diff = maxAbsDiffSym(Kb1, Kb2);
  expect(diff < 1e-8, "blocked matches unblocked (block=4)");
  cout << "  max|Δ|=" << diff << endl;
}

int main()
{
  cout << "==== testCholAdjoint (CIP-0012) ====" << endl;
  testUnblockedMatchesFD();
  testBlockedMatchesUnblocked();
  cout << "Number of failures: " << fails << "." << endl;
  return fails ? 1 : 0;
}
