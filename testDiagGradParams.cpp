/* CIP-002D follow-up: bulk getDiagGradParams — regularise-once + O(1)-in-N allocs. */
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include "CKern.h"
#include "CDist.h"
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

string str(unsigned int x)
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

void fillX(CMatrix& X, unsigned int N, unsigned int Q)
{
  X.resize(N, Q);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int q = 0; q < Q; q++)
      X.setVal(0.1 * (i + 1) - 0.07 * q, i, q);
}

void fillW(CMatrix& w, unsigned int N, double scale = 1.0)
{
  w.resize(N, 1);
  for(unsigned int i = 0; i < N; i++)
    w.setVal(scale * (0.5 + 0.01 * i), i);
}

// Finite-difference reference for diagonal parameter gradients.
void fdDiagGrad(CMatrix& g, CKern& kern, const CMatrix& X, const CMatrix& w, double eps = 1e-6)
{
  const unsigned int np = kern.getNumParams();
  g.resize(1, np);
  g.zeros();
  CMatrix dplus(X.getRows(), 1);
  CMatrix dminus(X.getRows(), 1);
  for(unsigned int p = 0; p < np; p++)
  {
    const double theta = kern.getParam(p);
    kern.setParam(theta + eps, p);
    kern.diagCompute(dplus, X);
    kern.setParam(theta - eps, p);
    kern.diagCompute(dminus, X);
    kern.setParam(theta, p);
    double acc = 0.0;
    for(unsigned int i = 0; i < X.getRows(); i++)
      acc += w.getVal(i) * (dplus.getVal(i) - dminus.getVal(i)) / (2.0 * eps);
    g.setVal(acc, p);
  }
}

void testStationaryClosedForm()
{
  const unsigned int N = 17;
  const unsigned int Q = 3;
  CMatrix X;
  CMatrix w;
  fillX(X, N, Q);
  fillW(w, N);

  CRbfKern rbf(Q);
  rbf.setParam(1.7, 0); // inverseWidth
  rbf.setParam(2.3, 1); // variance
  CMatrix g(1, rbf.getNumParams());
  CMatrix gfd(1, rbf.getNumParams());
  rbf.getDiagGradParams(g, X, w, false);
  fdDiagGrad(gfd, rbf, X, w);
  expectNear(g.getVal(0), 0.0, 1e-12, "RBF inverseWidth diag grad is 0");
  expectNear(g.getVal(1), gfd.getVal(1), 1e-6, "RBF variance diag grad vs FD");
  expectNear(g.getVal(0), gfd.getVal(0), 1e-6, "RBF inverseWidth diag grad vs FD");

  double sumW = 0.0;
  for(unsigned int i = 0; i < N; i++)
    sumW += w.getVal(i);
  expectNear(g.getVal(1), sumW, 1e-12, "RBF variance diag grad equals sum(w)");

  CWhiteKern white(Q);
  white.setParam(0.4, 0);
  CMatrix gw(1, 1);
  white.getDiagGradParams(gw, X, w, false);
  expectNear(gw.getVal(0), sumW, 1e-12, "White variance diag grad equals sum(w)");

  CLinKern lin(Q);
  lin.setParam(1.5, 0);
  CMatrix gl(1, 1);
  CMatrix glfd(1, 1);
  lin.getDiagGradParams(gl, X, w, false);
  fdDiagGrad(glfd, lin, X, w);
  expectNear(gl.getVal(0), glfd.getVal(0), 1e-5, "Lin variance diag grad vs FD");
}

void testCompoundVsFd()
{
  const unsigned int N = 11;
  const unsigned int Q = 2;
  CMatrix X;
  CMatrix w;
  fillX(X, N, Q);
  fillW(w, N, 0.8);

  CCmpndKern cmpnd(Q);
  CRbfKern rbf(Q);
  CBiasKern bias(Q);
  CWhiteKern white(Q);
  rbf.setParam(0.9, 0);
  rbf.setParam(1.2, 1);
  bias.setParam(0.3, 0);
  white.setParam(0.05, 0);
  cmpnd.addKern(&rbf);
  cmpnd.addKern(&bias);
  cmpnd.addKern(&white);

  CMatrix g(1, cmpnd.getNumParams());
  CMatrix gfd(1, cmpnd.getNumParams());
  cmpnd.getDiagGradParams(g, X, w, false);
  fdDiagGrad(gfd, cmpnd, X, w);
  for(unsigned int p = 0; p < cmpnd.getNumParams(); p++)
    expectNear(g.getVal(p), gfd.getVal(p), 1e-5,
               string("cmpnd diag grad param ") + str(p));
}

void testTensorVsFd()
{
  const unsigned int N = 9;
  const unsigned int Q = 2;
  CMatrix X;
  CMatrix w;
  fillX(X, N, Q);
  fillW(w, N);

  CTensorKern tensor(Q);
  CRbfKern rbf(Q);
  CBiasKern bias(Q);
  rbf.setParam(1.1, 0);
  rbf.setParam(0.8, 1);
  bias.setParam(1.4, 0);
  tensor.addKern(&rbf);
  tensor.addKern(&bias);

  CMatrix g(1, tensor.getNumParams());
  CMatrix gfd(1, tensor.getNumParams());
  tensor.getDiagGradParams(g, X, w, false);
  fdDiagGrad(gfd, tensor, X, w);
  for(unsigned int p = 0; p < tensor.getNumParams(); p++)
    expectNear(g.getVal(p), gfd.getVal(p), 1e-5,
               string("tensor diag grad param ") + str(p));
}

void testRegulariseOnce()
{
  const unsigned int Q = 2;
  CRbfKern rbf(Q);
  rbf.setParam(1.0, 0);
  rbf.setParam(2.0, 1);
  CGaussianDist* prior = new CGaussianDist();
  prior->setParam(1.0, 0); // precision
  rbf.addPrior(prior, 1);  // on variance

  CMatrix priorOnly(1, 2);
  priorOnly.zeros();
  rbf.addPriorGrad(priorOnly);

  for(unsigned int N : {1u, 7u, 50u})
  {
    CMatrix X;
    CMatrix w;
    fillX(X, N, Q);
    fillW(w, N);
    CMatrix greg(1, 2);
    CMatrix gnoreg(1, 2);
    rbf.getDiagGradParams(greg, X, w, true);
    rbf.getDiagGradParams(gnoreg, X, w, false);
    for(unsigned int p = 0; p < 2; p++)
    {
      const double delta = greg.getVal(p) - gnoreg.getVal(p);
      expectNear(delta, priorOnly.getVal(p), 1e-12,
                 string("regularise once N=") + str(N)
                 + " param " + str(p));
    }
  }
}

void testAllocO1InN()
{
  const unsigned int Q = 2;
  CCmpndKern cmpnd(Q);
  CRbfKern rbf(Q);
  CBiasKern bias(Q);
  CWhiteKern white(Q);
  cmpnd.addKern(&rbf);
  cmpnd.addKern(&bias);
  cmpnd.addKern(&white);

  // Warm-up so first-call workspace / name tables are not charged.
  {
    CMatrix X;
    CMatrix w;
    fillX(X, 8, Q);
    fillW(w, 8);
    CMatrix g(1, cmpnd.getNumParams());
    cmpnd.getDiagGradParams(g, X, w, false);
  }

  unsigned int allocSmall = 0;
  unsigned int allocLarge = 0;
  {
    const unsigned int N = 128;
    CMatrix X;
    CMatrix w;
    fillX(X, N, Q);
    fillW(w, N);
    CMatrix g(1, cmpnd.getNumParams());
    CMatrix::resetTotalAlloc();
    cmpnd.getDiagGradParams(g, X, w, false);
    allocSmall = CMatrix::getTotalAlloc();
  }
  {
    const unsigned int N = 2048;
    CMatrix X;
    CMatrix w;
    fillX(X, N, Q);
    fillW(w, N);
    CMatrix g(1, cmpnd.getNumParams());
    CMatrix::resetTotalAlloc();
    cmpnd.getDiagGradParams(g, X, w, false);
    allocLarge = CMatrix::getTotalAlloc();
  }

  cout << "alloc N=128: " << allocSmall << ", N=2048: " << allocLarge << endl;
  // Pre-fix path was ~2N (subg temps per point through compound). Bulk must
  // stay flat in N — allow a small constant slack for any incidental temps.
  expect(allocLarge <= allocSmall + 4,
         "diag grad allocs O(1) in N (large within +4 of small)");
  expect(allocLarge < 32, "diag grad allocs stay small absolute constant");
}

} // namespace

int main()
{
  try
  {
    testStationaryClosedForm();
    testCompoundVsFd();
    testTensorVsFd();
    testRegulariseOnce();
    testAllocO1InN();
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
