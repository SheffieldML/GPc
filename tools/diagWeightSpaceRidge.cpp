/* CIP-0028 ridge diagnostics
 *
 * 1) Free vs fixed centres: analytic grads tighten the DTCVAR bound.
 * 2) M-sweep: more free ridges → LL* / RMSE approach FTC exact.
 *    Window width scales as s ∝ 1/√M so Kuu overlap stays comparable.
 *
 *   make diagWeightSpaceRidge && ./diagWeightSpaceRidge
 */
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "CGp.h"
#include "CInducing.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlutil.h"

using namespace std;

namespace {

double rmse(CGp& m, const CMatrix& X, const CMatrix& y)
{
  CMatrix mu(X.getRows(), 1);
  m.out(mu, X);
  double sse = 0.0;
  for(unsigned int i = 0; i < X.getRows(); i++)
  {
    const double e = mu.getVal(i, 0) - y.getVal(i, 0);
    sse += e * e;
  }
  return sqrt(sse / (double)X.getRows());
}

// Condition number via Cholesky diagonal extremes (SPD Kuu).
double kuuKappaChol(const CInducing& ind)
{
  const unsigned int M = ind.numInducing();
  CMatrix Kuu(M, M);
  CMlpKern dummy(1);
  ind.computeKuu(Kuu, dummy);
  for(unsigned int i = 0; i < M; i++)
    Kuu.addVal(1e-10, i, i);
  Kuu.setSymmetric(true);
  try
  {
    Kuu.chol(); // upper triangular
  }
  catch(...)
  {
    return 1e16;
  }
  double dmin = fabs(Kuu.getVal(0, 0));
  double dmax = dmin;
  for(unsigned int i = 1; i < M; i++)
  {
    const double d = fabs(Kuu.getVal(i, i));
    if(d < dmin) dmin = d;
    if(d > dmax) dmax = d;
  }
  // For Chol U, diag(U)^2 track eigenvalues order-of-magnitude; report (dmax/dmin)^2.
  const double ratio = dmax / max(dmin, 1e-18);
  return ratio * ratio;
}

struct RunResult
{
  double ll0;
  double llStar;
  double teRmse;
  double kappa0;
};

void fillXu(CMatrix& Xu, const CMatrix& X)
{
  const unsigned int M = Xu.getRows();
  const unsigned int N = X.getRows();
  for(unsigned int i = 0; i < M; i++)
  {
    const unsigned int idx = (M == 1) ? 0 : (i * (N - 1)) / (M - 1);
    Xu.setVal(X.getVal(idx, 0), i, 0);
  }
}

RunResult runOne(CMatrix& X, CMatrix& y, CMatrix& Xu,
                 const CMatrix& Xte, const CMatrix& yte,
                 int approx, const string& ind, bool freeInd,
                 double windowWidth, unsigned int iters)
{
  CCmpndKern kern(X);
  kern.addKern(new CMlpKern(X));
  kern.addKern(new CWhiteKern(X));
  CGaussianNoise noise(&y);
  noise.setBiasVal(0.0, 0);
  const unsigned int M = Xu.getRows();
  CGp m(&kern, &noise, &X, approx, (approx == CGp::FTC) ? 0 : M, 0);
  RunResult r;
  r.kappa0 = 0.0;
  if(approx != CGp::FTC)
  {
    m.setInducingVals(Xu);
    m.setInducingFixed(!freeInd);
    m.setBetaVal(20.0);
    m.setInducingType(ind);
    if(ind == "weightspace")
    {
      m.getInducing()->setWindowWidth(windowWidth);
      r.kappa0 = kuuKappaChol(*m.getInducing());
    }
  }
  m.setVerbosity(0);
  m.setDefaultOptimiser(CGp::SCG);
  r.ll0 = m.logLikelihood();
  m.optimise(iters);
  r.llStar = m.logLikelihood();
  r.teRmse = rmse(m, Xte, yte);
  return r;
}

void printRow(const string& tag, const RunResult& r, double ftcStar)
{
  const double gap = ftcStar - r.llStar;
  cout << setw(24) << left << tag
       << " LL0=" << setw(10) << right << fixed << setprecision(4) << r.ll0
       << " LL*=" << setw(10) << r.llStar
       << " FTC-LL*=" << setw(9) << setprecision(4) << gap
       << " RMSE_te=" << setw(7) << setprecision(4) << r.teRmse << "\n";
}

} // namespace

int main()
{
  const unsigned int N = 80, Nte = 60, d = 1;
  const unsigned int itersFreeFix = 80;
  const double sRef = 0.55;
  const unsigned int Mref = 12;
  CMatrix X(N, d), y(N, 1), Xte(Nte, d), yte(Nte, 1);
  ndlutil::init_genrand(42u);
  for(unsigned int i = 0; i < N; i++)
  {
    const double x = -1.5 + 3.0 * (double)i / (double)(N - 1);
    X.setVal(x, i, 0);
    y.setVal(sin(2.0 * x) + 0.1 * cos(3.0 * x) + 0.05 * ndlutil::randn(), i, 0);
  }
  for(unsigned int i = 0; i < Nte; i++)
  {
    const double x = -1.5 + 3.0 * (double)i / (double)(Nte - 1);
    Xte.setVal(x, i, 0);
    yte.setVal(sin(2.0 * x) + 0.1 * cos(3.0 * x), i, 0);
  }

  // ---- Part A: free vs fixed at fixed M ----
  {
    const unsigned int M = Mref;
    CMatrix Xu(M, d);
    fillXu(Xu, X);
    cout << "==== A. Free vs fixed centres (M=" << M << ", s=" << sRef << ") ====\n"
         << "  Same Xu → same ridge init; free optimises directions/offsets.\n\n";

    CMatrix XuFtc(1, d);
    fillXu(XuFtc, X);
    const RunResult ftc = runOne(X, y, XuFtc, Xte, yte, CGp::FTC, "points", false, sRef, itersFreeFix);
    const RunResult ridgeFix = runOne(X, y, Xu, Xte, yte, CGp::DTCVAR, "weightspace", false, sRef, itersFreeFix);
    const RunResult ridgeFree = runOne(X, y, Xu, Xte, yte, CGp::DTCVAR, "weightspace", true, sRef, itersFreeFix);

    printRow("FTC exact", ftc, ftc.llStar);
    printRow("DTCVAR ridge fixed", ridgeFix, ftc.llStar);
    printRow("DTCVAR ridge free", ridgeFree, ftc.llStar);
    cout << fixed << setprecision(4)
         << "\nBound tighten (free − fixed) = "
         << (ridgeFree.llStar - ridgeFix.llStar) << "\n\n";
  }

  // ---- Part B: more ridges → approach FTC ----
  // Keep window overlap roughly constant: s(M) = sRef * sqrt(Mref / M).
  const unsigned int Ms[] = {4, 8, 16, 32, 48};
  const unsigned int nM = sizeof(Ms) / sizeof(Ms[0]);
  cout << "==== B. M-sweep: free ridges approach FTC ====\n"
       << "  N=" << N << "  s(M)=" << sRef << "*sqrt(" << Mref << "/M)"
       << "  iters grow with M\n\n";

  CMatrix XuDummy(1, d);
  fillXu(XuDummy, X);
  const RunResult ftc = runOne(X, y, XuDummy, Xte, yte, CGp::FTC, "points", false, sRef, 120);
  cout << left
       << setw(6) << "M"
       << right
       << setw(8) << "s"
       << setw(10) << "kappa0"
       << setw(12) << "LL*"
       << setw(12) << "FTC-LL*"
       << setw(10) << "RMSE_te"
       << setw(12) << "gap/|FTC|" << "\n";
  cout << left << setw(6) << "FTC" << right
       << setw(8) << "-"
       << setw(10) << "-"
       << setw(12) << fixed << setprecision(4) << ftc.llStar
       << setw(12) << 0.0
       << setw(10) << ftc.teRmse
       << setw(12) << 0.0 << "\n";

  vector<double> gaps;
  vector<double> rmses;
  for(unsigned int t = 0; t < nM; t++)
  {
    const unsigned int M = Ms[t];
    const double s = sRef * sqrt((double)Mref / (double)M);
    const unsigned int iters = 100 + 4 * M;
    CMatrix Xu(M, d);
    fillXu(Xu, X);
    const RunResult r = runOne(X, y, Xu, Xte, yte, CGp::DTCVAR, "weightspace", true, s, iters);
    const double gap = ftc.llStar - r.llStar;
    const double rel = gap / max(1.0, fabs(ftc.llStar));
    gaps.push_back(gap);
    rmses.push_back(r.teRmse);
    cout << left << setw(6) << M << right << fixed
         << setw(8) << setprecision(3) << s
         << setw(10) << setprecision(1) << scientific << r.kappa0
         << setw(12) << fixed << setprecision(4) << r.llStar
         << setw(12) << gap
         << setw(10) << r.teRmse
         << setw(12) << setprecision(4) << rel << "\n";
  }

  const double gapFirst = gaps.front();
  const double gapLast = gaps.back();
  const double rmseFirst = rmses.front();
  const double rmseLast = rmses.back();

  cout << "\n" << fixed << setprecision(4);
  cout << "LL* gap to FTC:  M=" << Ms[0] << " → M=" << Ms[nM - 1] << ": "
       << gapFirst << " → " << gapLast
       << "  (Δ = " << (gapFirst - gapLast) << ")\n";
  cout << "Test RMSE:       M=" << Ms[0] << " → M=" << Ms[nM - 1] << ": "
       << rmseFirst << " → " << rmseLast
       << "  (FTC " << ftc.teRmse << ")\n";

  const bool llApproaches = (gapLast < 0.30 * gapFirst) && (gapLast < 30.0);
  const bool rmseApproaches = (rmseLast < 0.40 * rmseFirst)
                              && (rmseLast < 2.0 * ftc.teRmse);
  if(llApproaches && rmseApproaches)
    cout << "→ more free ridges approach the FTC truth (bound + predictions).\n";
  else if(llApproaches || rmseApproaches)
    cout << "→ more free ridges approach FTC on "
         << (llApproaches ? "the bound" : "predictions") << ".\n";
  else
    cout << "→ did not approach FTC enough (check iters / width schedule).\n";

  return (llApproaches && rmseApproaches) ? 0 : 1;
}
