/* CIP-0028 ridge diagnostics
 *
 * 1) Free vs fixed centres: analytic grads tighten the DTCVAR bound.
 * 2) M-sweep: more free ridges → bound approaches the *exact* GP
 *    marginal at the same hypers: log N(y; 0, K + σ²I) with σ² = 1/β.
 *
 * Important: do NOT compare DTCVAR LL* to a separately fitted FTC LL*.
 * GPc FTC never puts β into K (noise is White-in-kernel if present);
 * sparse DTCVAR optimises β. That apples-to-oranges gap is misleading.
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

// Exact GP marginal log N(y; 0, K_f + σ² I) for the current MLP (no White).
double exactMarginalLL(CKern& kern, const CMatrix& X, const CMatrix& y, double sigma2)
{
  const unsigned int N = X.getRows();
  CMatrix K(N, N);
  kern.compute(K, X);
  for(unsigned int i = 0; i < N; i++)
    K.addVal(sigma2, i, i);
  K.setSymmetric(true);
  CMatrix L(K);
  for(unsigned int i = 0; i < N; i++)
    L.addVal(1e-12, i, i);
  L.setSymmetric(true);
  L.chol(); // upper
  double logDet = 0.0;
  for(unsigned int i = 0; i < N; i++)
    logDet += 2.0 * log(fabs(L.getVal(i, i)));
  CMatrix y2(y);
  y2.trsm(L, 1.0, "l", "u", "t", "n");
  y2.trsm(L, 1.0, "l", "u", "n", "n");
  double quad = 0.0;
  for(unsigned int i = 0; i < N; i++)
    quad += y.getVal(i, 0) * y2.getVal(i, 0);
  return -0.5 * (quad + logDet + (double)N * ndlutil::LOGTWOPI);
}

struct RunResult
{
  double ll0;
  double llStar;
  double exactStar; // exact marginal at same kern + β*
  double teRmse;
  double betaStar;
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

// MLP-only compound (no White): observation noise is β alone.
RunResult runRidge(CMatrix& X, CMatrix& y, CMatrix& Xu,
                   const CMatrix& Xte, const CMatrix& yte,
                   bool freeInd, double windowWidth, unsigned int iters)
{
  CCmpndKern kern(X);
  kern.addKern(new CMlpKern(X));
  CGaussianNoise noise(&y);
  noise.setBiasVal(0.0, 0);
  const unsigned int M = Xu.getRows();
  CGp m(&kern, &noise, &X, CGp::DTCVAR, M, 0);
  m.setInducingVals(Xu);
  m.setInducingFixed(!freeInd);
  m.setBetaVal(20.0);
  m.setInducingType("weightspace");
  m.getInducing()->setWindowWidth(windowWidth);
  m.setVerbosity(0);
  m.setDefaultOptimiser(CGp::SCG);

  RunResult r;
  r.ll0 = m.logLikelihood();
  m.optimise(iters);
  r.llStar = m.logLikelihood();
  r.betaStar = m.getBetaVal();
  r.exactStar = exactMarginalLL(kern, X, y, 1.0 / r.betaStar);
  r.teRmse = rmse(m, Xte, yte);
  return r;
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

  // ---- Part A: free vs fixed at fixed M (same approx → fair) ----
  {
    const unsigned int M = Mref;
    CMatrix Xu(M, d);
    fillXu(Xu, X);
    cout << "==== A. Free vs fixed centres (M=" << M << ", s=" << sRef << ") ====\n"
         << "  Same approx; free optimises directions/offsets.\n"
         << "  Gap = exact(K+I/β*) − DTCVAR at shared hypers.\n\n";

    const RunResult ridgeFix = runRidge(X, y, Xu, Xte, yte, false, sRef, itersFreeFix);
    const RunResult ridgeFree = runRidge(X, y, Xu, Xte, yte, true, sRef, itersFreeFix);

    cout << left << setw(18) << "method" << right
         << setw(10) << "LL*"
         << setw(10) << "exact*"
         << setw(10) << "gap"
         << setw(10) << "RMSE_te" << "\n";
    cout << left << setw(18) << "ridge fixed" << right << fixed << setprecision(4)
         << setw(10) << ridgeFix.llStar
         << setw(10) << ridgeFix.exactStar
         << setw(10) << (ridgeFix.exactStar - ridgeFix.llStar)
         << setw(10) << ridgeFix.teRmse << "\n";
    cout << left << setw(18) << "ridge free" << right
         << setw(10) << ridgeFree.llStar
         << setw(10) << ridgeFree.exactStar
         << setw(10) << (ridgeFree.exactStar - ridgeFree.llStar)
         << setw(10) << ridgeFree.teRmse << "\n";
    cout << "\nBound tighten (free − fixed LL*) = "
         << (ridgeFree.llStar - ridgeFix.llStar) << "\n"
         << "Gap shrink (fixed − free)     = "
         << ((ridgeFix.exactStar - ridgeFix.llStar)
             - (ridgeFree.exactStar - ridgeFree.llStar)) << "\n\n";
  }

  // ---- Part B: M-sweep vs exact at shared hypers ----
  const unsigned int Ms[] = {4, 8, 16, 32, 48};
  const unsigned int nM = sizeof(Ms) / sizeof(Ms[0]);
  cout << "==== B. M-sweep: free ridges → exact marginal ====\n"
       << "  N=" << N << "  s(M)=" << sRef << "*sqrt(" << Mref << "/M)\n"
       << "  gap = exact(K+I/β*) − DTCVAR  (same kern, same β*)\n\n";

  cout << left << setw(6) << "M" << right
       << setw(8) << "s"
       << setw(10) << "LL*"
       << setw(10) << "exact*"
       << setw(10) << "gap"
       << setw(10) << "RMSE_te" << "\n";

  vector<double> gaps;
  vector<double> rmses;
  for(unsigned int t = 0; t < nM; t++)
  {
    const unsigned int M = Ms[t];
    const double s = sRef * sqrt((double)Mref / (double)M);
    const unsigned int iters = 100 + 4 * M;
    CMatrix Xu(M, d);
    fillXu(Xu, X);
    const RunResult r = runRidge(X, y, Xu, Xte, yte, true, s, iters);
    const double gap = r.exactStar - r.llStar;
    gaps.push_back(gap);
    rmses.push_back(r.teRmse);
    cout << left << setw(6) << M << right << fixed
         << setw(8) << setprecision(3) << s
         << setw(10) << setprecision(4) << r.llStar
         << setw(10) << r.exactStar
         << setw(10) << gap
         << setw(10) << r.teRmse << "\n";
  }

  // Gap is not monotone from tiny M: each M optimises to different hypers
  // (exact* changes). Read the large-M trend (M≥16) and RMSE.
  const double gap16 = gaps[2];
  const double gapLast = gaps.back();
  cout << "\n" << fixed << setprecision(4);
  cout << "True bound gap: M=" << Ms[2] << " → M=" << Ms[nM - 1] << ": "
       << gap16 << " → " << gapLast
       << "  (Δ = " << (gap16 - gapLast) << ")\n";
  cout << "Test RMSE:      M=" << Ms[0] << " → M=" << Ms[nM - 1] << ": "
       << rmses.front() << " → " << rmses.back() << "\n";
  cout << "Note: gap vs separately fitted FTC is not a bound test "
       << "(FTC ignores β; DTCVAR learns β).\n";

  const bool gapsOk = (gapLast < gap16) && (gapLast >= -1e-3);
  const bool rmseOk = rmses.back() < 0.4 * rmses.front();
  if(gapsOk && rmseOk)
    cout << "→ with more free ridges the true bound gap shrinks and predictions improve.\n";
  else if(rmseOk)
    cout << "→ predictions improve with M; bound gap still mixed (hypers differ per M).\n";
  else
    cout << "→ did not improve cleanly (check iters / width schedule).\n";
  return (gapsOk && rmseOk) ? 0 : 1;
}
