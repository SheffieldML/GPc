/* CIP-0028: FTC vs DTCVAR points vs ridge weight-space (fixed vs free centres).
 *
 * Shows that analytic grads on ridge directions/offsets tighten the DTCVAR
 * bound relative to the same initialisation with inducing fixed.
 *
 *   make diagWeightSpaceRidge && ./diagWeightSpaceRidge
 */
#include <cmath>
#include <iomanip>
#include <iostream>
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

struct RunResult
{
  double ll0;
  double llStar;
  double teRmse;
};

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
  if(approx != CGp::FTC)
  {
    m.setInducingVals(Xu);
    m.setInducingFixed(!freeInd);
    m.setBetaVal(20.0);
    m.setInducingType(ind); // weightspace reseeds W from Xu (deterministic)
    if(ind == "weightspace")
      m.getInducing()->setWindowWidth(windowWidth);
  }
  m.setVerbosity(0);
  m.setDefaultOptimiser(CGp::SCG);
  RunResult r;
  r.ll0 = m.logLikelihood();
  m.optimise(iters);
  r.llStar = m.logLikelihood();
  r.teRmse = rmse(m, Xte, yte);
  return r;
}

void printRow(const string& tag, const RunResult& r, double ftcStar)
{
  const double gap = ftcStar - r.llStar; // >= 0 if DTCVAR is a lower bound
  cout << setw(24) << left << tag
       << " LL0=" << setw(10) << right << fixed << setprecision(4) << r.ll0
       << " LL*=" << setw(10) << r.llStar
       << " FTC-LL*=" << setw(9) << setprecision(4) << gap
       << " RMSE_te=" << setw(7) << setprecision(4) << r.teRmse << "\n";
}

} // namespace

int main()
{
  const unsigned int N = 80, M = 12, Nte = 60, d = 1, iters = 80;
  const double s = 0.55;
  CMatrix X(N, d), y(N, 1), Xu(M, d), Xte(Nte, d), yte(Nte, 1);
  ndlutil::init_genrand(42u);
  for(unsigned int i = 0; i < N; i++)
  {
    const double x = -1.5 + 3.0 * (double)i / (double)(N - 1);
    X.setVal(x, i, 0);
    y.setVal(sin(2.0 * x) + 0.1 * cos(3.0 * x) + 0.05 * ndlutil::randn(), i, 0);
  }
  for(unsigned int i = 0; i < M; i++)
    Xu.setVal(X.getVal((i * (N - 1)) / (M - 1), 0), i, 0);
  for(unsigned int i = 0; i < Nte; i++)
  {
    const double x = -1.5 + 3.0 * (double)i / (double)(Nte - 1);
    Xte.setVal(x, i, 0);
    yte.setVal(sin(2.0 * x) + 0.1 * cos(3.0 * x), i, 0);
  }

  cout << "CIP-0028 ridge bound vs free centres\n"
       << "  1-D MLP regression, N=" << N << " M=" << M
       << " iters=" << iters << " windowWidth=" << s << "\n"
       << "  Same Xu → same ridge init (setInducingType seed); free vs fixed only.\n\n";

  const RunResult ftc = runOne(X, y, Xu, Xte, yte, CGp::FTC, "points", false, s, iters);
  const RunResult pts = runOne(X, y, Xu, Xte, yte, CGp::DTCVAR, "points", true, s, iters);
  const RunResult ridgeFix = runOne(X, y, Xu, Xte, yte, CGp::DTCVAR, "weightspace", false, s, iters);
  const RunResult ridgeFree = runOne(X, y, Xu, Xte, yte, CGp::DTCVAR, "weightspace", true, s, iters);

  printRow("FTC exact", ftc, ftc.llStar);
  printRow("DTCVAR points free", pts, ftc.llStar);
  printRow("DTCVAR ridge fixed", ridgeFix, ftc.llStar);
  printRow("DTCVAR ridge free", ridgeFree, ftc.llStar);

  const double tighten = ridgeFree.llStar - ridgeFix.llStar;
  const double gapFix = ftc.llStar - ridgeFix.llStar;
  const double gapFree = ftc.llStar - ridgeFree.llStar;
  cout << "\n";
  cout << fixed << setprecision(4);
  cout << "Bound tighten (free − fixed LL*) = " << tighten << "\n";
  cout << "Gap to FTC: fixed=" << gapFix << "  free=" << gapFree;
  if(tighten > 1e-3 && gapFree < gapFix)
    cout << "  → free centres recover a tighter lower bound.\n";
  else if(tighten > 0.0)
    cout << "  → free centres improve LL* (small).\n";
  else
    cout << "  → no improvement (check init / iters).\n";
  return (tighten > 1e-3 && gapFree < gapFix) ? 0 : 1;
}
