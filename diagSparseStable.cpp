/* CIP-0012 / CIP-001B: Legacy vs StableFactor vs LQ on a γ×M sweep.
 *
 * Real-world map: 1-D sparse regression while varying RBF inverseWidth (γ) and
 * inducing count M — from smooth/global kernels (small γ) to short-scale local
 * structure (large γ), and from coarse to dense inducing grids. Separated Z on
 * a grid (not crowdZ). Asks whether the three linalg modes still implement the
 * same DTC / DTCVAR / FITC / PITC model as conditioning changes.
 *
 * Pair dumps with scripts/sparse_mp_reference.py for mpmath gates.
 */
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"

using namespace std;

namespace {

// Evenly spaced inputs and inducing points on [-1,1] — the default “place Z on
// a grid over the observed range” workflow in sparse GP tutorials.
void fillGrid1d(CMatrix& X, CMatrix& y, CMatrix& Xu, unsigned int N, unsigned int M)
{
  for(unsigned int i = 0; i < N; i++)
  {
    const double x = -1.0 + 2.0 * (double)i / (double)(N - 1);
    X.setVal(x, i, 0);
    y.setVal(sin(2.0 * x), i, 0);
  }
  for(unsigned int i = 0; i < M; i++)
  {
    const double x = -1.0 + 2.0 * (double)i / (double)(M - 1);
    Xu.setVal(x, i, 0);
  }
}

struct CellResult
{
  double gamma;
  unsigned int M;
  double llDiff;
  double gMax;
  bool ok;
};

CellResult runCell(int approx, double gamma, unsigned int M, unsigned int N, double beta)
{
  CMatrix X(N, 1);
  CMatrix y(N, 1);
  CMatrix Xu(M, 1);
  fillGrid1d(X, y, Xu, N, M);

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CWhiteKern(X));
  kern.setParam(gamma, 0); // RBF inverseWidth
  kern.setParam(1.0, 1);   // RBF variance
  kern.setParam(1e-4, 2);  // white

  CGaussianNoise noise(&y);
  for(unsigned int i = 0; i < noise.getOutputDim(); i++)
    noise.setBiasVal(0.0, i);

  CGp model(&kern, &noise, &X, approx, M, 0);
  model.setInducingFixed(true);
  model.setInducingVals(Xu);
  model.setBetaVal(beta);

  model.setStableSparseLinalg(false);
  const double llL = model.logLikelihood();
  CMatrix gL(1, model.getOptNumParams());
  model.logLikelihoodGradient(gL);

  model.setStableSparseLinalg(true);
  const double llS = model.logLikelihood();
  CMatrix gS(1, model.getOptNumParams());
  model.logLikelihoodGradient(gS);

  // CIP-001B: on this grid/γ sweep LQ must match StableFactor (same sparse GP,
  // different factorisation). PITC still uses Gram under lq mode.
  model.setSparseLinalgMode(CGp::SparseLinalgLQ);
  const double llQ = model.logLikelihood();
  CMatrix gQ(1, model.getOptNumParams());
  model.logLikelihoodGradient(gQ);
  const double llSQ = fabs(llS - llQ);
  double gSQ = 0.0;
  for(unsigned int j = 0; j < gS.getCols(); j++)
    gSQ = max(gSQ, fabs(gS.getVal(j) - gQ.getVal(j)));

  CellResult r;
  r.gamma = gamma;
  r.M = M;
  r.llDiff = fabs(llL - llS);
  r.gMax = 0.0;
  for(unsigned int j = 0; j < gL.getCols(); j++)
    r.gMax = max(r.gMax, fabs(gL.getVal(j) - gS.getVal(j)));
  // LL dual-path is tight; M=20 PITC can sit ~1e-6. Grad diffs need mpmath (CIP-0012).
  const bool finite = std::isfinite(llL) && std::isfinite(llS) && std::isfinite(llQ)
    && std::isfinite(r.gMax) && std::isfinite(gSQ);
  r.ok = finite && (r.llDiff < 5e-6) && (r.gMax < 5e-3)
    && (llSQ < 5e-6) && (gSQ < 5e-3);
  return r;
}

int sweepApprox(int approx, const string& name)
{
  cout << "==== diagSparseStable " << name << " ====" << endl;
  const unsigned int N = 40;
  const double beta = 100.0;
  const double gammas[] = {1.0, 10.0, 100.0, 1000.0};
  const unsigned int Ms[] = {5u, 10u, 20u};
  int fail = 0;
  cout << setprecision(6);
  cout << "gamma M ll_abs_diff grad_max_abs status" << endl;
  for(unsigned int gi = 0; gi < 4; gi++)
  {
    for(unsigned int mi = 0; mi < 3; mi++)
    {
      CellResult r = runCell(approx, gammas[gi], Ms[mi], N, beta);
      cout << r.gamma << " " << r.M << " " << r.llDiff << " " << r.gMax
           << " " << (r.ok ? "PASS" : "FAIL") << endl;
      if(!r.ok)
        fail++;
    }
  }
  return fail;
}

} // namespace

int main()
{
  int fail = 0;
  fail += sweepApprox(CGp::DTC, "dtc");
  fail += sweepApprox(CGp::DTCVAR, "dtcvar");
  fail += sweepApprox(CGp::FITC, "fitc");
  fail += sweepApprox(CGp::PITC, "pitc");
  cout << "Total failing cells: " << fail << endl;
  return fail == 0 ? 0 : 1;
}
