/* CIP-001B: head-to-head Legacy / StableFactor / LQ on hard sparse cells.
 * Prints LL and central-FD grad relative error.
 *
 * Cells and the real-world GP situations they stand for:
 *   crowdZ_*                  — inducing points collapsed (optimiser stacked Z,
 *                               or bad init). Stresses K_uu, not C; LQ ≡ Stable.
 *   mild_dtcvar               — textbook 1-D sparse regression: separated Z,
 *                               moderate lengthscale and noise. Expect agreement.
 *   gamma1000_M20_dtcvar      — short lengthscale / many inducing points on a
 *                               compact domain (local kernels, dense inducing).
 *   largeN_clust_kufkfu_*     — large N with clustered inputs, Z on a grid.
 *                               Ill-conditioned K_uf K_fu with OK K_uu — the
 *                               Seeger/C-side regime (large-data sparse GPs).
 */
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using namespace std;

static void dumpState(const string& path, const CMatrix& X, const CMatrix& y,
                      const CMatrix& Xu, CGp& model, const string& approx)
{
  ofstream out(path.c_str());
  out << setprecision(16);
  out << "approx " << approx << endl;
  out << "N " << X.getRows() << endl;
  out << "q " << X.getCols() << endl;
  out << "d " << y.getCols() << endl;
  out << "M " << Xu.getRows() << endl;
  out << "inverseWidth " << model.getKernel()->getParam(0) << endl;
  out << "variance " << model.getKernel()->getParam(1) << endl;
  // Compound RBF+White: white is param 2 (0 if kernel has fewer params).
  const double white =
    (model.getKernel()->getNumParams() > 2) ? model.getKernel()->getParam(2) : 0.0;
  out << "white " << white << endl;
  out << "beta " << model.getBetaVal() << endl;
  out << "ll " << model.logLikelihood() << endl;
  out << "X" << endl;
  for(unsigned int i = 0; i < X.getRows(); i++)
  {
    for(unsigned int j = 0; j < X.getCols(); j++)
      out << X.getVal(i, j) << (j + 1 == X.getCols() ? "" : " ");
    out << endl;
  }
  out << "y" << endl;
  for(unsigned int i = 0; i < y.getRows(); i++)
    out << y.getVal(i, 0) << endl;
  out << "Xu" << endl;
  for(unsigned int i = 0; i < Xu.getRows(); i++)
  {
    for(unsigned int j = 0; j < Xu.getCols(); j++)
      out << Xu.getVal(i, j) << (j + 1 == Xu.getCols() ? "" : " ");
    out << endl;
  }
  CMatrix g(1, model.getOptNumParams());
  model.logLikelihoodGradient(g);
  out << "grads_ll " << g.getCols() << endl;
  for(unsigned int j = 0; j < g.getCols(); j++)
    out << g.getVal(0, j) << endl;
}

static bool finiteVal(double x)
{
  return std::isfinite(x);
}

static double fdGradCheckMaxRel(CGp& model, double change = 1e-6)
{
  const unsigned int n = model.getOptNumParams();
  CMatrix params(1, n);
  CMatrix analytic(1, n);
  model.getOptParams(params);
  model.logLikelihoodGradient(analytic);
  double maxRel = 0.0;
  for(unsigned int j = 0; j < n; j++)
  {
    const double orig = params.getVal(j);
    params.setVal(orig + change, j);
    model.setOptParams(params);
    const double plus = model.logLikelihood();
    params.setVal(orig - change, j);
    model.setOptParams(params);
    const double minus = model.logLikelihood();
    params.setVal(orig, j);
    model.setOptParams(params);
    const double num = 0.5 * (plus - minus) / change;
    const double an = analytic.getVal(j);
    const double denom = max(1.0, max(fabs(num), fabs(an)));
    maxRel = max(maxRel, fabs(num - an) / denom);
  }
  return maxRel;
}

struct ModeResult
{
  bool ok;
  double ll;
  double fdRel;
  string err;
};

static ModeResult runMode(CGp& model, int mode)
{
  ModeResult r;
  r.ok = false;
  r.ll = numeric_limits<double>::quiet_NaN();
  r.fdRel = numeric_limits<double>::infinity();
  try
  {
    model.setSparseLinalgMode(mode);
    r.ll = model.logLikelihood();
    if(!finiteVal(r.ll))
    {
      r.err = "nonfinite_ll";
      return r;
    }
    CMatrix g(1, model.getOptNumParams());
    if(!finiteVal(model.logLikelihoodGradient(g)))
    {
      r.err = "nonfinite_grad";
      return r;
    }
    r.fdRel = fdGradCheckMaxRel(model);
    if(!finiteVal(r.fdRel))
    {
      r.err = "nonfinite_fd";
      return r;
    }
    r.ok = true;
  }
  catch(ndlexceptions::Error& e)
  {
    r.err = e.getMessage();
  }
  catch(...)
  {
    r.err = "unknown";
  }
  return r;
}

static void printRow(const string& cell, const string& mode, const ModeResult& r)
{
  cout << left << setw(28) << cell << setw(10) << mode;
  if(!r.ok)
  {
    cout << "FAIL (" << r.err << ")" << endl;
    return;
  }
  cout << scientific << setprecision(6)
       << "ll=" << r.ll
       << "  fd_rel=" << r.fdRel << endl;
}

static void compareCell(CGp& model, const string& cell)
{
  const int modes[] = {
    CGp::SparseLinalgLegacy,
    CGp::SparseLinalgStableFactor,
    CGp::SparseLinalgLQ
  };
  const char* names[] = {"legacy", "stable", "lq"};
  ModeResult rs[3];
  for(int i = 0; i < 3; i++)
  {
    rs[i] = runMode(model, modes[i]);
    printRow(cell, names[i], rs[i]);
  }
  if(rs[1].ok && rs[2].ok)
  {
    const double dll = fabs(rs[1].ll - rs[2].ll);
    const double dfd = rs[2].fdRel - rs[1].fdRel;
    cout << "  stable→lq  |Δll|=" << scientific << dll
         << "  Δfd_rel(lq-stable)=" << dfd;
    if(dfd < -1e-12)
      cout << "  (lq better FD)";
    else if(dfd > 1e-12)
      cout << "  (stable better FD)";
    else
      cout << "  (FD ~equal)";
    cout << endl;
  }
  cout << endl;
}

int main()
{
  cout << "==== diagLqVsStable (CIP-001B) ====\n" << endl;

  // Inducing collapse: all Z piled near the origin (failed inducing optimisation
  // or duplicate init). K_uu is the bottleneck; C ≈ I so LQ cannot help.
  {
    const unsigned int n = 60, m = 20;
    CMatrix X(n, 1), Y(n, 1), Xu(m, 1);
    for(unsigned int i = 0; i < n; i++)
    {
      const double x = -1.0 + 2.0 * (double)i / (double)(n - 1);
      X.setVal(x, i, 0);
      Y.setVal(sin(2.0 * x), i, 0);
    }
    for(unsigned int i = 0; i < m; i++)
      Xu.setVal(1e-9 * (double)(i + 1), i, 0);

    const int approxs[] = {CGp::FITC, CGp::DTCVAR, CGp::DTC};
    const char* tags[] = {"crowdZ_fitc", "crowdZ_dtcvar", "crowdZ_dtc"};
    for(unsigned int k = 0; k < 3; k++)
    {
      CCmpndKern kern(X);
      kern.addKern(new CRbfKern(X));
      kern.addKern(new CWhiteKern(X));
      kern.setParam(1e4, 0);
      kern.setParam(1.0, 1);
      kern.setParam(1e-6, 2);
      CGaussianNoise noise(&Y);
      for(unsigned int i = 0; i < noise.getOutputDim(); i++)
        noise.setBiasVal(0.0, i);
      CGp model(&kern, &noise, &X, approxs[k], m, 0);
      model.setInducingVals(Xu);
      model.setInducingFixed(true);
      model.setBetaVal(1e6);
      model.setVerbosity(0);
      compareCell(model, tags[k]);
    }
  }

  // Routine sparse GP: evenly spaced data and inducing inputs, moderate RBF
  // lengthscale and noise. All three linalg modes should agree closely.
  {
    const unsigned int n = 40, m = 10;
    CMatrix X(n, 1), Y(n, 1), Xu(m, 1);
    for(unsigned int i = 0; i < n; i++)
    {
      X.setVal(-2.0 + 4.0 * (double)i / (double)(n - 1), i, 0);
      Y.setVal(sin(X.getVal(i, 0)), i, 0);
    }
    for(unsigned int i = 0; i < m; i++)
      Xu.setVal(X.getVal((i * n) / m, 0), i, 0);

    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CWhiteKern(X));
    kern.setParam(10.0, 0);
    kern.setParam(1.0, 1);
    kern.setParam(1e-4, 2);
    CGaussianNoise noise(&Y);
    for(unsigned int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, m, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(100.0);
    model.setVerbosity(0);
    compareCell(model, "mild_dtcvar");
  }

  // Short-lengthscale regression with a dense inducing grid on a small interval
  // (e.g. modelling rapid local variation). Separated Z, but K_uu / K_uf get
  // sharper structure as inverseWidth grows.
  {
    const unsigned int n = 40, m = 20;
    CMatrix X(n, 1), Y(n, 1), Xu(m, 1);
    for(unsigned int i = 0; i < n; i++)
    {
      X.setVal(-3.0 + 6.0 * (double)i / (double)(n - 1), i, 0);
      Y.setVal(sin(X.getVal(i, 0)), i, 0);
    }
    for(unsigned int i = 0; i < m; i++)
      Xu.setVal(X.getVal((i * (n - 1)) / (m - 1), 0), i, 0);

    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CWhiteKern(X));
    kern.setParam(1000.0, 0);
    kern.setParam(1.0, 1);
    kern.setParam(1e-4, 2);
    CGaussianNoise noise(&Y);
    for(unsigned int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, m, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(100.0);
    model.setVerbosity(0);
    compareCell(model, "gamma1000_M20_dtcvar");
  }

  // Large-data sparse GP: thousands of points in a few spatial/temporal clusters,
  // inducing grid across the domain, low observation noise (large β). Makes
  // K_uf K_fu (hence C) ill-conditioned while K_uu stays fine — Seeger LQ regime.
  {
    const unsigned int n = 2000, m = 25;
    CMatrix X(n, 1), Y(n, 1), Xu(m, 1);
    const unsigned int nPer = n / 5;
    for(unsigned int c = 0; c < 5; c++)
    {
      const double centre = -3.0 + 6.0 * (double)c / 4.0;
      for(unsigned int i = 0; i < nPer; i++)
      {
        const double u = (double)((i * 1103515245u + 12345u + c * 97u) % 10000) / 10000.0;
        const double x = centre + 0.05 * (u - 0.5);
        X.setVal(x, c * nPer + i, 0);
        Y.setVal(sin(2.0 * x), c * nPer + i, 0);
      }
    }
    for(unsigned int i = 0; i < m; i++)
      Xu.setVal(-3.0 + 6.0 * (double)i / (double)(m - 1), i, 0);

    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CWhiteKern(X));
    kern.setParam(20.0, 0);
    kern.setParam(1.0, 1);
    kern.setParam(1e-4, 2);
    CGaussianNoise noise(&Y);
    for(unsigned int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, m, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(1e6);
    model.setVerbosity(0);
    compareCell(model, "largeN_clust_kufkfu_dtcvar");

    // Dumps for scripts/sparse_mp_reference.py (CIP-001B soak).
    model.setSparseLinalgMode(CGp::SparseLinalgStableFactor);
    dumpState("bench/cip001B_soak/largeN_clust_kufkfu_dtcvar_stable_state.txt",
              X, Y, Xu, model, "dtcvar");
    model.setSparseLinalgMode(CGp::SparseLinalgLQ);
    dumpState("bench/cip001B_soak/largeN_clust_kufkfu_dtcvar_lq_state.txt",
              X, Y, Xu, model, "dtcvar");
    cout << "Wrote bench/cip001B_soak/largeN_clust_kufkfu_dtcvar_{stable,lq}_state.txt"
         << endl;
  }

  cout << "Notes:\n"
       << "  crowdZ_*: LQ ≡ StableFactor (Δll=0); FD util limited by K_uu/K_mm,\n"
       << "            not the Seeger C-side regime.\n"
       << "  largeN_clust_*: C = I+βBB^T stressed; pair with testLQAdjoint κ/SVD\n"
       << "            gate and sparse_mp_reference.py on the dumped state.\n";
  return 0;
}
