/* CIP-003D phase 1: Hermite layout, capacity rule, audit (E1, E5) and layout
   warnings (W1, W2). K_uf / gradients arrive in phases 2 and 3. */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "CInducing.h"
#include "CKern.h"
#include "CSvgp.h"
#include "CGp.h"
#include "CDeepGp.h"
#include "CNoise.h"
#include <random>
#include "ndlutil.h"

using namespace std;

namespace {

int fails = 0;

void expect(bool ok, const string& msg)
{
  if(!ok)
  {
    cerr << "FAIL: " << msg << endl;
    ++fails;
  }
  else
    cout << msg << " matches." << endl;
}

// Run f; return the exception message, or "" if none thrown.
template <class F>
string thrown(F f)
{
  try
  {
    f();
  }
  catch(ndlexceptions::Error& e)
  {
    return e.getMessage();
  }
  return "";
}

bool has(const string& hay, const string& needle)
{
  return hay.find(needle) != string::npos;
}

CMatrix rows(const vector<vector<double> >& r)
{
  CMatrix V((unsigned int)r.size(), (unsigned int)r[0].size());
  for(size_t i = 0; i < r.size(); i++)
    for(size_t j = 0; j < r[0].size(); j++)
      V.setVal(r[i][j], (unsigned int)i, (unsigned int)j);
  return V;
}

void testCapacityTable()
{
  // Values quoted in CIP-003D (floor(C(D'+k-1,k)/2), degree 1 = D').
  struct { unsigned int d, total; } t[] = {
    {1, 16}, {2, 63}, {3, 212}, {5, 1557}, {8, 16106}};
  for(size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++)
    expect(CInducingHermite::totalCapacity(t[i].d + 1) == t[i].total,
           "capacity total d=" + to_string(t[i].d) + " is " + to_string(t[i].total));
  expect(CInducingHermite::degreeCapacity(2, 1) == 2, "degree-1 capacity is D'");
  expect(CInducingHermite::degreeCapacity(2, 3) == 2, "d=1 degree-3 capacity 2");
  expect(CInducingHermite::degreeCapacity(6, 5) == 126, "D'=6 degree-5 capacity 126");
  // Large D' must saturate, not overflow.
  expect(CInducingHermite::totalCapacity(2000) ==
             numeric_limits<unsigned int>::max(),
         "capacity saturates at UINT_MAX for huge D'");
  expect(!thrown([] { CInducingHermite::degreeCapacity(3, 2); }).empty(),
         "even degree rejected by capacity rule");
}

// Sweep (CIP-003C lesson): every (M, d) either builds with exactly M windows,
// or throws E1 and M really exceeds capacity. Anything else is a failure.
void testSweep()
{
  const unsigned int dims[3] = {1, 2, 8};
  unsigned int built = 0, e1 = 0, other = 0, countBad = 0, w1 = 0, w2 = 0;
  double kmax = 0.0, kmedSum = 0.0;
  vector<double> kappas;
  for(unsigned int di = 0; di < 3; di++)
  {
    const unsigned int cap = CInducingHermite::totalCapacity(dims[di] + 1);
    for(unsigned int M = 1; M <= 300; M++)
    {
      CMatrix V;
      vector<unsigned int> ks;
      string err = thrown([&] { CInducingHermite::makeLayout(V, ks, M, dims[di]); });
      if(err.empty())
      {
        built++;
        if(M > cap || V.getRows() != M || ks.size() != M)
          countBad++;
        CInducingHermite ind(V, ks);
        const HermiteLayoutReport& r = ind.report();
        if(ind.numInducing() != M || r.numInducing != M)
          countBad++;
        kappas.push_back(r.kappa);
        kmax = max(kmax, r.kappa);
        if(r.warnConditioning) w1++;
        if(r.warnNearCapacity) w2++;
      }
      else if(has(err, "(E1)") && M > cap)
        e1++;
      else
      {
        other++;
        cerr << "  unexpected failure d=" << dims[di] << " M=" << M << ": " << err << endl;
      }
    }
  }
  sort(kappas.begin(), kappas.end());
  const double kmed = kappas[kappas.size() / 2];
  (void)kmedSum;
  cout << "  sweep: built=" << built << " E1=" << e1 << " other=" << other
       << "  kappa median=" << kmed << " max=" << kmax
       << "  W1 fired=" << w1 << " W2 fired=" << w2 << endl;
  expect(other == 0, "sweep M=1..300, d in {1,2,8}: only success or E1 (no other error)");
  expect(countBad == 0, "sweep: delivered window count equals requested M whenever it builds");
  // d=1 has capacity 16 so M=17..300 must all be refused; d=2: M=64..300.
  expect(e1 == (300 - 16) + (300 - 63), "sweep: E1 exactly where M exceeds capacity");
}

void testE1Message()
{
  CMatrix V;
  vector<unsigned int> ks;
  string e = thrown([&] { CInducingHermite::makeLayout(V, ks, 17, 1); });
  expect(has(e, "(E1)") && has(e, "M=17") && has(e, "capacity is 16") &&
             has(e, "d=1") && has(e, "3:2") && has(e, "9:5"),
         "E1 names M, capacity, d and per-degree capacities");
  expect(!thrown([&] { CInducingHermite::makeLayout(V, ks, 0, 2); }).empty(),
         "M=0 rejected");
  expect(!thrown([&] { CInducingHermite::makeLayout(V, ks, 4, 0); }).empty(),
         "inputDim=0 rejected");
  expect(!thrown([&] { CInducingHermite::makeLayout(V, ks, 4, 2, 7u, 4); }).empty(),
         "even kmax rejected");
}

void testDeterminism()
{
  CMatrix A, B, C;
  vector<unsigned int> ka, kb, kc;
  CInducingHermite::makeLayout(A, ka, 40, 3, 11u);
  CInducingHermite::makeLayout(B, kb, 40, 3, 11u);
  CInducingHermite::makeLayout(C, kc, 40, 3, 12u);
  expect(A.equals(B) && ka == kb, "layout deterministic for a given seed");
  expect(!A.equals(C), "layout changes with the seed");
  // Regression: a cached Box-Muller spare from an earlier randn() call used to
  // leak into the layout, so the same seed gave different layouts per process.
  ndlutil::randn();
  CMatrix D;
  vector<unsigned int> kd;
  CInducingHermite::makeLayout(D, kd, 40, 3, 11u);
  expect(A.equals(D) && ka == kd, "layout independent of prior randn() parity");
}

void testAudit()
{
  const double s = 1.0 / sqrt(2.0);
  CMatrix good = rows({{1.0, 0.0}, {0.0, 1.0}});
  expect(thrown([&] { CInducingHermite i(good, vector<unsigned int>(2, 1)); }).empty(),
         "orthonormal degree-1 pair passes the audit");
  expect(has(thrown([&] { CInducingHermite i(good, vector<unsigned int>(2, 2)); }), "(E5)"),
         "E5: even degree");
  expect(has(thrown([&] { CInducingHermite i(good, vector<unsigned int>(2, 0)); }), "(E5)"),
         "E5: degree zero");
  expect(has(thrown([&] { CInducingHermite i(good, vector<unsigned int>(3, 1)); }), "(E5)"),
         "E5: degree count differs from rows");
  CMatrix nonunit = rows({{2.0, 0.0}, {0.0, 1.0}});
  expect(has(thrown([&] { CInducingHermite i(nonunit, vector<unsigned int>(2, 1)); }), "unit"),
         "E5: non-unit direction");
  CMatrix dup = rows({{1.0, 0.0}, {1.0, 0.0}});
  expect(has(thrown([&] { CInducingHermite i(dup, vector<unsigned int>(2, 1)); }), "positive definite"),
         "E5: duplicate direction makes K_uu singular");
  CMatrix anti = rows({{s, s}, {-s, -s}}); // v and -v: same window up to sign
  expect(has(thrown([&] { CInducingHermite i(anti, vector<unsigned int>(2, 3)); }), "positive definite"),
         "E5: antiparallel pair of odd degree is a duplicate window");
  // Three degree-3 windows in R^2 exceed the capacity of 2.
  CMatrix three = rows({{1.0, 0.0}, {s, s}, {0.0, 1.0}});
  expect(has(thrown([&] { CInducingHermite i(three, vector<unsigned int>(3, 3)); }), "capacity is 2"),
         "E5: per-degree capacity enforced on direct construction");
}

void testWarnings()
{
  // W1: two degree-1 directions 1e-3 rad apart -> kappa ~ 4e6, still PD.
  CMatrix close = rows({{1.0, 0.0}, {cos(1e-3), sin(1e-3)}});
  CInducingHermite w1(close, vector<unsigned int>(2, 1));
  expect(w1.report().warnConditioning && w1.report().kappa > 1e6, "W1 fires on near-parallel pair");
  expect(w1.report().warnings().size() >= 1 && has(w1.report().warnings()[0], "W1"),
         "W1 text names the condition number");

  // W2: d=1 capacity 16; M=15 is 94% full.
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 15, 1);
  CInducingHermite nearCap(V, ks);
  expect(nearCap.report().warnNearCapacity, "W2 fires at 94% of capacity (d=1, M=15)");

  // Control: roomy layout fires nothing.
  CInducingHermite::makeLayout(V, ks, 8, 5);
  CInducingHermite roomy(V, ks);
  expect(!roomy.report().warnNearCapacity && !roomy.report().warnConditioning &&
             roomy.report().warnings().empty(),
         "no W1/W2 on a roomy well-conditioned layout (d=5, M=8)");
  cout << "  " << roomy.report().summary() << endl;
  cout << "  " << nearCap.report().summary() << endl;
}

void testKuu()
{
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 30, 3, 5u);
  CInducingHermite ind(V, ks);
  CMlpKern kern(3);
  CMatrix Kuu;
  ind.computeKuu(Kuu, kern);
  bool diagOk = true, blockOk = true, symOk = true;
  const unsigned int M = ind.numInducing();
  for(unsigned int m = 0; m < M; m++)
  {
    if(fabs(Kuu.getVal(m, m) - 1.0) > 1e-12)
      diagOk = false;
    for(unsigned int n = 0; n < M; n++)
    {
      double dot = 0.0;
      for(unsigned int j = 0; j < V.getCols(); j++)
        dot += V.getVal(m, j) * V.getVal(n, j);
      const double want = (ks[m] == ks[n]) ? pow(dot, (int)ks[m]) : 0.0;
      if(fabs(Kuu.getVal(m, n) - want) > 1e-12)
        blockOk = false;
      if(fabs(Kuu.getVal(m, n) - Kuu.getVal(n, m)) > 1e-14)
        symOk = false;
    }
  }
  expect(diagOk, "K_uu has unit diagonal (unit directions)");
  expect(blockOk, "K_uu[m,n] = delta_{k_m k_n} (v_m . v_n)^k");
  expect(symOk, "K_uu symmetric");
  // Degree-1 frame is orthonormal => its block is the identity.
  bool ident = true;
  const unsigned int n1 = ind.report().count[0];
  for(unsigned int m = 0; m < n1; m++)
    for(unsigned int n = 0; n < n1; n++)
      if(fabs(Kuu.getVal(m, n) - (m == n ? 1.0 : 0.0)) > 1e-12)
        ident = false;
  expect(n1 == 4 && ident, "degree-1 block is the identity (orthonormal frame, D'=4)");
  expect(ind.typeName() == "hermite" && !ind.kuuDependsOnKernel() &&
             ind.getNumParams() == 0,
         "type 'hermite', kernel-free K_uu, no learnable parameters");
}

// ---- phase 2: K_fu against quadrature over omega ~ N(0, I_{D'}) ----------
double heNorm(unsigned int k, double x) // He_k(x) / sqrt(k!)
{
  double p0 = 1.0, p1 = x;
  if(k == 0) return 1.0;
  for(unsigned int i = 1; i < k; i++)
  {
    const double p2 = x * p1 - i * p0;
    p0 = p1;
    p1 = p2;
  }
  return p1 / sqrt(tgamma(k + 1.0));
}

// xhat_n (D'-vector) = sqrt(sigmaDiag) .* (x_n, 1), as in the ridge code.
CMatrix xhatOf(const CMatrix& X, const vector<double>& sigmaDiag)
{
  const unsigned int d = X.getCols();
  CMatrix Xh(X.getRows(), d + 1);
  for(unsigned int n = 0; n < X.getRows(); n++)
  {
    for(unsigned int j = 0; j < d; j++)
      Xh.setVal(sqrt(sigmaDiag[j]) * X.getVal(n, j), n, j);
    Xh.setVal(sqrt(sigmaDiag[d]), n, d);
  }
  return Xh;
}

// Tensor-grid trapezoid (spectrally accurate for Gaussian-weighted smooth
// integrands). Returns B * E[erf(w.xhat_n/sqrt2) He_k(v_m.w)/sqrt(k!)] in KQ
// and B^2 E[erf(w.xhat_i/sqrt2) erf(w.xhat_j/sqrt2)] in KffQ.
void quadrature(CMatrix& KQ, CMatrix& KffQ, const CMatrix& Xh, const CMatrix& V,
                const vector<unsigned int>& ks, double variance, double h, double L)
{
  const unsigned int D = V.getCols();
  const unsigned int N = Xh.getRows();
  const unsigned int M = V.getRows();
  const int G = (int)floor(2 * L / h + 0.5) + 1;
  const double B = sqrt(variance * M_PI / 2.0);
  KQ.resize(N, M);
  KQ.zeros();
  KffQ.resize(N, N);
  KffQ.zeros();
  vector<int> idx(D, 0);
  vector<double> w(D), e(N);
  const double norm = pow(2 * M_PI, -0.5 * D);
  bool done = false;
  while(!done)
  {
    double r2 = 0.0;
    for(unsigned int c = 0; c < D; c++)
    {
      w[c] = -L + idx[c] * h;
      r2 += w[c] * w[c];
    }
    const double dens = norm * exp(-0.5 * r2) * pow(h, (int)D);
    for(unsigned int n = 0; n < N; n++)
    {
      double t = 0.0;
      for(unsigned int c = 0; c < D; c++)
        t += w[c] * Xh.getVal(n, c);
      e[n] = erf(t / sqrt(2.0));
    }
    for(unsigned int m = 0; m < M; m++)
    {
      double s = 0.0;
      for(unsigned int c = 0; c < D; c++)
        s += V.getVal(m, c) * w[c];
      const double he = heNorm(ks[m], s);
      for(unsigned int n = 0; n < N; n++)
        KQ.addVal(B * dens * e[n] * he, n, m);
    }
    for(unsigned int i = 0; i < N; i++)
      for(unsigned int j = 0; j < N; j++)
        KffQ.addVal(B * B * dens * e[i] * e[j], i, j);
    unsigned int c = 0;
    while(c < D && ++idx[c] == G)
      idx[c++] = 0;
    done = (c == D);
  }
}

double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

// Gate one kernel: (1) the erf feature identity reproduces kern.computeElement,
// (2) wsHermiteCrossCompute reproduces the quadrature of the window integral.
template <class K>
void gateKernel(K& kern, const vector<double>& sigmaDiag, double variance,
                const string& label, double h, double L, double tol)
{
  const unsigned int d = kern.getInputDim();
  CMatrix X(3, d);
  const double vals[] = {0.3, -0.7, 0.55, 0.9, -0.2, 0.4, -0.6, 0.1, 0.8};
  for(unsigned int n = 0; n < 3; n++)
    for(unsigned int j = 0; j < d; j++)
      X.setVal(vals[(n * d + j) % 9], n, j);
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 9, d);
  // makeLayout may return only odd degrees up to capacity; ensure degrees 1,3,5.. appear
  CMatrix Kfu, KQ, KffQ, Kff(X.getRows(), X.getRows());
  kern.wsHermiteCrossCompute(Kfu, X, V, ks);
  quadrature(KQ, KffQ, xhatOf(X, sigmaDiag), V, ks, variance, h, L);
  kern.compute(Kff, X);
  const double eFeat = maxAbsDiff(Kff, KffQ);
  const double eHerm = maxAbsDiff(Kfu, KQ);
  cout << label << ": |K_ff - erf-feature quadrature| = " << eFeat
       << ", |K_fu - Hermite quadrature| = " << eHerm << endl;
  expect(eFeat < tol, label + ": erf-feature identity reproduces computeElement");
  expect(eHerm < tol, label + ": K_fu matches quadrature of the window integral");
  // Nyström capture phi_n = L^{-1} K_u,n has |phi_n|^2 <= k(x_n,x_n) (K_ff - Q_ff is PSD).
  CInducingHermite ind(V, ks);
  CMatrix Kuu(V.getRows(), V.getRows()), Kuf, kd(X.getRows(), 1);
  ind.computeKuu(Kuu, kern);
  ind.computeKuf(Kuf, kern, X);
  kern.diagCompute(kd, X);
  const unsigned int M = V.getRows();
  vector<vector<double> > Lc(M, vector<double>(M, 0.0));
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j <= i; j++)
    {
      double a = Kuu.getVal(i, j);
      for(unsigned int c = 0; c < j; c++)
        a -= Lc[i][c] * Lc[j][c];
      Lc[i][j] = (i == j) ? sqrt(a) : a / Lc[j][j];
    }
  bool bounded = true;
  double ratio = 0.0;
  for(unsigned int n = 0; n < X.getRows(); n++)
  {
    vector<double> phi(M);
    double cap = 0.0;
    for(unsigned int i = 0; i < M; i++)
    {
      double a = Kuf.getVal(i, n);
      for(unsigned int c = 0; c < i; c++)
        a -= Lc[i][c] * phi[c];
      phi[i] = a / Lc[i][i];
      cap += phi[i] * phi[i];
    }
    if(cap > kd.getVal(n, 0) * (1 + 1e-10)) bounded = false;
    ratio = max(ratio, cap / kd.getVal(n, 0));
  }
  cout << label << ": max capture / prior variance = " << ratio << endl;
  expect(bounded, label + ": Nystrom capture <= prior variance");
}

void testKfu()
{
  { // isotropic MLP, non-default parameters, d=1 (D'=2)
    CMlpKern kern(1);
    kern.setParam(1.7, 0); // weight variance
    kern.setParam(0.6, 1); // bias variance
    kern.setParam(2.3, 2); // variance
    gateKernel(kern, {1.7, 0.6}, 2.3, "mlp d=1", 0.05, 9.0, 1e-9);
  }
  { // ARD MLP, d=2 (D'=3)
    CMlpardKern kern(2);
    kern.setParam(0.8, 0);
    kern.setParam(0.5, 1);
    kern.setParam(1.4, 2);
    kern.setParam(1.3, 3);
    kern.setParam(0.4, 4);
    gateKernel(kern, {0.8 * 1.3, 0.8 * 0.4, 0.5}, 1.4, "mlpard d=2", 0.1, 8.0, 1e-6);
  }
}

void testKernelRouting()
{
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 6, 2);
  CInducingHermite ind(V, ks);
  CMatrix X(4, 2), K, Kx;
  for(unsigned int n = 0; n < 4; n++)
  {
    X.setVal(0.3 * n - 0.4, n, 0);
    X.setVal(0.5 - 0.2 * n, n, 1);
  }
  CMlpKern mlp(2);
  mlp.setParam(1.3, 2);
  // E2: a kernel without Hermite features is refused, with a message naming it.
  CRbfKern rbf(2);
  const string e2 = thrown([&] { ind.computeKuf(K, rbf, X); });
  expect(has(e2, "E2") && has(e2, "mlp"), "E2: non-MLP kernel refused with an explicit message");
  // Compound mlp + white routes to the signal component and matches mlp alone.
  CCmpndKern cmp(2);
  cmp.addKern(&mlp);
  CWhiteKern white(2);
  cmp.addKern(&white);
  CMatrix Kmlp, Kcmp;
  ind.computeKuf(Kmlp, mlp, X);
  ind.computeKuf(Kcmp, cmp, X);
  expect(maxAbsDiff(Kmlp, Kcmp) == 0.0, "compound (mlp + white) routes to the signal component");
  // Layout contract: Kuf is M x N, Kux equals Kuf on the same inputs.
  ind.computeKux(Kx, mlp, X);
  expect(Kmlp.getRows() == ind.numInducing() && Kmlp.getCols() == 4 && maxAbsDiff(Kmlp, Kx) == 0.0,
         "Kuf is M x N and Kux equals Kuf on the same inputs");
}

// ---- phase 3: gradients against central finite differences ---------------
double relErr(double a, double b)
{
  return fabs(a - b) / max(1.0, max(fabs(a), fabs(b)));
}

// <gKfu, Kfu> as a function of the kernel parameters and X.
double contraction(const CKern& kern, const CMatrix& X, const CMatrix& V,
                   const vector<unsigned int>& ks, const CMatrix& gKfu)
{
  CMatrix Kfu;
  kern.wsHermiteCrossCompute(Kfu, X, V, ks);
  double acc = 0.0;
  for(unsigned int n = 0; n < Kfu.getRows(); n++)
    for(unsigned int m = 0; m < Kfu.getCols(); m++)
      acc += gKfu.getVal(n, m) * Kfu.getVal(n, m);
  return acc;
}

void gateHooksFd(CKern& kern, const string& label)
{
  const unsigned int d = kern.getInputDim();
  mt19937 rng(11);
  normal_distribution<double> nd(0.0, 1.0);
  CMatrix X(5, d), V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 12, d);
  for(unsigned int n = 0; n < 5; n++)
    for(unsigned int j = 0; j < d; j++)
      X.setVal(0.8 * nd(rng), n, j);
  CMatrix gKfu(5, V.getRows());
  for(unsigned int n = 0; n < 5; n++)
    for(unsigned int m = 0; m < V.getRows(); m++)
      gKfu.setVal(nd(rng), n, m);
  const unsigned int P = kern.getNumParams();
  CMatrix gP(1, P), gX;
  kern.wsHermiteGradParams(gP, X, V, ks, gKfu);
  kern.wsHermiteGradX(gX, X, V, ks, gKfu);
  const double eps = 1e-6;
  double worstP = 0.0, worstX = 0.0;
  for(unsigned int p = 0; p < P; p++)
  {
    const double p0 = kern.getParam(p);
    kern.setParam(p0 + eps, p);
    const double fp = contraction(kern, X, V, ks, gKfu);
    kern.setParam(p0 - eps, p);
    const double fm = contraction(kern, X, V, ks, gKfu);
    kern.setParam(p0, p);
    worstP = max(worstP, relErr(gP.getVal(p), (fp - fm) / (2 * eps)));
  }
  for(unsigned int n = 0; n < 5; n++)
    for(unsigned int j = 0; j < d; j++)
    {
      const double x0 = X.getVal(n, j);
      X.setVal(x0 + eps, n, j);
      const double fp = contraction(kern, X, V, ks, gKfu);
      X.setVal(x0 - eps, n, j);
      const double fm = contraction(kern, X, V, ks, gKfu);
      X.setVal(x0, n, j);
      worstX = max(worstX, relErr(gX.getVal(n, j), (fp - fm) / (2 * eps)));
    }
  cout << label << ": max rel err params = " << worstP << ", X = " << worstX << endl;
  expect(worstP < 1e-6, label + ": wsHermiteGradParams matches finite differences");
  expect(worstX < 1e-6, label + ": wsHermiteGradX matches finite differences");
}

void testHookGradients()
{
  CMlpKern mlp(2);
  mlp.setParam(1.3, 0);
  mlp.setParam(0.7, 1);
  mlp.setParam(1.9, 2);
  gateHooksFd(mlp, "mlp d=2");
  CMlpardKern ard(3);
  ard.setParam(0.9, 0);
  ard.setParam(0.4, 1);
  ard.setParam(1.2, 2);
  ard.setParam(1.5, 3);
  ard.setParam(0.6, 4);
  ard.setParam(2.0, 5);
  gateHooksFd(ard, "mlpard d=3");
  // Compound: params of the white component get zero gradient from K_fu.
  CMlpKern sig(2);
  CWhiteKern white(2);
  CCmpndKern cmp(2);
  cmp.addKern(&sig);
  cmp.addKern(&white);
  gateHooksFd(cmp, "compound mlp+white d=2");
}

// Layer-level: CSvgp with Hermite inducing. Kernel-parameter gradient (including
// the packing guard for zero inducing parameters) and data-X gradient vs FD.
void testSvgpGradients()
{
  const unsigned int N = 18, q = 2;
  mt19937 rng(5);
  normal_distribution<double> nd(0.0, 1.0);
  CMatrix X(N, q), y(N, 1), Z(4, q);
  for(unsigned int i = 0; i < N; i++)
  {
    for(unsigned int d = 0; d < q; d++)
      X.setVal(nd(rng), i, d);
    y.setVal(tanh(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
  }
  Z.zeros();
  CMlpardKern kern(X);
  kern.setParam(1.0, 0);
  kern.setParam(1.0, 1);
  kern.setParam(1.0, 2);
  CSvgp sv(&kern, &X, &y, Z, 4.0, 0);
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 14, q);
  sv.setInducing(new CInducingHermite(V, ks));
  for(unsigned int t = 0; t < 5; t++)
    sv.naturalGradientStep(0.3);
  vector<unsigned int> all(N);
  for(unsigned int i = 0; i < N; i++)
    all[i] = i;

  const unsigned int nOpt = sv.getOptNumParams();
  expect(nOpt == kern.getNumParams() + 1, "Hermite layer reserves no inducing parameters");
  CMatrix g(1, nOpt);
  const double L0 = sv.objectiveGradBatch(all, g);
  CMatrix p(1, nOpt), p0(1, nOpt);
  sv.getOptParams(p0);
  double worst = 0.0;
  const double eps = 1e-6;
  for(unsigned int i = 0; i < nOpt; i++)
  {
    p.deepCopy(p0);
    p.setVal(p0.getVal(i) + eps, i);
    sv.setOptParams(p);
    CMatrix gt(1, nOpt);
    const double Lp = sv.objectiveGradBatch(all, gt);
    p.setVal(p0.getVal(i) - eps, i);
    sv.setOptParams(p);
    const double Lm = sv.objectiveGradBatch(all, gt);
    worst = max(worst, relErr(g.getVal(i), (Lp - Lm) / (2 * eps)));
  }
  sv.setOptParams(p0);
  cout << "CSvgp+Hermite: L0 = " << L0 << ", max rel err opt params = " << worst << endl;
  expect(worst < 1e-5, "CSvgp with Hermite inducing: objective gradient matches finite differences");

  CMatrix XB(X), gK, gZ, gX;
  double gB = 0.0;
  sv.elboGradAt(gK, gZ, gB, &gX, XB, all, 1.0);
  double worstX = 0.0;
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int d = 0; d < q; d++)
    {
      const double x0 = XB.getVal(i, d);
      XB.setVal(x0 + eps, i, d);
      const double Lp = sv.dataTermAt(XB, all, 1.0);
      XB.setVal(x0 - eps, i, d);
      const double Lm = sv.dataTermAt(XB, all, 1.0);
      XB.setVal(x0, i, d);
      worstX = max(worstX, relErr(gX.getVal(i, d), (Lp - Lm) / (2 * eps)));
    }
  cout << "CSvgp+Hermite: max rel err gX = " << worstX << endl;
  expect(worstX < 1e-5, "CSvgp with Hermite inducing: data-X gradient matches finite differences");
}

// ---- phase 4: CGp / CDeepGp wiring, E3, E4, E6, stream round trip -------
struct SparseToy
{
  CMatrix X, y, Xu;
  CMlpKern kern;
  CRbfKern rbf;
  CGaussianNoise noise;
  CGp* model;
  SparseToy(unsigned int N, unsigned int q, unsigned int M, bool useRbf = false)
      : X(N, q), y(N, 1), Xu(M, q), kern(q), rbf(q), noise(&y), model(0)
  {
    mt19937 rng(21);
    normal_distribution<double> nd(0.0, 1.0);
    for(unsigned int i = 0; i < N; i++)
    {
      for(unsigned int j = 0; j < q; j++)
        X.setVal(nd(rng), i, j);
      y.setVal(tanh(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
    }
    for(unsigned int m = 0; m < M; m++)
      for(unsigned int j = 0; j < q; j++)
        Xu.setVal(X.getVal(m % N, j), m, j);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CKern* k = useRbf ? (CKern*)&rbf : (CKern*)&kern;
    model = new CGp(k, &noise, &X, CGp::DTCVAR, M, 0);
    model->setInducingVals(Xu);
    model->setBetaVal(10.0);
  }
  ~SparseToy() { delete model; }
};

void testCGpWiring()
{
  // E4: refuse while inducing is free, and leave the model unchanged.
  {
    SparseToy t(20, 2, 8);
    const string e4 = thrown([&] { t.model->setInducingType("hermite"); });
    expect(has(e4, "E4") && t.model->getInducingType() == "points",
           "E4: hermite refused while inducing is not fixed; type unchanged");
    t.model->setInducingFixed(true);
    t.model->setInducingType("hermite");
    expect(t.model->getInducingType() == "hermite" && t.model->getInducing()->numInducing() == 8,
           "CGp hermite: delivered numInducing equals requested numActive");
    expect(has(thrown([&] { t.model->setInducingFixed(false); }), "E4"),
           "E4: unfixing a hermite model refused");
    // E3
    expect(has(thrown([&] { t.model->getInducing()->setWindowWidth(0.3); }), "E3"),
           "E3: window width refused");
    expect(has(thrown([&] { t.model->getInducing()->setLearnWindowWidth(true); }), "E3"),
           "E3: learn window width refused");
    expect(!thrown([&] { t.model->setByPath("inducing.window_width", 0.3); }).empty(),
           "E3: window_width path refused");
    expect(!has(thrown([&] { t.model->getInducing()->setLearnWindowWidth(false); }), "E3"),
           "learn_window_width=false is not an error");
    // The optimiser vector holds no inducing parameters.
    const double ll = t.model->logLikelihood();
    expect(std::isfinite(ll), "CGp hermite: logLikelihood finite");
    const unsigned int nOpt = t.model->getOptNumParams();
    CMatrix g(1, nOpt), p(1, nOpt), p0(1, nOpt);
    t.model->computeObjectiveGradParams(g);
    t.model->getOptParams(p0);
    double worst = 0.0;
    const double eps = 1e-6;
    for(unsigned int i = 0; i < nOpt; i++)
    {
      p.deepCopy(p0);
      p.setVal(p0.getVal(i) + eps, i);
      t.model->setOptParams(p);
      const double fp = t.model->computeObjectiveVal();
      p.setVal(p0.getVal(i) - eps, i);
      t.model->setOptParams(p);
      const double fm = t.model->computeObjectiveVal();
      worst = max(worst, relErr(g.getVal(i), (fp - fm) / (2 * eps)));
    }
    t.model->setOptParams(p0);
    cout << "CGp+Hermite (DTCVAR): nOpt=" << nOpt << ", max rel err = " << worst << endl;
    expect(worst < 1e-5, "CGp hermite: objective gradient matches finite differences");
  }
  // E2 and E1 through the CGp setter, each leaving the type unchanged.
  {
    SparseToy t(20, 2, 8, true);
    t.model->setInducingFixed(true);
    const string e2 = thrown([&] { t.model->setInducingType("hermite"); });
    expect(has(e2, "E2") && t.model->getInducingType() == "points",
           "E2: RBF kernel refused at the CGp setter; type unchanged");
  }
  {
    SparseToy t(30, 1, 20);
    t.model->setInducingFixed(true);
    const string e1 = thrown([&] { t.model->setInducingType("hermite"); });
    expect(has(e1, "E1") && t.model->getInducingType() == "points",
           "E1: M=20 > capacity 16 at d=1 refused at the CGp setter; type unchanged");
  }
}

void testCGpStream()
{
  SparseToy t(20, 2, 8);
  t.model->setInducingFixed(true);
  t.model->setInducingType("hermite");
  ostringstream out;
  writeGpToStream(*t.model, out);
  const string blob = out.str();
  expect(has(blob, "inducingType=hermite") && has(blob, "hermiteDirections=") &&
             has(blob, "hermiteDegrees="),
         "stream writes the hermite block (V and ks)");
  istringstream in(blob);
  CGp* round = readGpFromStream(in);
  expect(round != 0 && round->getInducingType() == "hermite" &&
             round->getInducing()->numInducing() == 8,
         "stream restores hermite type and M");
  if(round != 0)
  {
    const CInducingHermite* a = dynamic_cast<const CInducingHermite*>(t.model->getInducing());
    const CInducingHermite* b = dynamic_cast<const CInducingHermite*>(round->getInducing());
    expect(a && b && a->degrees() == b->degrees() &&
               maxAbsDiff(a->directions(), b->directions()) < 1e-12,
           "stream round trip preserves V and ks");
    CMatrix K1(8, 8), K2(8, 8), Ku1, Ku2;
    a->computeKuu(K1, *t.model->getKernel());
    b->computeKuu(K2, *round->getKernel());
    a->computeKux(Ku1, *t.model->getKernel(), t.X);
    b->computeKux(Ku2, *round->getKernel(), t.X);
    expect(maxAbsDiff(K1, K2) < ndlutil::MATCHTOL && maxAbsDiff(Ku1, Ku2) < ndlutil::MATCHTOL,
           "stream round trip: K_uu and K_ux equal within MATCHTOL");
    delete round;
  }
  // E6 at stream level.
  {
    string noDeg = blob.substr(0, blob.find("hermiteDegrees="));
    istringstream bin(noDeg);
    const string e = thrown([&] { delete readGpFromStream(bin); });
    expect(has(e, "E6"), "E6: missing hermiteDegrees block");
  }
  {
    string unfixed = blob;
    const size_t p = unfixed.find("fixInducing=1");
    expect(p != string::npos, "blob has fixInducing=1");
    if(p != string::npos)
    {
      unfixed.replace(p, 13, "fixInducing=0");
      istringstream bin(unfixed);
      const string e = thrown([&] { delete readGpFromStream(bin); });
      expect(has(e, "E6"), "E6: hermite with fixInducing=0");
    }
  }
  // E6 at data level.
  {
    CMatrix V;
    vector<unsigned int> ks;
    CInducingHermite::makeLayout(V, ks, 6, 2);
    CInducingHermite good(V, ks);
    CMatrix row = good.degreesAsRow();
    expect(thrown([&] { delete CInducingHermite::fromStreamData(V, row, 6, 2); }).empty(),
           "fromStreamData accepts a consistent block");
    expect(has(thrown([&] { delete CInducingHermite::fromStreamData(V, row, 7, 2); }), "E6"),
           "E6: numActive disagrees with V rows");
    expect(has(thrown([&] { delete CInducingHermite::fromStreamData(V, row, 6, 3); }), "E6"),
           "E6: V columns disagree with inputDim+1");
    CMatrix shortRow(1, 5);
    shortRow.zeros();
    expect(has(thrown([&] { delete CInducingHermite::fromStreamData(V, shortRow, 6, 2); }), "E6"),
           "E6: ks length disagrees");
    CMatrix even(row);
    even.setVal(2.0, 0, 3);
    expect(has(thrown([&] { delete CInducingHermite::fromStreamData(V, even, 6, 2); }), "E6"),
           "E6: even degree");
    CMatrix frac(row);
    frac.setVal(3.5, 0, 3);
    expect(has(thrown([&] { delete CInducingHermite::fromStreamData(V, frac, 6, 2); }), "E6"),
           "E6: non-integer degree");
    CMatrix dup(V);
    for(unsigned int j = 0; j < dup.getCols(); j++)
      dup.setVal(V.getVal(0, j), 1, j);
    CMatrix row2 = row;
    row2.setVal(row.getVal(0, 0), 0, 1);
    expect(has(thrown([&] { delete CInducingHermite::fromStreamData(dup, row2, 6, 2); }), "E6"),
           "E6: stored layout that fails the audit (duplicate window)");
  }
  // Old model files (no hermite block) still load.
  {
    SparseToy p(20, 2, 8);
    p.model->setInducingFixed(true);
    ostringstream o2;
    writeGpToStream(*p.model, o2);
    istringstream i2(o2.str());
    CGp* r = readGpFromStream(i2);
    expect(r != 0 && r->getInducingType() == "points" && !has(o2.str(), "hermite"),
           "points model stream (no hermite block) still loads");
    delete r;
  }
}

void testDeepWiring()
{
  const unsigned int N = 14, M = 6;
  mt19937 rng(3);
  normal_distribution<double> nd(0.0, 1.0);
  CMatrix X(N, 1), y(N, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    X.setVal(nd(rng), i, 0);
    y.setVal(tanh(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
  }
  { // E2 at the layer setter: type recorded unchanged.
    CRbfKern r0(1), r1(1);
    CDeepGp dgp(X, y, vector<CKern*>{&r0, &r1}, vector<unsigned int>{1u, 1u, 1u},
                vector<unsigned int>{M, M}, 4.0, 0.5, 0);
    const string e = thrown([&] { dgp.setLayerInducingType(0, "hermite"); });
    expect(has(e, "E2") && dgp.getLayerInducingType(0) == "points",
           "CDeepGp: E2 at the layer setter leaves the recorded type unchanged");
  }
  CMlpKern k0(1), k1(1);
  CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1}, vector<unsigned int>{1u, 1u, 1u},
              vector<unsigned int>{M, M}, 4.0, 0.5, 0);
  dgp.setLayerInducingType(0, "hermite");
  dgp.setLayerInducingType(1, "hermite");
  dgp.setNumSamples(1);
  dgp.setSeed(5);
  expect(has(thrown([&] { dgp.setLayerLearnWindowWidth(0, true, false); }), "E3"),
         "CDeepGp: E3 for learn_window_width on a hermite layer");
  expect(dgp.layer(0).getInducing()->numInducing() == M, "CDeepGp hermite: delivered M");
  CMatrix pBefore;
  dgp.getStochasticParams(pBefore);
  const double L0 = dgp.elboEstimate();
  ostringstream os;
  dgp.toStream(os);
  expect(has(os.str(), "inducingType=hermite"), "CDeepGp stream records hermite layers");
  istringstream is(os.str());
  CMatrix Xt(1, 1), Yt(1, 1);
  CRbfKern kt(1);
  CDeepGp loaded(Xt, Yt, vector<CKern*>{&kt}, vector<unsigned int>{1u, 1u},
                 vector<unsigned int>{1u}, 1.0, 1.0, 0);
  loaded.fromStream(is);
  bool same = loaded.getLayerInducingType(0) == "hermite" && loaded.getLayerInducingType(1) == "hermite";
  for(unsigned int l = 0; l < 2 && same; l++)
  {
    const CInducingHermite* a = dynamic_cast<const CInducingHermite*>(dgp.layer(l).getInducing());
    const CInducingHermite* b = dynamic_cast<const CInducingHermite*>(loaded.layer(l).getInducing());
    same = a && b && a->degrees() == b->degrees() &&
           maxAbsDiff(a->directions(), b->directions()) < 1e-12;
  }
  expect(same, "CDeepGp stream round trip preserves V and ks per layer");
  loaded.setNumSamples(1);
  loaded.setSeed(5);
  const double L1 = loaded.elboEstimate();
  cout << "CDeepGp+Hermite ELBO before/after stream: " << L0 << " / " << L1 << endl;
  expect(fabs(L0 - L1) < 1e-6, "CDeepGp stream round trip: ELBO equal");

  // Layer gradients (deferred from phase 3): frozen noise, all stochastic params.
  {
    CMatrix eps(N, 1);
    for(unsigned int i = 0; i < N; i++)
      eps.setVal(nd(rng), i, 0);
    dgp.setFrozenEps(eps);
    vector<unsigned int> all(N);
    for(unsigned int i = 0; i < N; i++)
      all[i] = i;
    CMatrix gA;
    dgp.objectiveGradBatch(all, gA);
    CMatrix p0;
    dgp.getStochasticParams(p0);
    const double h = 1e-5;
    double worst = 0.0;
    for(unsigned int i = 0; i < p0.getCols(); i++)
    {
      CMatrix pp(p0), pm(p0), gT;
      pp.setVal(p0.getVal(i) + h, i);
      pm.setVal(p0.getVal(i) - h, i);
      dgp.setStochasticParams(pp);
      const double Lp = dgp.objectiveGradBatch(all, gT);
      dgp.setStochasticParams(pm);
      const double Lm = dgp.objectiveGradBatch(all, gT);
      worst = max(worst, relErr(gA.getVal(i), (Lp - Lm) / (2 * h)));
    }
    dgp.setStochasticParams(p0);
    cout << "CDeepGp+Hermite: " << p0.getCols() << " stochastic params, max rel err = " << worst << endl;
    expect(worst < 1e-4, "CDeepGp with Hermite layers: gradient matches finite differences");
  }
}

void testGating()
{
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 6, 2);
  CInducingHermite ind(V, ks);
  CMatrix K;
  expect(!thrown([&] { ind.setCentres(V); }).empty(), "setCentres refused (windows fixed)");
}

// ---- phase 5: W3/W4, delivered-layout description ------------------------
bool anyHas(const vector<string>& v, const string& needle)
{
  for(const string& s : v)
    if(has(s, needle))
      return true;
  return false;
}

void testPhase5Warnings()
{
  { // Well-conditioned control: nothing to warn about, layout always described.
    SparseToy t(20, 2, 8);
    t.model->setInducingFixed(true);
    t.model->setInducingType("hermite");
    expect(t.model->hermiteWarnings().empty(), "control: no warnings at M=8, d=2");
    const ModelDescription d = t.model->describe();
    bool layout = false, fixedLine = false;
    for(const string& h : d.headerLines)
    {
      layout = layout || has(h, "Hermite layout: ");
      fixedLine = fixedLine || has(h, "no free parameters");
    }
    expect(layout && fixedLine, "describe() shows layout and the fixed-by-construction line");
    const CInducingHermite* ph = dynamic_cast<const CInducingHermite*>(t.model->getInducing());
    expect(ph != 0, "hermite inducing object present");
    if(ph)
    {
      const double frac = ph->capturedVariance(t.kern, t.X);
      expect(frac > 0.0 && frac <= 1.0 + 1e-9, "captured variance fraction in (0, 1]");
      expect(!anyHas(ph->warnings(t.kern, t.X), "W3"), "W3 silent at the default threshold here");
      expect(anyHas(ph->warnings(t.kern, t.X, 1.01), "W3"), "W3 fires when threshold is above the fraction");
      expect(frac > CInducingHermite::capturedVarianceWarnThreshold(),
             "calibrated W3 threshold is below the fraction at M >= D'");
    }
  }
  { // White noise is not part of the signal the windows represent: it must not
    // change the captured-variance fraction (it once inflated the denominator).
    SparseToy t(20, 2, 8);
    t.model->setInducingFixed(true);
    t.model->setInducingType("hermite");
    const CInducingHermite* ph = dynamic_cast<const CInducingHermite*>(t.model->getInducing());
    CMlpKern mlp(2);
    CWhiteKern white(2);
    white.setVariance(5.0);
    CCmpndKern cmp(2);
    cmp.addKern(&mlp);
    cmp.addKern(&white);
    const double alone = ph->capturedVariance(mlp, t.X);
    const double withWhite = ph->capturedVariance(cmp, t.X);
    expect(fabs(alone - withWhite) < 1e-10, "captured variance ignores white-noise components");
  }
  { // W1 and W2 reach CGp::hermiteWarnings at d=1 full capacity.
    SparseToy t(40, 1, 16);
    t.model->setInducingFixed(true);
    t.model->setInducingType("hermite");
    const vector<string> w = t.model->hermiteWarnings();
    expect(anyHas(w, "W1") && anyHas(w, "W2"), "CGp: W1 and W2 at d=1, M=16");
  }
  { // Non-hermite model: no hermite warnings.
    SparseToy t(20, 2, 8, true);
    expect(t.model->hermiteWarnings().empty(), "points model has no hermite warnings");
  }
  { // W4: only hidden layers of a multi-layer deep GP.
    const unsigned int N = 14;
    mt19937 rng(3);
    normal_distribution<double> nd(0.0, 1.0);
    CMatrix X(N, 1), y(N, 1);
    for(unsigned int i = 0; i < N; i++)
    {
      X.setVal(nd(rng), i, 0);
      y.setVal(tanh(X.getVal(i, 0)), i, 0);
    }
    CMlpKern k0(1), k1(1);
    CDeepGp two(X, y, vector<CKern*>{&k0, &k1}, vector<unsigned int>{1u, 1u, 1u},
                vector<unsigned int>{6, 6}, 4.0, 0.5, 0);
    expect(two.hermiteWarnings().empty(), "deep GP with points layers: no hermite warnings");
    two.setLayerInducingType(0, "hermite");
    two.setLayerInducingType(1, "hermite");
    const vector<string> w = two.hermiteWarnings();
    expect(anyHas(w, "W4") && anyHas(w, "hidden layer 0"), "W4 on hidden layer 0");
    int w4 = 0;
    for(const string& s : w)
      w4 += has(s, "W4") ? 1 : 0;
    expect(w4 == 1, "W4 only for the hidden layer, not the output layer");
    CMlpKern kk(1);
    CDeepGp one(X, y, vector<CKern*>{&kk}, vector<unsigned int>{1u, 1u},
                vector<unsigned int>{6}, 4.0, 0.5, 0);
    one.setLayerInducingType(0, "hermite");
    expect(!anyHas(one.hermiteWarnings(), "W4"), "single-layer deep GP: no W4");
  }
}

} // namespace

int main()
{
  try
  {
    cout << "==== testInducingHermite (CIP-003D phases 1-5) ====" << endl;
    testCapacityTable();
    testSweep();
    testE1Message();
    testDeterminism();
    testAudit();
    testWarnings();
    testKuu();
    testKfu();
    testKernelRouting();
    testHookGradients();
    testSvgpGradients();
    testCGpWiring();
    testCGpStream();
    testDeepWiring();
    testPhase5Warnings();
    testGating();
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
