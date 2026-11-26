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

void testGating()
{
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 6, 2);
  CInducingHermite ind(V, ks);
  CMatrix K;
  expect(!thrown([&] { ind.setCentres(V); }).empty(), "setCentres refused (windows fixed)");
}

} // namespace

int main()
{
  try
  {
    cout << "==== testInducingHermite (CIP-003D phases 1-2) ====" << endl;
    testCapacityTable();
    testSweep();
    testE1Message();
    testDeterminism();
    testAudit();
    testWarnings();
    testKuu();
    testKfu();
    testKernelRouting();
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
