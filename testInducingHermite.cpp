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

void testGating()
{
  CMatrix V;
  vector<unsigned int> ks;
  CInducingHermite::makeLayout(V, ks, 6, 2);
  CInducingHermite ind(V, ks);
  CMlpKern kern(2);
  CMatrix X(4, 2), K;
  X.zeros();
  expect(!thrown([&] { ind.computeKuf(K, kern, X); }).empty(),
         "computeKuf throws until phase 2");
  expect(!thrown([&] { ind.setCentres(V); }).empty(), "setCentres refused (windows fixed)");
}

} // namespace

int main()
{
  try
  {
    cout << "==== testInducingHermite (CIP-003D phase 1) ====" << endl;
    testCapacityTable();
    testSweep();
    testE1Message();
    testDeterminism();
    testAudit();
    testWarnings();
    testKuu();
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
