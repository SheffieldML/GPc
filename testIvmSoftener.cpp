/* CIP-0027: deeper IVM White-softener regression checks.
 *
 * Complements the short Jaccard gate in testCip0027Perf with:
 *   - getWhite() / White-in-K selection effect
 *   - fixed probit σ² (bias-only trainable params)
 *   - longer unitsquare budget (stronger active-set divergence)
 *   - RBF+White stream round-trip
 *
 *   make testIvmSoftener && ./testIvmSoftener
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "CClctrl.h"
#include "CIvm.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlutil.h"

using namespace std;

namespace {

class SoftCtrl : public CClctrl {
 public:
  SoftCtrl(int c, char** v) : CClctrl(c, v) {}
  void helpInfo() {}
  void helpHeader() {}
};

enum CompKind { COMP_RBW = 0, COMP_RBF = 1, COMP_RW = 2 };

int failTrue(bool cond, const string& what)
{
  if(cond)
  {
    cout << what << " passes." << endl;
    return 0;
  }
  cout << "FAILURE: " << what << "." << endl;
  return 1;
}

int checkAbove(const string& label, double val, double floor)
{
  if(val >= floor)
  {
    cout << label << " passes (" << val << " >= " << floor << ")." << endl;
    return 0;
  }
  cout << "FAILURE: " << label << ": " << val << " < " << floor << endl;
  return 1;
}

int checkBelow(const string& label, double val, double ceil)
{
  if(val <= ceil)
  {
    cout << label << " passes (" << val << " <= " << ceil << ")." << endl;
    return 0;
  }
  cout << "FAILURE: " << label << ": " << val << " > " << ceil << endl;
  return 1;
}

int checkClose(const string& label, double a, double b, double tol = ndlutil::MATCHTOL)
{
  const double diff = fabs(a - b);
  if(diff <= tol)
  {
    cout << label << " passes." << endl;
    return 0;
  }
  cout << "FAILURE: " << label << ": |a-b|=" << diff
       << " (a=" << a << ", b=" << b << ")" << endl;
  return 1;
}

double jaccard(const vector<unsigned int>& a, const vector<unsigned int>& b)
{
  set<unsigned int> sa(a.begin(), a.end());
  set<unsigned int> sb(b.begin(), b.end());
  unsigned int inter = 0;
  for(set<unsigned int>::const_iterator it = sa.begin(); it != sa.end(); ++it)
  {
    if(sb.count(*it))
      inter++;
  }
  const unsigned int uni = (unsigned int)(sa.size() + sb.size() - inter);
  if(uni == 0)
    return 1.0;
  return (double)inter / (double)uni;
}

void buildKern(CCmpndKern& kern, const CMatrix& X, CompKind kind)
{
  kern.addKern(new CRbfKern(X));
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
  if(kind == COMP_RBW)
  {
    kern.addKern(new CBiasKern(X));
    kern.addKern(new CWhiteKern(X));
  }
  else if(kind == COMP_RW)
  {
    kern.addKern(new CWhiteKern(X));
  }
}

vector<unsigned int> activeOf(CIvm& model)
{
  vector<unsigned int> a(model.getNumActive());
  for(unsigned int i = 0; i < model.getNumActive(); i++)
    a[i] = model.getActivePoint(i);
  return a;
}

struct SoftRun {
  vector<unsigned int> active;
  double ll;
  double white;
};

SoftRun runComp(CompKind kind, CMatrix& X, CMatrix& y,
                unsigned int M, unsigned int seed,
                unsigned int outer, unsigned int ki, unsigned int ni)
{
  ndlutil::init_genrand(seed);
  CCmpndKern kern(X);
  buildKern(kern, X, kind);
  CProbitNoise noise(&y);
  for(unsigned int j = 0; j < noise.getOutputDim(); j++)
    noise.setParam(0.0, j);
  CIvm model(&X, &y, &kern, &noise, CIvm::ENTROPY, M, 0);
  model.selectPoints();
  model.optimise(outer, ki, ni);
  SoftRun r;
  r.active = activeOf(model);
  r.ll = model.logLikelihood(y, X);
  r.white = 0.0;
  try { r.white = kern.getByPath("white.variance"); } catch(...) {}
  return r;
}

int testGetWhiteAndProbitFixed(SoftCtrl& ctrl)
{
  cout << "==== IVM getWhite + fixed probit σ² ====" << endl;
  int fail = 0;
  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/unitsquaregp.svml");

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CWhiteKern(X));
  kern.setByPath("white.variance", 0.37);
  fail += checkClose("compound getWhite == white.variance", kern.getWhite(), 0.37);

  CProbitNoise noise(&y);
  fail += failTrue(noise.getNumParams() == noise.getOutputDim(),
                   "probit trainable params are bias-only");
  fail += checkClose("probit sigma2 init 1e-6", noise.getSigma2(), 1e-6);

  ndlutil::init_genrand(7);
  CIvm model(&X, &y, &kern, &noise, CIvm::ENTROPY, 20, 0);
  model.selectPoints();
  model.optimise(2, 20, 10);
  fail += checkClose("probit sigma2 fixed after optimise", noise.getSigma2(), 1e-6);
  fail += failTrue(kern.getWhite() > 0.0, "White softener still present after optimise");
  return fail;
}

int testWhiteChangesSelection(SoftCtrl& ctrl)
{
  cout << "==== IVM White changes active-set selection ====" << endl;
  int fail = 0;
  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/unitsquaregp.svml");
  const unsigned int M = 30;
  const unsigned int seed = 11;

  auto selectOnly = [&](bool withWhite, double whiteVar) -> vector<unsigned int> {
    ndlutil::init_genrand(seed);
    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    kern.setByPath("rbf.lengthscale", 1.0);
    kern.setByPath("rbf.variance", 1.0);
    if(withWhite)
    {
      kern.addKern(new CWhiteKern(X));
      kern.setByPath("white.variance", whiteVar);
    }
    CProbitNoise noise(&y);
    for(unsigned int j = 0; j < noise.getOutputDim(); j++)
      noise.setParam(0.0, j);
    CIvm model(&X, &y, &kern, &noise, CIvm::ENTROPY, M, 0);
    model.selectPoints();
    return activeOf(model);
  };

  const vector<unsigned int> withW = selectOnly(true, 0.5);
  const vector<unsigned int> noW = selectOnly(false, 0.0);
  const double jac = jaccard(withW, noW);
  // White enters updateM via getWhite(); selection must diverge from RBF-only.
  fail += checkBelow("selectPoints Jaccard White vs RBF-only", jac, 0.85);
  return fail;
}

int testLongBudgetUnitsquare(SoftCtrl& ctrl)
{
  cout << "==== IVM long-budget unitsquare softener gate ====" << endl;
  int fail = 0;
  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/unitsquaregp.svml");
  const unsigned int N = X.getRows();
  // Heavier than the short B′ gate (M=50, 3×40×15) but lighter than a full
  // tutorial bake-off — still enough for RBF-only to diverge hard.
  const unsigned int M = min(80u, N / 5);
  const unsigned int seed = 42;

  SoftRun base = runComp(COMP_RBW, X, y, M, seed, 4, 50, 15);
  SoftRun rbf = runComp(COMP_RBF, X, y, M, seed, 4, 50, 15);
  SoftRun rw = runComp(COMP_RW, X, y, M, seed, 4, 50, 15);

  const double jacRbf = jaccard(base.active, rbf.active);
  const double jacRw = jaccard(base.active, rw.active);
  // Observed: RBF-only ≪ 0.3; RBF+White ≈ 1.0; LL_RW ≈ LL_RBW.
  fail += checkBelow("long-budget RBF-only Jaccard", jacRbf, 0.35);
  fail += checkAbove("long-budget RBF+White Jaccard", jacRw, 0.80);
  fail += checkBelow("long-budget |LL_RW - LL_RBW|",
                     fabs(rw.ll - base.ll), 1.0);
  fail += failTrue(fabs(rbf.ll - base.ll) > 0.5,
                   "long-budget RBF-only LL diverges from status quo");
  return fail;
}

int testStreamRoundTrip(SoftCtrl& ctrl)
{
  cout << "==== IVM RBF+White stream round-trip ====" << endl;
  int fail = 0;
  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/unitsquaregp.svml");

  ndlutil::init_genrand(3);
  CCmpndKern kern(X);
  buildKern(kern, X, COMP_RW);
  CProbitNoise noise(&y);
  for(unsigned int j = 0; j < noise.getOutputDim(); j++)
    noise.setParam(0.0, j);
  CIvm model(&X, &y, &kern, &noise, CIvm::ENTROPY, 25, 0);
  model.selectPoints();
  model.optimise(2, 15, 10);

  const string path = "crap_ivm_softener";
  model.toFile(path);
  CIvm loaded;
  loaded.fromFile(path);
  // Full equals() also compares inactiveSet, which the stream format does not
  // currently persist — gate the softener-relevant payload instead.
  fail += failTrue(loaded.getNumActive() == model.getNumActive(),
                   "stream numActive matches");
  bool actOk = true;
  for(unsigned int i = 0; i < model.getNumActive(); i++)
  {
    if(loaded.getActivePoint(i) != model.getActivePoint(i))
      actOk = false;
  }
  fail += failTrue(actOk, "stream activeSet matches");
  fail += failTrue(kern.getWhite() > 0.0, "trained model retains White softener");
  return fail;
}

} // namespace

int main(int argc, char** argv)
{
  SoftCtrl ctrl(argc, argv);
  ctrl.setVerbosity(0);
  ctrl.setSeed(1);

  int fail = 0;
  try
  {
    fail += testGetWhiteAndProbitFixed(ctrl);
    fail += testWhiteChangesSelection(ctrl);
    fail += testLongBudgetUnitsquare(ctrl);
    fail += testStreamRoundTrip(ctrl);
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
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
