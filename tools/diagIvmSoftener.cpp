/* CIP-0027 Phase B′: IVM kernel-composition softener matrix on unitsquare.
 *
 * Compares:
 *   (1) RBF + Bias + White  — status quo CLI default
 *   (2) RBF only            — no softener in K
 *   (3) RBF + White         — keep White, drop Bias
 *
 * Metrics after selectPoints + short optimise: train error, white.variance,
 * active-set Jaccard vs (1).
 *
 *   make diagIvmSoftener && ./diagIvmSoftener
 */
#include <algorithm>
#include <cmath>
#include <iomanip>
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

class DiagCtrl : public CClctrl {
 public:
  DiagCtrl(int c, char** v) : CClctrl(c, v) {}
  void helpInfo() {}
  void helpHeader() {}
};

enum CompKind { COMP_RBW = 0, COMP_RBF = 1, COMP_RW = 2 };

const char* compName(CompKind k)
{
  switch(k)
  {
  case COMP_RBW: return "RBF+Bias+White";
  case COMP_RBF: return "RBF-only";
  case COMP_RW:  return "RBF+White";
  }
  return "?";
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

double trainError(CIvm& model, const CMatrix& X, const CMatrix& y)
{
  CMatrix yPred(y.getRows(), y.getCols());
  model.out(yPred, X);
  unsigned int wrong = 0;
  for(unsigned int i = 0; i < y.getRows(); i++)
  {
    const double t = y.getVal(i, 0);
    const double p = yPred.getVal(i, 0);
    if((t > 0.0 && p <= 0.0) || (t <= 0.0 && p > 0.0))
      wrong++;
  }
  return (double)wrong / (double)y.getRows();
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

struct RunResult {
  string name;
  double err;
  double white;
  double biasKern;
  double ll;
  vector<unsigned int> active;
};

RunResult runOne(CompKind kind, CMatrix& X, CMatrix& y,
                 unsigned int numActive, unsigned int seed)
{
  ndlutil::init_genrand(seed); // reset RNG for fair active-set comparison
  CCmpndKern kern(X);
  buildKern(kern, X, kind);
  CProbitNoise noise(&y);
  for(unsigned int j = 0; j < noise.getOutputDim(); j++)
    noise.setParam(0.0, j); // bias offsets (probit trainable params)

  CIvm model(&X, &y, &kern, &noise, CIvm::ENTROPY, numActive, 0);
  model.selectPoints();
  // Several outer cycles so White / RBF can separate from RBF-only.
  model.optimise(3, 40, 15);

  RunResult r;
  r.name = compName(kind);
  r.err = trainError(model, X, y);
  r.ll = model.logLikelihood(y, X);
  r.white = 0.0;
  r.biasKern = 0.0;
  try { r.white = kern.getByPath("white.variance"); } catch(...) {}
  try { r.biasKern = kern.getByPath("bias.variance"); } catch(...) {}
  r.active.resize(model.getNumActive());
  for(unsigned int i = 0; i < model.getNumActive(); i++)
    r.active[i] = model.getActivePoint(i);
  return r;
}

} // namespace

int main(int argc, char** argv)
{
  DiagCtrl ctrl(argc, argv);
  ctrl.setVerbosity(0);
  ctrl.setSeed(1);

  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/unitsquaregp.svml");

  const unsigned int N = X.getRows();
  const unsigned int M = min(50u, N / 4);
  cout << "CIP-0027 B′ IVM softener matrix (unitsquare N=" << N
       << " M=" << M << ")" << endl;
  cout << setprecision(6) << fixed;

  const unsigned int seed = 42;
  RunResult base = runOne(COMP_RBW, X, y, M, seed);
  RunResult rbf = runOne(COMP_RBF, X, y, M, seed);
  RunResult rw = runOne(COMP_RW, X, y, M, seed);

  cout << "\nconfig              err     white      biasK     LL        Jaccard@(1)\n";
  const RunResult* rows[3] = { &base, &rbf, &rw };
  for(int i = 0; i < 3; i++)
  {
    const RunResult& r = *rows[i];
    const double jac = jaccard(base.active, r.active);
    cout << left << setw(18) << r.name
         << right << setw(8) << r.err
         << setw(10) << r.white
         << setw(10) << r.biasKern
         << setw(10) << r.ll
         << setw(12) << jac
         << endl;
  }

  cout << "\nDecision hints (CIP-0027):\n";
  cout << "  Keep White if RBF-only err >> status-quo err or Jaccard << 1.\n";
  cout << "  Drop Bias if RBF+White ≈ status quo on err and Jaccard.\n";
  return 0;
}
