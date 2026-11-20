/* CIP-0027 Phase C: harden dual-noise policy + algorithm performance gates.
 *
 * Not helper-only: trains FTC / sparse GP on sinc and IVM softener on
 * unitsquare, then asserts LL / RMSE / Jaccard floors that catch regressions
 * in the likelihood-owned noise path.
 *
 *   make testCip0027Perf && ./testCip0027Perf
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "CClctrl.h"
#include "CGp.h"
#include "CIvm.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlutil.h"

using namespace std;

namespace {

class PerfCtrl : public CClctrl {
 public:
  PerfCtrl(int c, char** v) : CClctrl(c, v) {}
  void helpInfo() {}
  void helpHeader() {}
};

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

double rmsePred(CGp& gp, const CMatrix& Xte, const CMatrix& yte)
{
  CMatrix mu(yte.getRows(), 1);
  CMatrix var(yte.getRows(), 1);
  gp.posteriorMeanVar(mu, var, Xte);
  double sse = 0.0;
  for(unsigned int i = 0; i < yte.getRows(); i++)
  {
    const double d = mu.getVal(i, 0) - yte.getVal(i, 0);
    sse += d * d;
  }
  return sqrt(sse / (double)yte.getRows());
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

int testDualNoisePolicy()
{
  cout << "==== Phase C dual-noise policy ====" << endl;
  int fail = 0;
  CMatrix X(6, 1);
  CMatrix y(6, 1);
  for(unsigned int i = 0; i < 6; i++)
  {
    X.setVal(-1.0 + 2.0 * (double)i / 5.0, i, 0);
    y.setVal(sin(3.0 * X.getVal(i, 0)), i, 0);
  }

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CWhiteKern(X));
  CGaussianNoise noise(&y);
  noise.setBias(0.0);
  CGp gp(&kern, &noise, &X, CGp::FTC, 0, 0);
  // Free White + free likelihood.variance must throw on optimise / grad.
  bool threw = false;
  try
  {
    gp.optimise(1);
  }
  catch(ndlexceptions::Error&)
  {
    threw = true;
  }
  fail += failTrue(threw, "dual free noise throws without allow");

  gp.fixLikelihoodVariance(true);
  bool okFixed = true;
  try
  {
    gp.optimise(2);
  }
  catch(ndlexceptions::Error&)
  {
    okFixed = false;
  }
  fail += failTrue(okFixed, "legacy fixLikelihoodVariance allows optimise");

  CCmpndKern kern2(X);
  kern2.addKern(new CRbfKern(X));
  kern2.addKern(new CWhiteKern(X));
  CGaussianNoise noise2(&y);
  noise2.setBias(0.0);
  CGp gpAllow(&kern2, &noise2, &X, CGp::FTC, 0, 0);
  gpAllow.setAllowWhiteAndLikelihoodNoise(true);
  bool okAllow = true;
  try
  {
    gpAllow.optimise(2);
  }
  catch(ndlexceptions::Error&)
  {
    okAllow = false;
  }
  fail += failTrue(okAllow, "allowWhiteAndLikelihoodNoise permits optimise");
  return fail;
}

int testFtcSincPerf(PerfCtrl& ctrl)
{
  cout << "==== FTC sinc algorithm performance ====" << endl;
  int fail = 0;
  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/sinc.csv");
  const unsigned int N = X.getRows();
  const unsigned int Nte = 8;
  const unsigned int Ntr = N - Nte;
  CMatrix Xtr(Ntr, X.getCols());
  CMatrix ytr(Ntr, 1);
  CMatrix Xte(Nte, X.getCols());
  CMatrix yte(Nte, 1);
  for(unsigned int i = 0; i < Ntr; i++)
  {
    Xtr.copyRowRow(i, X, i);
    ytr.copyRowRow(i, y, i);
  }
  for(unsigned int i = 0; i < Nte; i++)
  {
    Xte.copyRowRow(i, X, Ntr + i);
    yte.copyRowRow(i, y, Ntr + i);
  }

  CCmpndKern kern(Xtr);
  kern.addKern(new CRbfKern(Xtr));
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
  CGaussianNoise noise(&ytr);
  noise.setBias(0.0);
  CGp gp(&kern, &noise, &Xtr, CGp::FTC, 0, 0);
  gp.setLikelihoodVariance(1.0);
  const double ll0 = gp.logLikelihood();
  gp.setDefaultOptimiser(CGp::SCG);
  gp.optimise(40);
  const double ll1 = gp.logLikelihood();
  const double s2 = gp.getLikelihoodVariance();
  const double rmse = rmsePred(gp, Xte, yte);

  fail += failTrue(ll1 > ll0 + 1.0, "FTC LL improves under SCG");
  fail += checkAbove("FTC trained LL floor", ll1, 15.0);
  fail += checkBelow("FTC test RMSE ceiling", rmse, 0.25);
  fail += checkBelow("FTC likelihood.variance ceiling", s2, 0.05);

  // Post-train equivalence: RBF + White(σ²) with fixed likelihood.variance.
  CCmpndKern kernW(Xtr);
  kernW.addKern(new CRbfKern(Xtr));
  kernW.addKern(new CWhiteKern(Xtr));
  kernW.setByPath("rbf.lengthscale", kern.getByPath("rbf.lengthscale"));
  kernW.setByPath("rbf.variance", kern.getByPath("rbf.variance"));
  kernW.setByPath("white.variance", s2);
  CGaussianNoise noiseW(&ytr);
  noiseW.setBias(0.0);
  CGp gpW(&kernW, &noiseW, &Xtr, CGp::FTC, 0, 0);
  gpW.fixLikelihoodVariance(true);
  fail += checkClose("FTC trained LL vs legacy White", ll1, gpW.logLikelihood());
  fail += checkClose("FTC trained RMSE vs legacy White",
                     rmse, rmsePred(gpW, Xte, yte), 1e-8);
  return fail;
}

int testSparseDtcPerf(PerfCtrl& ctrl)
{
  cout << "==== sparse DTC sinc algorithm performance ====" << endl;
  int fail = 0;
  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/sinc.csv");
  const unsigned int N = X.getRows();
  const unsigned int Nte = 8;
  const unsigned int Ntr = N - Nte;
  CMatrix Xtr(Ntr, X.getCols());
  CMatrix ytr(Ntr, 1);
  CMatrix Xte(Nte, X.getCols());
  CMatrix yte(Nte, 1);
  for(unsigned int i = 0; i < Ntr; i++)
  {
    Xtr.copyRowRow(i, X, i);
    ytr.copyRowRow(i, y, i);
  }
  for(unsigned int i = 0; i < Nte; i++)
  {
    Xte.copyRowRow(i, X, Ntr + i);
    yte.copyRowRow(i, y, Ntr + i);
  }

  const unsigned int M = 8;
  CMatrix Xu(M, Xtr.getCols());
  for(unsigned int i = 0; i < M; i++)
    Xu.copyRowRow(i, Xtr, i * (Ntr / M));

  CCmpndKern kern(Xtr);
  kern.addKern(new CRbfKern(Xtr));
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
  CGaussianNoise noise(&ytr);
  noise.setBias(0.0);
  CGp dtc(&kern, &noise, &Xtr, CGp::DTC, M, 0);
  dtc.setInducingVals(Xu);
  dtc.setInducingFixed(true);
  dtc.setLikelihoodVariance(1.0);
  const double ll0 = dtc.logLikelihood();
  dtc.setDefaultOptimiser(CGp::SCG);
  dtc.optimise(40);
  const double ll1 = dtc.logLikelihood();
  const double s2 = dtc.getLikelihoodVariance();
  const double beta = dtc.getBetaVal();
  const double rmse = rmsePred(dtc, Xte, yte);

  fail += failTrue(ll1 > ll0 + 1.0, "DTC LL improves under SCG");
  fail += checkAbove("DTC trained LL floor", ll1, 10.0);
  fail += checkBelow("DTC test RMSE ceiling", rmse, 0.5);
  fail += checkClose("DTC beta ↔ variance after train", beta, 1.0 / s2, 1e-8);
  return fail;
}

vector<unsigned int> ivmActive(CIvm& model)
{
  vector<unsigned int> a(model.getNumActive());
  for(unsigned int i = 0; i < model.getNumActive(); i++)
    a[i] = model.getActivePoint(i);
  return a;
}

int testIvmWhiteSoftenerGate(PerfCtrl& ctrl)
{
  cout << "==== IVM White softener performance gate ====" << endl;
  int fail = 0;
  CMatrix X(1, 1);
  CMatrix y(1, 1);
  ctrl.readData(X, y, "examples/unitsquaregp.svml");
  const unsigned int N = X.getRows();
  const unsigned int M = min(50u, N / 4);
  const unsigned int seed = 42;

  auto run = [&](bool withBias, bool withWhite) -> vector<unsigned int> {
    ndlutil::init_genrand(seed);
    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    kern.setByPath("rbf.lengthscale", 1.0);
    kern.setByPath("rbf.variance", 1.0);
    if(withBias)
      kern.addKern(new CBiasKern(X));
    if(withWhite)
      kern.addKern(new CWhiteKern(X));
    CProbitNoise noise(&y);
    for(unsigned int j = 0; j < noise.getOutputDim(); j++)
      noise.setParam(0.0, j);
    CIvm model(&X, &y, &kern, &noise, CIvm::ENTROPY, M, 0);
    model.selectPoints();
    model.optimise(3, 40, 15);
    return ivmActive(model);
  };

  const vector<unsigned int> base = run(true, true);   // RBF+Bias+White
  const vector<unsigned int> rbf = run(false, false);  // RBF only
  const vector<unsigned int> rw = run(false, true);    // RBF+White

  const double jacRbf = jaccard(base, rbf);
  const double jacRw = jaccard(base, rw);
  // B′ soak: RBF-only ≈ 0.32, RBF+White ≈ 0.92 — gate softener regression.
  fail += checkBelow("IVM RBF-only Jaccard vs status quo", jacRbf, 0.50);
  fail += checkAbove("IVM RBF+White Jaccard vs status quo", jacRw, 0.70);
  return fail;
}

} // namespace

int main(int argc, char** argv)
{
  PerfCtrl ctrl(argc, argv);
  ctrl.setVerbosity(0);
  ctrl.setSeed(1);

  int fail = 0;
  try
  {
    fail += testDualNoisePolicy();
    fail += testFtcSincPerf(ctrl);
    fail += testSparseDtcPerf(ctrl);
    fail += testIvmWhiteSoftenerGate(ctrl);
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
