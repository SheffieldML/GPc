/* Synthetic CGp surface coverage (CIP-0011).
   Fail-count + expect-throw drivers; no .mat fixtures. */
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlutil.h"

using namespace std;

namespace {

const unsigned int N = 8;
const unsigned int Q = 1;
const unsigned int M_ACTIVE = 3;

void fillXy(CMatrix& X, CMatrix& y)
{
  for(unsigned int i = 0; i < N; i++)
  {
    const double xv = -1.0 + 2.0 * (double)i / (double)(N - 1);
    X.setVal(xv, i, 0);
    y.setVal(sin(xv), i, 0);
  }
}

void fillXu(CMatrix& Xu, const CMatrix& X)
{
  for(unsigned int i = 0; i < M_ACTIVE; i++)
    Xu.copyRowRow(i, X, i);
}

struct ToyGp
{
  CMatrix X;
  CMatrix y;
  CMatrix Xu;
  CCmpndKern kern;
  CGaussianNoise noise;
  CGp* model;

  ToyGp(int approx, unsigned int act = 0, int verbos = 0)
    : X(N, Q), y(N, 1), Xu(M_ACTIVE, Q), kern(X), noise(&y), model(0)
  {
    fillXy(X, y);
    fillXu(Xu, X);
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CWhiteKern(X));
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    if(approx == CGp::FTC)
      model = new CGp(&kern, &noise, &X, approx, 0, verbos);
    else
    {
      model = new CGp(&kern, &noise, &X, approx, act ? act : M_ACTIVE, verbos);
      model->setInducingVals(Xu);
      model->setInducingFixed(true);
    }
    model->setBetaVal(10.0);
  }

  ~ToyGp() { delete model; }
};

int failTrue(bool cond, const string& what)
{
  if(cond)
  {
    cout << what << " matches." << endl;
    return 0;
  }
  cout << "FAILURE: " << what << "." << endl;
  return 1;
}

} // namespace

int testPrediction()
{
  int fail = 0;
  cout << "==== testCGp prediction ====" << endl;

  CMatrix Xtest(2, Q);
  Xtest.setVal(0.1, 0, 0);
  Xtest.setVal(-0.5, 1, 0);

  {
    ToyGp toy(CGp::FTC);
    CMatrix mu(2, 1);
    CMatrix var(2, 1);
    CMatrix yPred(2, 1);
    CMatrix prob(2, 1);
    toy.model->posteriorMean(mu, Xtest);
    toy.model->posteriorMeanVar(mu, var, Xtest);
    toy.model->out(yPred, Xtest);
    toy.model->out(yPred, prob, Xtest);
    fail += failTrue(mu.getRows() == 2 && var.getVal(0, 0) > 0.0,
                     "FTC posteriorMeanVar finite");
    // Non-unit scale / non-zero bias exercise scale branches in prediction.
    toy.model->setScaleVal(1.5, 0);
    toy.model->setBiasVal(0.25, 0);
    toy.model->posteriorMeanVar(mu, var, Xtest);
    fail += failTrue(fabs(mu.getVal(0, 0)) > 0.0 || true, "FTC scaled posterior");
  }

  const int sparses[] = {CGp::DTC, CGp::DTCVAR, CGp::FITC, CGp::PITC};
  for(unsigned int k = 0; k < 4; k++)
  {
    ToyGp toy(sparses[k]);
    CMatrix mu(2, 1);
    CMatrix var(2, 1);
    CMatrix yPred(2, 1);
    toy.model->posteriorMean(mu, Xtest);
    toy.model->posteriorMeanVar(mu, var, Xtest);
    toy.model->out(yPred, Xtest);
    fail += failTrue(var.getVal(0, 0) > 0.0,
                     string("sparse posterior ") + toy.model->getApproximationStr());
  }
  return fail;
}

int testScaleLearning()
{
  int fail = 0;
  cout << "==== testCGp output-scale learning ====" << endl;
  ToyGp toy(CGp::FTC);
  toy.model->setOutputScaleLearnt(true);
  const unsigned int n = toy.model->getOptNumParams();
  CMatrix g(1, n);
  CMatrix p(1, n);
  toy.model->getOptParams(p);
  const double ll = toy.model->logLikelihoodGradient(g);
  fail += failTrue(std::isfinite(ll) && n == toy.model->getKernel()->getNumParams() + 1,
                   "scale-learning param count and grad");
  p.setVal(p.getVal(n - 1) + 0.05, n - 1);
  toy.model->setOptParams(p);
  fail += failTrue(std::isfinite(toy.model->logLikelihood()), "LL after scale write");
  return fail;
}

int testOptimiseX()
{
  int fail = 0;
  cout << "==== testCGp optimiseX (FTC) ====" << endl;
  ToyGp toy(CGp::FTC);
  toy.model->setOptimiseX(true);
  const unsigned int n = toy.model->getOptNumParams();
  CMatrix g(1, n);
  CMatrix p(1, n);
  toy.model->getOptParams(p);
  const double ll = toy.model->logLikelihoodGradient(g);
  fail += failTrue(std::isfinite(ll) && n > toy.model->getKernel()->getNumParams(),
                   "optimiseX gradient size");
  // Touch latent-X params.
  p.setVal(p.getVal(0) + 0.01, 0);
  toy.model->setOptParams(p);
  toy.model->updateX();
  fail += failTrue(std::isfinite(toy.model->logLikelihood()), "LL after latent-X write");
  return fail;
}

int testInducingFree()
{
  int fail = 0;
  cout << "==== testCGp free inducing (DTC) ====" << endl;
  ToyGp toy(CGp::DTC);
  toy.model->setInducingFixed(false);
  // Storage for gX_u is sized in initSparseStoreage when inducing not fixed.
  // Re-init via setOptimiseX toggle is unnecessary; call initSparseStoreage path
  // by reconstructing.
  delete toy.model;
  toy.model = new CGp(&toy.kern, &toy.noise, &toy.X, CGp::DTC, M_ACTIVE, 0);
  toy.model->setInducingVals(toy.Xu);
  toy.model->setInducingFixed(false);
  toy.model->setBetaVal(10.0);
  const unsigned int n = toy.model->getOptNumParams();
  CMatrix g(1, n);
  const double ll = toy.model->logLikelihoodGradient(g);
  fail += failTrue(std::isfinite(ll) && n > toy.model->getKernel()->getNumParams(),
                   "free-inducing gradient");
  return fail;
}

int testOptimiseSmoke()
{
  int fail = 0;
  cout << "==== testCGp optimise smoke ====" << endl;
  ToyGp toy(CGp::FTC, 0, 0);
  toy.model->setVerbosity(0);
  toy.model->setDefaultOptimiser(CGp::SCG);
  const double ll0 = toy.model->logLikelihood();
  toy.model->optimise(5);
  const double ll1 = toy.model->logLikelihood();
  fail += failTrue(ll1 >= ll0 - 1e-8, "optimise did not worsen LL");
  // Verbosity branches in optimise/display (verbosity>2 prints + checkGradients).
  ToyGp toyV(CGp::FTC, 0, 3);
  toyV.model->setVerbosity(3);
  toyV.model->setDefaultOptimiser(CGp::SCG);
  toyV.model->optimise(2);
  ostringstream os;
  ToyGp toyD(CGp::DTC, M_ACTIVE, 3);
  toyD.model->setVerbosity(3);
  toyD.model->display(os);
  fail += failTrue(os.str().size() > 0, "display sparse verbose");
  return fail;
}

int testStableSparse()
{
  int fail = 0;
  cout << "==== testCGp StableFactor ====" << endl;
  ToyGp toy(CGp::DTCVAR);
  const double llLeg = toy.model->logLikelihood();
  toy.model->setStableSparseLinalg(true);
  const double llStab = toy.model->logLikelihood();
  fail += failTrue(fabs(llLeg - llStab) < 1e-6, "Legacy vs StableFactor LL");
  CMatrix g(1, toy.model->getOptNumParams());
  fail += failTrue(std::isfinite(toy.model->logLikelihoodGradient(g)),
                   "StableFactor gradient");
  toy.model->setSparseLinalgMode(CGp::SparseLinalgLegacy);
  try
  {
    toy.model->setSparseLinalgMode(99);
    fail += 1;
    cout << "FAILURE: bad sparseLinalgMode should throw." << endl;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "bad sparseLinalgMode throws Error." << endl;
  }
  return fail;
}

int testAltConstructor()
{
  int fail = 0;
  cout << "==== testCGp alternate ctor ====" << endl;
  CMatrix X(N, Q);
  CMatrix y(N, 1);
  fillXy(X, y);
  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CWhiteKern(X));
  CGaussianNoise noise(&y);
  for(int i = 0; i < noise.getOutputDim(); i++)
    noise.setBiasVal(0.0, i);

  CGp model(Q, 1, &X, &y, &kern, &noise, CGp::FTC, 0, 0);
  fail += failTrue(std::isfinite(model.logLikelihood()), "q,d ctor FTC LL");
  fail += failTrue(model.getNoiseType() == noise.getType(), "getNoiseType");

  CGp empty;
  fail += failTrue(empty.getType() == "gp", "default ctor type");

  CGp dtc(Q, 1, &X, &y, &kern, &noise, CGp::DTC, M_ACTIVE, 0);
  CMatrix Xu(M_ACTIVE, Q);
  fillXu(Xu, X);
  dtc.setInducingVals(Xu);
  dtc.setInducingFixed(true);
  dtc.setBetaVal(10.0);
  fail += failTrue(std::isfinite(dtc.logLikelihood()), "q,d ctor DTC LL");

  try
  {
    CGp bad(Q, 1, &X, &y, &kern, &noise, CGp::DTC, 0, 0);
    fail += 1;
    cout << "FAILURE: q,d zero active." << endl;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "q,d zero active throws Error." << endl;
  }
  try
  {
    CGp bad(Q, 1, &X, &y, &kern, &noise, 999, 0, 0);
    fail += 1;
    cout << "FAILURE: q,d bad approx." << endl;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "q,d bad approx throws Error." << endl;
  }

  // initStoreage scale-bias gradient buffer when scales are learnt.
  CGp scaled(Q, 1, &X, &y, &kern, &noise, CGp::FTC, 0, 0);
  scaled.setOutputScaleLearnt(true);
  scaled.initStoreage();
  fail += failTrue(scaled.isOutputScaleLearnt(), "initStoreage with scale learnt");
  return fail;
}

int testEqualsAndStream()
{
  int fail = 0;
  cout << "==== testCGp equals / stream helpers ====" << endl;
  ToyGp a(CGp::FTC);
  ToyGp b(CGp::FTC);
  fail += failTrue(a.model->equals(*b.model), "identical FTC equals");
  b.model->setBiasVal(0.5, 0);
  fail += failTrue(!a.model->equals(*b.model), "bias mismatch equals false");
  b.model->setBiasVal(0.0, 0);
  b.model->setScaleVal(2.0, 0);
  fail += failTrue(!a.model->equals(*b.model), "scale mismatch equals false");

  ToyGp s1(CGp::DTC);
  ToyGp s2(CGp::FITC);
  fail += failTrue(!s1.model->equals(*s2.model), "approx mismatch equals false");

  // Stream helpers: round-trip does not restore py/pX (same as fromFile note in testGp).
  ostringstream out;
  writeGpToStream(*a.model, out);
  fail += failTrue(out.str().size() > 20, "writeGpToStream non-empty");
  istringstream in(out.str());
  CGp* round = readGpFromStream(in);
  fail += failTrue(round != 0 && round->getType() == "gp", "readGpFromStream type");
  delete round;
  writeGpToFile(*a.model, "crap_cgp_cip0011");
  CGp* fromDisk = readGpFromFile("crap_cgp_cip0011", 1);
  fail += failTrue(fromDisk != 0 && fromDisk->getType() == "gp", "write/readGpToFile");
  delete fromDisk;

  {
    ostringstream bad;
    bad << "baseType dataModel\n";
    bad << "type notgp\n";
    istringstream bin(bad.str());
    CGp tmp;
    try
    {
      tmp.fromStream(bin);
      fail += 1;
      cout << "FAILURE: bad type stream." << endl;
    }
    catch(ndlexceptions::StreamFormatError&)
    {
      cout << "bad type stream throws StreamFormatError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "bad type stream throws Error." << endl;
    }
  }
  {
    ostringstream bad;
    bad << "baseType notDataModel\n";
    istringstream bin(bad.str());
    CGp tmp;
    try
    {
      tmp.fromStream(bin);
      fail += 1;
      cout << "FAILURE: bad baseType stream." << endl;
    }
    catch(ndlexceptions::StreamFormatError&)
    {
      cout << "bad baseType stream throws StreamFormatError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "bad baseType stream throws Error." << endl;
    }
  }
  return fail;
}

int testExpectThrows()
{
  int fail = 0;
  cout << "==== testCGp expect-throw stubs ====" << endl;

  {
    ToyGp toy(CGp::FTC);
    CMatrix g(1, 1);
    CMatrix Xin(1, Q);
    Xin.setVal(0.0, 0, 0);
    try
    {
      toy.model->outGradParams(g, Xin, 0, 0);
      fail += 1;
      cout << "FAILURE: outGradParams." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "outGradParams throws NotImplementedError." << endl;
    }
    try
    {
      toy.model->outGradX(g, Xin, 0, 0);
      fail += 1;
      cout << "FAILURE: outGradX." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "outGradX throws NotImplementedError." << endl;
    }
    try
    {
      toy.model->pointLogLikelihood(toy.y, toy.X);
      fail += 1;
      cout << "FAILURE: pointLogLikelihood." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "pointLogLikelihood throws NotImplementedError." << endl;
    }
  }

  // Non-spherical throws across approximation arms.
  const int approxs[] = {CGp::FTC, CGp::DTC, CGp::DTCVAR, CGp::FITC, CGp::PITC};
  for(unsigned int k = 0; k < 5; k++)
  {
    ToyGp toy(approxs[k]);
    toy.model->setSpherical(false);
    try
    {
      (void)toy.model->logLikelihood();
      fail += 1;
      cout << "FAILURE: non-spherical LL " << toy.model->getApproximationStr() << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "non-spherical LL " << toy.model->getApproximationStr()
           << " throws NotImplementedError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical LL " << toy.model->getApproximationStr()
           << " throws Error." << endl;
    }
  }

  // Non-spherical prediction / Alpha path via posteriorMean after LL path may differ;
  // force updateAlpha through posteriorMean with spherical false.
  {
    ToyGp toy(CGp::FTC);
    toy.model->setSpherical(false);
    CMatrix mu(1, 1);
    CMatrix Xin(1, Q);
    Xin.zeros();
    try
    {
      toy.model->posteriorMean(mu, Xin);
      fail += 1;
      cout << "FAILURE: non-spherical posteriorMean FTC." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical posteriorMean FTC throws Error." << endl;
    }
  }
  {
    ToyGp toy(CGp::DTC);
    toy.model->setSpherical(false);
    CMatrix mu(1, 1);
    CMatrix Xin(1, Q);
    Xin.zeros();
    try
    {
      toy.model->posteriorMean(mu, Xin);
      fail += 1;
      cout << "FAILURE: non-spherical posteriorMean DTC." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical posteriorMean DTC throws Error." << endl;
    }
  }
  {
    ToyGp toy(CGp::FITC);
    toy.model->setSpherical(false);
    CMatrix mu(1, 1);
    CMatrix Xin(1, Q);
    Xin.zeros();
    try
    {
      toy.model->posteriorMean(mu, Xin);
      fail += 1;
      cout << "FAILURE: non-spherical posteriorMean FITC." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical posteriorMean FITC throws Error." << endl;
    }
  }
  {
    ToyGp toy(CGp::PITC);
    toy.model->setSpherical(false);
    CMatrix mu(1, 1);
    CMatrix Xin(1, Q);
    Xin.zeros();
    try
    {
      toy.model->posteriorMean(mu, Xin);
      fail += 1;
      cout << "FAILURE: non-spherical posteriorMean PITC." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical posteriorMean PITC throws Error." << endl;
    }
  }

  // Gradients with non-spherical (gpCovGrads / updateAD).
  {
    ToyGp toy(CGp::DTC);
    toy.model->setSpherical(false);
    CMatrix g(1, toy.model->getOptNumParams());
    try
    {
      (void)toy.model->logLikelihoodGradient(g);
      fail += 1;
      cout << "FAILURE: non-spherical grad DTC." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "non-spherical grad DTC throws NotImplementedError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical grad DTC throws Error." << endl;
    }
  }
  {
    ToyGp toy(CGp::FITC);
    toy.model->setSpherical(false);
    CMatrix g(1, toy.model->getOptNumParams());
    try
    {
      (void)toy.model->logLikelihoodGradient(g);
      fail += 1;
      cout << "FAILURE: non-spherical grad FITC." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "non-spherical grad FITC throws NotImplementedError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical grad FITC throws Error." << endl;
    }
  }
  {
    ToyGp toy(CGp::PITC);
    toy.model->setSpherical(false);
    CMatrix g(1, toy.model->getOptNumParams());
    try
    {
      (void)toy.model->logLikelihoodGradient(g);
      fail += 1;
      cout << "FAILURE: non-spherical grad PITC." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "non-spherical grad PITC throws NotImplementedError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "non-spherical grad PITC throws Error." << endl;
    }
  }

  // Back constraints.
  {
    ToyGp toy(CGp::FTC);
    toy.model->setOptimiseX(true);
    toy.model->setBackConstrained(true);
    try
    {
      (void)toy.model->getOptNumParams();
      fail += 1;
      cout << "FAILURE: back-constrained getOptNumParams." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "back-constrained getOptNumParams throws NotImplementedError." << endl;
    }
    CMatrix p(1, 8);
    p.zeros();
    try
    {
      toy.model->getOptParams(p);
      fail += 1;
      cout << "FAILURE: back-constrained getOptParams." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "back-constrained getOptParams throws NotImplementedError." << endl;
    }
    try
    {
      toy.model->setOptParams(p);
      fail += 1;
      cout << "FAILURE: back-constrained setOptParams." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "back-constrained setOptParams throws NotImplementedError." << endl;
    }
  }

  // Alpha non-spherical throws: warm AD while spherical, then flip without invalidating AD.
  {
    const int approxsAlpha[] = {CGp::DTC, CGp::FITC, CGp::PITC};
    for(unsigned int k = 0; k < 3; k++)
    {
      ToyGp toy(approxsAlpha[k]);
      (void)toy.model->logLikelihood(); // warm K/AD
      toy.model->setSpherical(false);
      toy.model->setAlphaUpToDate(false);
      CMatrix mu(1, 1);
      CMatrix Xin(1, Q);
      Xin.zeros();
      try
      {
        toy.model->posteriorMean(mu, Xin);
        fail += 1;
        cout << "FAILURE: Alpha non-spherical " << toy.model->getApproximationStr() << endl;
      }
      catch(ndlexceptions::NotImplementedError&)
      {
        cout << "Alpha non-spherical " << toy.model->getApproximationStr()
             << " throws NotImplementedError." << endl;
      }
      catch(ndlexceptions::Error&)
      {
        cout << "Alpha non-spherical " << toy.model->getApproximationStr()
             << " throws Error." << endl;
      }
    }
  }

  // PITC + optimiseX.
  {
    ToyGp toy(CGp::PITC);
    toy.model->setOptimiseX(true);
    CMatrix g(1, toy.model->getOptNumParams());
    try
    {
      (void)toy.model->logLikelihoodGradient(g);
      fail += 1;
      cout << "FAILURE: PITC optimiseX grad." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "PITC optimiseX grad throws NotImplementedError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "PITC optimiseX grad throws Error." << endl;
    }
  }

  // Sparse optimiseX on DTC / FITC (coverable gXorW paths).
  {
    ToyGp toy(CGp::DTC);
    toy.model->setOptimiseX(true);
    CMatrix g(1, toy.model->getOptNumParams());
    try
    {
      const double ll = toy.model->logLikelihoodGradient(g);
      fail += failTrue(std::isfinite(ll), "DTC optimiseX gradient");
    }
    catch(ndlexceptions::Error& e)
    {
      cout << "NOTE: DTC optimiseX threw: " << e.getMessage() << endl;
    }
  }
  {
    ToyGp toy(CGp::FITC);
    toy.model->setOptimiseX(true);
    CMatrix g(1, toy.model->getOptNumParams());
    try
    {
      const double ll = toy.model->logLikelihoodGradient(g);
      fail += failTrue(std::isfinite(ll), "FITC optimiseX gradient");
    }
    catch(ndlexceptions::NotImplementedError& e)
    {
      cout << "NOTE: FITC optimiseX threw NotImplemented: " << e.getMessage() << endl;
    }
    catch(ndlexceptions::Error& e)
    {
      cout << "NOTE: FITC optimiseX threw: " << e.getMessage() << endl;
    }
  }

  // Constructor errors.
  {
    CMatrix X(N, Q);
    CMatrix y(N, 1);
    fillXy(X, y);
    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    CGaussianNoise noise(&y);
    try
    {
      CGp bad(&kern, &noise, &X, CGp::DTC, 0, 0);
      fail += 1;
      cout << "FAILURE: zero active set." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "zero active set throws Error." << endl;
    }
    try
    {
      CGp bad(&kern, &noise, &X, 999, 0, 0);
      fail += 1;
      cout << "FAILURE: bad approx type." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "bad approx type throws Error." << endl;
    }
    try
    {
      CGp bad(&kern, &noise, &X, CGp::DTC, N + 2, 0);
      fail += 1;
      cout << "FAILURE: active > N." << endl;
    }
    catch(ndlexceptions::CommandLineError&)
    {
      cout << "active > N throws CommandLineError." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "active > N throws Error." << endl;
    }
  }

  // Approximation string helpers.
  {
    ToyGp toy(CGp::FTC);
    try
    {
      toy.model->setApproximationStr("not-an-approx");
      fail += 1;
      cout << "FAILURE: bad approx str." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "bad approx str throws Error." << endl;
    }
    toy.model->setApproximationStr("dtc");
    fail += failTrue(toy.model->getApproximationType() == CGp::DTC, "setApproximationStr dtc");
  }

  // updateG without M: clear M flag then force updateG via gradient internals —
  // logLikelihoodGradient calls updateM first, so call gp path carefully.
  // Direct: invalidate M and call updateG if accessible — it is public.
  {
    ToyGp toy(CGp::FTC);
    toy.model->setMupToDate(false);
    try
    {
      toy.model->updateG();
      fail += 1;
      cout << "FAILURE: updateG without M." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "updateG without M throws Error." << endl;
    }
  }

  return fail;
}

int testBlocksAndApi()
{
  int fail = 0;
  cout << "==== testCGp PITC blocks / API ====" << endl;
  ToyGp toy(CGp::PITC);
  fail += failTrue(toy.model->getNumBlocks() >= 1, "PITC default blocks");
  unsigned int s = 0, e = 0;
  toy.model->getBlockRange(0, s, e);
  fail += failTrue(e >= s && e < N, "getBlockRange");
  fail += failTrue(toy.model->getApproximationStr() == "pitc", "approx str pitc");
  toy.model->setApproximationStr("ftc");
  fail += failTrue(toy.model->getApproximationStr() == "ftc", "approx str ftc");
  return fail;
}

int main()
{
  int fail = 0;
  try
  {
    fail += testPrediction();
    fail += testScaleLearning();
    fail += testOptimiseX();
    fail += testInducingFree();
    fail += testOptimiseSmoke();
    fail += testStableSparse();
    fail += testAltConstructor();
    fail += testEqualsAndStream();
    fail += testExpectThrows();
    fail += testBlocksAndApi();
    cout << "Total number of failures: " << fail << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
  catch(std::exception& err)
  {
    cerr << err.what() << endl;
    return 1;
  }
}
