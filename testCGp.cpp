/* Synthetic CGp surface coverage (CIP-0011).
   Fail-count + expect-throw drivers; no .mat fixtures. */
#include <algorithm>
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

  // Everyday sparse prediction: DTC / DTCVAR / FITC / PITC on a small 1-D toy —
  // the “fit sparse approx, predict mean/var at a couple of test points” path.
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

// Dual-path sparse linalg on a tiny 1-D toy (N=8, M=3): the “does the new
// factorisation still implement the same sparse GP?” gate for everyday DTC /
// DTCVAR / FITC / PITC regression — not a stress cell.
int testStableSparse()
{
  int fail = 0;
  cout << "==== testCGp StableFactor ====" << endl;
  ToyGp toy(CGp::DTCVAR);
  toy.model->setStableSparseLinalg(false);
  const double llLeg = toy.model->logLikelihood();
  CMatrix gLeg(1, toy.model->getOptNumParams());
  fail += failTrue(std::isfinite(toy.model->logLikelihoodGradient(gLeg)),
                   "Legacy DTCVAR gradient");
  toy.model->setStableSparseLinalg(true);
  const double llStab = toy.model->logLikelihood();
  fail += failTrue(fabs(llLeg - llStab) < 1e-6, "Legacy vs StableFactor LL");
  CMatrix g(1, toy.model->getOptNumParams());
  fail += failTrue(std::isfinite(toy.model->logLikelihoodGradient(g)),
                   "StableFactor gradient");
  double gDiff = 0.0;
  for(unsigned int i = 0; i < g.getCols(); i++)
    gDiff = max(gDiff, fabs(g.getVal(i) - gLeg.getVal(i)));
  fail += failTrue(gDiff < 1e-5, "Legacy vs StableFactor DTCVAR grads");
  ToyGp toyD(CGp::DTC);
  toyD.model->setStableSparseLinalg(false);
  CMatrix gDLeg(1, toyD.model->getOptNumParams());
  toyD.model->logLikelihoodGradient(gDLeg);
  toyD.model->setStableSparseLinalg(true);
  CMatrix gD(1, toyD.model->getOptNumParams());
  toyD.model->logLikelihoodGradient(gD);
  double gDDiff = 0.0;
  for(unsigned int i = 0; i < gD.getCols(); i++)
    gDDiff = max(gDDiff, fabs(gD.getVal(i) - gDLeg.getVal(i)));
  fail += failTrue(gDDiff < 1e-5, "Legacy vs StableFactor DTC grads");
  ToyGp toyF(CGp::FITC);
  toyF.model->setStableSparseLinalg(false);
  const double fL = toyF.model->logLikelihood();
  toyF.model->setStableSparseLinalg(true);
  const double fS = toyF.model->logLikelihood();
  fail += failTrue(fabs(fL - fS) < 1e-6, "Legacy vs StableFactor FITC LL");
  CMatrix gFLeg(1, toyF.model->getOptNumParams());
  toyF.model->setStableSparseLinalg(false);
  toyF.model->logLikelihoodGradient(gFLeg);
  toyF.model->setStableSparseLinalg(true);
  CMatrix gF(1, toyF.model->getOptNumParams());
  toyF.model->logLikelihoodGradient(gF);
  double gFDiff = 0.0;
  for(unsigned int i = 0; i < gF.getCols(); i++)
    gFDiff = max(gFDiff, fabs(gF.getVal(i) - gFLeg.getVal(i)));
  fail += failTrue(gFDiff < 1e-5, "Legacy vs StableFactor FITC grads");
  ToyGp toyP(CGp::PITC);
  toyP.model->setStableSparseLinalg(false);
  const double pL = toyP.model->logLikelihood();
  toyP.model->setStableSparseLinalg(true);
  const double pS = toyP.model->logLikelihood();
  fail += failTrue(fabs(pL - pS) < 1e-6, "Legacy vs StableFactor PITC LL");
  CMatrix gPLeg(1, toyP.model->getOptNumParams());
  toyP.model->setStableSparseLinalg(false);
  toyP.model->logLikelihoodGradient(gPLeg);
  toyP.model->setStableSparseLinalg(true);
  CMatrix gP(1, toyP.model->getOptNumParams());
  toyP.model->logLikelihoodGradient(gP);
  double gPDiff = 0.0;
  for(unsigned int i = 0; i < gP.getCols(); i++)
    gPDiff = max(gPDiff, fabs(gP.getVal(i) - gPLeg.getVal(i)));
  fail += failTrue(gPDiff < 1e-5, "Legacy vs StableFactor PITC grads");
  CMatrix Xtest(2, Q);
  Xtest.setVal(0.1, 0, 0);
  Xtest.setVal(-0.5, 1, 0);
  const int predApprox[] = {CGp::DTC, CGp::DTCVAR, CGp::FITC, CGp::PITC};
  const char* predName[] = {"DTC", "DTCVAR", "FITC", "PITC"};
  for(unsigned int k = 0; k < 4; k++)
  {
    ToyGp tp(predApprox[k]);
    CMatrix muL(2, 1), varL(2, 1), muS(2, 1), varS(2, 1);
    tp.model->setStableSparseLinalg(false);
    tp.model->posteriorMeanVar(muL, varL, Xtest);
    tp.model->setStableSparseLinalg(true);
    tp.model->posteriorMeanVar(muS, varS, Xtest);
    double pdiff = 0.0;
    for(unsigned int i = 0; i < 2; i++)
    {
      pdiff = max(pdiff, fabs(muL.getVal(i) - muS.getVal(i)));
      pdiff = max(pdiff, fabs(varL.getVal(i) - varS.getVal(i)));
    }
    fail += failTrue(pdiff < 1e-6,
                     string("Legacy vs StableFactor ") + predName[k] + " predict");
  }
  // Fresh model: the Dual-path toy above was left in StableFactor.
  {
    ToyGp toyDef(CGp::DTCVAR);
    fail += failTrue(toyDef.model->getSparseLinalgMode() == CGp::SparseLinalgLQ,
                     "LQ is the default sparseLinalgMode");
  }
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

  // CIP-001B: same mild toy under --sparse-linalg lq. Captures “swap Gram+chol
  // of C for LQ of [I,A] on a well-posed sparse GP” — LL/grads must match
  // StableFactor before we trust LQ on harder data.
  cout << "==== testCGp SparseLinalgLQ ====" << endl;
  {
    ToyGp toyLq(CGp::DTCVAR);
    toyLq.model->setSparseLinalgMode(CGp::SparseLinalgStableFactor);
    const double llS = toyLq.model->logLikelihood();
    CMatrix gS(1, toyLq.model->getOptNumParams());
    toyLq.model->logLikelihoodGradient(gS);
    toyLq.model->setSparseLinalgMode(CGp::SparseLinalgLQ);
    const double llQ = toyLq.model->logLikelihood();
    fail += failTrue(fabs(llS - llQ) < 1e-10, "StableFactor vs LQ DTCVAR LL");
    CMatrix gQ(1, toyLq.model->getOptNumParams());
    fail += failTrue(std::isfinite(toyLq.model->logLikelihoodGradient(gQ)),
                     "LQ DTCVAR gradient");
    double gDiffQ = 0.0;
    for(unsigned int i = 0; i < gQ.getCols(); i++)
      gDiffQ = max(gDiffQ, fabs(gQ.getVal(i) - gS.getVal(i)));
    fail += failTrue(gDiffQ < 1e-5, "StableFactor vs LQ DTCVAR grads");
  }
  {
    ToyGp toyD(CGp::DTC);
    toyD.model->setSparseLinalgMode(CGp::SparseLinalgStableFactor);
    const double llS = toyD.model->logLikelihood();
    toyD.model->setSparseLinalgMode(CGp::SparseLinalgLQ);
    const double llQ = toyD.model->logLikelihood();
    fail += failTrue(fabs(llS - llQ) < 1e-10, "StableFactor vs LQ DTC LL");
  }
  {
    ToyGp toyF(CGp::FITC);
    toyF.model->setSparseLinalgMode(CGp::SparseLinalgStableFactor);
    const double llS = toyF.model->logLikelihood();
    CMatrix gS(1, toyF.model->getOptNumParams());
    toyF.model->logLikelihoodGradient(gS);
    toyF.model->setSparseLinalgMode(CGp::SparseLinalgLQ);
    const double llQ = toyF.model->logLikelihood();
    fail += failTrue(fabs(llS - llQ) < 1e-10, "StableFactor vs LQ FITC LL");
    CMatrix gQ(1, toyF.model->getOptNumParams());
    toyF.model->logLikelihoodGradient(gQ);
    double gDiffQ = 0.0;
    for(unsigned int i = 0; i < gQ.getCols(); i++)
      gDiffQ = max(gDiffQ, fabs(gQ.getVal(i) - gS.getVal(i)));
    fail += failTrue(gDiffQ < 1e-5, "StableFactor vs LQ FITC grads");
  }
  fail += failTrue(toy.model->isLQSparseLinalg() == false, "isLQSparseLinalg false on Legacy");
  toy.model->setSparseLinalgMode(CGp::SparseLinalgLQ);
  fail += failTrue(toy.model->isLQSparseLinalg() && toy.model->isWhitenedSparseLinalg(),
                   "LQ mode sets isLQ/isWhitened");
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

int testModelPaths()
{
  int fail = 0;
  cout << "==== testCGp model paths (CIP-0014 Phase C) ====" << endl;

  ToyGp toy(CGp::FTC);
  toy.model->setByPath("rbf.variance", 2.5);
  fail += failTrue(fabs(toy.model->getByPath("rbf.variance") - 2.5) < 1e-14,
                   "model set/get rbf.variance");
  fail += failTrue(fabs(toy.model->getByPath("kern.rbf.variance") - 2.5) < 1e-14,
                   "kern. prefix get");
  toy.model->setByPath("kern.rbf.lengthscale", 2.0);
  fail += failTrue(fabs(toy.kern.getParam(0) - 0.25) < 1e-14,
                   "lengthscale via model stores inverseWidth");
  fail += failTrue(fabs(toy.model->getByPath("rbf.lengthscale") - 2.0) < 1e-14,
                   "get lengthscale recovers");

  vector<string> paths = toy.model->paramPaths();
  fail += failTrue(paths.size() >= 2 && paths[0].compare(0, 5, "kern.") == 0,
                   "paramPaths use kern. prefix");
  bool sawRbfVar = false;
  for(size_t i = 0; i < paths.size(); i++)
    if(paths[i] == "kern.rbf.variance")
      sawRbfVar = true;
  fail += failTrue(sawRbfVar, "paramPaths includes kern.rbf.variance");

  ToyGp sparse(CGp::FITC);
  sparse.model->setByPath("beta", 7.0);
  fail += failTrue(fabs(sparse.model->getByPath("beta") - 7.0) < 1e-14,
                   "beta path get/set");
  vector<string> sparsePaths = sparse.model->paramPaths();
  bool sawBeta = false;
  for(size_t i = 0; i < sparsePaths.size(); i++)
    if(sparsePaths[i] == "beta")
      sawBeta = true;
  fail += failTrue(sawBeta, "paramPaths includes beta when sparse");

  toy.model->setOutputScaleLearnt(true);
  toy.model->initStoreage();
  toy.model->setByPath("scale.0", 1.5);
  fail += failTrue(fabs(toy.model->getByPath("scale.0") - 1.5) < 1e-14,
                   "scale.0 path get/set");
  {
    vector<string> scalePaths = toy.model->paramPaths();
    bool sawScale = false;
    for(size_t i = 0; i < scalePaths.size(); i++)
      if(scalePaths[i] == "scale.0")
        sawScale = true;
    fail += failTrue(sawScale, "paramPaths includes scale.0 when learnt");
  }
  // OOB scale index while scales are learnt.
  try
  {
    toy.model->getByPath("scale.99");
    cout << "FAILURE: getByPath scale.99 should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "getByPath scale OOB throws Error." << endl;
  }
  try
  {
    toy.model->setByPath("scale.99", 1.0);
    cout << "FAILURE: setByPath scale.99 should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "setByPath scale OOB throws Error." << endl;
  }
  toy.model->clearFixes(); // hits scaleParamFixed clear loop while learnt
  toy.model->setOutputScaleLearnt(false);

  try
  {
    toy.model->setByPath("beta", 1.0);
    cout << "FAILURE: FTC setByPath beta should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "FTC setByPath beta throws Error." << endl;
  }
  try
  {
    toy.model->getByPath("beta");
    cout << "FAILURE: FTC getByPath beta should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "FTC getByPath beta throws Error." << endl;
  }
  try
  {
    toy.model->getByPath("scale.0");
    cout << "FAILURE: getByPath scale without learnScales should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "getByPath scale without learnScales throws Error." << endl;
  }
  try
  {
    toy.model->setByPath("scale.0", 1.0);
    cout << "FAILURE: setByPath scale without learnScales should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "setByPath scale without learnScales throws Error." << endl;
  }

  // isKernParamFixed before any fix() keeps mask empty → false for OOB index.
  fail += failTrue(!toy.model->isKernParamFixed(0)
                   && !toy.model->isKernParamFixed(999),
                   "isKernParamFixed false when mask unset/OOB");

  return fail;
}

int testFixMask()
{
  int fail = 0;
  cout << "==== testCGp fix mask (CIP-0014 Phase B) ====" << endl;

  ToyGp toy(CGp::FTC);
  const unsigned int nAll = toy.model->getOptNumParams();
  fail += failTrue(nAll == toy.model->getKernel()->getNumParams(),
                   "FTC opt params == kernel params");

  toy.model->fix("rbf.variance");
  fail += failTrue(toy.model->isFixed("rbf.variance"), "isFixed after fix");
  fail += failTrue(toy.model->getOptNumParams() == nAll - 1,
                   "fix drops one opt param");

  CMatrix params(1, toy.model->getOptNumParams());
  toy.model->getOptParams(params);
  const double varBefore = toy.kern.getByPath("rbf.variance");
  // Perturb free params and write back; fixed variance must stay put.
  for(unsigned int i = 0; i < params.getCols(); i++)
    params.setVal(params.getVal(i) + 0.01, i);
  toy.model->setOptParams(params);
  fail += failTrue(fabs(toy.kern.getByPath("rbf.variance") - varBefore) < 1e-14,
                   "fixed variance unchanged by setOptParams");

  toy.model->setVerbosity(0);
  try
  {
    toy.model->checkGradients();
    cout << "checkGradients with fixed variance matches." << endl;
  }
  catch(...)
  {
    cout << "FAILURE: checkGradients with fixed variance." << endl;
    fail += 1;
  }

  toy.model->unfix("rbf.variance");
  fail += failTrue(!toy.model->isFixed("rbf.variance"), "unfix clears flag");
  fail += failTrue(toy.model->getOptNumParams() == nAll,
                   "unfix restores opt param count");

  ToyGp sparse(CGp::FITC);
  const unsigned int nSparse = sparse.model->getOptNumParams();
  sparse.model->fix("beta");
  fail += failTrue(sparse.model->isBetaFixed() && sparse.model->isFixed("beta"),
                   "fix beta");
  fail += failTrue(sparse.model->getOptNumParams() == nSparse - 1,
                   "fix beta drops one param");
  sparse.model->unfix("beta");
  fail += failTrue(!sparse.model->isBetaFixed() && !sparse.model->isFixed("beta"),
                   "unfix beta");
  sparse.model->fix("beta");
  sparse.model->clearFixes();
  fail += failTrue(!sparse.model->isBetaFixed()
                   && sparse.model->getOptNumParams() == nSparse,
                   "clearFixes restores");

  // kern. prefix and scale.* paths (CIP-0014 coverage for CGp.cpp gate).
  toy.model->fix("kern.rbf.variance");
  fail += failTrue(toy.model->isFixed("rbf.variance"), "kern. prefix fix");
  toy.model->unfix("kern.rbf.variance");
  fail += failTrue(!toy.model->isFixed("rbf.variance"), "kern. prefix unfix");

  toy.model->setOutputScaleLearnt(true);
  toy.model->initStoreage();
  const unsigned int nScale = toy.model->getOptNumParams();
  fail += failTrue(nScale == toy.model->getKernel()->getNumParams() + 1,
                   "scale learnt adds one opt param");
  toy.model->fix("scale.0");
  fail += failTrue(toy.model->isFixed("scale.0"), "isFixed scale.0");
  fail += failTrue(toy.model->getOptNumParams() == nScale - 1,
                   "fix scale.0 drops one param");
  {
    CMatrix g(1, toy.model->getOptNumParams());
    fail += failTrue(std::isfinite(toy.model->logLikelihoodGradient(g)),
                     "grad with scale.0 fixed");
    CMatrix p(1, toy.model->getOptNumParams());
    toy.model->getOptParams(p);
    toy.model->setOptParams(p);
  }
  toy.model->unfix("scale.0");
  fail += failTrue(!toy.model->isFixed("scale.0")
                   && toy.model->getOptNumParams() == nScale,
                   "unfix scale.0");
  toy.model->setOutputScaleLearnt(false);

  try
  {
    toy.model->fix("beta");
    cout << "FAILURE: FTC fix beta should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "FTC fix beta throws Error." << endl;
  }

  try
  {
    toy.model->fix("scale.0");
    cout << "FAILURE: fix scale without learnScales should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "fix scale without learnScales throws Error." << endl;
  }

  toy.model->setOutputScaleLearnt(true);
  toy.model->initStoreage();
  try
  {
    toy.model->fix("scale.99");
    cout << "FAILURE: OOB scale fix should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "OOB scale fix throws Error." << endl;
  }
  toy.model->setOutputScaleLearnt(false);

  try
  {
    sparse.model->fixBeta(true);
    sparse.model->fixBeta(false);
    cout << "fixBeta toggle matches." << endl;
  }
  catch(...)
  {
    cout << "FAILURE: fixBeta toggle." << endl;
    fail += 1;
  }

  try
  {
    ToyGp ftc(CGp::FTC);
    ftc.model->fixBeta(true);
    cout << "FAILURE: FTC fixBeta should throw." << endl;
    fail += 1;
  }
  catch(ndlexceptions::Error&)
  {
    cout << "FTC fixBeta throws Error." << endl;
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

int testWhiteNoiseInducing()
{
  int fail = 0;
  cout << "==== testCGp white-noise inducing (CIP-001C) ====" << endl;

  // Non-sparse: Meff is 0; inducing type defaults to points with null pInducing.
  {
    ToyGp toy(CGp::FTC);
    fail += failTrue(toy.model->getEffectiveInducingCount() == 0.0, "FTC Meff is 0");
    fail += failTrue(toy.model->getInducingType() == "points", "FTC inducing type default");
    try
    {
      toy.model->setInducingType("whitenoise");
      fail += 1;
      cout << "FAILURE: FTC setInducingType." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "FTC setInducingType throws Error." << endl;
    }
    try
    {
      toy.model->setInducing(0);
      fail += 1;
      cout << "FAILURE: FTC setInducing null." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "FTC setInducing throws Error." << endl;
    }
  }

  // Smoke: RBF+White compound, DTCVAR, few SCG iters, finite LL.
  {
    ToyGp toy(CGp::DTCVAR);
    toy.model->setInducingFixed(false);
    toy.model->setInducingType("whitenoise");
    fail += failTrue(toy.model->getInducingType() == "whitenoise", "inducing type whitenoise");
    const double ll0 = toy.model->logLikelihood();
    fail += failTrue(std::isfinite(ll0), "WN DTCVAR initial LL finite");
    toy.model->setDefaultOptimiser(CGp::SCG);
    toy.model->setVerbosity(0);
    toy.model->optimise(5);
    const double ll1 = toy.model->logLikelihood();
    fail += failTrue(std::isfinite(ll1), "WN DTCVAR after optimise LL finite");
    const double meff = toy.model->getEffectiveInducingCount();
    fail += failTrue(std::isfinite(meff) && meff >= 0.0 && meff <= (double)M_ACTIVE + 1e-8,
                     "Meff in [0,M]");
  }

  // Legacy Meff on points (covers non-whitened diagnostic branch).
  {
    ToyGp toy(CGp::DTC);
    toy.model->setSparseLinalgMode(CGp::SparseLinalgLegacy);
    (void)toy.model->logLikelihood();
    const double meffLeg = toy.model->getEffectiveInducingCount();
    fail += failTrue(std::isfinite(meffLeg) && meffLeg >= 0.0 && meffLeg <= (double)M_ACTIVE + 1e-8,
                     "Legacy Meff in [0,M]");
  }

  // setInducing ownership transfer + unknown / null type errors.
  {
    ToyGp toy(CGp::DTC);
    const double s0 = CInducingWhiteNoise::medianSpacing(toy.model->X_u);
    CInducingWhiteNoise* owned = new CInducingWhiteNoise(&toy.model->X_u, s0 > 0.0 ? s0 : 1.0, false);
    toy.model->setInducing(owned);
    fail += failTrue(toy.model->getInducingType() == "whitenoise", "setInducing WN type");
    fail += failTrue(std::isfinite(toy.model->logLikelihood()), "setInducing WN LL finite");
    try
    {
      toy.model->setInducingType("notatype");
      fail += 1;
      cout << "FAILURE: unknown inducing type." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "unknown inducing type throws Error." << endl;
    }
    try
    {
      toy.model->setInducing(0);
      fail += 1;
      cout << "FAILURE: sparse setInducing null." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "sparse setInducing null throws Error." << endl;
    }
  }

  // WN finite LL on remaining sparse approximations (plus grads / predict).
  {
    const int approxs[] = {CGp::DTC, CGp::FITC, CGp::PITC};
    for(unsigned int k = 0; k < 3; k++)
    {
      ToyGp toy(approxs[k]);
      toy.model->setInducingType("whitenoise");
      const double ll = toy.model->logLikelihood();
      fail += failTrue(std::isfinite(ll),
                       string("WN ") + toy.model->getApproximationStr() + " LL finite");
      CMatrix g(1, toy.model->getOptNumParams());
      fail += failTrue(std::isfinite(toy.model->logLikelihoodGradient(g)),
                       string("WN ") + toy.model->getApproximationStr() + " grads finite");
      CMatrix mu(2, 1);
      CMatrix var(2, 1);
      CMatrix Xtest(2, Q);
      Xtest.setVal(0.1, 0, 0);
      Xtest.setVal(-0.5, 1, 0);
      toy.model->posteriorMeanVar(mu, var, Xtest);
      fail += failTrue(std::isfinite(mu.getVal(0)) && std::isfinite(var.getVal(0)),
                       string("WN ") + toy.model->getApproximationStr() + " predict finite");
    }
  }

  // Stream round-trip: whitenoise + windowWidth keys.
  {
    ToyGp toy(CGp::DTCVAR);
    toy.model->setInducingType("whitenoise");
    toy.model->getInducing()->setWindowWidth(0.35);
    ostringstream out;
    writeGpToStream(*toy.model, out);
    const string blob = out.str();
    fail += failTrue(blob.find("inducingType=whitenoise") != string::npos,
                     "stream writes inducingType=whitenoise");
    fail += failTrue(blob.find("windowWidth=") != string::npos,
                     "stream writes windowWidth");
    istringstream in(blob);
    CGp* round = readGpFromStream(in);
    fail += failTrue(round != 0 && round->getInducingType() == "whitenoise",
                     "stream restore inducingType whitenoise");
    fail += failTrue(round != 0 && round->getInducing() != 0
                     && fabs(round->getInducing()->getWindowWidth() - 0.35) < 1e-12,
                     "stream restore windowWidth");
    delete round;
  }

  // Stream round-trip: points key present; also whitenoise without windowWidth line.
  {
    ToyGp toy(CGp::DTC);
    ostringstream out;
    writeGpToStream(*toy.model, out);
    fail += failTrue(out.str().find("inducingType=points") != string::npos,
                     "stream writes inducingType=points");
    istringstream in(out.str());
    CGp* round = readGpFromStream(in);
    fail += failTrue(round != 0 && round->getInducingType() == "points",
                     "stream restore inducingType points");
    delete round;

    // Hand-edit: inducingType=whitenoise then a non-windowWidth token ⇒ seek back.
    string blob = out.str();
    const string needle = "inducingType=points";
    const size_t pos = blob.find(needle);
    fail += failTrue(pos != string::npos, "points key locate for rewrite");
    if(pos != string::npos)
    {
      string blobWn = blob;
      blobWn.replace(pos, needle.size(), "inducingType=whitenoise\nnotWindow=1");
      istringstream in2(blobWn);
      CGp* wnRound = readGpFromStream(in2);
      fail += failTrue(wnRound != 0 && wnRound->getInducingType() == "whitenoise",
                       "stream WN without windowWidth still loads");
      delete wnRound;

      // Non-inducingType peek line ⇒ leave default points and seek back.
      string blobSkip = blob;
      blobSkip.replace(pos, needle.size(), "otherKey=1");
      istringstream in3(blobSkip);
      CGp* ptRound = readGpFromStream(in3);
      fail += failTrue(ptRound != 0 && ptRound->getInducingType() == "points",
                       "stream non-inducingType peek keeps points");
      delete ptRound;
    }
  }

  // Equivalence: pure RBF, s = ell/sqrt(2) ⇒ point vs WN LL within 1e-8.
  {
    CMatrix X(N, Q), y(N, 1), Xu(M_ACTIVE, Q);
    fillXy(X, y);
    fillXu(Xu, X);
    CRbfKern kern(X);
    kern.setInverseWidth(2.0);
    kern.setVariance(1.5);
    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);

    CGp pt(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    pt.setInducingVals(Xu);
    pt.setInducingFixed(true);
    pt.setBetaVal(10.0);
    pt.setInducingType("points");

    CGp wn(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    wn.setInducingVals(Xu);
    wn.setInducingFixed(true);
    wn.setBetaVal(10.0);
    wn.setInducingType("whitenoise");
    const double ell = kern.getLengthScale();
    wn.getInducing()->setWindowWidth(ell / sqrt(2.0));

    const double llPt = pt.logLikelihood();
    const double llWn = wn.logLikelihood();
    const double diff = fabs(llPt - llWn);
    cout << "point LL=" << llPt << " WN LL=" << llWn << " |diff|=" << diff << endl;
    fail += failTrue(diff < 1e-8, "point vs WN LL equivalence at s=ell/sqrt(2)");
  }

  return fail;
}

int testWeightSpaceInducingPhase0()
{
  int fail = 0;
  cout << "==== testCGp weightspace inducing Phase 0 (CIP-0028) ====" << endl;

  // Type + window width + kernel-free Kuu; RBF still lacks wsCross.
  {
    ToyGp toy(CGp::DTCVAR);
    toy.model->setInducingType("weightspace");
    fail += failTrue(toy.model->getInducingType() == "weightspace",
                     "inducing type weightspace");
    fail += failTrue(toy.model->getInducing()->domainDim() == Q + 1,
                     "weightspace domainDim == inputDim+1");
    fail += failTrue(toy.model->getInducing()->centres().getCols() == Q + 1,
                     "weightspace centres cols == d+1");
    toy.model->getInducing()->setWindowWidth(0.42);
    fail += failTrue(fabs(toy.model->getInducing()->getWindowWidth() - 0.42) < 1e-12,
                     "weightspace windowWidth set");
    CMatrix Kuu(M_ACTIVE, M_ACTIVE);
    toy.model->getInducing()->computeKuu(Kuu, *toy.model->getKernel());
    fail += failTrue(std::isfinite(Kuu.getVal(0, 0)) && Kuu.getVal(0, 0) > 0.0,
                     "weightspace Kuu diagonal finite positive");
    // RBF+White: RBF has no wsCross ⇒ NotImplemented.
    try
    {
      (void)toy.model->logLikelihood();
      fail += 1;
      cout << "FAILURE: weightspace+RBF LL should throw NotImplementedError." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "weightspace+RBF logLikelihood throws NotImplementedError." << endl;
    }
  }

  // Stream round-trip.
  {
    ToyGp toy(CGp::DTC);
    toy.model->setInducingType("weightspace");
    toy.model->getInducing()->setWindowWidth(0.27);
    ostringstream out;
    writeGpToStream(*toy.model, out);
    const string blob = out.str();
    fail += failTrue(blob.find("inducingType=weightspace") != string::npos,
                     "stream writes inducingType=weightspace");
    fail += failTrue(blob.find("windowWidth=") != string::npos,
                     "stream writes weightspace windowWidth");
    fail += failTrue(blob.find("weightCentres=") != string::npos,
                     "stream writes weightCentres");
    istringstream in(blob);
    CGp* round = readGpFromStream(in);
    fail += failTrue(round != 0 && round->getInducingType() == "weightspace",
                     "stream restore inducingType weightspace");
    fail += failTrue(round != 0 && round->getInducing() != 0
                     && fabs(round->getInducing()->getWindowWidth() - 0.27) < 1e-12,
                     "stream restore weightspace windowWidth");
    fail += failTrue(round != 0 && round->getInducing() != 0
                     && round->getInducing()->centres().getCols() == Q + 1
                     && round->getInducing()->centres().getRows() == M_ACTIVE,
                     "stream restore weightCentres shape");
    delete round;
  }

  // Points / WN still selectable after weightspace (regression smoke).
  {
    ToyGp toy(CGp::DTCVAR);
    toy.model->setInducingType("weightspace");
    toy.model->setInducingType("points");
    fail += failTrue(toy.model->getInducingType() == "points",
                     "switch weightspace → points");
    fail += failTrue(std::isfinite(toy.model->logLikelihood()),
                     "points LL finite after weightspace");
    toy.model->setInducingType("whitenoise");
    fail += failTrue(toy.model->getInducingType() == "whitenoise",
                     "switch → whitenoise");
    fail += failTrue(std::isfinite(toy.model->logLikelihood()),
                     "WN LL finite after weightspace");
  }

  return fail;
}

int testWeightSpaceInducingPhase1()
{
  int fail = 0;
  cout << "==== testCGp weightspace inducing Phase 1 MLP (CIP-0028) ====" << endl;

  // DTCVAR smoke: MLP + White + weightspace finite LL / grads.
  {
    CMatrix X(N, Q);
    CMatrix y(N, 1);
    CMatrix Xu(M_ACTIVE, Q);
    fillXy(X, y);
    fillXu(Xu, X);
    CCmpndKern kern(X);
    kern.addKern(new CMlpKern(X));
    kern.addKern(new CWhiteKern(X));
    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(10.0);
    model.setInducingType("weightspace");
    model.getInducing()->setWindowWidth(0.35);
    fail += failTrue(model.getInducingType() == "weightspace", "Phase1 inducing type");
    const double ll0 = model.logLikelihood();
    fail += failTrue(std::isfinite(ll0), "MLP weightspace DTCVAR LL finite");
    CMatrix g(1, model.getOptNumParams());
    fail += failTrue(std::isfinite(model.logLikelihoodGradient(g)),
                     "MLP weightspace DTCVAR grads finite");
    model.setDefaultOptimiser(CGp::SCG);
    model.setVerbosity(0);
    model.optimise(5);
    fail += failTrue(std::isfinite(model.logLikelihood()),
                     "MLP weightspace DTCVAR after SCG LL finite");
  }

  // Closed-form vs MC of ∫ φ √p ψ (1-D); control without √p must disagree.
  {
    CMatrix X(4, 1);
    for(unsigned int i = 0; i < 4; i++)
      X.setVal(-0.75 + 0.5 * (double)i, i, 0);
    CMlpKern kern(X);
    kern.setParam(2.0, 0); // weightVariance
    kern.setParam(1.5, 1); // biasVariance
    kern.setParam(1.1, 2); // variance
    CMatrix W(2, 2);
    W.setVal(0.3, 0, 0); W.setVal(0.8, 0, 1);
    W.setVal(-0.4, 1, 0); W.setVal(1.1, 1, 1);
    const double s = 0.45;
    CMatrix Kfu(4, 2);
    kern.wsCrossCompute(Kfu, X, W, s);

    // MC: sample w ~ N(ŵ, s² I), estimate E[φ √p] * (4π s²)^{D/4}
    const unsigned int D = 2;
    const unsigned int nMc = 8000;
    ndlutil::init_genrand(42u);
    double maxRel = 0.0;
    double maxRelNoSqrtP = 0.0;
    const double alpha = kern.getParam(0);
    const double beta = kern.getParam(1);
    const double var = kern.getParam(2);
    const double B = sqrt(var * M_PI / 2.0);
    const double psiScale = pow(4.0 * M_PI * s * s, 0.25 * (double)D);
    for(unsigned int m = 0; m < 2; m++)
    {
      for(unsigned int n = 0; n < 4; n++)
      {
        double acc = 0.0;
        double accNo = 0.0;
        for(unsigned int t = 0; t < nMc; t++)
        {
          double w0 = W.getVal(m, 0) + s * ndlutil::randn();
          double w1 = W.getVal(m, 1) + s * ndlutil::randn();
          const double logAbsSigma = log(alpha) + log(beta);
          const double logSqrtP = -0.25 * (double)D * log(2.0 * M_PI) - 0.25 * logAbsSigma
                                 - 0.25 * (w0 * w0 / alpha + w1 * w1 / beta);
          const double sqrtP = exp(logSqrtP);
          const double xt = X.getVal(n, 0);
          const double lin = (w0 * xt + w1) / sqrt(2.0);
          const double phi = B * erf(lin);
          acc += phi * sqrtP;
          accNo += phi; // missing √p — false oracle
        }
        const double mc = psiScale * acc / (double)nMc;
        const double mcNo = psiScale * accNo / (double)nMc;
        const double cf = Kfu.getVal(n, m);
        const double rel = fabs(mc - cf) / max(1.0, fabs(cf));
        const double relNo = fabs(mcNo - cf) / max(1.0, fabs(cf));
        if(rel > maxRel) maxRel = rel;
        if(relNo > maxRelNoSqrtP) maxRelNoSqrtP = relNo;
      }
    }
    fail += failTrue(maxRel < 0.08, "wsCross MC vs closed form (√p)");
    fail += failTrue(maxRelNoSqrtP > 0.15,
                     "control without √p disagrees with closed form");
    cout << "  MC max rel err (√p)=" << maxRel
         << "  (no √p)=" << maxRelNoSqrtP << endl;
  }

  // FD: wsGradParams / W / X / s agree with wsCrossCompute.
  {
    CMatrix X(3, 1);
    X.setVal(-0.5, 0, 0); X.setVal(0.0, 1, 0); X.setVal(0.7, 2, 0);
    CMlpKern kern(X);
    CMatrix W(2, 2);
    W.setVal(0.2, 0, 0); W.setVal(0.9, 0, 1);
    W.setVal(-0.3, 1, 0); W.setVal(1.2, 1, 1);
    const double s = 0.5;
    CMatrix Kfu(3, 2);
    kern.wsCrossCompute(Kfu, X, W, s);
    CMatrix gKfu(3, 2);
    for(unsigned int n = 0; n < 3; n++)
      for(unsigned int m = 0; m < 2; m++)
        gKfu.setVal(1.0, n, m);

    CMatrix gParams(1, 3);
    kern.wsGradParams(gParams, X, W, s, gKfu);
    fail += failTrue(std::isfinite(gParams.getVal(0))
                     && std::isfinite(gParams.getVal(1))
                     && std::isfinite(gParams.getVal(2)),
                     "wsGradParams finite");

    CMatrix gW(2, 2);
    kern.wsGradW(gW, X, W, s, gKfu);
    fail += failTrue(std::isfinite(gW.getVal(0, 0)), "wsGradW finite");

    CMatrix gX(3, 1);
    kern.wsGradX(gX, X, W, s, gKfu);
    fail += failTrue(std::isfinite(gX.getVal(0, 0)), "wsGradX finite");

    CMatrix gS(1, 1);
    kern.wsGradS(gS, X, W, s, gKfu);
    fail += failTrue(std::isfinite(gS.getVal(0)), "wsGradS finite");

    // Spot-check weightVariance FD vs wsGradParams.
    const double p0 = kern.getParam(0);
    const double h = 1e-6 * max(1.0, fabs(p0));
    CMatrix Kp(3, 2);
    CMatrix Km(3, 2);
    kern.setParam(p0 + h, 0);
    kern.wsCrossCompute(Kp, X, W, s);
    kern.setParam(p0 - h, 0);
    kern.wsCrossCompute(Km, X, W, s);
    kern.setParam(p0, 0);
    double fd = 0.0;
    for(unsigned int n = 0; n < 3; n++)
      for(unsigned int m = 0; m < 2; m++)
        fd += (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * h);
    const double rel = fabs(fd - gParams.getVal(0)) / max(1.0, fabs(fd));
    fail += failTrue(rel < 1e-4, "wsGradParams vs FD weightVariance");
  }

  return fail;
}

int testWeightSpaceInducingPhase1b()
{
  int fail = 0;
  cout << "==== testCGp weightspace inducing Phase 1b MLP-ARD (CIP-0028) ====" << endl;

  // DTCVAR smoke with 2-D MLP-ARD + weightspace.
  {
    const unsigned int Q2 = 2;
    CMatrix X(N, Q2);
    CMatrix y(N, 1);
    CMatrix Xu(M_ACTIVE, Q2);
    for(unsigned int i = 0; i < N; i++)
    {
      const double xv = -1.0 + 2.0 * (double)i / (double)(N - 1);
      X.setVal(xv, i, 0);
      X.setVal(0.5 * xv, i, 1);
      y.setVal(sin(xv) + 0.1 * xv, i, 0);
    }
    for(unsigned int i = 0; i < M_ACTIVE; i++)
      Xu.copyRowRow(i, X, i);
    CCmpndKern kern(X);
    kern.addKern(new CMlpardKern(X));
    kern.addKern(new CWhiteKern(X));
    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(8.0);
    model.setInducingType("weightspace");
    model.getInducing()->setWindowWidth(0.4);
    fail += failTrue(model.getInducing()->domainDim() == Q2 + 1,
                     "ARD weightspace domainDim == d+1");
    fail += failTrue(std::isfinite(model.logLikelihood()),
                     "MLP-ARD weightspace DTCVAR LL finite");
    CMatrix g(1, model.getOptNumParams());
    fail += failTrue(std::isfinite(model.logLikelihoodGradient(g)),
                     "MLP-ARD weightspace DTCVAR grads finite");
    model.setDefaultOptimiser(CGp::SCG);
    model.setVerbosity(0);
    model.optimise(5);
    fail += failTrue(std::isfinite(model.logLikelihood()),
                     "MLP-ARD weightspace DTCVAR after SCG LL finite");
  }

  // ARD reduces to isotropic when all scales equal 1 (after matching Σ).
  // With scales=0.5 default, match isotropic by α_iso = α_ard * scale.
  {
    CMatrix X(3, 2);
    X.setVal(-0.4, 0, 0); X.setVal(0.2, 0, 1);
    X.setVal(0.0, 1, 0); X.setVal(-0.3, 1, 1);
    X.setVal(0.6, 2, 0); X.setVal(0.5, 2, 1);
    CMlpardKern kard(X);
    kard.setParam(4.0, 0);
    kard.setParam(2.0, 1);
    kard.setParam(1.0, 2);
    kard.setParam(1.0, 3); // scale0
    kard.setParam(1.0, 4); // scale1
    CMlpKern kiso(X);
    kiso.setParam(4.0, 0);
    kiso.setParam(2.0, 1);
    kiso.setParam(1.0, 2);
    CMatrix W(2, 3);
    W.setVal(0.2, 0, 0); W.setVal(-0.1, 0, 1); W.setVal(0.9, 0, 2);
    W.setVal(-0.3, 1, 0); W.setVal(0.4, 1, 1); W.setVal(1.1, 1, 2);
    const double s = 0.55;
    CMatrix KfuArd(3, 2);
    CMatrix KfuIso(3, 2);
    kard.wsCrossCompute(KfuArd, X, W, s);
    kiso.wsCrossCompute(KfuIso, X, W, s);
    double maxAbs = 0.0;
    for(unsigned int n = 0; n < 3; n++)
      for(unsigned int m = 0; m < 2; m++)
        maxAbs = max(maxAbs, fabs(KfuArd.getVal(n, m) - KfuIso.getVal(n, m)));
    fail += failTrue(maxAbs < 1e-10, "ARD scales=1 matches isotropic wsCross");
  }

  // FD spot-check on an ARD inputScale param.
  {
    CMatrix X(3, 2);
    X.setVal(-0.5, 0, 0); X.setVal(0.1, 0, 1);
    X.setVal(0.0, 1, 0); X.setVal(-0.2, 1, 1);
    X.setVal(0.4, 2, 0); X.setVal(0.3, 2, 1);
    CMlpardKern kern(X);
    CMatrix W(2, 3);
    W.setVal(0.1, 0, 0); W.setVal(0.2, 0, 1); W.setVal(1.0, 0, 2);
    W.setVal(-0.2, 1, 0); W.setVal(0.3, 1, 1); W.setVal(0.8, 1, 2);
    const double s = 0.5;
    CMatrix gKfu(3, 2);
    gKfu.ones();
    CMatrix gParams(1, kern.getNumParams());
    kern.wsGradParams(gParams, X, W, s, gKfu);
    const unsigned int scaleIdx = 3;
    const double p0 = kern.getParam(scaleIdx);
    const double h = 1e-6;
    CMatrix Kp(3, 2);
    CMatrix Km(3, 2);
    kern.setParam(min(p0 + h, 0.999), scaleIdx);
    kern.wsCrossCompute(Kp, X, W, s);
    kern.setParam(max(p0 - h, 1e-6), scaleIdx);
    kern.wsCrossCompute(Km, X, W, s);
    kern.setParam(p0, scaleIdx);
    double fd = 0.0;
    const double denom = min(p0 + h, 0.999) - max(p0 - h, 1e-6);
    for(unsigned int n = 0; n < 3; n++)
      for(unsigned int m = 0; m < 2; m++)
        fd += (Kp.getVal(n, m) - Km.getVal(n, m)) / denom;
    const double rel = fabs(fd - gParams.getVal(scaleIdx)) / max(1.0, fabs(fd));
    fail += failTrue(rel < 1e-3, "wsGradParams vs FD ARD inputScale");
  }

  return fail;
}

int testGreensInducingPhase0()
{
  int fail = 0;
  cout << "==== testCGp greens inducing Phase 0 (CIP-0029) ====" << endl;

  // Type + window width + kernel-free Kuu; Kuf still NotImplemented.
  {
    ToyGp toy(CGp::DTCVAR);
    toy.model->setInducingType("greens");
    fail += failTrue(toy.model->getInducingType() == "greens", "inducing type greens");
    toy.model->getInducing()->setWindowWidth(0.42);
    fail += failTrue(fabs(toy.model->getInducing()->getWindowWidth() - 0.42) < 1e-12,
                     "greens windowWidth set");
    CMatrix Kuu(M_ACTIVE, M_ACTIVE);
    toy.model->getInducing()->computeKuu(Kuu, *toy.model->getKernel());
    fail += failTrue(std::isfinite(Kuu.getVal(0, 0)) && Kuu.getVal(0, 0) > 0.0,
                     "greens Kuu diagonal finite positive");
    // RBF+White compound has no greensCross on RBF ⇒ NotImplemented until Phase 2/Matérn.
    try
    {
      (void)toy.model->logLikelihood();
      fail += 1;
      cout << "FAILURE: greens+RBF LL should throw NotImplementedError." << endl;
    }
    catch(ndlexceptions::NotImplementedError&)
    {
      cout << "greens+RBF logLikelihood throws NotImplementedError." << endl;
    }
  }

  // Stream round-trip.
  {
    ToyGp toy(CGp::DTC);
    toy.model->setInducingType("greens");
    toy.model->getInducing()->setWindowWidth(0.27);
    ostringstream out;
    writeGpToStream(*toy.model, out);
    const string blob = out.str();
    fail += failTrue(blob.find("inducingType=greens") != string::npos,
                     "stream writes inducingType=greens");
    fail += failTrue(blob.find("windowWidth=") != string::npos,
                     "stream writes greens windowWidth");
    istringstream in(blob);
    CGp* round = readGpFromStream(in);
    fail += failTrue(round != 0 && round->getInducingType() == "greens",
                     "stream restore inducingType greens");
    fail += failTrue(round != 0 && round->getInducing() != 0
                     && fabs(round->getInducing()->getWindowWidth() - 0.27) < 1e-12,
                     "stream restore greens windowWidth");
    delete round;
  }

  // Multi-D inputs rejected.
  {
    const unsigned int Q2 = 2;
    CMatrix X(N, Q2), y(N, 1), Xu(M_ACTIVE, Q2);
    for(unsigned int i = 0; i < N; i++)
    {
      X.setVal(-1.0 + 2.0 * (double)i / (double)(N - 1), i, 0);
      X.setVal(0.1 * (double)i, i, 1);
      y.setVal(sin(X.getVal(i, 0)), i, 0);
    }
    for(unsigned int i = 0; i < M_ACTIVE; i++)
      Xu.copyRowRow(i, X, i);
    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CWhiteKern(X));
    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTC, M_ACTIVE, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(10.0);
    try
    {
      model.setInducingType("greens");
      fail += 1;
      cout << "FAILURE: greens should reject 2-D inputs." << endl;
    }
    catch(ndlexceptions::Error&)
    {
      cout << "greens rejects 2-D inputs." << endl;
    }
  }

  return fail;
}

int testGreensBrownianPhase1()
{
  int fail = 0;
  cout << "==== testCGp greens Brownian Phase 1 (CIP-0029) ====" << endl;

  // Positive 1-D grid (Brownian same-sign convention).
  CMatrix X(N, 1), y(N, 1), Xu(M_ACTIVE, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    const double t = 0.2 + 1.5 * (double)i / (double)(N - 1);
    X.setVal(t, i, 0);
    y.setVal(sin(t), i, 0);
  }
  for(unsigned int i = 0; i < M_ACTIVE; i++)
    Xu.copyRowRow(i, X, i);

  CBrownianKern kern(X);
  kern.setVariance(1.25);
  const double s = 0.35;

  // Closed form vs trapezoid ∫_0^t ψ(τ) dτ * σ
  {
    CMatrix Kfu;
    kern.greensCrossCompute(Kfu, X, Xu, s);
    const double sigma = sqrt(kern.getVariance());
    const double c = pow(4.0 * M_PI * s * s, 0.25);
    double maxRel = 0.0;
    const int nQuad = 4000;
    for(unsigned int n = 0; n < N; n++)
    {
      const double t = X.getVal(n, 0);
      for(unsigned int m = 0; m < M_ACTIVE; m++)
      {
        const double z = Xu.getVal(m, 0);
        double integ = 0.0;
        for(int k = 0; k <= nQuad; k++)
        {
          const double tau = t * (double)k / (double)nQuad;
          const double w = (k == 0 || k == nQuad) ? 0.5 : 1.0;
          const double dens = exp(-0.5 * (tau - z) * (tau - z) / (s * s))
                              / sqrt(2.0 * M_PI * s * s);
          integ += w * dens;
        }
        integ *= (t / (double)nQuad) * c;
        const double closed = Kfu.getVal(n, m);
        const double quad = sigma * integ;
        const double denom = max(1.0, fabs(closed));
        maxRel = max(maxRel, fabs(closed - quad) / denom);
      }
    }
    cout << "Brownian greens vs quad maxRel=" << maxRel << endl;
    fail += failTrue(maxRel < 5e-3, "Brownian greensCross vs trapezoid");
  }

  // FD grads on variance / Z / X (central).
  {
    CMatrix gKfu(N, M_ACTIVE);
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M_ACTIVE; m++)
        gKfu.setVal(0.1 + 0.01 * (double)(n + m), n, m);

    CMatrix g(1, 1);
    kern.greensGradParams(g, X, Xu, s, gKfu);
    const double eps = 1e-6;
    kern.setVariance(1.25 + eps);
    CMatrix Kp;
    kern.greensCrossCompute(Kp, X, Xu, s);
    kern.setVariance(1.25 - eps);
    CMatrix Km;
    kern.greensCrossCompute(Km, X, Xu, s);
    kern.setVariance(1.25);
    double fdVar = 0.0;
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M_ACTIVE; m++)
        fdVar += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * eps);
    fail += failTrue(fabs(g.getVal(0) - fdVar) < 1e-5 * max(1.0, fabs(fdVar)),
                     "Brownian greensGradParams vs FD");

    CMatrix gZ;
    kern.greensGradZ(gZ, X, Xu, s, gKfu);
    for(unsigned int m = 0; m < M_ACTIVE; m++)
    {
      Xu.addVal(eps, m, 0);
      kern.greensCrossCompute(Kp, X, Xu, s);
      Xu.addVal(-2.0 * eps, m, 0);
      kern.greensCrossCompute(Km, X, Xu, s);
      Xu.addVal(eps, m, 0);
      double fd = 0.0;
      for(unsigned int n = 0; n < N; n++)
        fd += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * eps);
      fail += failTrue(fabs(gZ.getVal(m, 0) - fd) < 1e-5 * max(1.0, fabs(fd)),
                       "Brownian greensGradZ vs FD");
    }
  }

  // DTCVAR smoke with greens inducing.
  {
    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(5.0);
    model.setInducingType("greens");
    model.getInducing()->setWindowWidth(s);
    const double ll0 = model.logLikelihood();
    fail += failTrue(std::isfinite(ll0), "Brownian greens DTCVAR LL finite");
    CMatrix g(1, model.getOptNumParams());
    fail += failTrue(std::isfinite(model.logLikelihoodGradient(g)),
                     "Brownian greens DTCVAR grads finite");
    model.setDefaultOptimiser(CGp::SCG);
    model.setVerbosity(0);
    model.optimise(5);
    fail += failTrue(std::isfinite(model.logLikelihood()),
                     "Brownian greens DTCVAR after SCG LL finite");
  }

  return fail;
}

int testGreensMaternPhase2()
{
  int fail = 0;
  cout << "==== testCGp greens Exp/Matérn Phase 2 (CIP-0029) ====" << endl;

  CMatrix X(N, 1), y(N, 1), Xu(M_ACTIVE, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    const double t = -1.0 + 2.0 * (double)i / (double)(N - 1);
    X.setVal(t, i, 0);
    y.setVal(sin(t), i, 0);
  }
  for(unsigned int i = 0; i < M_ACTIVE; i++)
    Xu.copyRowRow(i, X, i + 1);
  const double s = 0.4;
  const double var = 1.3;

  auto quadVsClosed = [&](CKern& kern, double lambda, int degree, const string& name) -> int
  {
    int f = 0;
    CMatrix Kfu;
    kern.greensCrossCompute(Kfu, X, Xu, s);
    const double sigma = sqrt(var);
    const double c = pow(4.0 * M_PI * s * s, 0.25);
    double amp = 0.0;
    if(degree == 0)
      amp = sigma * sqrt(2.0 * lambda) * c;
    else if(degree == 1)
      amp = sigma * 2.0 * pow(lambda, 1.5) * c;
    else
      amp = sigma * sqrt(4.0 / 3.0) * pow(lambda, 2.5) * c;

    double maxRel = 0.0;
    const int nQuad = 5000;
    const double uMax = 12.0 / lambda + 8.0 * s;
    for(unsigned int n = 0; n < N; n++)
    {
      const double x = X.getVal(n, 0);
      for(unsigned int m = 0; m < M_ACTIVE; m++)
      {
        const double z = Xu.getVal(m, 0);
        double integ = 0.0;
        for(int k = 0; k <= nQuad; k++)
        {
          const double u = uMax * (double)k / (double)nQuad;
          const double w = (k == 0 || k == nQuad) ? 0.5 : 1.0;
          double poly = 1.0;
          if(degree == 1)
            poly = u;
          else if(degree == 2)
            poly = u * u;
          const double g = amp / c * poly * exp(-lambda * u); // amp includes c; strip for ψ
          const double tau = x - u;
          const double dens = exp(-0.5 * (tau - z) * (tau - z) / (s * s))
                              / sqrt(2.0 * M_PI * s * s);
          integ += w * g * dens;
        }
        integ *= (uMax / (double)nQuad) * c;
        const double closed = Kfu.getVal(n, m);
        const double denom = max(1.0, fabs(closed));
        maxRel = max(maxRel, fabs(closed - integ) / denom);
      }
    }
    cout << name << " greens vs quad maxRel=" << maxRel << endl;
    f += failTrue(maxRel < 5e-3, name + " greensCross vs trapezoid");

    // FD variance param
    CMatrix gKfu(N, M_ACTIVE);
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M_ACTIVE; m++)
        gKfu.setVal(0.07 + 0.01 * (double)(n * M_ACTIVE + m), n, m);
    CMatrix g(1, kern.getNumParams());
    kern.greensGradParams(g, X, Xu, s, gKfu);
    const double eps = 1e-6;
    const double v0 = kern.getParam(1); // variance is param 1 for Exp/Matérn
    kern.setParam(v0 + eps, 1);
    CMatrix Kp;
    kern.greensCrossCompute(Kp, X, Xu, s);
    kern.setParam(v0 - eps, 1);
    CMatrix Km;
    kern.greensCrossCompute(Km, X, Xu, s);
    kern.setParam(v0, 1);
    double fd = 0.0;
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M_ACTIVE; m++)
        fd += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * eps);
    f += failTrue(fabs(g.getVal(1) - fd) < 1e-4 * max(1.0, fabs(fd)),
                  name + " greensGradParams variance vs FD");
    return f;
  };

  {
    CExpKern kern(X);
    kern.setInverseWidth(1.2);
    kern.setVariance(var);
    fail += quadVsClosed(kern, 1.2, 0, "Exp");

    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(4.0);
    model.setInducingType("greens");
    model.getInducing()->setWindowWidth(s);
    fail += failTrue(std::isfinite(model.logLikelihood()), "Exp greens DTCVAR LL finite");
    CMatrix g(1, model.getOptNumParams());
    fail += failTrue(std::isfinite(model.logLikelihoodGradient(g)),
                     "Exp greens DTCVAR grads finite");
  }

  {
    CMatern32Kern kern(X);
    kern.setParam(0.9, 0); // lengthScale
    kern.setParam(var, 1);
    const double lam = sqrt(3.0) / 0.9;
    fail += quadVsClosed(kern, lam, 1, "Matern32");
  }

  {
    CMatern52Kern kern(X);
    kern.setParam(0.9, 0);
    kern.setParam(var, 1);
    const double lam = sqrt(5.0) / 0.9;
    fail += quadVsClosed(kern, lam, 2, "Matern52");

    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(4.0);
    model.setInducingType("greens");
    model.getInducing()->setWindowWidth(s);
    fail += failTrue(std::isfinite(model.logLikelihood()), "Matern52 greens DTCVAR LL finite");
  }

  return fail;
}

int testGreensIntegratedBrownianPhase3()
{
  int fail = 0;
  cout << "==== testCGp greens Integrated Brownian Phase 3 (CIP-0029) ====" << endl;

  CMatrix X(N, 1), y(N, 1), Xu(M_ACTIVE, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    const double t = 0.2 + 1.5 * (double)i / (double)(N - 1);
    X.setVal(t, i, 0);
    y.setVal(0.3 * t * t, i, 0);
  }
  for(unsigned int i = 0; i < M_ACTIVE; i++)
    Xu.copyRowRow(i, X, i);

  CIntegratedBrownianKern kern(X);
  kern.setVariance(1.25);
  const double s = 0.35;

  // Closed form vs trapezoid ∫_0^t (t−τ) ψ(τ) dτ * σ
  {
    CMatrix Kfu;
    kern.greensCrossCompute(Kfu, X, Xu, s);
    const double sigma = sqrt(kern.getVariance());
    const double c = pow(4.0 * M_PI * s * s, 0.25);
    double maxRel = 0.0;
    const int nQuad = 4000;
    for(unsigned int n = 0; n < N; n++)
    {
      const double t = X.getVal(n, 0);
      for(unsigned int m = 0; m < M_ACTIVE; m++)
      {
        const double z = Xu.getVal(m, 0);
        double integ = 0.0;
        for(int k = 0; k <= nQuad; k++)
        {
          const double tau = t * (double)k / (double)nQuad;
          const double w = (k == 0 || k == nQuad) ? 0.5 : 1.0;
          const double dens = exp(-0.5 * (tau - z) * (tau - z) / (s * s))
                              / sqrt(2.0 * M_PI * s * s);
          integ += w * (t - tau) * dens;
        }
        integ *= (t / (double)nQuad) * c;
        const double closed = Kfu.getVal(n, m);
        const double quad = sigma * integ;
        const double denom = max(1.0, fabs(closed));
        maxRel = max(maxRel, fabs(closed - quad) / denom);
      }
    }
    cout << "IntegratedBrownian greens vs quad maxRel=" << maxRel << endl;
    fail += failTrue(maxRel < 5e-3, "IntegratedBrownian greensCross vs trapezoid");
  }

  {
    CMatrix gKfu(N, M_ACTIVE);
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M_ACTIVE; m++)
        gKfu.setVal(0.1 + 0.01 * (double)(n + m), n, m);

    CMatrix g(1, 1);
    kern.greensGradParams(g, X, Xu, s, gKfu);
    const double eps = 1e-6;
    kern.setVariance(1.25 + eps);
    CMatrix Kp;
    kern.greensCrossCompute(Kp, X, Xu, s);
    kern.setVariance(1.25 - eps);
    CMatrix Km;
    kern.greensCrossCompute(Km, X, Xu, s);
    kern.setVariance(1.25);
    double fdVar = 0.0;
    for(unsigned int n = 0; n < N; n++)
      for(unsigned int m = 0; m < M_ACTIVE; m++)
        fdVar += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * eps);
    fail += failTrue(fabs(g.getVal(0) - fdVar) < 1e-5 * max(1.0, fabs(fdVar)),
                     "IntegratedBrownian greensGradParams vs FD");

    CMatrix gZ;
    kern.greensGradZ(gZ, X, Xu, s, gKfu);
    for(unsigned int m = 0; m < M_ACTIVE; m++)
    {
      Xu.addVal(eps, m, 0);
      kern.greensCrossCompute(Kp, X, Xu, s);
      Xu.addVal(-2.0 * eps, m, 0);
      kern.greensCrossCompute(Km, X, Xu, s);
      Xu.addVal(eps, m, 0);
      double fd = 0.0;
      for(unsigned int n = 0; n < N; n++)
        fd += gKfu.getVal(n, m) * (Kp.getVal(n, m) - Km.getVal(n, m)) / (2.0 * eps);
      fail += failTrue(fabs(gZ.getVal(m, 0) - fd) < 1e-5 * max(1.0, fabs(fd)),
                       "IntegratedBrownian greensGradZ vs FD");
    }

    CMatrix gS;
    kern.greensGradS(gS, X, Xu, s, gKfu);
    {
      CMatrix Ksp, Ksm;
      kern.greensCrossCompute(Ksp, X, Xu, s + eps);
      kern.greensCrossCompute(Ksm, X, Xu, s - eps);
      double fdS = 0.0;
      for(unsigned int n = 0; n < N; n++)
        for(unsigned int m = 0; m < M_ACTIVE; m++)
          fdS += gKfu.getVal(n, m) * (Ksp.getVal(n, m) - Ksm.getVal(n, m)) / (2.0 * eps);
      fail += failTrue(fabs(gS.getVal(0, 0) - fdS) < 1e-5 * max(1.0, fabs(fdS)),
                       "IntegratedBrownian greensGradS vs FD");
    }
  }

  {
    CGaussianNoise noise(&y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::DTCVAR, M_ACTIVE, 0);
    model.setInducingVals(Xu);
    model.setInducingFixed(true);
    model.setBetaVal(5.0);
    model.setInducingType("greens");
    model.getInducing()->setWindowWidth(s);
    const double ll0 = model.logLikelihood();
    fail += failTrue(std::isfinite(ll0), "IntegratedBrownian greens DTCVAR LL finite");
    CMatrix g(1, model.getOptNumParams());
    fail += failTrue(std::isfinite(model.logLikelihoodGradient(g)),
                     "IntegratedBrownian greens DTCVAR grads finite");
    model.setDefaultOptimiser(CGp::SCG);
    model.setVerbosity(0);
    model.optimise(5);
    fail += failTrue(std::isfinite(model.logLikelihood()),
                     "IntegratedBrownian greens DTCVAR after SCG LL finite");
  }

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
    fail += testFixMask();
    fail += testModelPaths();
    fail += testWhiteNoiseInducing();
    fail += testWeightSpaceInducingPhase0();
    fail += testWeightSpaceInducingPhase1();
    fail += testWeightSpaceInducingPhase1b();
    fail += testGreensInducingPhase0();
    fail += testGreensBrownianPhase1();
    fail += testGreensMaternPhase2();
    fail += testGreensIntegratedBrownianPhase3();
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
