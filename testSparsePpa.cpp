/* Prototype gates for CSparsePpa (CIP-0026 experiment).
   White-kernel caveat: CWhiteKern::compute(X,X2) is always zero, so DTC with
   M=N does not recover the dense Gram when the compound kernel includes white.
   Full-data recovery gates use RBF+Bias only; β carries the observation noise. */

#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CPpa.h"
#include "CSparsePpa.h"
#include "ndlstrutil.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

static int fails = 0;

static void expect(bool cond, const string& msg)
{
  if(cond)
    cout << "PASS: " << msg << endl;
  else
    {
      cout << "FAIL: " << msg << endl;
      fails++;
    }
}

static CMatrix loadDat(const string& path, int rows, int cols)
{
  CMatrix A(rows, cols);
  ifstream in(path.c_str());
  if(!in)
    throw ndlexceptions::FileReadError(path);
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      {
        double v = 0.0;
        if(!(in >> v))
          throw ndlexceptions::FileFormatError(path);
        A.setVal(v, i, j);
      }
  return A;
}

static double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

static void makeKernNoWhite(CCmpndKern& kern, const CMatrix& X)
{
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CBiasKern(X));
  CMatrix kp(1, 3);
  // Transformed params akin to Phase-0 rbf/bias (drop white's -2).
  kp.setVal(0.0, 0); kp.setVal(0.0, 1); kp.setVal(-2.0, 2);
  kern.setTransParams(kp);
}

static void makeKernWithWhite(CCmpndKern& kern, const CMatrix& X)
{
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CBiasKern(X));
  kern.addKern(new CWhiteKern(X));
  CMatrix kp(1, 4);
  kp.setVal(0.0, 0); kp.setVal(0.0, 1); kp.setVal(-2.0, 2); kp.setVal(-2.0, 3);
  kern.setTransParams(kp);
}

int main()
{
  const string dir = "testdata" + ndlstrutil::dirSep() + "ppa" + ndlstrutil::dirSep();
  const int N = 30;
  const int D = 2;
  const double tol = 1e-5;
  const double tolEm = 1e-4;

  CMatrix X = loadDat(dir + "phase0_X.txt", N, D);
  CMatrix y = loadDat(dir + "phase0_y.txt", N, 1);

  // ---- Gate: DTC M=N vs dense CPpa (no white) after init ----
  {
    CCmpndKern kernD(X); makeKernNoWhite(kernD, X);
    CCmpndKern kernS(X); makeKernNoWhite(kernS, X);
    CProbitNoise noiseD(&y); noiseD.initParams();
    CProbitNoise noiseS(&y); noiseS.initParams();
    CPpa dense(X, y, kernD, noiseD, 0);
    CSparsePpa sp(X, y, kernS, noiseS, CGp::DTC, static_cast<unsigned int>(N), 1.0, 0);
    expect(maxAbsDiff(sp.getFBar(), dense.fBar) < tol, "DTC M=N fBar vs dense (init, no white)");
    expect(maxAbsDiff(sp.getF(), dense.f) < tol, "DTC M=N f vs dense (init, no white)");
    expect(maxAbsDiff(sp.getG(), dense.g) < tol, "DTC M=N g vs dense (init, no white)");
  }

  // ---- Gate: DTC M=N vs dense after several E-steps (real posterior exercise) ----
  {
    CCmpndKern kernD(X); makeKernNoWhite(kernD, X);
    CCmpndKern kernS(X); makeKernNoWhite(kernS, X);
    CProbitNoise noiseD(&y); noiseD.initParams();
    CProbitNoise noiseS(&y); noiseS.initParams();
    CPpa dense(X, y, kernD, noiseD, 0);
    CSparsePpa sp(X, y, kernS, noiseS, CGp::DTC, static_cast<unsigned int>(N), 1.0, 0);
    dense.optimise(3, 0, 0);
    sp.optimise(3, 0);
    expect(maxAbsDiff(sp.getFBar(), dense.fBar) < tolEm,
           "DTC M=N fBar vs dense (3 E-steps, no white)");
    expect(maxAbsDiff(sp.getF(), dense.f) < tolEm,
           "DTC M=N f vs dense (3 E-steps, no white)");
  }

  // ---- Gate: FTC request remaps to DTC(M=N) ----
  {
    CCmpndKern kernD(X); makeKernNoWhite(kernD, X);
    CCmpndKern kernS(X); makeKernNoWhite(kernS, X);
    CProbitNoise noiseD(&y); noiseD.initParams();
    CProbitNoise noiseS(&y); noiseS.initParams();
    CPpa dense(X, y, kernD, noiseD, 0);
    CSparsePpa sp(X, y, kernS, noiseS, CGp::FTC, 0, 1.0, 0);
    expect(sp.getApproxType() == CGp::DTC, "FTC request remaps to DTC");
    expect(sp.getNumActive() == static_cast<unsigned int>(N), "FTC remap uses M=N");
    expect(maxAbsDiff(sp.getFBar(), dense.fBar) < tol, "FTC-remap fBar vs dense (init)");
  }

  // ---- Sanity: DTC M < N runs ----
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CProbitNoise noise(&y); noise.initParams();
    CSparsePpa sp(X, y, kern, noise, CGp::DTC, 10, 1.0, 0);
    sp.optimise(2, 0);
    bool finite = true;
    for(int i = 0; i < N; i++)
      if(!std::isfinite(sp.getFBar().getVal(i, 0)))
        finite = false;
    expect(finite, "DTC M<N fBar finite after 2 E-steps");
  }

  // ---- Document: white kernel breaks DTC M=N recovery ----
  {
    CCmpndKern kernD(X); makeKernWithWhite(kernD, X);
    CCmpndKern kernS(X); makeKernWithWhite(kernS, X);
    CProbitNoise noiseD(&y); noiseD.initParams();
    CProbitNoise noiseS(&y); noiseS.initParams();
    CPpa dense(X, y, kernD, noiseD, 0);
    CSparsePpa sp(X, y, kernS, noiseS, CGp::DTC, static_cast<unsigned int>(N), 1.0, 0);
    // Init matches vacuously (both start at f=0 → fBar=0 → same sites).
    expect(maxAbsDiff(sp.getFBar(), dense.fBar) < tol, "with-white init still matches (f=0 vacuous)");
    dense.eStep();
    sp.eStep();
    const double drift = maxAbsDiff(sp.getFBar(), dense.fBar);
    expect(drift > 1e-2, "with-white DTC M=N drifts after 2nd E-step (K_uf omits white)");
    cout << "  info: white-kernel M=N drift = " << drift << endl;
  }

  // ---- Document: raw CGp::FTC omits additive β^{-1}I ----
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CCmpndKern kernP(X); makeKernNoWhite(kernP, X);
    CProbitNoise noiseP(&y); noiseP.initParams();
    CPpa dense(X, y, kernP, noiseP, 0);
    CGaussianNoise gnoise(&dense.f);
    for(unsigned int j = 0; j < gnoise.getOutputDim(); j++)
      gnoise.setBiasVal(0.0, j);
    CMatrix Xcopy(X);
    CMatrix fcopy(dense.f);
    gnoise.setTarget(&fcopy);
    CGp ftc(static_cast<unsigned int>(D), 1, &Xcopy, &fcopy, &kern, &gnoise, CGp::FTC, 0, 0);
    ftc.setBetaVal(1.0);
    CMatrix muFtc(N, 1);
    ftc.posteriorMean(muFtc, Xcopy);
    const double dFtc = maxAbsDiff(muFtc, dense.fBar);
    expect(dFtc > 1e-2, "CGp::FTC mean diverges from PPA fBar (no beta on Gram)");
    cout << "  info: max|FTC mean - PPA fBar| = " << dFtc << endl;
  }

  // ---- Full q(u) ELBO: DTC M=N matches dense approxLogLikelihood ----
  {
    CCmpndKern kernD(X); makeKernNoWhite(kernD, X);
    CCmpndKern kernS(X); makeKernNoWhite(kernS, X);
    CProbitNoise noiseD(&y); noiseD.initParams();
    CProbitNoise noiseS(&y); noiseS.initParams();
    CPpa dense(X, y, kernD, noiseD, 0);
    CSparsePpa sp(X, y, kernS, noiseS, CGp::DTC, static_cast<unsigned int>(N), 1.0, 0);
    for(int i = 0; i < 3; i++)
      {
        dense.eStep();
        sp.eStep();
      }
    const double denseLL = dense.approxLogLikelihood();
    const double sparseLL = sp.ppaBound();
    expect(fabs(sparseLL - denseLL) < 1e-6,
           "DTC M=N ppaBound matches dense approxLogLikelihood");
    cout << "  info: dense approx=" << denseLL << " sparse ELBO=" << sparseLL
         << " diff=" << (sparseLL - denseLL) << endl;
  }
  {
    CCmpndKern kernD(X); makeKernNoWhite(kernD, X);
    CCmpndKern kernS(X); makeKernNoWhite(kernS, X);
    CProbitNoise noiseD(&y); noiseD.initParams();
    CProbitNoise noiseS(&y); noiseS.initParams();
    CPpa dense(X, y, kernD, noiseD, 0);
    CSparsePpa sp(X, y, kernS, noiseS, CGp::DTCVAR, static_cast<unsigned int>(N), 1.0, 0);
    for(int i = 0; i < 3; i++)
      {
        dense.eStep();
        sp.eStep();
      }
    expect(fabs(sp.ppaBound() - dense.approxLogLikelihood()) < 1e-6,
           "DTCVAR M=N ppaBound matches dense approxLogLikelihood");
  }

  // ---- Full ELBO finite for M<N; DTCVAR selectable ----
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CProbitNoise noise(&y); noise.initParams();
    CSparsePpa sp(X, y, kern, noise, CGp::DTC, 10, 1.0, 0);
    expect(std::isfinite(sp.ppaBound()), "DTC M<N ppaBound finite");
  }
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CProbitNoise noise(&y); noise.initParams();
    CSparsePpa sp(X, y, kern, noise, CGp::DTCVAR, 10, 1.0, 0);
    expect(sp.getApproxType() == CGp::DTCVAR, "DTCVAR approx selected");
    expect(std::isfinite(sp.ppaBound()), "DTCVAR M<N ppaBound finite");
  }

  // ---- Phase B: sparse .model stream round-trip ----
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CProbitNoise noise(&y); noise.initParams();
    CSparsePpa sp(X, y, kern, noise, CGp::DTC, 10, 1.0, 0);
    sp.optimise(1, 0);
    const string path = "/tmp/test_sparse_ppa_0.3.model";
    writeSparsePpaToFile(sp, path, "testSparsePpa round-trip");
    CSparsePpa* loaded = readSparsePpaFromFile(path, 0);
    expect(loaded->getApproxType() == CGp::DTC, "stream round-trip approx");
    expect(loaded->getNumActive() == 10u, "stream round-trip M");
    expect(fabs(loaded->getBeta() - 1.0) < 1e-12, "stream round-trip beta");
    expect(maxAbsDiff(loaded->getFBar(), sp.getFBar()) < 1e-8, "stream round-trip fBar");
    expect(maxAbsDiff(loaded->getF(), sp.getF()) < 1e-8, "stream round-trip f");
    expect(std::isfinite(loaded->ppaBound()), "loaded ppaBound finite");
    delete loaded;
  }

  // ---- Phase D: FITC / PITC smoke; free inducing ----
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CProbitNoise noise(&y); noise.initParams();
    CSparsePpa fitc(X, y, kern, noise, CGp::FITC, 10, 1.0, 0);
    fitc.optimise(1, 0);
    expect(std::isfinite(fitc.ppaBound()), "FITC M=10 ppaBound finite");
  }
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CProbitNoise noise(&y); noise.initParams();
    CSparsePpa pitc(X, y, kern, noise, CGp::PITC, 10, 1.0, 0);
    pitc.optimise(1, 0);
    expect(std::isfinite(pitc.ppaBound()), "PITC M=10 ppaBound finite");
  }
  {
    CCmpndKern kern(X); makeKernNoWhite(kern, X);
    CProbitNoise noise(&y); noise.initParams();
    CSparsePpa sp(X, y, kern, noise, CGp::DTC, 8, 1.0, 0);
    sp.fixInducing(false);
    sp.optimise(1, 5);
    expect(std::isfinite(sp.ppaBound()), "free inducing DTC optimise finite");
  }

  cout << "Number of failures: " << fails << endl;
  return fails > 0 ? 1 : 0;
}
