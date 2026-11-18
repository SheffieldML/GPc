/* CIP-0021: compare CPpa init state to Octave Phase 0 fixtures. */
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CPpa.h"
#include "ndlstrutil.h"
#include <cmath>
#include <iostream>
#include <fstream>
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
  for(int i=0; i<rows; i++)
    for(int j=0; j<cols; j++)
      {
        double v=0.0;
        if(!(in >> v))
          throw ndlexceptions::FileFormatError(path);
        A.setVal(v, i, j);
      }
  return A;
}

static double loadScalar(const string& path)
{
  ifstream in(path.c_str());
  if(!in)
    throw ndlexceptions::FileReadError(path);
  double v=0.0;
  in >> v;
  return v;
}

static double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i=0; i<A.getRows(); i++)
    for(unsigned int j=0; j<A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

int main()
{
  const string dir = "testdata" + ndlstrutil::dirSep() + "ppa" + ndlstrutil::dirSep();
  const int N = 30;
  const int D = 2;

  CMatrix X = loadDat(dir + "phase0_X.txt", N, D);
  CMatrix y = loadDat(dir + "phase0_y.txt", N, 1);

  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CBiasKern(X));
  kern.addKern(new CWhiteKern(X));
  // Match MATLAB kernExtractParam init: [0 0 -2 -2] in transformed space.
  CMatrix kp(1, 4);
  kp.setVal(0.0, 0); kp.setVal(0.0, 1); kp.setVal(-2.0, 2); kp.setVal(-2.0, 3);
  kern.setTransParams(kp);

  CProbitNoise noise(&y);
  noise.initParams();

  CPpa model(X, y, kern, noise, 0);

  CMatrix fRef = loadDat(dir + "phase0_f_init.txt", N, 1);
  CMatrix fBarRef = loadDat(dir + "phase0_fBar_init.txt", N, 1);
  CMatrix gRef = loadDat(dir + "phase0_g_init.txt", N, 1);
  CMatrix KRef = loadDat(dir + "phase0_Kstore_init.txt", N, N);
  const double llRef = loadScalar(dir + "phase0_loglike_init.txt");

  const double tol = 1e-5;
  expect(maxAbsDiff(model.K, KRef) < tol, "Kstore/K vs fixture (init)");
  expect(maxAbsDiff(model.fBar, fBarRef) < tol, "fBar vs fixture (init)");
  expect(maxAbsDiff(model.f, fRef) < tol, "f vs fixture (init)");
  expect(maxAbsDiff(model.g, gRef) < tol, "g vs fixture (init)");
  expect(fabs(model.approxLogLikelihood() - llRef) < 1e-4, "approx log-likelihood vs fixture (init)");

  // After fixtures are E-step-only (upstream MATLAB kernel M-step is a no-op).
  model.optimise(3, 0, 0);

  CMatrix fAfter = loadDat(dir + "phase0_f_after.txt", N, 1);
  CMatrix fBarAfter = loadDat(dir + "phase0_fBar_after.txt", N, 1);
  CMatrix gAfter = loadDat(dir + "phase0_g_after.txt", N, 1);
  CMatrix KAfter = loadDat(dir + "phase0_Kstore_after.txt", N, N);
  const double llAfter = loadScalar(dir + "phase0_loglike_after.txt");

  expect(maxAbsDiff(model.K, KAfter) < 1e-4, "K vs fixture (after E-steps)");
  expect(maxAbsDiff(model.fBar, fBarAfter) < 1e-4, "fBar vs fixture (after E-steps)");
  expect(maxAbsDiff(model.f, fAfter) < 1e-4, "f vs fixture (after E-steps)");
  expect(maxAbsDiff(model.g, gAfter) < 1e-4, "g vs fixture (after E-steps)");
  expect(fabs(model.approxLogLikelihood() - llAfter) < 1e-3, "approx log-likelihood vs fixture (after E-steps)");

  // Kernel M-step analytic gradient vs Octave ppaKernelLogLikeGrad.
  CCmpndKern kern2(X);
  kern2.addKern(new CRbfKern(X));
  kern2.addKern(new CBiasKern(X));
  kern2.addKern(new CWhiteKern(X));
  kern2.setTransParams(kp);
  CProbitNoise noise2(&y);
  noise2.initParams();
  CPpa model2(X, y, kern2, noise2, 0);
  CMatrix gKern(1, 4);
  model2.kernelLogLikelihoodGradient(gKern);
  CMatrix gKernRef = loadDat(dir + "phase0_kernGrad_init.txt", 1, 4);
  expect(maxAbsDiff(gKern, gKernRef) < 1e-5, "kernel LL gradient vs Octave");

  cout << "Number of failures: " << fails << endl;
  return fails > 0 ? 1 : 0;
}
