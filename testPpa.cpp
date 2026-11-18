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
  expect(maxAbsDiff(model.K, KRef) < tol, "Kstore/K vs fixture");
  expect(maxAbsDiff(model.fBar, fBarRef) < tol, "fBar vs fixture");
  expect(maxAbsDiff(model.f, fRef) < tol, "f vs fixture");
  expect(maxAbsDiff(model.g, gRef) < tol, "g vs fixture");
  expect(fabs(model.logLikelihood() - llRef) < 1e-4, "approx log-likelihood vs fixture");

  cout << "Number of failures: " << fails << endl;
  return fails > 0 ? 1 : 0;
}
