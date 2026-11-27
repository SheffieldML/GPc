/* CIP-0040 Phase 1a: CConvKern against exact reductions (C1), a direct double
   loop over patches (C2), finite-difference gradients (C3), a shift sanity
   check (C6) and a stream round trip. */
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include "CKern.h"
#include "ndlutil.h"

using namespace std;

namespace {

int fails = 0;

string str(double x)
{
  ostringstream os;
  os.precision(12);
  os << x;
  return os.str();
}

void expect(bool ok, const string& msg)
{
  if(!ok)
  {
    cerr << "FAIL: " << msg << endl;
    ++fails;
  }
  else
    cout << "ok: " << msg << endl;
}

void expectNear(double a, double b, double tol, const string& msg)
{
  expect(fabs(a - b) <= tol, msg + " (got " + str(a) + " vs " + str(b) + ", diff " + str(a - b) + ")");
}

void fillRand(CMatrix& A, double scale, double phase)
{
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      A.setVal(scale * sin(1.7 * i + 0.9 * j + phase + 0.3 * i * j), i, j);
}

double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

CRbfKern* makeRbf(unsigned int q)
{
  CRbfKern* k = new CRbfKern(q);
  k->setInitParam();
  k->setInverseWidth(0.7);
  k->setVariance(1.3);
  return k;
}

CMlpardKern* makeMlpard(unsigned int q)
{
  CMlpardKern* k = new CMlpardKern(q);
  k->setInitParam();
  k->setParam(0.8, 0);
  k->setParam(0.6, 1);
  k->setParam(1.2, 2);
  for(unsigned int j = 0; j < q; j++)
    k->setParam(0.5 + 0.1 * (j % 4), 3 + j);
  return k;
}

const char* modeName(CConvKern::WeightMode m)
{
  return m == CConvKern::WEIGHTS_LEARNED ? "learned"
         : (m == CConvKern::WEIGHTS_TIED ? "tied" : "fixed");
}

// C1: one patch equal to the whole image, weight 1.
void testReduction(CKern* patchKern, unsigned int H, unsigned int W, unsigned int C,
                   const string& label)
{
  CConvKern conv(*patchKern, H, W, C, H, W, 1, CConvKern::WEIGHTS_FIXED);
  expect(conv.numPatches() == 1, label + ": single patch");
  const unsigned int N = 4, N2 = 3, D = H * W * C;
  CMatrix X(N, D), X2(N2, D);
  fillRand(X, 1.0, 0.1);
  fillRand(X2, 1.0, 1.1);
  CMatrix Kc(N, N2), Kp(N, N2), Ksc(N, N), Ksp(N, N), dc(N, 1), dp(N, 1);
  conv.compute(Kc, X, X2);
  patchKern->compute(Kp, X, X2);
  expectNear(maxAbsDiff(Kc, Kp), 0.0, 1e-13, label + ": K(X,X2) equals patch kernel");
  conv.compute(Ksc, X);
  patchKern->compute(Ksp, X);
  expectNear(maxAbsDiff(Ksc, Ksp), 0.0, 1e-13, label + ": K(X,X) equals patch kernel");
  conv.diagCompute(dc, X);
  patchKern->diagCompute(dp, X);
  expectNear(maxAbsDiff(dc, dp), 0.0, 1e-13, label + ": diag equals patch kernel");

  CMatrix cv(N, N2);
  fillRand(cv, 1.0, 2.0);
  CMatrix gc(1, conv.getNumParams()), gp(1, patchKern->getNumParams());
  conv.getGradParams(gc, X, X2, cv, false);
  patchKern->getGradParams(gp, X, X2, cv, false);
  expect(gc.getCols() == gp.getCols(), label + ": fixed weights add no parameters");
  expectNear(maxAbsDiff(gc, gp), 0.0, 1e-12, label + ": gradients equal patch kernel");
}

// C2: direct double loop over patches, one patch-kernel call per pair.
void testDoubleLoop(CKern* patchKern, unsigned int H, unsigned int W, unsigned int C,
                    unsigned int ph, unsigned int pw, unsigned int stride,
                    CConvKern::WeightMode mode, const string& label)
{
  CConvKern conv(*patchKern, H, W, C, ph, pw, stride, mode);
  const unsigned int P = conv.numPatches(), q = conv.patchDim(), D = H * W * C;
  // Non-uniform weights so the contraction is exercised.
  for(unsigned int i = conv.patchKernel().getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.3 + 0.1 * sin(2.0 * i), i);
  const unsigned int N = 3, N2 = 4;
  CMatrix X(N, D), X2(N2, D);
  fillRand(X, 1.0, 0.3);
  fillRand(X2, 1.0, 1.9);
  CMatrix Xp, X2p;
  conv.extractPatches(Xp, X);
  conv.extractPatches(X2p, X2);
  const vector<double>& w = conv.patchWeights();
  CMatrix K(N, N2), Kref(N, N2);
  conv.compute(K, X, X2);
  CMatrix a(1, q), b(1, q);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < N2; m++)
    {
      double s = 0.0;
      for(unsigned int p = 0; p < P; p++)
        for(unsigned int pp = 0; pp < P; pp++)
          s += w[p] * w[pp] * conv.patchKernel().computeElement(Xp, n * P + p, X2p, m * P + pp);
      Kref.setVal(s, n, m);
    }
  expectNear(maxAbsDiff(K, Kref), 0.0, 1e-12, label + ": K equals double loop");
  CMatrix d(N, 1);
  conv.diagCompute(d, X);
  double worst = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    double s = 0.0;
    for(unsigned int p = 0; p < P; p++)
      for(unsigned int pp = 0; pp < P; pp++)
        s += w[p] * w[pp] * conv.patchKernel().computeElement(Xp, n * P + p, Xp, n * P + pp);
    worst = max(worst, fabs(d.getVal(n) - s));
    worst = max(worst, fabs(conv.diagComputeElement(X, n) - s));
  }
  expectNear(worst, 0.0, 1e-12, label + ": diag equals double loop");
  CMatrix Ks(N, N);
  conv.compute(Ks, X);
  double asym = 0.0;
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < N; j++)
      asym = max(asym, fabs(Ks.getVal(i, j) - Ks.getVal(j, i)));
  expectNear(asym, 0.0, 1e-13, label + ": K(X,X) symmetric");
  for(unsigned int i = 0; i < N; i++)
    expectNear(Ks.getVal(i, i), d.getVal(i), 1e-12, label + ": diag equals diagonal of K");
}

// C3: finite differences of  L = sum(cv .* K)  and  L = sum(c .* diag)  in every parameter.
void testGradients(CKern* patchKern, unsigned int H, unsigned int W, unsigned int C,
                   unsigned int ph, unsigned int pw, unsigned int stride,
                   CConvKern::WeightMode mode, const string& label)
{
  CConvKern conv(*patchKern, H, W, C, ph, pw, stride, mode);
  const unsigned int nParams = conv.getNumParams(), D = H * W * C;
  for(unsigned int i = conv.patchKernel().getNumParams(); i < nParams; i++)
    conv.setParam(0.3 + 0.1 * sin(2.0 * i), i);
  const unsigned int N = 3, N2 = 4;
  CMatrix X(N, D), X2(N2, D), cv(N, N2), cvSq(N, N), cd(N, 1);
  fillRand(X, 1.0, 0.7);
  fillRand(X2, 1.0, 2.3);
  fillRand(cv, 1.0, 0.2);
  fillRand(cvSq, 1.0, 0.9);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = 0.5 * (cvSq.getVal(i, j) + cvSq.getVal(j, i));
      cvSq.setVal(v, i, j);
      cvSq.setVal(v, j, i);
    }
  cvSq.setSymmetric(true);
  fillRand(cd, 1.0, 1.4);

  CMatrix g(1, nParams), gSq(1, nParams), gD(1, nParams);
  conv.getGradParams(g, X, X2, cv, false);
  conv.getGradParams(gSq, X, cvSq, false);
  conv.getDiagGradParams(gD, X, cd, false);

  auto objective = [&](int which) {
    CMatrix K(N, which == 1 ? N : N2), d(N, 1);
    double s = 0.0;
    if(which == 0)
    {
      conv.compute(K, X, X2);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N2; j++)
          s += cv.getVal(i, j) * K.getVal(i, j);
    }
    else if(which == 1)
    {
      conv.compute(K, X, X);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
          s += cvSq.getVal(i, j) * K.getVal(i, j);
    }
    else
    {
      conv.diagCompute(d, X);
      for(unsigned int i = 0; i < N; i++)
        s += cd.getVal(i) * d.getVal(i);
    }
    return s;
  };

  double worst = 0.0, worstSq = 0.0, worstD = 0.0;
  const double h = 1e-6;
  for(unsigned int i = 0; i < nParams; i++)
  {
    const double p0 = conv.getParam(i);
    double fd[3];
    for(int which = 0; which < 3; which++)
    {
      conv.setParam(p0 + h, i);
      const double fp = objective(which);
      conv.setParam(p0 - h, i);
      const double fm = objective(which);
      conv.setParam(p0, i);
      fd[which] = (fp - fm) / (2.0 * h);
    }
    worst = max(worst, fabs(fd[0] - g.getVal(i)) / (1.0 + fabs(fd[0])));
    worstSq = max(worstSq, fabs(fd[1] - gSq.getVal(i)) / (1.0 + fabs(fd[1])));
    worstD = max(worstD, fabs(fd[2] - gD.getVal(i)) / (1.0 + fabs(fd[2])));
  }
  expectNear(worst, 0.0, 1e-6, label + ": gradient of K(X,X2) matches finite differences");
  expectNear(worstSq, 0.0, 1e-6, label + ": gradient of K(X,X) matches finite differences");
  expectNear(worstD, 0.0, 1e-6, label + ": gradient of diag matches finite differences");
}

// C6: content away from the border, tied weights, stride 1: shifting by one pixel
// leaves k(x,x) and k(x, shift(x)) unchanged.
void testShift()
{
  const unsigned int H = 8, W = 8;
  CRbfKern* base = makeRbf(9);
  CConvKern conv(*base, H, W, 1, 3, 3, 1, CConvKern::WEIGHTS_TIED);
  delete base;
  CMatrix X(2, H * W);
  X.zeros();
  const double blob[3][3] = {{0.2, 0.9, 0.4}, {0.7, 1.5, 0.1}, {0.3, 0.6, 0.8}};
  for(unsigned int i = 0; i < 3; i++)
    for(unsigned int j = 0; j < 3; j++)
    {
      X.setVal(blob[i][j], 0, (2 + i) * W + (2 + j));
      X.setVal(blob[i][j], 1, (3 + i) * W + (3 + j));  // shifted down-right by one
    }
  CMatrix d(2, 1), K(2, 2);
  conv.diagCompute(d, X);
  expectNear(d.getVal(0), d.getVal(1), 1e-13, "shift: k(x,x) unchanged by a one-pixel shift");
  conv.compute(K, X);
  expectNear(K.getVal(0, 1), K.getVal(0, 0), 1e-13, "shift: k(x, shift(x)) equals k(x,x)");
}

void testStream()
{
  CMlpardKern* base = makeMlpard(6);
  CConvKern conv(*base, 4, 5, 2, 1, 3, 1, CConvKern::WEIGHTS_LEARNED);
  delete base;
  for(unsigned int i = conv.patchKernel().getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.2 + 0.05 * i, i);
  CMatrix X(3, 4 * 5 * 2);
  fillRand(X, 1.0, 0.4);
  CMatrix K0(3, 3);
  conv.compute(K0, X);
  ostringstream os;
  conv.toStream(os);
  istringstream is(os.str());
  CKern* loaded = readKernFromStream(is);
  expect(loaded->getType() == "conv", "stream: type is conv");
  CMatrix K1(3, 3);
  loaded->compute(K1, X);
  expectNear(maxAbsDiff(K0, K1), 0.0, 1e-13, "stream: round trip reproduces K");
  expect(loaded->getNumParams() == conv.getNumParams(), "stream: parameter count preserved");
  CKern* cl = conv.clone();
  CMatrix K2(3, 3);
  cl->compute(K2, X);
  expectNear(maxAbsDiff(K0, K2), 0.0, 0.0, "clone: reproduces K");
  delete cl;
  delete loaded;
}

void testErrors()
{
  CRbfKern* base = makeRbf(5);
  bool threw = false;
  try { CConvKern bad(*base, 4, 4, 1, 2, 2, 1); }
  catch(const ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: patch kernel dimension mismatch is rejected");
  threw = false;
  try { CConvKern bad(*base, 2, 2, 1, 3, 3, 1); }
  catch(const ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: patch larger than image is rejected");
  delete base;
}


// C5: the diagonal estimator is unbiased (value and gradients), is a fixed function of
// (seed, row) so its gradient matches finite differences, and is exact when S >= P.
void testDiagEstimator()
{
  const unsigned int H = 8, W = 8, C = 1, ph = 3, pw = 3, q = 9, N = 3;
  CRbfKern* base = makeRbf(q);
  CConvKern conv(*base, H, W, C, ph, pw, 1, CConvKern::WEIGHTS_LEARNED);
  delete base;
  for(unsigned int i = conv.patchKernel().getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.02 + 0.01 * sin(2.0 * i), i);
  const unsigned int P = conv.numPatches(), np = conv.getNumParams();
  CMatrix X(N, H * W * C);
  fillRand(X, 1.0, 0.4);
  CMatrix cv(N, 1);
  for(unsigned int n = 0; n < N; n++)
    cv.setVal(0.5 + 0.3 * n, n);

  CMatrix dExact(N, 1), gExact(1, np);
  conv.diagCompute(dExact, X);
  conv.getDiagGradParams(gExact, X, cv, false);

  // S >= P is exact.
  conv.setDiagEstimator(P, 3);
  CMatrix d(N, 1);
  conv.diagCompute(d, X);
  expect(maxAbsDiff(d, dExact) < 1e-13, "diag estimator: S >= P is exact");

  // Unbiasedness over seeds, using the standard error of the mean.
  const unsigned int S = 14, R = 4000;
  vector<double> sd(N, 0.0), sd2(N, 0.0), sg(np, 0.0), sg2(np, 0.0);
  CMatrix g(1, np);
  for(unsigned int r = 0; r < R; r++)
  {
    conv.setDiagEstimator(S, 1000 + r);
    conv.diagCompute(d, X);
    for(unsigned int n = 0; n < N; n++)
    {
      sd[n] += d.getVal(n);
      sd2[n] += d.getVal(n) * d.getVal(n);
    }
    conv.getDiagGradParams(g, X, cv, false);
    for(unsigned int i = 0; i < np; i++)
    {
      sg[i] += g.getVal(i);
      sg2[i] += g.getVal(i) * g.getVal(i);
    }
  }
  double worstD = 0.0, worstG = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    const double mean = sd[n] / R, var = sd2[n] / R - mean * mean;
    worstD = max(worstD, fabs(mean - dExact.getVal(n)) / sqrt(var / R));
  }
  for(unsigned int i = 0; i < np; i++)
  {
    const double mean = sg[i] / R, var = sg2[i] / R - mean * mean;
    if(var > 1e-30)
      worstG = max(worstG, fabs(mean - gExact.getVal(i)) / sqrt(var / R));
  }
  cout << "  info: worst |mean - exact| / s.e.: diag " << worstD << ", gradient " << worstG << endl;
  expect(worstD < 4.0, "C5: estimated diagonal unbiased within 4 standard errors");
  expect(worstG < 5.0, "C5: estimated diagonal gradients unbiased within 5 standard errors");

  // Gradient consistency at fixed seed: FD of sum_n cv_n d_n against getDiagGradParams.
  conv.setDiagEstimator(S, 77);
  conv.getDiagGradParams(g, X, cv, false);
  const double eps = 1e-6;
  double worst = 0.0;
  for(unsigned int i = 0; i < np; i++)
  {
    const double p0 = conv.getParam(i);
    double f[2];
    for(int s = 0; s < 2; s++)
    {
      conv.setParam(p0 + (s == 0 ? eps : -eps), i);
      conv.diagCompute(d, X);
      f[s] = 0.0;
      for(unsigned int n = 0; n < N; n++)
        f[s] += cv.getVal(n) * d.getVal(n);
    }
    conv.setParam(p0, i);
    const double fd = (f[0] - f[1]) / (2.0 * eps);
    worst = max(worst, fabs(fd - g.getVal(i)) / max(1.0, fabs(fd)));
  }
  expectNear(worst, 0.0, 1e-6, "C5: estimated diagonal gradient matches finite differences at fixed seed");

  // diagComputeElement (used by CSvgp) is the same estimate, row by row.
  conv.diagCompute(d, X);
  double eltDiff = 0.0;
  for(unsigned int n = 0; n < N; n++)
    eltDiff = max(eltDiff, fabs(conv.diagComputeElement(X, n) - d.getVal(n)));
  expect(eltDiff < 1e-13, "C5: diagComputeElement agrees with diagCompute when estimated");

  // Same seed, same value; copy keeps the estimator.
  CMatrix d1(N, 1), d2(N, 1);
  conv.diagCompute(d1, X);
  CConvKern copy(conv);
  copy.diagCompute(d2, X);
  expect(maxAbsDiff(d1, d2) == 0.0 && copy.getDiagSamples() == S, "C5: deterministic in (seed, row); copy keeps estimator");
  bool threw = false;
  try { conv.setDiagEstimator(1, 0); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "C5: S = 1 rejected");
}

} // namespace

int main()
{
  try
  {
    {
      CRbfKern* k = makeRbf(12);
      testReduction(k, 3, 2, 2, "reduction rbf");
      delete k;
      CMlpardKern* m = makeMlpard(12);
      testReduction(m, 3, 2, 2, "reduction mlpard");
      delete m;
    }
    const CConvKern::WeightMode modes[3] = {CConvKern::WEIGHTS_LEARNED, CConvKern::WEIGHTS_TIED,
                                            CConvKern::WEIGHTS_FIXED};
    for(int mi = 0; mi < 3; mi++)
      for(unsigned int stride = 1; stride <= 2; stride++)
      {
        const string tag = string("rbf ") + modeName(modes[mi]) + " stride " + str(stride);
        CRbfKern* k = makeRbf(12);  // 3 x 2 x 2 patch
        testDoubleLoop(k, 5, 6, 2, 3, 2, stride, modes[mi], tag);
        testGradients(k, 5, 6, 2, 3, 2, stride, modes[mi], tag);
        delete k;
        const string tag2 = string("mlpard ") + modeName(modes[mi]) + " stride " + str(stride);
        CMlpardKern* m = makeMlpard(12);
        testDoubleLoop(m, 5, 6, 2, 3, 2, stride, modes[mi], tag2);
        testGradients(m, 5, 6, 2, 3, 2, stride, modes[mi], tag2);
        delete m;
      }
    testShift();
    testDiagEstimator();
    testStream();
    testErrors();
  }
  catch(const ndlexceptions::Error& e)
  {
    cerr << "Unexpected exception: " << e.what() << endl;
    return 1;
  }
  cout << "Number of failures: " << fails << "." << endl;
  return fails == 0 ? 0 : 1;
}
