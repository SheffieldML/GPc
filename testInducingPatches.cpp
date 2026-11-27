/* CIP-0040 Phase 1b: CInducingPatches against exact reductions (C1), a direct
   double loop (C2) and finite differences of the SVGP ELBO (C3). */
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "CKern.h"
#include "CInducing.h"
#include "CSvgp.h"
#include "CDeepGp.h"
#include <sstream>
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
    cout << "ok: " << msg << endl;
}

void fillRand(CMatrix& A, unsigned int seed)
{
  mt19937 rng(seed);
  normal_distribution<double> nd(0.0, 1.0);
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      A.setVal(nd(rng), i, j);
}

double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

CRbfKern makeRbf(unsigned int q)
{
  CRbfKern k(q);
  k.setInitParam();
  k.setInverseWidth(0.7);
  k.setVariance(1.3);
  return k;
}

const char* modeName(CConvKern::WeightMode m)
{
  return m == CConvKern::WEIGHTS_LEARNED ? "learned" : (m == CConvKern::WEIGHTS_TIED ? "tied" : "fixed");
}

// C1: one patch equal to the image reduces to ordinary point inducing.
void testReduction()
{
  const unsigned int H = 3, W = 2, C = 2, q = 12, M = 4, N = 5;
  CRbfKern base = makeRbf(q);
  CConvKern conv(base, H, W, C, H, W, 1, CConvKern::WEIGHTS_FIXED);
  CMatrix Z(M, q), X(N, q), Xs(3, q);
  fillRand(Z, 1); fillRand(X, 2); fillRand(Xs, 3);
  CInducingPatches ip(Z);
  CMatrix Zc(Z);
  CInducingPoints pts(&Zc);
  CMatrix A(M, M), B(M, M);
  ip.computeKuu(A, conv); pts.computeKuu(B, base);
  expect(maxAbsDiff(A, B) < 1e-13, "reduction: Kuu equals point inducing");
  A.resize(M, N); B.resize(M, N);
  ip.computeKuf(A, conv, X); pts.computeKuf(B, base, X);
  expect(maxAbsDiff(A, B) < 1e-13, "reduction: Kuf equals point inducing");
  A.resize(M, 3); B.resize(M, 3);
  ip.computeKux(A, conv, Xs); pts.computeKux(B, base, Xs);
  expect(maxAbsDiff(A, B) < 1e-13, "reduction: Kux equals point inducing");

  CMatrix gKuu(M, M), gKuf(M, N);
  fillRand(gKuu, 4); fillRand(gKuf, 5);
  CMatrix gs(M, M);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < M; j++)
      gs.setVal(0.5 * (gKuu.getVal(i, j) + gKuu.getVal(j, i)), i, j);
  CMatrix gkc(1, conv.getNumParams()), gkp(1, base.getNumParams()), gZc, gZp;
  gkc.zeros(); gkp.zeros();
  ip.gradFromKuuKuf(gkc, gZc, 0, conv, X, gs, gKuf, false, false);
  pts.gradFromKuuKuf(gkp, gZp, 0, base, X, gs, gKuf, false, false);
  expect(gkc.getCols() == gkp.getCols() && maxAbsDiff(gkc, gkp) < 1e-12,
         "reduction: kernel gradients equal point inducing");
  expect(maxAbsDiff(gZc, gZp) < 1e-12, "reduction: inducing gradients equal point inducing");
}

// C2: Kuf by a double loop of patch-kernel element evaluations.
void testDoubleLoop(CConvKern::WeightMode mode, unsigned int stride)
{
  const unsigned int H = 5, W = 6, C = 1, ph = 3, pw = 2, q = 6, M = 4, N = 3;
  CRbfKern base = makeRbf(q);
  CConvKern conv(base, H, W, C, ph, pw, stride, mode);
  for(unsigned int i = base.getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.3 + 0.1 * sin(2.0 * i), i);
  CMatrix Z(M, q), X(N, H * W * C);
  fillRand(Z, 6); fillRand(X, 7);
  CInducingPatches ip(Z);
  CMatrix Kuf, Xp;
  ip.computeKuf(Kuf, conv, X);
  conv.extractPatches(Xp, X);
  const unsigned int P = conv.numPatches();
  double worst = 0.0;
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int n = 0; n < N; n++)
    {
      double s = 0.0;
      for(unsigned int p = 0; p < P; p++)
        s += conv.patchWeights()[p] * conv.patchKernel().computeElement(Z, m, Xp, n * P + p);
      worst = max(worst, fabs(s - Kuf.getVal(m, n)));
    }
  expect(worst < 1e-13, string("double loop: Kuf ") + modeName(mode) + " stride " + to_string(stride));
}

// C3: analytic ELBO gradient (kernel incl. weights, patches, beta) against finite differences.
void testSvgpGradients(CConvKern::WeightMode mode, unsigned int stride)
{
  const unsigned int H = 5, W = 6, C = 1, ph = 3, pw = 2, q = 6, M = 5, N = 8;
  CRbfKern base = makeRbf(q);
  CConvKern conv(base, H, W, C, ph, pw, stride, mode);
  for(unsigned int i = base.getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.3 + 0.1 * sin(2.0 * i), i);
  CMatrix X(N, H * W * C), y(N, 1), Zdummy(M, H * W * C);
  fillRand(X, 8); fillRand(y, 9); fillRand(Zdummy, 10);
  CSvgp sv(&conv, &X, &y, Zdummy, 5.0, 0);
  sv.setInducingType("patches");
  expect(sv.getInducing()->typeName() == "patches", "svgp: patches inducing selected");
  // Move the whitened q away from its prior so every term contributes.
  CMatrix qm(M, 1); fillRand(qm, 11);
  vector<CMatrix> qc(1, CMatrix(M, M, 0.0));
  qc[0].addDiag(0.7);
  qc[0].setSymmetric(true);
  sv.setWhitenedQ(qm, qc);

  vector<unsigned int> all(N);
  for(unsigned int i = 0; i < N; i++) all[i] = i;
  CMatrix g(1, sv.getOptNumParams()), p0(1, sv.getOptNumParams());
  sv.objectiveGradBatch(all, g);
  sv.getOptParams(p0);
  const double eps = 1e-6;
  double maxRel = 0.0;
  for(unsigned int i = 0; i < sv.getOptNumParams(); i++)
  {
    CMatrix pp(p0), pm(p0);
    pp.setVal(p0.getVal(i) + eps, i);
    pm.setVal(p0.getVal(i) - eps, i);
    sv.setOptParams(pp);
    const double Lp = sv.elbo();
    sv.setOptParams(pm);
    const double Lm = sv.elbo();
    sv.setOptParams(p0);
    const double fd = (Lp - Lm) / (2.0 * eps);
    const double an = g.getVal(i);
    const double scale = max(1.0, max(fabs(fd), fabs(an)));
    maxRel = max(maxRel, fabs(fd - an) / scale);
  }
  cout << "  info: max relative diff " << maxRel << " over " << sv.getOptNumParams() << " params" << endl;
  expect(maxRel < 2e-4, string("C3: ELBO gradient ") + modeName(mode) + " stride " + to_string(stride));

  // Prediction path runs and has the right shape.
  CMatrix mu, var, Xs(2, H * W * C);
  fillRand(Xs, 12);
  sv.predictiveMeanVar(mu, var, Xs);
  expect(mu.getRows() == 2 && mu.getCols() == 1, "svgp: predict shape");
}

// C3 through CDeepGp: conv first layer with inducing patches, FD at frozen noise, stream round trip.
void testDeepGp()
{
  const unsigned int H = 5, W = 6, C = 1, ph = 3, pw = 2, q = 6, N = 10, M = 4, hidden = 2;
  const double beta = 4.0;
  CRbfKern base = makeRbf(q);
  CConvKern conv(base, H, W, C, ph, pw, 2, CConvKern::WEIGHTS_LEARNED);
  for(unsigned int i = base.getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.3 + 0.1 * sin(2.0 * i), i);
  CRbfKern k1(hidden);
  k1.setInitParam();
  CMatrix X(N, H * W * C), y(N, 1);
  fillRand(X, 21); fillRand(y, 22);
  ndlutil::init_genrand(5u);
  vector<CKern*> kerns = {&conv, &k1};
  CDeepGp dgp(X, y, kerns, vector<unsigned int>{H * W * C, hidden, 1u},
              vector<unsigned int>{M, M}, beta, 0.5, 0);
  dgp.setLayerInducingType(0, "patches");
  expect(dgp.getLayerInducingType(0) == "patches", "deepgp: patches recorded on layer 0");
  dgp.setNumSamples(1);
  CMatrix eps(N * hidden, 1);
  fillRand(eps, 23);
  dgp.setFrozenEps(eps);

  vector<unsigned int> all(N);
  for(unsigned int i = 0; i < N; i++) all[i] = i;
  CMatrix g, p0;
  dgp.objectiveGradBatch(all, g);
  dgp.getStochasticParams(p0);
  const double h = 1e-5;
  double maxErr = 0.0;
  for(unsigned int i = 0; i < p0.getCols(); i++)
  {
    CMatrix pp(p0), pm(p0), tmp;
    pp.setVal(p0.getVal(i) + h, i);
    pm.setVal(p0.getVal(i) - h, i);
    dgp.setStochasticParams(pp);
    const double Lp = dgp.objectiveGradBatch(all, tmp);
    dgp.setStochasticParams(pm);
    const double Lm = dgp.objectiveGradBatch(all, tmp);
    dgp.setStochasticParams(p0);
    maxErr = max(maxErr, fabs(g.getVal(i) - (Lp - Lm) / (2.0 * h)));
  }
  cout << "  info: deepgp max|g-FD|=" << maxErr << " over " << p0.getCols() << " params" << endl;
  expect(maxErr < 2e-2, "C3: CDeepGp gradients with conv layer and inducing patches");

  ostringstream oss;
  dgp.toStream(oss);
  istringstream iss(oss.str());
  CMatrix Xt(1, 1), Yt(1, 1);
  CRbfKern kt(1);
  kt.setInitParam();
  vector<CKern*> tk{&kt};
  CDeepGp loaded(Xt, Yt, tk, vector<unsigned int>{1u, 1u}, vector<unsigned int>{1u}, 1.0, 1.0, 0);
  loaded.fromStream(iss);
  CMatrix p1;
  loaded.getStochasticParams(p1);
  expect(loaded.getLayerInducingType(0) == "patches" && p0.getCols() == p1.getCols()
         && maxAbsDiff(p0, p1) < 1e-10, "stream: conv layer and inducing patches round trip");
  CMatrix Xs(3, H * W * C), mu0, v0, mu1, v1;
  fillRand(Xs, 24);
  dgp.predictMeanVar(Xs, mu0, v0);
  loaded.predictMeanVar(Xs, mu1, v1);
  expect(maxAbsDiff(mu0, mu1) < 1e-9 && maxAbsDiff(v0, v1) < 1e-9, "stream: predictions reproduced");
}

// C5 through CSvgp: with a Gaussian likelihood the bound is linear in the kernel diagonal,
// so the ELBO and its gradient computed with the estimated diagonal are unbiased.
void testElboEstimator()
{
  const unsigned int H = 8, W = 8, C = 1, ph = 3, pw = 3, q = 9, M = 5, N = 6;
  CRbfKern base = makeRbf(q);
  CConvKern conv(base, H, W, C, ph, pw, 1, CConvKern::WEIGHTS_LEARNED);
  for(unsigned int i = base.getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.02 + 0.01 * sin(2.0 * i), i);
  CMatrix X(N, H * W * C), y(N, 1), Zd(M, H * W * C);
  fillRand(X, 31); fillRand(y, 32); fillRand(Zd, 33);
  CSvgp sv(&conv, &X, &y, Zd, 3.0, 0);
  sv.setInducingType("patches");
  CMatrix qm(M, 1); fillRand(qm, 34);
  vector<CMatrix> qc(1, CMatrix(M, M, 0.0));
  qc[0].addDiag(0.7);
  qc[0].setSymmetric(true);
  sv.setWhitenedQ(qm, qc);
  vector<unsigned int> all(N);
  for(unsigned int i = 0; i < N; i++) all[i] = i;
  CMatrix p0(1, sv.getOptNumParams()), g(1, sv.getOptNumParams());
  sv.getOptParams(p0);
  const double Lexact = sv.objectiveGradBatch(all, g);
  CMatrix gExact(g);

  const unsigned int S = 12, R = 3000;
  double sL = 0.0, sL2 = 0.0;
  vector<double> sg(g.getCols(), 0.0), sg2(g.getCols(), 0.0);
  for(unsigned int r = 0; r < R; r++)
  {
    conv.setDiagEstimator(S, 5000 + r);
    sv.setOptParams(p0);  // refresh cached kernel quantities
    const double L = sv.objectiveGradBatch(all, g);
    sL += L; sL2 += L * L;
    for(unsigned int i = 0; i < g.getCols(); i++)
    {
      sg[i] += g.getVal(i); sg2[i] += g.getVal(i) * g.getVal(i);
    }
  }
  const double mean = sL / R, var = sL2 / R - mean * mean;
  const double z = fabs(mean - Lexact) / sqrt(var / R);
  double zg = 0.0;
  for(unsigned int i = 0; i < g.getCols(); i++)
  {
    const double m = sg[i] / R, v = sg2[i] / R - m * m;
    if(v > 1e-30)
      zg = max(zg, fabs(m - gExact.getVal(i)) / sqrt(v / R));
  }
  cout << "  info: ELBO exact " << Lexact << ", estimated mean " << mean << " (sd " << sqrt(var) << ")" << ", z=" << z
       << ", worst gradient z=" << zg << endl;
  expect(z < 4.0, "C5: Gaussian-likelihood ELBO estimate unbiased within 4 standard errors");
  expect(zg < 5.0, "C5: Gaussian-likelihood ELBO gradient estimate unbiased within 5 standard errors");
}

void testErrors()
{
  CRbfKern base = makeRbf(4);
  CMatrix Z(3, 4), X(2, 4);
  fillRand(Z, 1); fillRand(X, 2);
  CInducingPatches ip(Z);
  CMatrix K(3, 2);
  bool threw = false;
  try { ip.computeKuf(K, base, X); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: non-convolutional kernel rejected");
  CConvKern conv(base, 2, 2, 1, 2, 2, 1, CConvKern::WEIGHTS_FIXED);
  CMatrix Zbad(3, 5);
  CInducingPatches ipb(Zbad);
  threw = false;
  try { ipb.computeKuu(K, conv); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: patch dimension mismatch rejected");
}

}  // namespace

int main()
{
  try
  {
    testReduction();
    const CConvKern::WeightMode modes[3] = {CConvKern::WEIGHTS_LEARNED, CConvKern::WEIGHTS_TIED,
                                            CConvKern::WEIGHTS_FIXED};
    for(int mi = 0; mi < 3; mi++)
      for(unsigned int stride = 1; stride <= 2; stride++)
      {
        testDoubleLoop(modes[mi], stride);
        testSvgpGradients(modes[mi], stride);
      }
    testDeepGp();
    testElboEstimator();
    testErrors();
  }
  catch(const ndlexceptions::Error& e)
  {
    cerr << "Unexpected exception: " << e.what() << " " << e.getMessage() << endl;
    return 1;
  }
  cout << "Number of failures: " << fails << "." << endl;
  return fails == 0 ? 0 : 1;
}
