/* CIP-0040 Phase 3: translation-insensitive convolutional kernel,
   k(x,x') = sum_{p,p'} k_g(x^[p], x'^[p']) k_loc(l_p, l_p'), with inducing (patch, location)
   pairs.  Oracles: k_loc == 1 recovers the weighted kernel with tied weight 1 (C1), direct
   double loops (C2), finite differences of the kernel, SVGP ELBO and deep-GP objective (C3),
   stream round trip. */
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "CKern.h"
#include "CInducing.h"
#include "CSvgp.h"
#include "CDeepGp.h"
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

const unsigned int H = 5, W = 6, C = 1, ph = 3, pw = 2, q = ph * pw * C;

// C1: very long location lengthscale equals the weighted kernel with tied weight 1.
void testReduction()
{
  CRbfKern base = makeRbf(q);
  CConvKern loc(base, H, W, C, ph, pw, 2, CConvKern::WEIGHTS_FIXED);
  loc.enableLocation(1e6);
  CConvKern tied(base, H, W, C, ph, pw, 2, CConvKern::WEIGHTS_TIED);
  tied.setParam(1.0, base.getNumParams());
  expect(loc.getNumParams() == base.getNumParams() + 1, "location kernel has one extra parameter");
  const unsigned int N = 4, M = 5;
  CMatrix X(N, H * W * C);
  fillRand(X, 1);
  CMatrix A(N, N), B(N, N);
  loc.compute(A, X); tied.compute(B, X);
  expect(maxAbsDiff(A, B) < 1e-8, "C1: K equals tied weight 1 when k_loc == 1");
  CMatrix da(N, 1), db(N, 1);
  loc.diagCompute(da, X); tied.diagCompute(db, X);
  expect(maxAbsDiff(da, db) < 1e-8, "C1: diagonal equals tied weight 1");
  CMatrix Z(M, q), Zl(M, q + 2);
  fillRand(Z, 2);
  for(unsigned int m = 0; m < M; m++)
  {
    for(unsigned int j = 0; j < q; j++) Zl.setVal(Z.getVal(m, j), m, j);
    Zl.setVal(0.1 * m, m, q); Zl.setVal(0.2 * m, m, q + 1);
  }
  CInducingPatches ipL(Zl), ipT(Z);
  CMatrix Ku1(M, M), Ku2(M, M), Kf1, Kf2;
  ipL.computeKuu(Ku1, loc); ipT.computeKuu(Ku2, tied);
  expect(maxAbsDiff(Ku1, Ku2) < 1e-8, "C1: K_uu equals plain patches when k_loc == 1");
  ipL.computeKuf(Kf1, loc, X); ipT.computeKuf(Kf2, tied, X);
  expect(maxAbsDiff(Kf1, Kf2) < 1e-8, "C1: K_uf equals plain patches when k_loc == 1");
}

// C2: direct double loops, with a finite lengthscale.
void testDoubleLoop(unsigned int stride)
{
  CRbfKern base = makeRbf(q);
  CConvKern loc(base, H, W, C, ph, pw, stride, CConvKern::WEIGHTS_FIXED);
  loc.enableLocation(0.45);
  const unsigned int N = 3, M = 4, P = loc.numPatches();
  CMatrix X(N, H * W * C), Xp;
  fillRand(X, 3);
  loc.extractPatches(Xp, X);
  const double l2 = 0.45 * 0.45;
  CMatrix K(N, N);
  loc.compute(K, X);
  double worst = 0.0;
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < N; m++)
    {
      double s = 0.0;
      for(unsigned int a = 0; a < P; a++)
        for(unsigned int b = 0; b < P; b++)
        {
          double ai, aj, bi, bj;
          loc.patchLocation(a, ai, aj); loc.patchLocation(b, bi, bj);
          const double kl = exp(-0.5 * ((ai - bi) * (ai - bi) + (aj - bj) * (aj - bj)) / l2);
          s += kl * base.computeElement(Xp, n * P + a, Xp, m * P + b);
        }
      worst = max(worst, fabs(s - K.getVal(n, m)));
    }
  expect(worst < 1e-12, "C2: K double loop, stride " + to_string(stride));

  CMatrix Z(M, q + 2), Zp(M, q), Kuf;
  fillRand(Z, 4);
  for(unsigned int m = 0; m < M; m++)
  {
    Z.setVal(0.5 + 0.2 * Z.getVal(m, q), m, q);
    Z.setVal(0.5 + 0.2 * Z.getVal(m, q + 1), m, q + 1);
    for(unsigned int j = 0; j < q; j++) Zp.setVal(Z.getVal(m, j), m, j);
  }
  CInducingPatches ip(Z);
  ip.computeKuf(Kuf, loc, X);
  worst = 0.0;
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int n = 0; n < N; n++)
    {
      double s = 0.0;
      for(unsigned int p = 0; p < P; p++)
      {
        double li, lj;
        loc.patchLocation(p, li, lj);
        const double d2 = (Z.getVal(m, q) - li) * (Z.getVal(m, q) - li)
                        + (Z.getVal(m, q + 1) - lj) * (Z.getVal(m, q + 1) - lj);
        s += exp(-0.5 * d2 / l2) * base.computeElement(Zp, m, Xp, n * P + p);
      }
      worst = max(worst, fabs(s - Kuf.getVal(m, n)));
    }
  expect(worst < 1e-12, "C2: K_uf double loop, stride " + to_string(stride));
}

// C3 at kernel level: gradients of K and of the diagonal (exact and estimated) with respect
// to every parameter, including the location lengthscale.
void testKernelGradients()
{
  CRbfKern base = makeRbf(q);
  CConvKern loc(base, H, W, C, ph, pw, 1, CConvKern::WEIGHTS_FIXED);
  loc.enableLocation(0.6);
  const unsigned int N = 3;
  CMatrix X(N, H * W * C), cv(N, N), cd(N, 1);
  fillRand(X, 5); fillRand(cv, 6); fillRand(cd, 7);
  for(int est = 0; est < 2; est++)
  {
    if(est) loc.setDiagEstimator(6, 11);
    CMatrix g(1, loc.getNumParams()), gd(1, loc.getNumParams());
    loc.getGradParams(g, X, X, cv, false);
    loc.getDiagGradParams(gd, X, cd, false);
    double worstK = 0.0, worstD = 0.0;
    const double h = 1e-6;
    for(unsigned int i = 0; i < loc.getNumParams(); i++)
    {
      const double p0 = loc.getParam(i);
      double fK[2], fD[2];
      for(int s = 0; s < 2; s++)
      {
        loc.setParam(p0 + (s ? h : -h), i);
        CMatrix K(N, N), d(N, 1);
        loc.compute(K, X);
        loc.diagCompute(d, X);
        double a = 0.0, b = 0.0;
        for(unsigned int n = 0; n < N; n++)
        {
          b += cd.getVal(n) * d.getVal(n);
          for(unsigned int m = 0; m < N; m++) a += cv.getVal(n, m) * K.getVal(n, m);
        }
        fK[s] = a; fD[s] = b;
      }
      loc.setParam(p0, i);
      worstK = max(worstK, fabs((fK[1] - fK[0]) / (2 * h) - g.getVal(i)) / max(1.0, fabs(g.getVal(i))));
      worstD = max(worstD, fabs((fD[1] - fD[0]) / (2 * h) - gd.getVal(i)) / max(1.0, fabs(gd.getVal(i))));
    }
    expect(worstK < 1e-6, "C3: getGradParams of K (location kernel)");
    expect(worstD < 1e-6, string("C3: getDiagGradParams (location kernel, ") + (est ? "estimated" : "exact") + " diagonal)");
  }
}

// C3 through CSvgp: ELBO finite differences over kernel, Z and inducing locations.
void testSvgpGradients(unsigned int stride)
{
  const unsigned int M = 5, N = 8;
  CRbfKern base = makeRbf(q);
  CConvKern loc(base, H, W, C, ph, pw, stride, CConvKern::WEIGHTS_FIXED);
  loc.enableLocation(0.5);
  CMatrix X(N, H * W * C), y(N, 1), Zdummy(M, H * W * C);
  fillRand(X, 8); fillRand(y, 9); fillRand(Zdummy, 10);
  CSvgp sv(&loc, &X, &y, Zdummy, 5.0, 0);
  sv.setInducingType("patches");
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
    maxRel = max(maxRel, fabs(fd - an) / max(1.0, max(fabs(fd), fabs(an))));
  }
  cout << "  info: max relative diff " << maxRel << " over " << sv.getOptNumParams() << " params" << endl;
  expect(maxRel < 5e-4, "C3: ELBO gradient, location kernel with inducing pairs, stride " + to_string(stride));
}

void testDeepGp()
{
  const unsigned int N = 10, M = 4, hidden = 2;
  CRbfKern base = makeRbf(q);
  CConvKern loc(base, H, W, C, ph, pw, 2, CConvKern::WEIGHTS_FIXED);
  loc.enableLocation(0.5);
  CRbfKern k1(hidden);
  k1.setInitParam();
  CMatrix X(N, H * W * C), y(N, 1);
  fillRand(X, 21); fillRand(y, 22);
  ndlutil::init_genrand(5u);
  vector<CKern*> kerns = {&loc, &k1};
  CDeepGp dgp(X, y, kerns, vector<unsigned int>{H * W * C, hidden, 1u},
              vector<unsigned int>{M, M}, 4.0, 0.5, 0);
  dgp.setLayerInducingType(0, "patches");
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
  expect(maxErr < 2e-2, "C3: CDeepGp gradients with a location-kernel conv layer");

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
  expect(p0.getCols() == p1.getCols() && maxAbsDiff(p0, p1) < 1e-10, "stream: location kernel and inducing pairs round trip");
  const CConvKern* lk = dynamic_cast<const CConvKern*>(loaded.layer(0).getKern());
  expect(lk != 0 && lk->hasLocation() && fabs(lk->locationLengthscale() - 0.5) < 1e-12,
         "stream: location lengthscale restored");
  CMatrix Xs(3, H * W * C), mu0, v0, mu1, v1;
  fillRand(Xs, 24);
  dgp.predictMeanVar(Xs, mu0, v0);
  loaded.predictMeanVar(Xs, mu1, v1);
  expect(maxAbsDiff(mu0, mu1) < 1e-9 && maxAbsDiff(v0, v1) < 1e-9, "stream: predictions reproduced");
}

void testErrors()
{
  CMlpKern mlp(q);
  mlp.setInitParam();
  CConvKern c(mlp, H, W, C, ph, pw, 1, CConvKern::WEIGHTS_FIXED);
  c.enableLocation(0.5);
  CMatrix X(2, H * W * C), V(3, q + 1), a(3, 1), K;
  fillRand(X, 1); fillRand(V, 2); fillRand(a, 3);
  bool threw = false;
  try { c.wsRidgeCrossCompute(K, X, V, a, 1.0); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: weight-space windows rejected for the location kernel");
  CConvKern learned(mlp, H, W, C, ph, pw, 1, CConvKern::WEIGHTS_LEARNED);
  threw = false;
  try { learned.enableLocation(0.5); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: location kernel requires fixed weights");
  threw = false;
  try { c.enableLocation(-1.0); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: non-positive lengthscale rejected");
}

}  // namespace

int main()
{
  try
  {
    testReduction();
    for(unsigned int stride = 1; stride <= 2; stride++)
    {
      testDoubleLoop(stride);
      testSvgpGradients(stride);
    }
    testKernelGradients();
    testDeepGp();
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
