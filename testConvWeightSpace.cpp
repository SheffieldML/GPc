/* CIP-0040 Phase 2: weight-space inducing variables (ridge and Hermite windows) over the
   filters of a convolutional kernel with an MLP patch kernel. Oracles: exact reduction to
   the plain MLP kernel (C1), a per-image double loop over patches (C2/C4), finite
   differences of the SVGP ELBO and of the deep GP objective (C3), stream round trip. */
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

CMlpKern makeMlp(unsigned int q)
{
  CMlpKern k(q);
  k.setInitParam();
  k.setParam(0.6, 0);
  k.setParam(0.8, 1);
  k.setParam(1.2, 2);
  return k;
}

const char* modeName(CConvKern::WeightMode m)
{
  return m == CConvKern::WEIGHTS_LEARNED ? "learned" : (m == CConvKern::WEIGHTS_TIED ? "tied" : "fixed");
}

void unitDirections(CMatrix& V, unsigned int M, unsigned int D, unsigned int seed)
{
  V.resize(M, D);
  fillRand(V, seed);
  for(unsigned int m = 0; m < M; m++)
  {
    double n = 0.0;
    for(unsigned int k = 0; k < D; k++)
      n += V.getVal(m, k) * V.getVal(m, k);
    n = sqrt(n);
    for(unsigned int k = 0; k < D; k++)
      V.setVal(V.getVal(m, k) / n, m, k);
  }
}

// C1: one patch equal to the image and w = 1 reduces to the plain MLP hooks.
void testReduction()
{
  const unsigned int H = 3, W = 2, C = 2, q = 12, M = 5, N = 4;
  CMlpKern base = makeMlp(q);
  CConvKern conv(base, H, W, C, H, W, 1, CConvKern::WEIGHTS_FIXED);
  expect(conv.wsDomainDim() == q, "reduction: window space is patch space");
  CMatrix X(N, q), V, a(M, 1);
  fillRand(X, 1);
  unitDirections(V, M, q + 1, 2);
  fillRand(a, 3);
  CMatrix A, B;
  conv.wsRidgeCrossCompute(A, X, V, a, 0.9);
  base.wsRidgeCrossCompute(B, X, V, a, 0.9);
  expect(maxAbsDiff(A, B) < 1e-13, "reduction: ridge K_fu equals plain MLP");
  vector<unsigned int> ks = {1, 3, 1, 5, 3};
  conv.wsHermiteCrossCompute(A, X, V, ks);
  base.wsHermiteCrossCompute(B, X, V, ks);
  expect(maxAbsDiff(A, B) < 1e-13, "reduction: Hermite K_fu equals plain MLP");
  CMatrix gK(N, M);
  fillRand(gK, 4);
  CMatrix gc(1, conv.getNumParams()), gb(1, base.getNumParams());
  conv.wsRidgeGradParams(gc, X, V, a, 0.9, gK);
  base.wsRidgeGradParams(gb, X, V, a, 0.9, gK);
  expect(maxAbsDiff(gc, gb) < 1e-12, "reduction: ridge parameter gradients equal plain MLP");
  conv.wsHermiteGradParams(gc, X, V, ks, gK);
  base.wsHermiteGradParams(gb, X, V, ks, gK);
  expect(maxAbsDiff(gc, gb) < 1e-12, "reduction: Hermite parameter gradients equal plain MLP");
}

// C2 / C4: K_fu by a per-image double loop of patch-kernel hook calls.
void testDoubleLoop(CConvKern::WeightMode mode, unsigned int stride)
{
  const unsigned int H = 5, W = 6, C = 1, ph = 3, pw = 2, q = 6, M = 7, N = 4;
  CMlpKern base = makeMlp(q);
  CConvKern conv(base, H, W, C, ph, pw, stride, mode);
  for(unsigned int i = base.getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.3 + 0.1 * sin(2.0 * i), i);
  CMatrix X(N, H * W * C), V, a(M, 1);
  fillRand(X, 5);
  unitDirections(V, M, q + 1, 6);
  fillRand(a, 7);
  CMatrix Xp;
  conv.extractPatches(Xp, X);
  const unsigned int P = conv.numPatches();
  const vector<double>& w = conv.patchWeights();
  CMatrix K;
  conv.wsRidgeCrossCompute(K, X, V, a, 0.8);
  double worst = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    CMatrix Xn(P, q);
    for(unsigned int p = 0; p < P; p++)
      for(unsigned int j = 0; j < q; j++)
        Xn.setVal(Xp.getVal(n * P + p, j), p, j);
    CMatrix Kn;
    conv.patchKernel().wsRidgeCrossCompute(Kn, Xn, V, a, 0.8);
    for(unsigned int m = 0; m < M; m++)
    {
      double s = 0.0;
      for(unsigned int p = 0; p < P; p++)
        s += w[p] * Kn.getVal(p, m);
      worst = max(worst, fabs(s - K.getVal(n, m)));
    }
  }
  expect(worst < 1e-12, string("C2: ridge K_fu double loop ") + modeName(mode) + " stride " + to_string(stride));
}

// C3 through CSvgp (ridge with learned widths, or Hermite): ELBO finite differences.
void testSvgpGradients(const string& family, CConvKern::WeightMode mode, unsigned int stride)
{
  const unsigned int H = 5, W = 6, C = 1, ph = 3, pw = 2, q = 6, M = 10, N = 8;
  CMlpKern base = makeMlp(q);
  CConvKern conv(base, H, W, C, ph, pw, stride, mode);
  for(unsigned int i = base.getNumParams(); i < conv.getNumParams(); i++)
    conv.setParam(0.3 + 0.1 * sin(2.0 * i), i);
  CMatrix X(N, H * W * C), y(N, 1), Zdummy(M, H * W * C);
  fillRand(X, 8); fillRand(y, 9); fillRand(Zdummy, 10);
  CSvgp sv(&conv, &X, &y, Zdummy, 5.0, 0);
  sv.setInducingType(family);
  expect(sv.getInducing()->typeName() == family, "svgp: " + family + " inducing selected on a conv kernel");
  if(family == "weightspace")
  {
    CInducingWeightSpace* ws = dynamic_cast<CInducingWeightSpace*>(sv.getInducing());
    expect(ws != 0 && ws->domainDim() == q + 2, "svgp: windows live in patch space (q + 1 direction + offset columns)");
    ws->setLearnWindowWidth(true);
    sv.updateKernelMatrices();
  }
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
  expect(maxRel < 5e-4, "C3: ELBO gradient " + family + " " + modeName(mode) + " stride " + to_string(stride));
}

// C3 through CDeepGp plus stream round trip (exercises the Hermite reader's window dimension).
void testDeepGp(const string& family)
{
  const unsigned int H = 5, W = 6, C = 1, ph = 3, pw = 2, q = 6, N = 10, M = 10, hidden = 2;
  const double beta = 4.0;
  CMlpKern base = makeMlp(q);
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
  dgp.setLayerInducingType(0, family);
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
  cout << "  info: deepgp " << family << " max|g-FD|=" << maxErr << " over " << p0.getCols() << " params" << endl;
  expect(maxErr < 2e-2, "C3: CDeepGp gradients, conv layer with " + family + " inducing");

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
  expect(loaded.getLayerInducingType(0) == family && p0.getCols() == p1.getCols()
         && maxAbsDiff(p0, p1) < 1e-10, "stream: conv layer with " + family + " inducing round trip");
  CMatrix Xs(3, H * W * C), mu0, v0, mu1, v1;
  fillRand(Xs, 24);
  dgp.predictMeanVar(Xs, mu0, v0);
  loaded.predictMeanVar(Xs, mu1, v1);
  expect(maxAbsDiff(mu0, mu1) < 1e-9 && maxAbsDiff(v0, v1) < 1e-9, "stream: predictions reproduced (" + family + ")");
}

void testErrors()
{
  const unsigned int q = 4;
  CRbfKern rbf(q);
  rbf.setInitParam();
  CConvKern conv(rbf, 2, 2, 1, 2, 2, 1, CConvKern::WEIGHTS_FIXED);
  CMatrix X(2, 4), V, a(3, 1), K;
  fillRand(X, 1);
  unitDirections(V, 3, q + 1, 2);
  fillRand(a, 3);
  bool threw = false;
  try { conv.wsRidgeCrossCompute(K, X, V, a, 1.0); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: ridge windows over a non-MLP patch kernel are rejected");
  threw = false;
  try { conv.wsHermiteCrossCompute(K, X, V, vector<unsigned int>(3, 1u)); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: Hermite windows over a non-MLP patch kernel are rejected");
  CMlpKern mlp = makeMlp(q);
  CConvKern convM(mlp, 2, 2, 1, 2, 2, 1, CConvKern::WEIGHTS_FIXED);
  CMatrix gX;
  threw = false;
  try { convM.wsRidgeGradX(gX, X, V, a, 1.0, CMatrix(2, 3)); } catch(ndlexceptions::Error&) { threw = true; }
  expect(threw, "errors: input gradients are rejected (first layer only)");
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
        testSvgpGradients("weightspace", modes[mi], stride);
        testSvgpGradients("hermite", modes[mi], stride);
      }
    testDeepGp("weightspace");
    testDeepGp("hermite");
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
