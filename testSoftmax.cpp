/* CIP-003E Phase 5: CSoftmaxNoise and the block Laplace solver.

   - Fixed-draw (common random number) gradients against finite differences.
   - Monte Carlo accuracy against K=2 (logistic, quadrature).
   - Point derivatives: gradient, negative Hessian and its derivative by finite differences.
   - Block Laplace: mode stationarity, log marginal and dL/dK against finite
     differences, equality with the scalar solver at blockSize 1 (probit),
     predictive marginals.
   - Block EP with the sampled cavity moments (fixed seed, statistical tolerance).
   - .model round trip and capabilities. */
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "CEpInference.h"
#include "CLaplaceInference.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlexceptions.h"
#include "ndlutil.h"

using namespace std;

static int fails = 0;
static void expect(bool cond, const string& msg)
{
  cout << (cond ? "PASS: " : "FAIL: ") << msg << endl;
  if(!cond) fails++;
}
static string dtoa(double v) { ostringstream o; o << v; return o.str(); }

static CMatrix gram(const CMatrix& X, double s)
{
  const unsigned int N = X.getRows();
  CMatrix G(N, N);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < N; j++)
      G.setVal(4.0 * exp(-0.5 * pow(X.getVal(i, 0) - X.getVal(j, 0), 2.0) / s), i, j);
  return G;
}

int main()
{
  try {
    const unsigned int K = 4;
    CMatrix Y1(1, K);
    Y1.zeros();
    Y1.setVal(1.0, 0, 2);
    const double y[K] = {0, 0, 1, 0};
    double mu[K] = {0.3, -0.4, 0.8, 0.6}, var[K] = {0.5, 0.9, 0.3, 0.7};

    // --- fixed-draw gradients ---
    {
      CSoftmaxNoise sm(&Y1, K, 64, 7);
      double gm[K], gv[K], dm[K], dv[K], t1[K], t2[K];
      sm.expectedLogLikelihoodBlock(y, mu, var, 0, gm, gv);
      sm.cavityMomentsBlock(y, mu, var, 0, dm, dv);
      const double h = 1e-5;
      double e1 = 0, e2 = 0, e3 = 0, e4 = 0;
      for(unsigned int k = 0; k < K; k++)
      {
        double m1[K], m2[K], v1[K], v2[K];
        copy(mu, mu + K, m1); copy(mu, mu + K, m2);
        copy(var, var + K, v1); copy(var, var + K, v2);
        m1[k] += h; m2[k] -= h; v1[k] += h; v2[k] -= h;
        e1 = max(e1, fabs((sm.expectedLogLikelihoodBlock(y, m1, var, 0, t1, t2) - sm.expectedLogLikelihoodBlock(y, m2, var, 0, t1, t2)) / (2 * h) - gm[k]));
        e2 = max(e2, fabs((sm.expectedLogLikelihoodBlock(y, mu, v1, 0, t1, t2) - sm.expectedLogLikelihoodBlock(y, mu, v2, 0, t1, t2)) / (2 * h) - gv[k]));
        e3 = max(e3, fabs((sm.logZBlock(y, m1, var, 0) - sm.logZBlock(y, m2, var, 0)) / (2 * h) - dm[k]));
        e4 = max(e4, fabs((sm.logZBlock(y, mu, v1, 0) - sm.logZBlock(y, mu, v2, 0)) / (2 * h) - dv[k]));
      }
      expect(max(e1, e2) < 1e-7, "expected log-lik gradients vs FD, fixed draws (" + dtoa(max(e1, e2)) + ")");
      expect(max(e3, e4) < 1e-7, "log Z derivatives vs FD, fixed draws (" + dtoa(max(e3, e4)) + ")");
      double p1[K], p2[K];
      sm.predictBlock(mu, var, 0, p1);
      CSoftmaxNoise sm2(&Y1, K, 64, 7);
      sm2.predictBlock(mu, var, 0, p2);
      double sum = 0.0, same = 0.0;
      for(unsigned int k = 0; k < K; k++) { sum += p1[k]; same = max(same, fabs(p1[k] - p2[k])); }
      expect(fabs(sum - 1.0) < 1e-12 && same == 0.0, "predictBlock sums to 1 and is deterministic for a fixed seed");
    }

    // --- Monte Carlo accuracy: K=2 is the logistic likelihood ---
    {
      CMatrix Y2(1, 2);
      Y2.zeros(); Y2.setVal(1.0, 0, 0);
      const double m2[2] = {0.7, -0.2}, v2[2] = {0.6, 0.4}, yy[2] = {1.0, 0.0};
      // E[log sigma(d)], d ~ N(0.9, 1.0), by trapezoid on a fine grid.
      double ref = 0.0;
      const double dm0 = 0.9, dv0 = 1.0;
      const int n = 40001;
      for(int i = 0; i < n; i++)
      {
        const double z = -10.0 + 20.0 * i / (n - 1.0);
        const double dens = exp(-0.5 * z * z) / sqrt(2.0 * 3.14159265358979323846);
        const double d = dm0 + sqrt(dv0) * z;
        ref += -log1p(exp(-d)) * dens * (20.0 / (n - 1.0));
      }
      CSoftmaxNoise big(&Y2, 2, 400000, 3);
      double g1[2], g2[2];
      const double est = big.expectedLogLikelihoodBlock(yy, m2, v2, 0, g1, g2);
      expect(fabs(est - ref) < 2e-3, "K=2 expected log-lik vs logistic quadrature (" + dtoa(est) + " vs " + dtoa(ref) + ")");
      CSoftmaxNoise small(&Y2, 2, 256, 3);
      const double est2 = small.expectedLogLikelihoodBlock(yy, m2, v2, 0, g1, g2);
      expect(fabs(est2 - ref) < 2e-2, "S=256 within Monte Carlo error (" + dtoa(fabs(est2 - ref)) + ")");
    }

    // --- point derivatives ---
    {
      CSoftmaxNoise sm(&Y1, K, 64, 7);
      double f[K] = {0.2, -0.5, 1.0, 0.4}, g[K], lp;
      CMatrix W, dW;
      sm.pointDerivsBlock(y, f, 0, &lp, g, W, &dW);
      const double h = 1e-6;
      double eg = 0, eW = 0, edW = 0;
      for(unsigned int d = 0; d < K; d++)
      {
        double fp[K], fm[K], gp[K], gm[K], lpp, lpm;
        copy(f, f + K, fp); copy(f, f + K, fm);
        fp[d] += h; fm[d] -= h;
        CMatrix Wp, Wm;
        sm.pointDerivsBlock(y, fp, 0, &lpp, gp, Wp);
        sm.pointDerivsBlock(y, fm, 0, &lpm, gm, Wm);
        eg = max(eg, fabs((lpp - lpm) / (2 * h) - g[d]));
        for(unsigned int a = 0; a < K; a++)
        {
          eW = max(eW, fabs(-(gp[a] - gm[a]) / (2 * h) - W.getVal(a, d)));
          for(unsigned int b = 0; b < K; b++)
            edW = max(edW, fabs((Wp.getVal(a, b) - Wm.getVal(a, b)) / (2 * h) - dW.getVal(a * K + b, d)));
        }
      }
      expect(eg < 1e-8, "point gradient vs FD (" + dtoa(eg) + ")");
      expect(eW < 1e-8, "negative Hessian vs FD of the gradient (" + dtoa(eW) + ")");
      expect(edW < 1e-8, "dW/df vs FD of W (" + dtoa(edW) + ")");
    }

    // --- Block Laplace, K = 3 ---
    const unsigned int N = 20, C = 3;
    CMatrix X(N, 1), Y(N, C);
    Y.zeros();
    for(unsigned int i = 0; i < N; i++)
    {
      const double x = -3.0 + 6.0 * i / (N - 1.0);
      X.setVal(x, i, 0);
      unsigned int c = x < -1.0 ? 0 : (x < 1.0 ? 1 : 2);
      if(i == 4) c = 2;  // one conflicting label: Laplace has no precision floor
      Y.setVal(1.0, i, c);
    }
    const double s0 = 0.8;
    {
      CSoftmaxNoise sm(&Y, C, 64, 1);
      CLaplaceInference lap;
      lap.setNoise(&sm);
      lap.setMaxIters(200);
      lap.setTol(1e-13);
      lap.setJitter(1e-6);
      const CMatrix Kmat = gram(X, s0);
      const double L = lap.runBlock(Kmat, Y);
      expect(lap.converged() && L == L && L < 0.0, "block Laplace converged, log marginal " + dtoa(L));
      double worst = 0.0;
      {
        vector<double> yb(C), fb(C), gb(C);
        CMatrix H;
        for(unsigned int i = 0; i < N; i++)
        {
          for(unsigned int a = 0; a < C; a++) { yb[a] = Y.getVal(i, a); fb[a] = lap.lastFhat().getVal(i, a); }
          double lp;
          sm.pointDerivsBlock(&yb[0], &fb[0], 0, &lp, &gb[0], H);
          for(unsigned int a = 0; a < C; a++)
            worst = max(worst, fabs(gb[a] - lap.lastAlphaBlock().getVal(i, a)));
        }
      }
      expect(worst < 1e-7, "mode stationarity alpha == grad log p (" + dtoa(worst) + ")");
      CMatrix dK;
      lap.fillDLDK(dK);
      double analytic = 0.0, dAmp = 0.0;
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
        {
          analytic += dK.getVal(i, j) * Kmat.getVal(i, j) * 0.5 * pow(X.getVal(i, 0) - X.getVal(j, 0), 2.0) / (s0 * s0);
          dAmp += dK.getVal(i, j) * Kmat.getVal(i, j);
        }
      double Lv[2], La[2];
      const double h = 1e-5;
      for(int sg = 0; sg < 2; sg++)
      {
        const double sc = sg == 0 ? 1.0 + h : 1.0 - h;
        CLaplaceInference l2, l3;
        l2.setNoise(&sm); l2.setMaxIters(200); l2.setTol(1e-13);
        l3.setNoise(&sm); l3.setMaxIters(200); l3.setTol(1e-13);
        Lv[sg] = l2.runBlock(gram(X, s0 * sc), Y);
        CMatrix Ks(Kmat);
        Ks.scale(sc);
        La[sg] = l3.runBlock(Ks, Y);
      }
      const double fd = (Lv[0] - Lv[1]) / (2 * h * s0);
      expect(fabs(fd - analytic) < 1e-5 * (1.0 + fabs(fd)),
             "block Laplace dL/d(lengthscale) vs FD (fd " + dtoa(fd) + ", analytic " + dtoa(analytic) + ")");
      const double fdA = (La[0] - La[1]) / (2 * h);
      expect(fabs(fdA - dAmp) < 1e-5 * (1.0 + fabs(fdA)),
             "block Laplace amplitude gradient vs FD (" + dtoa(fdA) + " vs " + dtoa(dAmp) + ")");

      CMatrix kss(N, 1), pm, pv;
      for(unsigned int i = 0; i < N; i++) kss.setVal(4.0, i, 0);
      lap.predictiveMeanVarBlock(Kmat, kss, pm, pv);
      unsigned int correct = 0;
      bool pos = true;
      for(unsigned int i = 0; i < N; i++)
      {
        unsigned int best = 0, truth = 0;
        for(unsigned int a = 0; a < C; a++)
        {
          if(pm.getVal(i, a) > pm.getVal(i, best)) best = a;
          if(Y.getVal(i, a) > 0.5) truth = a;
          pos = pos && pv.getVal(i, a) > 0.0 && pv.getVal(i, a) <= 4.0 + 1e-9;
        }
        if(best == truth) correct++;
      }
      expect(correct >= N - 2 && pos, "block Laplace predictive marginals (" + ndlstrutil::itoa(correct) + "/20 correct)");
    }

    // --- blockSize 1: block solver == scalar solver (probit) ---
    {
      CMatrix yp(N, 1);
      for(unsigned int i = 0; i < N; i++) yp.setVal(Y.getVal(i, 0) > 0.5 ? 1.0 : -1.0, i, 0);
      CProbitNoise pn(&yp);
      pn.setGpyMode(true);
      CLaplaceInference a, b;
      a.setNoise(&pn); b.setNoise(&pn);
      a.setMaxIters(100); b.setMaxIters(100); a.setTol(1e-13); b.setTol(1e-13);
      const CMatrix Kmat = gram(X, s0);
      const double La = a.run(Kmat, yp), Lb = b.runBlock(Kmat, yp);
      CMatrix da, db;
      a.fillDLDK(da); b.fillDLDK(db);
      double dd = 0.0;
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
          dd = max(dd, fabs(da.getVal(i, j) - db.getVal(i, j)));
      expect(fabs(La - Lb) < 1e-8 * (1.0 + fabs(La)), "blockSize 1: log marginal == scalar Laplace (" + dtoa(La) + " vs " + dtoa(Lb) + ")");
      expect(dd < 1e-6, "blockSize 1: dL/dK == scalar Laplace (" + dtoa(dd) + ")");
    }

    // --- Block EP with sampled cavity moments ---
    {
      CMatrix Yc(N, C);
      Yc.zeros();
      for(unsigned int i = 0; i < N; i++)
      {
        const double x = X.getVal(i, 0);
        Yc.setVal(1.0, i, x < -1.0 ? 0 : (x < 1.0 ? 1 : 2));
      }
      CSoftmaxNoise sm(&Yc, C, 512, 5);
      CEpInference ep;
      ep.setJitter(1e-6); ep.setMaxIters(300); ep.setTol(1e-8); ep.setEta(0.5); ep.setDelta(0.5);
      CEpSites sites;
      CMatrix mu2, vd;
      const double L = ep.runBlock(vector<CMatrix>(1, gram(X, s0)), Yc, sm, sites, mu2, vd);
      unsigned int correct = 0;
      for(unsigned int i = 0; i < N; i++)
      {
        unsigned int best = 0;
        for(unsigned int a = 0; a < C; a++) if(mu2.getVal(i, a) > mu2.getVal(i, best)) best = a;
        if(Yc.getVal(i, best) > 0.5) correct++;
      }
      expect(L == L && L < 0.0 && correct >= N - 1,
             "softmax block EP (fixed seed): finite log marginal " + dtoa(L) + ", " + ndlstrutil::itoa(correct) + "/20 correct");
    }

    // --- capabilities and stream ---
    {
      CSoftmaxNoise sm(&Y, C, 128, 99);
      const unsigned int all = CNoise::CapExpectedLogLik | CNoise::CapPointDerivs | CNoise::CapCavityMoments | CNoise::CapPredictive;
      expect(sm.capabilities() == all, "softmax advertises all four capabilities");
      ostringstream os;
      sm.toStream(os);
      istringstream is(os.str());
      CNoise* back = readNoiseFromStream(is);
      CSoftmaxNoise* sb = dynamic_cast<CSoftmaxNoise*>(back);
      bool ok = sb != 0 && sb->getNumClasses() == C && sb->getNumSamples() == 128 && sb->getSeed() == 99 && sb->getOutputDim() == C;
      if(ok)
      {
        double m[3] = {0.1, 0.4, -0.2}, v[3] = {0.5, 0.2, 0.9}, p1[3], p2[3];
        sm.predictBlock(m, v, 0, p1);
        sb->predictBlock(m, v, 0, p2);
        for(unsigned int k = 0; k < 3; k++) ok = ok && p1[k] == p2[k];
      }
      expect(ok, ".model round trip reproduces the draws (identical predictions)");
      delete back;
    }
  }
  catch(ndlexceptions::Error& e) {
    cout << "ERROR: " << e.what() << endl;
    return 2;
  }
  cout << (fails == 0 ? "All tests passed." : "FAILURES") << endl;
  return fails == 0 ? 0 : 1;
}
