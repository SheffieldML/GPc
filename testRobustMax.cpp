/* CIP-003E Phase 4: CRobustMaxNoise.

   - P(argmax) against the closed form (K=2), a pure-stdlib reference fixture
     (testdata/robustmax, generator in the same directory), and Monte Carlo.
   - Finite-difference gradients of the block expected log-likelihood and of log Z
     in mu and var, and of the expected log-likelihood in eps.
   - K=2 reduces to the probit step likelihood (Z = eps + (1-2eps) Phi(d/sqrt(V))).
   - Class probabilities from predictBlock sum to one.
   - Block EP on a K=3 problem: dL/dK against finite differences of the log marginal.
   - .model round trip and the capability guard (no Laplace). */
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "CEpInference.h"
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
static double Phi(double z) { return 0.5 * erfc(-z / sqrt(2.0)); }

static CRobustMaxNoise makeNoise(unsigned int K, double eps, CMatrix& Y)
{
  Y = CMatrix(1, K);
  Y.zeros();
  Y.setVal(1.0, 0, 0);
  return CRobustMaxNoise(&Y, K, eps);
}

int main()
{
  try {
    // --- K = 2 closed form, including tiny and large variances ---
    {
      const double mus[4][2] = {{0.3, -0.2}, {-1.0, 1.0}, {0.0, 0.0}, {2.0, 2.1}};
      const double vs[4][2] = {{0.5, 0.8}, {1e-8, 1e-8}, {1.0, 1.0}, {30.0, 0.01}};
      double worst = 0.0;
      for(int t = 0; t < 4; t++)
        for(unsigned int c = 0; c < 2; c++)
        {
          const double d = mus[t][c] - mus[t][1 - c];
          const double ref = Phi(d / sqrt(vs[t][0] + vs[t][1]));
          worst = max(worst, fabs(CRobustMaxNoise::classProbability(mus[t], vs[t], 2, c, 0, 0) - ref));
        }
      expect(worst < 1e-10, "K=2 P(argmax) == Phi(d/sqrt(v1+v2)) (err " + dtoa(worst) + ")");
    }

    // --- K = 2: logZ equals the probit step likelihood with softening eps ---
    {
      CMatrix Y;
      CRobustMaxNoise rm = makeNoise(2, 1e-3, Y);
      const double mu[2] = {0.4, -0.3}, var[2] = {0.6, 0.2}, y[2] = {1.0, 0.0};
      const double ref = 1e-3 + (1.0 - 2e-3) * Phi(0.7 / sqrt(0.8));
      expect(fabs(exp(rm.logZBlock(y, mu, var, 0)) - ref) < 1e-10, "K=2 Z == eps + (1-2eps) Phi(d/sqrt(V))");
    }

    // --- Reference fixture ---
    {
      ifstream f("testdata/robustmax/robustmax_ref.txt");
      expect(f.good(), "fixture testdata/robustmax/robustmax_ref.txt readable");
      string line;
      int n = 0;
      double worstP = 0.0, worstL = 0.0;
      while(getline(f, line))
      {
        istringstream is(line);
        unsigned int K, c;
        double eps;
        is >> K >> c >> eps;
        vector<double> mu(K), var(K);
        for(unsigned int k = 0; k < K; k++) is >> mu[k];
        for(unsigned int k = 0; k < K; k++) is >> var[k];
        double P, ell;
        is >> P >> ell;
        CMatrix Y(1, K);
        Y.zeros();
        Y.setVal(1.0, 0, c);
        CRobustMaxNoise rm(&Y, K, eps);
        vector<double> gm(K), gv(K), y(K, 0.0);
        y[c] = 1.0;
        const double L = rm.expectedLogLikelihoodBlock(&y[0], &mu[0], &var[0], 0, &gm[0], &gv[0]);
        worstP = max(worstP, fabs(CRobustMaxNoise::classProbability(&mu[0], &var[0], K, c, 0, 0) - P));
        worstL = max(worstL, fabs(L - ell));
        n++;
      }
      expect(n == 8, "fixture has 8 cases");
      expect(worstP < 1e-9, "P(argmax) vs reference fixture (err " + dtoa(worstP) + ")");
      expect(worstL < 1e-8, "expected log-lik vs reference fixture (err " + dtoa(worstL) + ")");
    }

    // --- Monte Carlo ---
    {
      const unsigned int K = 4;
      const double mu[K] = {0.2, 0.5, -0.3, 0.1}, var[K] = {0.7, 0.3, 1.2, 0.5};
      vector<double> pr(K);
      CMatrix Y;
      CRobustMaxNoise rm = makeNoise(K, 1e-3, Y);
      rm.predictBlock(mu, var, 0, &pr[0]);
      srand48(7);
      const int S = 400000;
      vector<double> cnt(K, 0.0);
      for(int s = 0; s < S; s++)
      {
        unsigned int best = 0;
        double bv = -1e300;
        for(unsigned int k = 0; k < K; k++)
        {
          // Box-Muller
          const double u1 = drand48() + 1e-300, u2 = drand48();
          const double f = mu[k] + sqrt(var[k]) * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
          if(f > bv) { bv = f; best = k; }
        }
        cnt[best] += 1.0;
      }
      double worst = 0.0, sum = 0.0;
      const double a = 1e-3 / 3.0, kap = 1.0 - 1e-3 - a;
      for(unsigned int k = 0; k < K; k++)
      {
        worst = max(worst, fabs(pr[k] - (a + kap * cnt[k] / S)));
        sum += pr[k];
      }
      expect(worst < 4e-3, "predictBlock vs Monte Carlo (err " + dtoa(worst) + ")");
      expect(fabs(sum - 1.0) < 1e-10, "predictBlock probabilities sum to 1");
    }

    // --- Finite differences ---
    {
      const unsigned int K = 4;
      CMatrix Y(1, K);
      Y.zeros();
      Y.setVal(1.0, 0, 2);
      CRobustMaxNoise rm(&Y, K, 5e-3);
      const double y[K] = {0, 0, 1, 0};
      double mu[K] = {0.3, -0.4, 0.2, 0.6}, var[K] = {0.5, 0.9, 0.3, 0.7};
      double gm[K], gv[K], dm[K], dv[K];
      rm.expectedLogLikelihoodBlock(y, mu, var, 0, gm, gv);
      rm.cavityMomentsBlock(y, mu, var, 0, dm, dv);
      const double h = 1e-5;
      double e1 = 0, e2 = 0, e3 = 0, e4 = 0;
      for(unsigned int k = 0; k < K; k++)
      {
        double gtmp[K], gtmp2[K];
        double m1[K], m2[K], v1[K], v2[K];
        copy(mu, mu + K, m1); copy(mu, mu + K, m2);
        copy(var, var + K, v1); copy(var, var + K, v2);
        m1[k] += h; m2[k] -= h; v1[k] += h; v2[k] -= h;
        const double fdMu = (rm.expectedLogLikelihoodBlock(y, m1, var, 0, gtmp, gtmp2)
                             - rm.expectedLogLikelihoodBlock(y, m2, var, 0, gtmp, gtmp2)) / (2 * h);
        const double fdVar = (rm.expectedLogLikelihoodBlock(y, mu, v1, 0, gtmp, gtmp2)
                              - rm.expectedLogLikelihoodBlock(y, mu, v2, 0, gtmp, gtmp2)) / (2 * h);
        e1 = max(e1, fabs(fdMu - gm[k]));
        e2 = max(e2, fabs(fdVar - gv[k]));
        const double zMu = (rm.logZBlock(y, m1, var, 0) - rm.logZBlock(y, m2, var, 0)) / (2 * h);
        const double zVar = (rm.logZBlock(y, mu, v1, 0) - rm.logZBlock(y, mu, v2, 0)) / (2 * h);
        e3 = max(e3, fabs(zMu - dm[k]));
        e4 = max(e4, fabs(zVar - dv[k]));
      }
      expect(e1 < 1e-7 && e2 < 1e-7, "expected log-lik gradients vs FD (" + dtoa(max(e1, e2)) + ")");
      expect(e3 < 1e-7 && e4 < 1e-7, "log Z derivatives vs FD (" + dtoa(max(e3, e4)) + ")");
      double gt[K], gt2[K];
      CRobustMaxNoise rp(&Y, K, 5e-3 + 1e-6), rn(&Y, K, 5e-3 - 1e-6);
      const double fd = (rp.expectedLogLikelihoodBlock(y, mu, var, 0, gt, gt2)
                         - rn.expectedLogLikelihoodBlock(y, mu, var, 0, gt, gt2)) / 2e-6;
      expect(fabs(fd - rm.expectedLogLikelihoodDEps(y, mu, var)) < 1e-7 * (1.0 + fabs(fd)), "d/d eps vs FD");
    }

    // --- Capabilities, refusals, stream ---
    {
      CMatrix Y;
      CRobustMaxNoise rm = makeNoise(3, 2e-3, Y);
      bool threw = false;
      try { rm.requireCapability(CNoise::CapPointDerivs, "Laplace"); }
      catch(ndlexceptions::NotImplementedError&) { threw = true; }
      expect(threw, "Laplace refused (no point derivatives)");
      expect((rm.capabilities() & CNoise::CapCavityMoments) && (rm.capabilities() & CNoise::CapExpectedLogLik),
             "advertises EP/ADF and VI capabilities");
      const double mu[3] = {0, 0, 0}, var[3] = {1, 1, 1}, bad[3] = {0.5, 0.5, 0.0};
      threw = false;
      double a[3], b[3];
      try { rm.cavityMomentsBlock(bad, mu, var, 0, a, b); }
      catch(ndlexceptions::Error&) { threw = true; }
      expect(threw, "non one-hot target refused");

      ostringstream os;
      rm.toStream(os);
      istringstream is(os.str());
      CNoise* back = readNoiseFromStream(is);
      CRobustMaxNoise* rb = dynamic_cast<CRobustMaxNoise*>(back);
      expect(rb != 0 && rb->getNumClasses() == 3 && fabs(rb->getEps() - 2e-3) < ndlutil::MATCHTOL
                 && rb->getType() == "robustmax" && rb->getOutputDim() == 3,
             ".model round trip (type, K, eps, outputDim)");
      delete back;
    }

    // --- Block EP, K = 3 ---
    {
      const unsigned int N = 24, K = 3;
      CMatrix X(N, 1), Y(N, K), Kmat(N, N);
      Y.zeros();
      for(unsigned int i = 0; i < N; i++)
      {
        const double x = -3.0 + 6.0 * i / (N - 1.0);
        X.setVal(x, i, 0);
        // class by region (clean labels: with flipped labels the robust-max sites hit the
        // precision floor and the log marginal is not stationary in the sites)
        unsigned int c = x < -1.0 ? 0 : (x < 1.0 ? 1 : 2);
        Y.setVal(1.0, i, c);
      }
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
          Kmat.setVal(4.0 * exp(-0.5 * pow(X.getVal(i, 0) - X.getVal(j, 0), 2.0) / 0.8), i, j);
      CRobustMaxNoise rm(&Y, K, 1e-2);
      CEpInference ep;
      ep.setJitter(1e-6);
      ep.setMaxIters(500);
      ep.setTol(1e-10);
      CEpSites sites;
      CMatrix mu, vd;
      const double L = ep.runBlock(vector<CMatrix>(1, Kmat), Y, rm, sites, mu, vd);
      expect(L == L && L < 0.0, "K=3 block EP log marginal finite (" + dtoa(L) + ")");
      { double mt = 1e300; for(unsigned int i = 0; i < N; i++) for(unsigned int k = 0; k < K; k++) mt = min(mt, sites.tau.getVal(i, k));
        expect(ep.converged() && mt > 1e-8, "K=3 EP converged with every site precision above the floor"); }
      unsigned int correct = 0;
      for(unsigned int i = 0; i < N; i++)
      {
        unsigned int best = 0, truth = 0;
        for(unsigned int k = 0; k < K; k++)
        {
          if(mu.getVal(i, k) > mu.getVal(i, best)) best = k;
          if(Y.getVal(i, k) > 0.5) truth = k;
        }
        if(best == truth) correct++;
      }
      expect(correct == N, "K=3 EP posterior recovers the labels (" + ndlstrutil::itoa(correct) + "/24)");
      // Gradient of the log marginal in the lengthscale parameter s (K = 4 exp(-d^2/(2s))),
      // summed over the K latents that share the kernel. (Scaling K is no test:
      // robust-max depends on f only through its argmax, so L is flat in the amplitude.)
      auto gram = [&](double s) {
        CMatrix G(N, N);
        for(unsigned int i = 0; i < N; i++)
          for(unsigned int j = 0; j < N; j++)
            G.setVal(4.0 * exp(-0.5 * pow(X.getVal(i, 0) - X.getVal(j, 0), 2.0) / s), i, j);
        return G;
      };
      const double s0 = 0.8;
      CMatrix dK;
      double analytic = 0.0;
      for(unsigned int k = 0; k < K; k++)
      {
        ep.fillDLDKBlock(k, dK);
        for(unsigned int i = 0; i < N; i++)
          for(unsigned int j = 0; j < N; j++)
          {
            const double d2 = pow(X.getVal(i, 0) - X.getVal(j, 0), 2.0);
            analytic += dK.getVal(i, j) * Kmat.getVal(i, j) * 0.5 * d2 / (s0 * s0);
          }
      }
      double Lv[2];
      const double h = 1e-4;
      for(int sgn = 0; sgn < 2; sgn++)
      {
        CMatrix Ks = gram(s0 * (sgn == 0 ? 1.0 + h : 1.0 - h));
        CEpInference e2;
        e2.setJitter(1e-6); e2.setMaxIters(500); e2.setTol(1e-10);
        CEpSites st;
        CMatrix m2, v2;
        Lv[sgn] = e2.runBlock(vector<CMatrix>(1, Ks), Y, rm, st, m2, v2);
      }
      const double fd = (Lv[0] - Lv[1]) / (2 * h * s0);
      expect(fabs(fd - analytic) < 1e-3 * (1.0 + fabs(fd)),
             "K=3 EP dL/d(lengthscale) vs finite difference (fd " + dtoa(fd) + ", analytic " + dtoa(analytic) + ")");
    }
  }
  catch(ndlexceptions::Error& e) {
    cout << "ERROR: " << e.what() << endl;
    return 2;
  }
  cout << (fails == 0 ? "All tests passed." : "FAILURES: ") << (fails ? ndlstrutil::itoa(fails) : "") << endl;
  return fails == 0 ? 0 : 1;
}
