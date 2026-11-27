/* CIP-003E Phase 2a: block EP (N x K sites) over a CNoise.

   Verification against references that do not depend on EP itself:
   - Gaussian noise: EP is exact, so log marginal, posterior mean/variance,
     dL/dK and predictions equal the closed-form GP results, for K = 1, K = 2 with
     a shared Gram matrix, and K = 2 with per-latent Gram matrices.
   - K = 1 probit through the block path equals the existing probit-only run().
   - A coupled-block stand-in (blockSize 2, still factorising) must equal the
     independent blockSize-1 run: exercises the block loop and bookkeeping.
   - Likelihoods without cavity moments, and block/column mismatches, are refused.
   Genuinely coupled likelihoods (robust-max) are verified in Phase 4. */
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "CEpInference.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlexceptions.h"

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

static double maxAbs(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

static const unsigned int N = 7;

static double rbf(double x, double z, double var, double ell)
{
  const double d = x - z;
  return var * exp(-0.5 * d * d / (ell * ell));
}

static CMatrix gram(const CMatrix& X, double var, double ell)
{
  CMatrix K(X.getRows(), X.getRows());
  for(unsigned int i = 0; i < X.getRows(); i++)
    for(unsigned int j = 0; j < X.getRows(); j++)
      K.setVal(rbf(X.getVal(i, 0), X.getVal(j, 0), var, ell), i, j);
  K.setSymmetric(true);
  return K;
}

static CMatrix crossGram(const CMatrix& X, const CMatrix& Xs, double var, double ell)
{
  CMatrix K(X.getRows(), Xs.getRows());
  for(unsigned int i = 0; i < X.getRows(); i++)
    for(unsigned int j = 0; j < Xs.getRows(); j++)
      K.setVal(rbf(X.getVal(i, 0), Xs.getVal(j, 0), var, ell), i, j);
  return K;
}

// Exact GP regression with Gaussian noise s2 and per-column bias b.
struct Exact
{
  double logMarg;
  CMatrix mu, var;  // N x 1 posterior of the latent
};

static Exact exactGp(const CMatrix& K, const CMatrix& Y, unsigned int col,
                     double s2, double bias)
{
  const unsigned int n = K.getRows();
  CMatrix A(K);
  A.setSymmetric(false);
  for(unsigned int i = 0; i < n; i++)
    A.setVal(A.getVal(i, i) + s2, i, i);
  A.setSymmetric(true);
  CMatrix U(A);
  U.chol();
  CMatrix Ai(n, n);
  Ai.setSymmetric(true);
  Ai.pdinv(U);
  CMatrix r(n, 1);
  for(unsigned int i = 0; i < n; i++)
    r.setVal(Y.getVal(i, col) - bias, i, 0);
  CMatrix a(n, 1);
  a.gemvColCol(0, Ai, r, 0, 1.0, 0.0, "n");
  Exact e;
  e.logMarg = -0.5 * r.dotColCol(0, a, 0) - 0.5 * logDet(U) - 0.5 * n * log(2.0 * M_PI);
  e.mu.resize(n, 1);
  e.mu.gemvColCol(0, K, a, 0, 1.0, 0.0, "n");
  e.var.resize(n, 1);
  CMatrix KA(n, n);
  KA.gemm(K, Ai, 1.0, 0.0, "n", "n");
  for(unsigned int i = 0; i < n; i++)
  {
    double s = 0.0;
    for(unsigned int j = 0; j < n; j++)
      s += KA.getVal(i, j) * K.getVal(j, i);
    e.var.setVal(K.getVal(i, i) - s, i, 0);
  }
  return e;
}

class CTwoBlockGaussian : public CGaussianNoise
{
 public:
  CTwoBlockGaussian(CMatrix* y) : CGaussianNoise(y) {}
  unsigned int blockSize() const { return 2; }
};

int main()
{
  try
  {
    CMatrix X(N, 1);
    CMatrix Y(N, 2);
    for(unsigned int i = 0; i < N; i++)
    {
      X.setVal(-1.5 + 3.0 * i / (N - 1.0), i, 0);
      Y.setVal(sin(2.0 * X.getVal(i, 0)) + 0.1 * i, i, 0);
      Y.setVal(cos(1.3 * X.getVal(i, 0)) - 0.2, i, 1);
    }
    const double s2 = 0.3;
    const double b0 = 0.0, b1 = 0.15;
    CMatrix K1 = gram(X, 1.2, 0.9);
    CMatrix K2 = gram(X, 0.8, 1.4);

    CEpInference ep;
    ep.setTol(1e-12);
    ep.setJitter(0.0);
    ep.setMaxIters(50);

    // --- Gaussian, K = 1: exact ---
    {
      CMatrix y1(N, 1);
      for(unsigned int i = 0; i < N; i++)
        y1.setVal(Y.getVal(i, 0), i, 0);
      CGaussianNoise g(&y1);
      g.setSigma2(s2);
      g.setBiasVal(b0, 0);
      CEpSites sites;
      CMatrix mu, vd;
      const double L = ep.runBlock(vector<CMatrix>(1, K1), y1, g, sites, mu, vd);
      const Exact e = exactGp(K1, y1, 0, s2, b0);
      expect(fabs(L - e.logMarg) < 1e-9, "Gaussian K=1: EP log marginal == exact");
      CMatrix mc(N, 1), vc(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        mc.setVal(mu.getVal(i, 0), i, 0);
        vc.setVal(vd.getVal(i, 0), i, 0);
      }
      expect(maxAbs(mc, e.mu) < 1e-9 && maxAbs(vc, e.var) < 1e-9,
             "Gaussian K=1: posterior mean and variance == exact");

      // dL/dK against finite differences of the exact marginal.
      CMatrix dLdK;
      ep.fillDLDKBlock(0, dLdK);
      double worst = 0.0;
      for(unsigned int i = 0; i < N; i += 2)
        for(unsigned int j = i; j < N; j += 3)
        {
          const double eps = 1e-6;
          CMatrix Kp(K1), Km(K1);
          Kp.setSymmetric(false);
          Km.setSymmetric(false);
          Kp.setVal(Kp.getVal(i, j) + eps, i, j);
          Km.setVal(Km.getVal(i, j) - eps, i, j);
          if(i != j)
          {
            Kp.setVal(Kp.getVal(j, i) + eps, j, i);
            Km.setVal(Km.getVal(j, i) - eps, j, i);
          }
          Kp.setSymmetric(true);
          Km.setSymmetric(true);
          const double fd = (exactGp(Kp, y1, 0, s2, b0).logMarg
                             - exactGp(Km, y1, 0, s2, b0).logMarg) / (2 * eps);
          const double an = (i == j) ? dLdK.getVal(i, j) : 2.0 * dLdK.getVal(i, j);
          worst = max(worst, fabs(fd - an));
        }
      expect(worst < 1e-6, "Gaussian K=1: dL/dK == finite differences of exact marginal");

      // Prediction.
      CMatrix Xs(3, 1);
      Xs.setVal(-0.7, 0, 0);
      Xs.setVal(0.2, 1, 0);
      Xs.setVal(1.1, 2, 0);
      CMatrix Kx = crossGram(X, Xs, 1.2, 0.9), kss(3, 1), pm, pv;
      for(unsigned int t = 0; t < 3; t++)
        kss.setVal(1.2, t, 0);
      ep.predictiveMeanVarBlock(0, Kx, kss, pm, pv);
      CMatrix A(K1);
      A.setSymmetric(false);
      for(unsigned int i = 0; i < N; i++)
        A.setVal(A.getVal(i, i) + s2, i, i);
      A.setSymmetric(true);
      CMatrix U(A);
      U.chol();
      CMatrix Ai(N, N);
      Ai.setSymmetric(true);
      Ai.pdinv(U);
      CMatrix a(N, 1);
      a.gemvColCol(0, Ai, y1, 0, 1.0, 0.0, "n");
      double perr = 0.0;
      for(unsigned int t = 0; t < 3; t++)
      {
        double m = 0.0, q = 0.0;
        for(unsigned int i = 0; i < N; i++)
        {
          m += Kx.getVal(i, t) * a.getVal(i, 0);
          for(unsigned int j = 0; j < N; j++)
            q += Kx.getVal(i, t) * Ai.getVal(i, j) * Kx.getVal(j, t);
        }
        perr = max(perr, fabs(m - pm.getVal(t, 0)));
        perr = max(perr, fabs(1.2 - q - pv.getVal(t, 0)));
      }
      expect(perr < 1e-8, "Gaussian K=1: predictive mean and variance == exact");
    }

    // --- Gaussian, K = 2, shared Gram and per-latent Gram ---
    for(int mode = 0; mode < 2; mode++)
    {
      const string tag = mode == 0 ? " (shared Gram)" : " (per-latent Gram)";
      CGaussianNoise g(&Y);
      g.setSigma2(s2);
      g.setBiasVal(b0, 0);
      g.setBiasVal(b1, 1);
      vector<CMatrix> Ks;
      Ks.push_back(K1);
      if(mode == 1)
        Ks.push_back(K2);
      CEpSites sites;
      CMatrix mu, vd;
      const double L = ep.runBlock(Ks, Y, g, sites, mu, vd);
      const Exact e0 = exactGp(K1, Y, 0, s2, b0);
      const Exact e1 = exactGp(mode == 0 ? K1 : K2, Y, 1, s2, b1);
      expect(fabs(L - (e0.logMarg + e1.logMarg)) < 1e-8,
             "Gaussian K=2: log marginal == sum of exact" + tag);
      double d = 0.0;
      for(unsigned int i = 0; i < N; i++)
      {
        d = max(d, fabs(mu.getVal(i, 0) - e0.mu.getVal(i, 0)));
        d = max(d, fabs(mu.getVal(i, 1) - e1.mu.getVal(i, 0)));
        d = max(d, fabs(vd.getVal(i, 0) - e0.var.getVal(i, 0)));
        d = max(d, fabs(vd.getVal(i, 1) - e1.var.getVal(i, 0)));
      }
      expect(d < 1e-8, "Gaussian K=2: posterior == exact for both latents" + tag);
    }

    // --- Block plumbing: blockSize 2 stand-in equals independent blockSize 1 ---
    {
      CGaussianNoise g1(&Y);
      g1.setSigma2(s2);
      g1.setBiasVal(b0, 0);
      g1.setBiasVal(b1, 1);
      CTwoBlockGaussian g2(&Y);
      g2.setSigma2(s2);
      g2.setBiasVal(b0, 0);
      g2.setBiasVal(b1, 1);
      CEpSites s1, s2s;
      CMatrix m1, v1, m2, v2;
      const double L1 = ep.runBlock(vector<CMatrix>(1, K1), Y, g1, s1, m1, v1);
      const double L2 = ep.runBlock(vector<CMatrix>(1, K1), Y, g2, s2s, m2, v2);
      expect(fabs(L1 - L2) < 1e-12 && maxAbs(m1, m2) < 1e-12 && maxAbs(v1, v2) < 1e-12 &&
                 maxAbs(s1.tau, s2s.tau) < 1e-12 && maxAbs(s1.v, s2s.v) < 1e-12,
             "blockSize 2 stand-in == independent blockSize 1");
    }

    // --- Probit K = 1 ---
    {
      CMatrix yp(N, 1), y01(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        yp.setVal(Y.getVal(i, 0) + 0.2 > 0.0 ? 1.0 : -1.0, i, 0);
        y01.setVal(yp.getVal(i, 0) > 0.0 ? 1.0 : 0.0, i, 0);
      }
      // GPy mode: Phi(y f) on f, so the block path on plain K must equal the
      // legacy probit-only run() on everything: log marginal, posterior, dL/dK.
      CProbitNoise pn(&yp);
      pn.setGpyMode(true);
      expect(pn.isGpyMode() && pn.getSigma2() == 1.0 && pn.getBiasVal(0) == 0.0,
             "GPy mode: sigma2 = 1, zero bias");
      CEpInference epA, epB, epC;
      epA.setJitter(1e-6); epB.setJitter(1e-6); epC.setJitter(1e-6);
      CEpSites sA, sB, sC;
      CMatrix mA, vA, mB, vB, mC, vC;
      const double LA = epA.run(K1, yp, sA, mA, vA);
      const double LB = epB.runBlock(vector<CMatrix>(1, K1), yp, pn, sB, mB, vB);
      expect(fabs(LA - LB) < 1e-6 * (1.0 + fabs(LA)), "GPy probit: block log marginal == run()");
      expect(maxAbs(mA, mB) < 1e-6 && maxAbs(vA, vB) < 1e-6, "GPy probit: block posterior == run()");
      CMatrix dA, dB;
      epA.fillDLDK(dA);
      epB.fillDLDKBlock(0, dB);
      expect(maxAbs(dA, dB) < 1e-6, "GPy probit: block dL/dK == run()");
      // {0,1} targets (GPy / pods) mean the same thing as +-1.
      CProbitNoise pn01(&y01);
      pn01.setGpyMode(true);
      const double LC = epC.runBlock(vector<CMatrix>(1, K1), y01, pn01, sC, mC, vC);
      expect(fabs(LB - LC) < 1e-12 && maxAbs(mB, mC) < 1e-12, "GPy probit: {0,1} targets == +-1");

      // Legacy CIP-0027 convention (step on f + eps, White in K): the same
      // model, so on K + I the log marginal and dL/dK still agree.
      CProbitNoise pl(&yp);
      pl.setParam(0.0, 0);
      expect(!pl.isGpyMode(), "default probit is the legacy convention");
      CMatrix Kw(K1);
      for(unsigned int i = 0; i < N; i++)
        Kw.setVal(K1.getVal(i, i) + 1.0, i, i);
      CEpInference epD;
      epD.setJitter(1e-6);
      CEpSites sD;
      CMatrix mD, vD;
      const double LD = epD.runBlock(vector<CMatrix>(1, Kw), yp, pl, sD, mD, vD);
      expect(fabs(LA - LD) < 1e-5 * (1.0 + fabs(LA)), "legacy probit on K+I: log marginal == GPy on K");
      CMatrix dD;
      epD.fillDLDKBlock(0, dD);
      expect(maxAbs(dA, dD) < 1e-5, "legacy probit on K+I: dL/dK == GPy on K");
    }

    // --- Refusals ---
    {
      CStudentTNoise st(&Y);
      CEpSites s;
      CMatrix m, v;
      bool threw = false;
      try { ep.runBlock(vector<CMatrix>(1, K1), Y, st, s, m, v); }
      catch(ndlexceptions::NotImplementedError&) { threw = true; }
      expect(threw, "likelihood without cavity moments is refused");

      CMatrix Y3(N, 3);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < 3; j++)
          Y3.setVal(0.1 * i + j, i, j);
      CTwoBlockGaussian two(&Y3);
      threw = false;
      try { ep.runBlock(vector<CMatrix>(1, K1), Y3, two, s, m, v); }
      catch(ndlexceptions::Error&) { threw = true; }
      expect(threw, "blockSize not dividing the output columns is refused");

      CGaussianNoise g(&Y);
      threw = false;
      try { ep.runBlock(vector<CMatrix>(3, K1), Y, g, s, m, v); }
      catch(ndlexceptions::Error&) { threw = true; }
      expect(threw, "wrong number of Gram matrices is refused");
    }
  }
  catch(ndlexceptions::Error& e)
  {
    cout << "FAIL: exception: " << e.getMessage() << endl;
    fails++;
  }
  cout << "Number of failures: " << fails << endl;
  return fails > 0 ? 1 : 0;
}
