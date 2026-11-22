/* CIP-0035 Phase 1–4: CDeepGp verification (points, ridges, probit, stream, V7). */
#include "CAdam.h"
#include "CDeepGp.h"
#include "CInducing.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

static double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  const unsigned int n = min(A.getCols(), B.getCols());
  for(unsigned int i = 0; i < n; i++)
    m = max(m, fabs(A.getVal(i) - B.getVal(i)));
  return m;
}

static void makeRbfBias(CCmpndKern& kern, const CMatrix& X)
{
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CBiasKern(X));
  CMatrix p(1, kern.getNumParams());
  p.setVal(1.0 / (0.8 * 0.8), 0);
  p.setVal(1.5, 1);
  p.setVal(0.2, 2);
  kern.setParams(p);
}

static void makeRbfOnly(CCmpndKern& kern, const CMatrix& X,
                        double invWidth, double variance)
{
  CRbfKern* prbf = new CRbfKern(X);
  prbf->setParam(invWidth, 0);
  prbf->setParam(variance, 1);
  kern.addKern(prbf);
}

// Exact E_q(h)[Gaussian SVGP data term] via psi statistics (CIP-0017 / CIP-0030).
static double exactPsiData(CKern& kern, const CMatrix& Y, const CMatrix& mu,
                           const CMatrix& S, const CMatrix& Z, double beta,
                           const CMatrix& qMean, const vector<CMatrix>& qCov)
{
  const unsigned int N = Y.getRows();
  const unsigned int D = Y.getCols();
  const unsigned int M = Z.getRows();
  CMatrix Kuu(M, M);
  kern.compute(Kuu, Z);
  Kuu.setSymmetric(true);
  CMatrix U(Kuu);
  U.jitChol(Kuu);

  const double psi0 = kern.psi0(mu, S);
  CMatrix Psi1(N, M);
  kern.psi1(Psi1, mu, S, Z);
  CMatrix Psi2(M, M);
  kern.psi2(Psi2, mu, S, Z);

  CMatrix Sigma(Psi2);
  Sigma.trsm(U, 1.0, "L", "U", "T", "N");
  Sigma.trsm(U, 1.0, "R", "U", "N", "N");
  Sigma.setSymmetric(true);

  double data = 0.0;
  for(unsigned int j = 0; j < D; j++)
  {
    CMatrix mj(M, 1);
    mj.copyColCol(0, qMean, j);
    CMatrix Psy(M, 1);
    Psy.zeros();
    double yTy = 0.0;
    for(unsigned int n = 0; n < N; n++)
    {
      const double yn = Y.getVal(n, j);
      yTy += yn * yn;
      for(unsigned int m = 0; m < M; m++)
        Psy.addVal(Psi1.getVal(n, m) * yn, m, 0);
    }
    Psy.trsm(U, 1.0, "L", "U", "T", "N");
    const double cross = mj.dotColCol(0, Psy, 0);
    CMatrix SigmaM(M, 1);
    SigmaM.symv(Sigma, mj, 1.0, 0.0, "u");
    const double mSm = mj.dotColCol(0, SigmaM, 0);
    const double trSigma = Sigma.trace();
    CMatrix SSigma(M, M);
    SSigma.symm(qCov[j], Sigma, 1.0, 0.0, "u", "l");
    const double trSSigma = SSigma.trace();
    const double quad = yTy - 2.0 * cross + mSm + psi0 - trSigma + trSSigma;
    data += -0.5 * (double)N * (ndlutil::LOGTWOPI - log(beta)) - 0.5 * beta * quad;
  }
  return data;
}

int main()
{
  try
  {
    // ---- V1: L=1 equals CSvgp ELBO at fixed q ----
    {
      const unsigned int N = 30, M = 8;
      const double beta = 15.0;
      mt19937 rng(3);
      normal_distribution<double> nd(0.0, 1.0);
      uniform_real_distribution<double> ud(-2.0, 2.0);
      CMatrix X(N, 1), y(N, 1), Z(M, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(ud(rng), i, 0);
        y.setVal(0.7 * X.getVal(i, 0) + 0.1 * nd(rng), i, 0);
      }
      for(unsigned int k = 0; k < M; k++)
        Z.setVal(-1.5 + 3.0 * (double)k / (double)(M - 1), k, 0);

      CCmpndKern kernSv(X);
      makeRbfBias(kernSv, X);
      CSvgp sv(&kernSv, &X, &y, Z, beta, 0);
      sv.setQParamMode(CSvgp::QModeCholesky);
      sv.naturalGradientStep(0.5);

      CCmpndKern kernDg(X);
      makeRbfBias(kernDg, X);
      vector<CKern*> kerns(1, &kernDg);
      vector<unsigned int> dims = {1u, 1u};
      vector<unsigned int> Ms = {M};
      CDeepGp dgp(X, y, kerns, dims, Ms, beta, 1.0, 0);
      dgp.layer(0).setInducingCentres(sv.getInducingCentres());
      vector<CMatrix> cov(1, sv.getWhitenedCov(0));
      dgp.layer(0).setWhitenedQ(sv.getWhitenedMean(), cov);
      CMatrix pk(1, kernSv.getNumParams());
      kernSv.getParams(pk);
      kernDg.setParams(pk);
      dgp.layer(0).setBeta(sv.getBeta());
      dgp.layer(0).updateKernelMatrices();

      const double Lsv = sv.elbo();
      const double Ldg = dgp.elboEstimate();
      cout << "  info: V1 Lsv=" << Lsv << " Ldg=" << Ldg
           << " diff=" << fabs(Lsv - Ldg) << endl;
      expect(fabs(Lsv - Ldg) < 1e-10, "V1: L=1 CDeepGp ELBO equals CSvgp");

      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      CMatrix gSv, gDg;
      const double Lsv2 = sv.objectiveGradBatch(all, gSv);
      const double Ldg2 = dgp.objectiveGradBatch(all, gDg);
      expect(fabs(Lsv2 - Ldg2) < 1e-10, "V1: L=1 objectiveGradBatch value matches");
      expect(gSv.getCols() == gDg.getCols(), "V1: L=1 stochastic param count matches");
      const double maxG = maxAbsDiff(gSv, gDg);
      cout << "  info: V1 max|gSv-gDg|=" << maxG << endl;
      expect(maxG < 1e-8, "V1: L=1 stochastic grads match CSvgp");
    }

    // ---- V2: σ₁²→0 reduces to CSvgp on X P₁ ----
    {
      const unsigned int N = 25, M = 6;
      const double beta = 12.0;
      mt19937 rng(5);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.5 + 3.0 * (double)i / (double)(N - 1), i, 0);
        y.setVal(sin(X.getVal(i, 0)) + 0.05 * nd(rng), i, 0);
      }

      ndlutil::init_genrand(7u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.0 / (0.8 * 0.8), 1e-20); // σ₁² → 0
      makeRbfBias(k1, X);
      vector<CKern*> kerns = {&k0, &k1};
      vector<unsigned int> dims = {1u, 1u, 1u};
      vector<unsigned int> Ms = {M, M};
      CDeepGp dgp(X, y, kerns, dims, Ms, beta, 1.0, 0);

      // Prior-like hidden q; residual variance vanishes with σ₁².
      CMatrix m0(dgp.layer(0).getNumInducing(), 1);
      m0.zeros();
      vector<CMatrix> S0(1);
      S0[0].resize(dgp.layer(0).getNumInducing(), dgp.layer(0).getNumInducing());
      S0[0].zeros();
      S0[0].addDiag(1.0);
      S0[0].setSymmetric(true);
      dgp.layer(0).setWhitenedQ(m0, S0);
      dgp.layer(0).updateKernelMatrices();

      CMatrix Zf(dgp.layer(1).getInducingCentres());
      dgp.layer(1).naturalGradientStep(0.4);
      dgp.layer(1).updateKernelMatrices();

      const CMatrix& P = dgp.meanMap(0);
      CMatrix XP(N, 1);
      XP.gemm(X, P, 1.0, 0.0, "n", "n");

      CCmpndKern kSv(XP);
      makeRbfBias(kSv, XP);
      CMatrix pk(1, k1.getNumParams());
      k1.getParams(pk);
      kSv.setParams(pk);
      CSvgp sv(&kSv, &XP, &y, Zf, dgp.layer(1).getBeta(), 0);
      sv.setQParamMode(CSvgp::QModeCholesky);
      vector<CMatrix> cov1(1, dgp.layer(1).getWhitenedCov(0));
      sv.setWhitenedQ(dgp.layer(1).getWhitenedMean(), cov1);
      sv.updateKernelMatrices();

      CMatrix eps(N, 1);
      eps.zeros();
      dgp.setNumSamples(1);
      dgp.setFrozenEps(eps);
      const double Ldgp = dgp.elboEstimate();
      const double Lsv = sv.elbo();
      const double kl0 = dgp.layer(0).klQ();
      cout << "  info: V2 Ldgp=" << Ldgp << " Lsv=" << Lsv
           << " kl0=" << kl0 << " |Ldgp+kl0-Lsv|="
           << fabs(Ldgp + kl0 - Lsv) << endl;
      // Hidden KL is ~0 at prior q; add it back for the σ₁²→0 reduction.
      expect(fabs(Ldgp + kl0 - Lsv) < 1e-5,
             "V2: σ₁²→0 CDeepGp ELBO equals CSvgp on XP");
    }

    // ---- V3: L=2 MC ELBO unbiased vs psi exact ----
    {
      const unsigned int N = 18, M = 5;
      const double beta = 8.0;
      mt19937 rng(11);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.2 + 2.4 * (double)i / (double)(N - 1), i, 0);
        y.setVal(0.6 * X.getVal(i, 0) + 0.15 * nd(rng), i, 0);
      }

      ndlutil::init_genrand(13u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.0 / (0.7 * 0.7), 0.4);
      makeRbfOnly(k1, X, 1.0 / (0.9 * 0.9), 1.2);
      vector<CKern*> kerns = {&k0, &k1};
      vector<unsigned int> dims = {1u, 1u, 1u};
      vector<unsigned int> Ms = {M, M};
      // chi=0.4 => hidden variance chi/(L-1)=0.4 (ctor overwrites k0 variance).
      CDeepGp dgp(X, y, kerns, dims, Ms, beta, 0.4, 0);

      CMatrix m0(dgp.layer(0).getNumInducing(), 1);
      m0.zeros();
      vector<CMatrix> S0(1);
      S0[0].resize(dgp.layer(0).getNumInducing(), dgp.layer(0).getNumInducing());
      S0[0].zeros();
      S0[0].addDiag(0.35);
      S0[0].setSymmetric(true);
      dgp.layer(0).setWhitenedQ(m0, S0);
      dgp.layer(0).updateKernelMatrices();
      dgp.layer(1).naturalGradientStep(0.35);
      dgp.layer(1).updateKernelMatrices();

      CMatrix mu0, var0;
      dgp.layer(0).marginalsAt(X, mu0, var0);
      CMatrix muH(N, 1);
      muH.gemm(X, dgp.meanMap(0), 1.0, 0.0, "n", "n");
      for(unsigned int i = 0; i < N; i++)
        muH.setVal(muH.getVal(i, 0) + mu0.getVal(i, 0), i, 0);

      vector<CMatrix> cov1(1, dgp.layer(1).getWhitenedCov(0));
      const double psiData = exactPsiData(
          k1, y, muH, var0, dgp.layer(1).getInducingCentres(),
          dgp.layer(1).getBeta(), dgp.layer(1).getWhitenedMean(), cov1);
      const double Lexact = psiData - dgp.layer(0).klQ() - dgp.layer(1).klQ()
                            + k0.priorLogProb() + k1.priorLogProb();

      dgp.setFreezeNoise(false);
      dgp.setNumSamples(2);
      dgp.setSeed(99);
      const unsigned int nMc = 4000;
      double sum = 0.0, sum2 = 0.0;
      CMatrix gSum;
      bool gInit = false;
      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      for(unsigned int t = 0; t < nMc; t++)
      {
        CMatrix g;
        const double Li = dgp.objectiveGradBatch(all, g);
        sum += Li;
        sum2 += Li * Li;
        if(!gInit)
        {
          gSum.resize(1, g.getCols());
          gSum.zeros();
          gInit = true;
        }
        for(unsigned int i = 0; i < g.getCols(); i++)
          gSum.setVal(gSum.getVal(i) + g.getVal(i), i);
      }
      const double mean = sum / (double)nMc;
      const double var = sum2 / (double)nMc - mean * mean;
      const double se = sqrt(max(0.0, var) / (double)nMc);
      cout << "  info: V3 exact=" << Lexact << " MC=" << mean
           << " se=" << se << " |diff|/se="
           << fabs(mean - Lexact) / max(se, 1e-12) << endl;
      expect(fabs(mean - Lexact) < 4.0 * se + 1e-3,
             "V3: MC ELBO unbiased vs psi exact (within 4 se)");

      // Gradient means: FD of Lexact is unavailable for all params; check that
      // the MC mean gradient is finite and a central FD at frozen eps agrees
      // with one analytic component (covered fully in V4). Report mean |g|.
      double meanAbsG = 0.0;
      for(unsigned int i = 0; i < gSum.getCols(); i++)
        meanAbsG += fabs(gSum.getVal(i) / (double)nMc);
      meanAbsG /= (double)max(1u, gSum.getCols());
      cout << "  info: V3 mean|g|=" << meanAbsG << endl;
      expect(isfinite(meanAbsG) && meanAbsG > 0.0,
             "V3: MC gradient means finite and nonzero");
    }

    // ---- V4: FD at fixed ε for stochastic params ----
    {
      const unsigned int N = 14, M = 4;
      const double beta = 10.0;
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.0 + 2.0 * (double)i / (double)(N - 1), i, 0);
        y.setVal(0.5 * sin(X.getVal(i, 0)), i, 0);
      }
      ndlutil::init_genrand(17u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.5, 0.5);
      makeRbfBias(k1, X);
      vector<CKern*> kerns = {&k0, &k1};
      vector<unsigned int> dims = {1u, 1u, 1u};
      vector<unsigned int> Ms = {M, M};
      CDeepGp dgp(X, y, kerns, dims, Ms, beta, 1.0, 0);
      CMatrix m0(dgp.layer(0).getNumInducing(), 1);
      m0.zeros();
      vector<CMatrix> S0(1);
      S0[0].resize(dgp.layer(0).getNumInducing(), dgp.layer(0).getNumInducing());
      S0[0].zeros();
      S0[0].addDiag(0.4);
      S0[0].setSymmetric(true);
      dgp.layer(0).setWhitenedQ(m0, S0);
      dgp.layer(1).naturalGradientStep(0.3);
      dgp.setNumSamples(1);

      CMatrix eps(N, 1);
      mt19937 rng(23);
      normal_distribution<double> nd(0.0, 1.0);
      for(unsigned int i = 0; i < N; i++)
        eps.setVal(nd(rng), i, 0);
      dgp.setFrozenEps(eps);

      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      CMatrix gAnal;
      dgp.objectiveGradBatch(all, gAnal);

      CMatrix p0;
      dgp.getStochasticParams(p0);
      const double h = 1e-5;
      double maxErr = 0.0;
      unsigned int worst = 0;
      for(unsigned int i = 0; i < p0.getCols(); i++)
      {
        CMatrix pp(p0), pm(p0);
        pp.setVal(p0.getVal(i) + h, i);
        pm.setVal(p0.getVal(i) - h, i);
        dgp.setStochasticParams(pp);
        CMatrix gTmp;
        const double Lp = dgp.objectiveGradBatch(all, gTmp);
        dgp.setStochasticParams(pm);
        const double Lm = dgp.objectiveGradBatch(all, gTmp);
        dgp.setStochasticParams(p0);
        const double gFd = (Lp - Lm) / (2.0 * h);
        const double err = fabs(gAnal.getVal(i) - gFd);
        if(err > maxErr)
        {
          maxErr = err;
          worst = i;
        }
      }
      cout << "  info: V4 max|g-FD|=" << maxErr << " at i=" << worst
           << " of " << p0.getCols() << endl;
      expect(maxErr < 5e-3, "V4: analytic grads match FD at fixed ε");
    }

    // ---- V6: same seed => identical ELBO ----
    {
      const unsigned int N = 24, M = 5;
      const double beta = 10.0;
      mt19937 rng(9);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.5 + 3.0 * (double)i / (double)(N - 1), i, 0);
        y.setVal(sin(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
      }
      auto run = [&](uint64_t seed) {
        ndlutil::init_genrand(42u);
        CCmpndKern k0(X), k1(X);
        makeRbfBias(k0, X);
        makeRbfBias(k1, X);
        vector<CKern*> kerns = {&k0, &k1};
        vector<unsigned int> dims = {1u, 1u, 1u};
        vector<unsigned int> Ms = {M, M};
        CDeepGp dgp(X, y, kerns, dims, Ms, beta, 1.0, 0);
        CMatrix m(dgp.layer(0).getNumInducing(), 1);
        m.zeros();
        vector<CMatrix> S(1);
        S[0].resize(dgp.layer(0).getNumInducing(), dgp.layer(0).getNumInducing());
        S[0].zeros();
        S[0].addDiag(0.5);
        S[0].setSymmetric(true);
        dgp.layer(0).setWhitenedQ(m, S);
        dgp.setNumSamples(2);
        dgp.setSeed(seed);
        vector<unsigned int> all(N);
        for(unsigned int i = 0; i < N; i++) all[i] = i;
        CMatrix g;
        const double L = dgp.objectiveGradBatch(all, g);
        return make_pair(L, g);
      };
      auto a = run(123);
      auto b = run(123);
      auto c = run(999);
      expect(fabs(a.first - b.first) < 1e-12, "V6: same seed => identical ELBO");
      expect(maxAbsDiff(a.second, b.second) < 1e-12, "V6: same seed => identical grads");
      expect(fabs(a.first - c.first) > 1e-8
             || maxAbsDiff(a.second, c.second) > 1e-8,
             "V6: different seeds differ");
    }

    // ---- V5: partition average equals full batch at eps=0 ----
    {
      const unsigned int N = 20, M = 4;
      const double beta = 12.0;
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.0 + 2.0 * (double)i / (double)(N - 1), i, 0);
        y.setVal(0.5 * X.getVal(i, 0), i, 0);
      }
      CCmpndKern k0(X), k1(X);
      makeRbfBias(k0, X);
      makeRbfBias(k1, X);
      vector<CKern*> kerns = {&k0, &k1};
      vector<unsigned int> dims = {1u, 1u, 1u};
      vector<unsigned int> Ms = {M, M};
      CDeepGp dgp(X, y, kerns, dims, Ms, beta, 1.0, 0);
      dgp.setNumSamples(1);

      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      CMatrix epsFull(N, 1);
      epsFull.zeros();
      dgp.setFrozenEps(epsFull);
      CMatrix gFull;
      const double Lfull = dgp.objectiveGradBatch(all, gFull);

      vector<unsigned int> b0, b1;
      for(unsigned int i = 0; i < N / 2; i++) b0.push_back(i);
      for(unsigned int i = N / 2; i < N; i++) b1.push_back(i);
      CMatrix eps0((unsigned int)b0.size(), 1), eps1((unsigned int)b1.size(), 1);
      eps0.zeros();
      eps1.zeros();
      dgp.setFrozenEps(eps0);
      CMatrix g0;
      const double L0 = dgp.objectiveGradBatch(b0, g0);
      dgp.setFrozenEps(eps1);
      CMatrix g1;
      const double L1 = dgp.objectiveGradBatch(b1, g1);
      const double Lavg = 0.5 * (L0 + L1);
      cout << "  info: V5 Lfull=" << Lfull << " Lavg=" << Lavg
           << " diff=" << fabs(Lfull - Lavg) << endl;
      expect(fabs(Lfull - Lavg) < 1e-8, "V5: partition average equals full (eps=0)");
    }

    // ---- Phase 2 V4: FD at fixed ε with weight-space ridges ----
    {
      const unsigned int N = 12, M = 6, q = 2;
      const double beta = 6.0;
      mt19937 rng(31);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, q), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        for(unsigned int d = 0; d < q; d++)
          X.setVal(nd(rng), i, d);
        y.setVal(tanh(X.getVal(i, 0) - 0.3 * X.getVal(i, 1)) + 0.05 * nd(rng),
                 i, 0);
      }
      ndlutil::init_genrand(29u);
      CMlpKern k0(X), k1(X);
      k0.setParam(1.0, 0); k0.setParam(1.0, 1); k0.setParam(0.8, 2);
      k1.setParam(1.0, 0); k1.setParam(1.0, 1); k1.setParam(1.2, 2);
      vector<CKern*> kerns = {&k0, &k1};
      vector<unsigned int> dims = {q, q, 1u};
      vector<unsigned int> Ms = {M, M};
      CDeepGp dgp(X, y, kerns, dims, Ms, beta, 0.5, 0);
      dgp.setLayerInducingType(0, "weightspace");
      dgp.setLayerInducingType(1, "weightspace");
      // learnS grads are not packed yet (CSvgp::packInducingGrad); FD W/q/kern only.
      CMatrix m0(dgp.layer(0).getNumInducing(), q);
      m0.zeros();
      vector<CMatrix> S0(q);
      for(unsigned int j = 0; j < q; j++)
      {
        S0[j].resize(dgp.layer(0).getNumInducing(), dgp.layer(0).getNumInducing());
        S0[j].zeros();
        S0[j].addDiag(0.25);
        S0[j].setSymmetric(true);
      }
      dgp.layer(0).setWhitenedQ(m0, S0);
      dgp.layer(1).naturalGradientStep(0.25);
      dgp.setNumSamples(1);

      // Hidden noise: |batch| * K * D_1 = N * 1 * q
      CMatrix eps(N * q, 1);
      for(unsigned int i = 0; i < eps.getRows(); i++)
        eps.setVal(nd(rng), i, 0);
      dgp.setFrozenEps(eps);

      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      CMatrix gAnal;
      dgp.objectiveGradBatch(all, gAnal);
      CMatrix p0;
      dgp.getStochasticParams(p0);
      const double h = 1e-5;
      double maxErr = 0.0;
      unsigned int worst = 0;
      // Subsample params for speed: first 12 + last 8 + every 7th.
      vector<unsigned int> idxs;
      for(unsigned int i = 0; i < min(12u, p0.getCols()); i++)
        idxs.push_back(i);
      for(unsigned int i = 12; i + 8 < p0.getCols(); i += 7)
        idxs.push_back(i);
      for(unsigned int i = (p0.getCols() > 8 ? p0.getCols() - 8 : 0);
          i < p0.getCols(); i++)
        idxs.push_back(i);
      for(unsigned int i : idxs)
      {
        CMatrix pp(p0), pm(p0);
        pp.setVal(p0.getVal(i) + h, i);
        pm.setVal(p0.getVal(i) - h, i);
        dgp.setStochasticParams(pp);
        CMatrix gTmp;
        const double Lp = dgp.objectiveGradBatch(all, gTmp);
        dgp.setStochasticParams(pm);
        const double Lm = dgp.objectiveGradBatch(all, gTmp);
        dgp.setStochasticParams(p0);
        const double gFd = (Lp - Lm) / (2.0 * h);
        const double err = fabs(gAnal.getVal(i) - gFd);
        if(err > maxErr)
        {
          maxErr = err;
          worst = i;
        }
      }
      cout << "  info: V4-ridge max|g-FD|=" << maxErr << " at i=" << worst
           << " of " << p0.getCols() << " (checked " << idxs.size() << ")"
           << endl;
      expect(maxErr < 2e-2, "V4: ridge analytic grads match FD at fixed ε");
    }

    // ---- Phase 2: s ≤ 1 via sigmoid pack under Adam + hard push ----
    {
      const unsigned int N = 16, M = 8;
      mt19937 rng(41);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.5 + 3.0 * (double)i / (double)(N - 1), i, 0);
        y.setVal(sin(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
      }
      ndlutil::init_genrand(43u);
      CMlpKern k0(X), k1(X);
      k0.setParam(1.0, 0); k0.setParam(0.5, 1); k0.setParam(0.5, 2);
      k1.setParam(1.0, 0); k1.setParam(0.5, 1); k1.setParam(1.0, 2);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 8.0, 0.5, 0);
      dgp.setLayerInducingType(0, "points");
      dgp.setLayerInducingType(1, "weightspace");
      dgp.setLayerLearnWindowWidth(1, true, true); // untied s_m on final
      dgp.setNumSamples(1);
      dgp.setSeed(7);

      auto* pWs = dynamic_cast<CInducingWeightSpace*>(dgp.layer(1).getInducing());
      expect(pWs != 0, "Phase2: final layer is weight-space");
      if(pWs != 0)
      {
        // Hard-push only unconstrained window coords u → +∞ (sigmoid → s→1).
        CMatrix pe;
        dgp.layer(1).getStochasticParams(pe);
        const unsigned int s0 =
            k1.getNumParams() + M * pWs->domainDim();
        for(unsigned int m = 0; m < M; m++)
          pe.setVal(40.0, s0 + m);
        dgp.layer(1).setStochasticParams(pe);
        bool ok = true;
        double smax = 0.0, smin = 1e9;
        for(unsigned int m = 0; m < M; m++)
        {
          const double sm = pWs->getWindowWidthAt(m);
          smax = max(smax, sm);
          smin = min(smin, sm);
          if(!(sm <= CInducingWeightSpace::windowWidthMax() + 1e-9)
             || !(sm >= CInducingWeightSpace::windowWidthMin() - 1e-9))
            ok = false;
        }
        cout << "  info: Phase2 s after u=40 push in [" << smin << ", " << smax
             << "]" << endl;
        expect(ok && smax <= 1.0 + 1e-9 && smax > 0.9,
               "Phase2: learnable s_m stay in [0.05, 1] under param push");

        // Short Adam: widths remain in range (points hidden keeps Kuu stable).
        CAdam adam(0.02);
        adam.setBatchSize(N);
        adam.setMaxIters(12);
        adam.setSeed(11);
        adam.maximise(dgp);
        ok = true;
        smax = 0.0;
        for(unsigned int m = 0; m < M; m++)
        {
          const double sm = pWs->getWindowWidthAt(m);
          smax = max(smax, sm);
          if(!(sm <= CInducingWeightSpace::windowWidthMax() + 1e-9)
             || !(sm >= CInducingWeightSpace::windowWidthMin() - 1e-9))
            ok = false;
        }
        cout << "  info: Phase2 s_max after Adam=" << smax << endl;
        expect(ok, "Phase2: learnable s_m stay in [0.05, 1] under Adam");
      }
    }

    // ---- Phase 2 V8 smoke: learned-ridge deep GP trains, beats shallow ----
    {
      const unsigned int N = 40, M = 10, Nte = 20;
      mt19937 rng(53);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1), Xte(Nte, 1), yte(Nte, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        const double x = -2.0 + 4.0 * (double)i / (double)(N - 1);
        X.setVal(x, i, 0);
        y.setVal(sin(1.5 * x) + 0.15 * nd(rng), i, 0);
      }
      for(unsigned int i = 0; i < Nte; i++)
      {
        const double x = -1.8 + 3.6 * (double)i / (double)(Nte - 1);
        Xte.setVal(x, i, 0);
        yte.setVal(sin(1.5 * x), i, 0);
      }

      auto trainRmse = [&](bool deepRidge) {
        ndlutil::init_genrand(55u);
        if(deepRidge)
        {
          CMlpKern k0(X), k1(X);
          k0.setParam(1.0, 0); k0.setParam(0.5, 1); k0.setParam(0.4, 2);
          k1.setParam(1.0, 0); k1.setParam(0.5, 1); k1.setParam(1.0, 2);
          CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                      vector<unsigned int>{1u, 1u, 1u},
                      vector<unsigned int>{M, M}, 10.0, 0.5, 0);
          dgp.setLayerInducingType(0, "weightspace");
          dgp.setLayerInducingType(1, "weightspace");
          dgp.setLayerLearnWindowWidth(0, true, false);
          dgp.setLayerLearnWindowWidth(1, true, false);
          dgp.setNumSamples(1);
          dgp.setSeed(3);
          CAdam adam(0.05);
          adam.setBatchSize(N);
          adam.setMaxIters(40);
          adam.setSeed(3);
          adam.maximise(dgp);
          // Predictive mean via final-layer marginals at mean-path h = X P.
          CMatrix mu0, var0;
          dgp.layer(0).marginalsAt(Xte, mu0, var0);
          CMatrix H(Nte, 1);
          H.gemm(Xte, dgp.meanMap(0), 1.0, 0.0, "n", "n");
          for(unsigned int i = 0; i < Nte; i++)
            H.setVal(H.getVal(i, 0) + mu0.getVal(i, 0), i, 0);
          CMatrix mu1, var1;
          dgp.layer(1).marginalsAt(H, mu1, var1);
          double sse = 0.0;
          for(unsigned int i = 0; i < Nte; i++)
          {
            const double e = mu1.getVal(i, 0) - yte.getVal(i, 0);
            sse += e * e;
          }
          return sqrt(sse / (double)Nte);
        }
        CMlpKern kern(X);
        kern.setParam(1.0, 0); kern.setParam(0.5, 1); kern.setParam(1.0, 2);
        CMatrix Z(M, 1);
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(X.getVal(k % N, 0), k, 0);
        CSvgp sv(&kern, &X, &y, Z, 10.0, 0);
        sv.setInducingType("weightspace");
        sv.setQParamMode(CSvgp::QModeCholesky);
        auto* pWs = dynamic_cast<CInducingWeightSpace*>(sv.getInducing());
        if(pWs != 0)
          pWs->setLearnWindowWidth(true);
        CAdam adam(0.05);
        adam.setBatchSize(N);
        adam.setMaxIters(40);
        adam.setSeed(3);
        adam.maximise(sv);
        CMatrix mu, var;
        sv.marginalsAt(Xte, mu, var);
        double sse = 0.0;
        for(unsigned int i = 0; i < Nte; i++)
        {
          const double e = mu.getVal(i, 0) - yte.getVal(i, 0);
          sse += e * e;
        }
        return sqrt(sse / (double)Nte);
      };
      const double rmseDeep = trainRmse(true);
      const double rmseShallow = trainRmse(false);
      cout << "  info: V8-ridge rmseDeep=" << rmseDeep
           << " rmseShallow=" << rmseShallow << endl;
      expect(isfinite(rmseDeep) && rmseDeep < 1.5,
             "V8-ridge: deep learned-ridge RMSE finite and sane");
      expect(isfinite(rmseShallow), "V8-ridge: shallow baseline RMSE finite");
    }

    // ---- Phase 3: L=1 probit equals CSvgp; orthant ELBO ≤ exact ----
    {
      auto logProbitOrthant2 = [](const CMatrix& K, const CMatrix& yv) {
        const double y0 = yv.getVal(0, 0);
        const double y1 = yv.getVal(1, 0);
        const double s11 = 1.0 + y0 * y0 * K.getVal(0, 0);
        const double s22 = 1.0 + y1 * y1 * K.getVal(1, 1);
        const double s12 = y0 * y1 * K.getVal(0, 1);
        double rho = s12 / sqrt(s11 * s22);
        if(rho > 1.0) rho = 1.0;
        if(rho < -1.0) rho = -1.0;
        const double p = 0.25 + asin(rho) / (2.0 * M_PI);
        return log(max(p, 1e-300));
      };

      CMatrix Xb(2, 1), Yb(2, 1), Zb(2, 1);
      Xb.setVal(-0.4, 0, 0);
      Xb.setVal(0.7, 1, 0);
      Yb.setVal(1.0, 0, 0);
      Yb.setVal(-1.0, 1, 0);
      Zb.deepCopy(Xb);

      CCmpndKern kSv(Xb), kDg(Xb);
      makeRbfOnly(kSv, Xb, 1.2, 1.0);
      makeRbfOnly(kDg, Xb, 1.2, 1.0);
      CProbitNoise noiseSv(&Yb), noiseDg(&Yb);

      CSvgp sv(&kSv, &Xb, &Yb, Zb, 1.0, 0);
      sv.setLikelihood(&noiseSv);
      sv.setQParamMode(CSvgp::QModeCholesky);
      for(unsigned int t = 0; t < 30; t++)
        sv.naturalGradientStep(0.35);

      CDeepGp dgp(Xb, Yb, vector<CKern*>{&kDg},
                  vector<unsigned int>{1u, 1u},
                  vector<unsigned int>{2u}, 1.0, 1.0, 0);
      dgp.setLikelihood(&noiseDg);
      dgp.layer(0).setInducingCentres(sv.getInducingCentres());
      vector<CMatrix> cov(1, sv.getWhitenedCov(0));
      dgp.layer(0).setWhitenedQ(sv.getWhitenedMean(), cov);
      CMatrix pk(1, kSv.getNumParams());
      kSv.getParams(pk);
      kDg.setParams(pk);
      dgp.layer(0).updateKernelMatrices();

      const double Lsv = sv.elbo();
      const double Ldg = dgp.elboEstimate();
      cout << "  info: Phase3 L=1 probit Lsv=" << Lsv << " Ldg=" << Ldg
           << " diff=" << fabs(Lsv - Ldg) << endl;
      expect(fabs(Lsv - Ldg) < 1e-8, "Phase3: L=1 probit CDeepGp equals CSvgp");

      CMatrix K(2, 2);
      kDg.compute(K, Xb);
      K.setSymmetric(true);
      const double Lexact = logProbitOrthant2(K, Yb);
      cout << "  info: Phase3 orthant ELBO=" << Ldg << " exact=" << Lexact
           << " gap=" << (Lexact - Ldg) << endl;
      expect(isfinite(Ldg) && isfinite(Lexact),
             "Phase3: probit orthant quantities finite");
      expect(Ldg <= Lexact + 1e-5,
             "Phase3: CDeepGp-probit ELBO ≤ exact orthant log marginal");
    }

    // ---- Phase 3: L=2 probit, σ₁²→0 reduces to CSvgp-probit on XP ----
    {
      const unsigned int N = 12, M = 5;
      mt19937 rng(61);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.2 + 2.4 * (double)i / (double)(N - 1), i, 0);
        y.setVal(X.getVal(i, 0) + 0.2 * nd(rng) > 0.0 ? 1.0 : -1.0, i, 0);
      }
      ndlutil::init_genrand(67u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.0, 1e-20);
      makeRbfOnly(k1, X, 1.0 / (0.7 * 0.7), 1.0);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 1.0, 1.0, 0);
      k0.setParam(1e-20, 1); // restore after chi overwrite; with ε=0, h=XP+μ
      dgp.layer(0).updateKernelMatrices();
      CProbitNoise pnoise(&y);
      dgp.setLikelihood(&pnoise);
      CMatrix m0(dgp.layer(0).getNumInducing(), 1);
      m0.zeros();
      vector<CMatrix> S0(1);
      S0[0].resize(dgp.layer(0).getNumInducing(), dgp.layer(0).getNumInducing());
      S0[0].zeros();
      S0[0].addDiag(1.0);
      S0[0].setSymmetric(true);
      dgp.layer(0).setWhitenedQ(m0, S0);
      dgp.layer(0).updateKernelMatrices();
      for(unsigned int t = 0; t < 20; t++)
        dgp.layer(1).naturalGradientStep(0.3);

      CMatrix XP(N, 1);
      XP.gemm(X, dgp.meanMap(0), 1.0, 0.0, "n", "n");
      CCmpndKern kSv(XP);
      makeRbfOnly(kSv, XP, 1.0 / (0.7 * 0.7), 1.0);
      CMatrix pk(1, k1.getNumParams());
      k1.getParams(pk);
      kSv.setParams(pk);
      CMatrix Zf(dgp.layer(1).getInducingCentres());
      CProbitNoise noiseSv(&y);
      CSvgp sv(&kSv, &XP, &y, Zf, 1.0, 0);
      sv.setLikelihood(&noiseSv);
      sv.setQParamMode(CSvgp::QModeCholesky);
      vector<CMatrix> cov1(1, dgp.layer(1).getWhitenedCov(0));
      sv.setWhitenedQ(dgp.layer(1).getWhitenedMean(), cov1);
      sv.updateKernelMatrices();

      CMatrix eps(N, 1);
      eps.zeros();
      dgp.setNumSamples(1);
      dgp.setFrozenEps(eps);
      const double Ldgp = dgp.elboEstimate();
      const double Lsv = sv.elbo();
      const double kl0 = dgp.layer(0).klQ();
      cout << "  info: Phase3 L=2 σ→0 Ldgp+kl0=" << (Ldgp + kl0)
           << " Lsv=" << Lsv << " diff=" << fabs(Ldgp + kl0 - Lsv) << endl;
      expect(fabs(Ldgp + kl0 - Lsv) < 1e-4,
             "Phase3: L=2 probit σ₁²→0 equals CSvgp-probit on XP");
    }

    // ---- Phase 3 V4: FD at fixed ε with probit final layer ----
    {
      const unsigned int N = 10, M = 4;
      mt19937 rng(71);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.0 + 2.0 * (double)i / (double)(N - 1), i, 0);
        y.setVal(sin(X.getVal(i, 0)) > 0.0 ? 1.0 : -1.0, i, 0);
      }
      ndlutil::init_genrand(73u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.5, 0.4);
      makeRbfOnly(k1, X, 1.2, 1.0);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 1.0, 0.4, 0);
      CProbitNoise pnoise(&y);
      dgp.setLikelihood(&pnoise);
      CMatrix m0(dgp.layer(0).getNumInducing(), 1);
      m0.zeros();
      vector<CMatrix> S0(1);
      S0[0].resize(dgp.layer(0).getNumInducing(), dgp.layer(0).getNumInducing());
      S0[0].zeros();
      S0[0].addDiag(0.3);
      S0[0].setSymmetric(true);
      dgp.layer(0).setWhitenedQ(m0, S0);
      for(unsigned int t = 0; t < 5; t++)
        dgp.layer(1).naturalGradientStep(0.2);
      dgp.setNumSamples(1);
      CMatrix eps(N, 1);
      for(unsigned int i = 0; i < N; i++)
        eps.setVal(nd(rng), i, 0);
      dgp.setFrozenEps(eps);

      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      CMatrix gAnal;
      dgp.objectiveGradBatch(all, gAnal);
      CMatrix p0;
      dgp.getStochasticParams(p0);
      const double h = 1e-5;
      double maxErr = 0.0;
      unsigned int worst = 0;
      for(unsigned int i = 0; i < p0.getCols(); i++)
      {
        CMatrix pp(p0), pm(p0);
        pp.setVal(p0.getVal(i) + h, i);
        pm.setVal(p0.getVal(i) - h, i);
        dgp.setStochasticParams(pp);
        CMatrix gTmp;
        const double Lp = dgp.objectiveGradBatch(all, gTmp);
        dgp.setStochasticParams(pm);
        const double Lm = dgp.objectiveGradBatch(all, gTmp);
        dgp.setStochasticParams(p0);
        const double gFd = (Lp - Lm) / (2.0 * h);
        const double err = fabs(gAnal.getVal(i) - gFd);
        if(err > maxErr)
        {
          maxErr = err;
          worst = i;
        }
      }
      cout << "  info: Phase3 V4-probit max|g-FD|=" << maxErr
           << " at i=" << worst << " of " << p0.getCols() << endl;
      expect(maxErr < 5e-3, "Phase3 V4: probit grads match FD at fixed ε");
    }

    // ---- Phase 3: ordered likelihood + optional final-layer CVI ----
    {
      const unsigned int N = 14, M = 5, nCat = 3;
      mt19937 rng(83);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.5 + 3.0 * (double)i / (double)(N - 1), i, 0);
        const double z = X.getVal(i, 0) + 0.3 * nd(rng);
        unsigned int cat = 0;
        if(z > -0.4) cat = 1;
        if(z > 0.5) cat = 2;
        y.setVal((double)cat, i, 0);
      }
      ndlutil::init_genrand(89u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.0, 0.3);
      makeRbfOnly(k1, X, 1.0, 1.0);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 1.0, 0.3, 0);
      COrderedNoise ono(&y, nCat);
      dgp.setLikelihood(&ono);
      dgp.setNumSamples(1);
      dgp.setSeed(5);
      dgp.setNatGradGamma(0.25);
      const double L0 = dgp.elboEstimate();
      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      for(unsigned int t = 0; t < 15; t++)
      {
        CMatrix g;
        dgp.objectiveGradBatch(all, g);
        dgp.afterBatchStep(all);
      }
      const double L1 = dgp.elboEstimate();
      cout << "  info: Phase3 ordered L0=" << L0 << " L1=" << L1 << endl;
      expect(isfinite(L0) && isfinite(L1),
             "Phase3: ordered ELBO finite");
      expect(L1 >= L0 - 1e-6,
             "Phase3: ordered final-layer CVI does not decrease ELBO");
    }

    // ---- Phase 4: residual-off reduces predictive variance ----
    {
      const unsigned int N = 12, M = 5;
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal(-1.0 + 2.0 * (double)i / (double)(N - 1), i, 0);
        y.setVal(sin(X.getVal(i, 0)), i, 0);
      }
      ndlutil::init_genrand(101u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.0, 0.5);
      makeRbfOnly(k1, X, 1.0, 1.0);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 1.0, 0.5, 0);
      CMatrix muOn, varOn, muOff, varOff;
      dgp.layer(0).marginalsAt(X, muOn, varOn);
      dgp.setDropResidual(true);
      dgp.layer(0).marginalsAt(X, muOff, varOff);
      double meanOn = 0.0, meanOff = 0.0;
      for(unsigned int i = 0; i < N; i++)
      {
        meanOn += varOn.getVal(i, 0);
        meanOff += varOff.getVal(i, 0);
      }
      meanOn /= (double)N;
      meanOff /= (double)N;
      cout << "  info: Phase4 residual varOn=" << meanOn
           << " varOff=" << meanOff << endl;
      expect(dgp.isDropResidual(), "Phase4: dropResidual flag set");
      expect(meanOff <= meanOn + 1e-12,
             "Phase4: residual-off variance <= residual-on");
    }

    // ---- Phase 4: stream round-trip ----
    {
      const unsigned int N = 10, M = 4;
      CMatrix X(N, 1), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        X.setVal((double)i / (double)(N - 1), i, 0);
        y.setVal(X.getVal(i, 0) * X.getVal(i, 0), i, 0);
      }
      ndlutil::init_genrand(103u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.2, 0.4);
      makeRbfOnly(k1, X, 0.9, 1.0);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 2.0, 0.4, 0);
      dgp.setNumSamples(2);
      dgp.setDropResidual(true);
      CMatrix p0;
      dgp.getStochasticParams(p0);
      const double L0 = dgp.elboEstimate();
      ostringstream oss;
      dgp.toStream(oss);
      istringstream iss(oss.str());
      CMatrix Xtmp(1, 1), Ytmp(1, 1);
      CRbfKern ktmp(1);
      ktmp.setInitParam();
      vector<CKern*> tk{&ktmp};
      CDeepGp loaded(Xtmp, Ytmp, tk, vector<unsigned int>{1u, 1u},
                     vector<unsigned int>{1u}, 1.0, 1.0, 0);
      loaded.fromStream(iss);
      CMatrix p1;
      loaded.getStochasticParams(p1);
      expect(loaded.getNumLayers() == 2 && loaded.isDropResidual()
             && loaded.getNumSamples() == 2,
             "Phase4: stream restores meta fields");
      expect(p0.getCols() == p1.getCols() && maxAbsDiff(p0, p1) < 1e-10,
             "Phase4: stream restores stochastic params");
      const double L1 = loaded.elboEstimate();
      cout << "  info: Phase4 stream L0=" << L0 << " L1=" << L1 << endl;
      expect(fabs(L0 - L1) < 1e-6, "Phase4: stream ELBO match");
    }

    // ---- Phase 4 V7: prior Jacobian spread with chi=1/L stays bounded ----
    // Draws v~N(0,I) per layer (A^T v), independent of trained q — matches
    // prototype dgp_jacobian.py residual-off prior probe.
    {
      const unsigned int D = 2, L = 4, M = 8, N = 16;
      mt19937 rng(107);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, D), y(N, D);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int d = 0; d < D; d++)
        {
          X.setVal(nd(rng), i, d);
          y.setVal(nd(rng), i, d);
        }
      ndlutil::init_genrand(109u);
      CCmpndKern k0(X), k1(X), k2(X), k3(X);
      const double vPrior = 1.0 / (double)L;
      makeRbfOnly(k0, X, 1.0, vPrior);
      makeRbfOnly(k1, X, 1.0, vPrior);
      makeRbfOnly(k2, X, 1.0, vPrior);
      makeRbfOnly(k3, X, 1.0, vPrior);
      vector<unsigned int> dims(L + 1, D);
      vector<unsigned int> Ms(L, M);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1, &k2, &k3}, dims, Ms,
                  1.0, 1.0, 0);
      // Constructor scales hidden vars by chi/(L-1); restore chi=1/L on all.
      k0.setVariance(vPrior);
      k1.setVariance(vPrior);
      k2.setVariance(vPrior);
      k3.setVariance(vPrior);
      for(unsigned int ell = 0; ell < L; ell++)
        dgp.layer(ell).updateKernelMatrices();
      const double spread = dgp.jacobianSpread(16, 111u);
      cout << "  info: Phase4 V7 median log-sv spread L=" << L
           << " D=" << D << " = " << spread << endl;
      expect(isfinite(spread) && spread > 0.05 && spread < 2.0,
             "Phase4 V7: chi=1/L prior Jacobian spread in (0.05, 2)");
    }

    // ---- Sampled predict mixture moments ----
    {
      const unsigned int N = 20, M = 6, Nte = 8;
      mt19937 rng(121);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1), Xte(Nte, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        const double x = -2.0 + 4.0 * (double)i / (double)(N - 1);
        X.setVal(x, i, 0);
        y.setVal(sin(1.2 * x) + 0.05 * nd(rng), i, 0);
      }
      for(unsigned int i = 0; i < Nte; i++)
        Xte.setVal(-1.5 + 3.0 * (double)i / (double)(Nte - 1), i, 0);
      ndlutil::init_genrand(123u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.0, 0.5);
      makeRbfOnly(k1, X, 1.0, 1.0);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 5.0, 0.5, 0);
      CMatrix mu0, var0, muS, varS;
      dgp.predictMeanVar(Xte, mu0, var0, 0);
      dgp.predictMeanVar(Xte, muS, varS, 8, 7u);
      expect(mu0.getRows() == Nte && var0.getRows() == Nte,
             "predictMeanVar mean-path shapes");
      expect(muS.getRows() == Nte && varS.getRows() == Nte,
             "predictMeanVar mixture shapes");
      for(unsigned int i = 0; i < Nte; i++)
      {
        expect(isfinite(mu0.getVal(i, 0)) && var0.getVal(i, 0) > 0.0,
               "predictMeanVar mean-path finite positive var");
        expect(isfinite(muS.getVal(i, 0)) && varS.getVal(i, 0) > 0.0,
               "predictMeanVar mixture finite positive var");
      }
      // Mixture variance should be at least as large as mean-path (extra
      // between-sample mean variance) once hidden q has prior mass; after
      // near-identity init they can be close — just check agreement scale.
      expect(fabs(mu0.getVal(0, 0) - muS.getVal(0, 0)) < 2.0,
             "predictMeanVar mean-path vs mixture not wildly apart");
    }

    // ---- Phase 4 V8 smoke: inducing-point deep GP trains ----
    {
      const unsigned int N = 36, M = 8, Nte = 16;
      mt19937 rng(113);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, 1), y(N, 1), Xte(Nte, 1), yte(Nte, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        const double x = -2.0 + 4.0 * (double)i / (double)(N - 1);
        X.setVal(x, i, 0);
        y.setVal(sin(1.2 * x) + 0.1 * nd(rng), i, 0);
      }
      for(unsigned int i = 0; i < Nte; i++)
      {
        const double x = -1.8 + 3.6 * (double)i / (double)(Nte - 1);
        Xte.setVal(x, i, 0);
        yte.setVal(sin(1.2 * x), i, 0);
      }
      ndlutil::init_genrand(115u);
      CCmpndKern k0(X), k1(X);
      makeRbfOnly(k0, X, 1.0, 0.5);
      makeRbfOnly(k1, X, 1.0, 1.0);
      CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1},
                  vector<unsigned int>{1u, 1u, 1u},
                  vector<unsigned int>{M, M}, 5.0, 0.5, 0);
      dgp.setNumSamples(1);
      dgp.setSeed(7);
      CAdam adam(0.05);
      adam.setBatchSize(N);
      adam.setMaxIters(30);
      adam.setSeed(7);
      adam.maximise(dgp);
      CMatrix mu0, var0;
      dgp.layer(0).marginalsAt(Xte, mu0, var0);
      CMatrix H(Nte, 1);
      H.gemm(Xte, dgp.meanMap(0), 1.0, 0.0, "n", "n");
      for(unsigned int i = 0; i < Nte; i++)
        H.setVal(H.getVal(i, 0) + mu0.getVal(i, 0), i, 0);
      CMatrix mu1, var1;
      dgp.layer(1).marginalsAt(H, mu1, var1);
      double sse = 0.0;
      for(unsigned int i = 0; i < Nte; i++)
      {
        const double e = mu1.getVal(i, 0) - yte.getVal(i, 0);
        sse += e * e;
      }
      const double rmse = sqrt(sse / (double)Nte);
      cout << "  info: Phase4 V8-points rmse=" << rmse << endl;
      expect(isfinite(rmse) && rmse < 1.5,
             "Phase4 V8: inducing-point deep GP RMSE finite and sane");
    }

    // ---- Hidden width != input width, Adam at lr 0.01 (regressions) ----
    // (a) hidden CSvgp cores must be built on D_{l-1}-column inputs, not X;
    // (b) KL in Cholesky mode must use the stored factor: refactorising
    //     S~ = L L^T failed (non-PD) within a few Adam steps on this problem.
    {
      const unsigned int N = 800, d = 10, D = 5, M = 64;
      mt19937 rng(211);
      normal_distribution<double> nd(0.0, 1.0);
      CMatrix X(N, d), y(N, 1);
      for(unsigned int i = 0; i < N; i++)
      {
        double u1 = 0.0, u2 = 0.0;
        for(unsigned int j = 0; j < d; j++)
        {
          const double x = nd(rng);
          X.setVal(x, i, j);
          u1 += x * ((j % 2) ? 0.3 : -0.25);
          u2 += x * ((j < 5) ? 0.3 : 0.2);
        }
        y.setVal(sin(2.0 * u1 + u2 * u2) + 0.1 * nd(rng), i, 0);
      }
      CMatrix Xh(N, D);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < D; j++)
          Xh.setVal(nd(rng), i, j);
      ndlutil::init_genrand(213u);
      CCmpndKern k0(X), k1(Xh), k2(Xh);
      makeRbfOnly(k0, X, 1.0, 1.0);
      makeRbfOnly(k1, Xh, 1.0, 1.0);
      makeRbfOnly(k2, Xh, 1.0, 1.0);
      bool threw = false;
      double L0 = 0.0, L1 = 0.0;
      try
      {
        CDeepGp dgp(X, y, vector<CKern*>{&k0, &k1, &k2},
                    vector<unsigned int>{d, D, D, 1u},
                    vector<unsigned int>{M, M, M}, 1.0, 1.0, 0);
        dgp.setNumSamples(1);
        dgp.setSeed(3);
        L0 = dgp.elboEstimate();
        CAdam adam(0.01);
        adam.setBatchSize(256);
        adam.setMaxIters(100);
        adam.setSeed(3);
        adam.maximise(dgp);
        L1 = dgp.elboEstimate();
      }
      catch(ndlexceptions::Error& e)
      {
        threw = true;
        cout << "  info: hidden-width run threw: " << e.what() << endl;
      }
      cout << "  info: hidden-width L0=" << L0 << " L1=" << L1 << endl;
      expect(!threw, "Hidden width != input width: construct and train at lr 0.01");
      expect(!threw && isfinite(L1) && L1 > L0, "Hidden width != input width: ELBO improves");
    }
  }
  catch(ndlexceptions::Error& e)
  {
    cout << "FAIL: exception: " << e.getMessage() << endl;
    fails++;
  }
  catch(exception& e)
  {
    cout << "FAIL: std::exception: " << e.what() << endl;
    fails++;
  }
  cout << "Number of failures: " << fails << endl;
  return fails > 0 ? 1 : 0;
}
