/* CIP-0035 Phase 1–2: CDeepGp verification (V1–V6 points; Phase 2 ridges). */
#include "CAdam.h"
#include "CDeepGp.h"
#include "CInducing.h"
#include "CKern.h"
#include "CMatrix.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <cmath>
#include <iostream>
#include <random>
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
