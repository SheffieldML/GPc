/* CIP-002A: SVGP q(u) natural gradients + numerics soak.

   Phase 1–4 / 0a–0b gates (existing) plus numerics soak (backlog
   2026-11-19_cip002A-numerics-soak):
   - hyperparameter grads at q* vs CGp DTCVAR (envelope theorem)
   - probit CVI natgrad vs FD on expectation parameters
   - N=2 probit orthant: ELBO ≤ exact; SVGP vs sparse PPA note
   - nested inducing sets ELBO(Z′) ≥ ELBO(Z)
   - marginal() clamp asserts near-nonnegativity
*/

#include "CAdam.h"
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CSparsePpa.h"
#include "CSvgp.h"
#include "ndlexceptions.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
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
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

static void makeKern(CCmpndKern& kern, const CMatrix& X, bool withWhite)
{
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CBiasKern(X));
  if(withWhite)
    kern.addKern(new CWhiteKern(X));
  CMatrix p(1, kern.getNumParams());
  p.setVal(1.0 / (0.8 * 0.8), 0);  // rbf inverse width
  p.setVal(1.5, 1);                // rbf variance
  p.setVal(0.2, 2);                // bias variance
  if(withWhite)
    p.setVal(0.05, 3);             // white variance
  kern.setParams(p);
}

static CMatrix precisionOf(const CMatrix& S)
{
  CMatrix Sc(S);
  Sc.setSymmetric(true);
  CMatrix U(Sc);
  U.chol();
  CMatrix P(S.getRows(), S.getCols());
  P.setSymmetric(true);
  P.pdinv(U);
  return P;
}

// Exact log N(y | 0, K + I/beta) summed over output columns.
static double exactLogMarginal(CKern& kern, const CMatrix& X, const CMatrix& y, double beta)
{
  const unsigned int N = X.getRows();
  CMatrix K(N, N);
  kern.compute(K, X);
  K.addDiag(1.0 / beta);
  K.setSymmetric(true);
  CMatrix U(K);
  U.chol();
  CMatrix invK(N, N);
  invK.setSymmetric(true);
  invK.pdinv(U);
  const double ld = logDet(U);
  double L = 0.0;
  CMatrix v(N, 1);
  for(unsigned int j = 0; j < y.getCols(); j++)
    {
      v.gemvColCol(0, invK, y, j, 1.0, 0.0, "n");
      L += -0.5 * v.dotColCol(0, y, j) - 0.5 * ld - (double)N * ndlutil::HALFLOGTWOPI;
    }
  return L + kern.priorLogProb();
}

static void runCase(bool withWhite)
{
  const string tag = withWhite ? " [rbf+bias+white]" : " [rbf+bias]";
  const unsigned int N = 60, M = 12, D = 2;
  const double beta = 25.0;
  mt19937 rng(42);
  normal_distribution<double> nd(0.0, 1.0);
  uniform_real_distribution<double> ud(-3.0, 3.0);

  CMatrix X(N, 1), y(N, D);
  for(unsigned int i = 0; i < N; i++)
    {
      const double x = ud(rng);
      X.setVal(x, i, 0);
      y.setVal(sin(2.0 * x) + 0.2 * nd(rng), i, 0);
      y.setVal(0.5 * x - cos(x) + 0.2 * nd(rng), i, 1);
    }
  CMatrix Z(M, 1);
  for(unsigned int k = 0; k < M; k++)
    Z.setVal(-2.9 + 5.8 * (double)k / (double)(M - 1), k, 0);

  // Reference: CGp DTCVAR with the same Z, beta, kernel, unit scale / zero bias.
  CCmpndKern kernG(X);
  makeKern(kernG, X, withWhite);
  CMatrix Xg(X), yg(y);
  CGaussianNoise gnoise(&yg);
  for(unsigned int j = 0; j < D; j++)
    gnoise.setBiasVal(0.0, j);
  CGp gp(1, D, &Xg, &yg, &kernG, &gnoise, CGp::DTCVAR, M, 0);
  gp.setActiveSet(Z);
  for(unsigned int j = 0; j < D; j++)
    {
      gp.setBiasVal(0.0, j);
      gp.setScaleVal(1.0, j);
    }
  gp.setBetaVal(beta);
  gp.invalidateCache();
  const double bound = gp.logLikelihood();

  CCmpndKern kern(X);
  makeKern(kern, X, withWhite);
  CSvgp sv(&kern, &X, &y, Z, beta, 0);

  // 1. One full-batch gamma = 1 step from the prior hits the collapsed bound.
  const double elbo0 = sv.elbo();
  expect(elbo0 < bound, "prior q(u) ELBO below DTCVAR bound" + tag);
  sv.naturalGradientStep(1.0);
  const double elbo1 = sv.elbo();
  cout << "  info: DTCVAR bound=" << bound << " SVGP ELBO after 1 step=" << elbo1
       << " diff=" << (elbo1 - bound) << endl;
  expect(fabs(elbo1 - bound) < 1e-8 * max(1.0, fabs(bound)),
         "gamma=1 natural-gradient step: ELBO == CGp DTCVAR logLikelihood" + tag);

  // A second gamma = 1 step is a fixed point.
  CMatrix m1(sv.getQMean());
  sv.naturalGradientStep(1.0);
  expect(maxAbsDiff(m1, sv.getQMean()) < 1e-9, "Titsias optimum is a fixed point" + tag);

  // Predictions agree with CGp DTCVAR.
  const unsigned int Ns = 25;
  CMatrix Xs(Ns, 1);
  for(unsigned int i = 0; i < Ns; i++)
    Xs.setVal(-3.5 + 7.0 * (double)i / (double)(Ns - 1), i, 0);
  CMatrix muG(Ns, D), varG(Ns, D), muS, varS;
  // CIP-0031: compare latent variances directly (no manual 1/β correction).
  gp.latentMeanVar(muG, varG, Xs);
  sv.latentMeanVar(muS, varS, Xs);
  const double dMu = maxAbsDiff(muG, muS);
  const double dVar = maxAbsDiff(varG, varS);
  cout << "  info: max|mean diff|=" << dMu << " max|var diff|=" << dVar << endl;
  expect(dMu < 1e-8, "latent mean matches CGp DTCVAR" + tag);
  expect(dVar < 1e-8, "latent variance matches CGp DTCVAR" + tag);
  CMatrix muGp(Ns, D), varGp(Ns, D), muSp, varSp;
  gp.predictiveMeanVar(muGp, varGp, Xs);
  sv.predictiveMeanVar(muSp, varSp, Xs);
  expect(maxAbsDiff(varGp, varSp) < 1e-8, "predictive variance matches CGp DTCVAR" + tag);
  expect(fabs(varSp.getVal(0) - varS.getVal(0) - 1.0 / beta) < 1e-12,
         "SVGP predictive − latent = 1/β" + tag);

  // 2. gamma < 1: monotone increase, converges to the bound.
  {
    CCmpndKern k2(X);
    makeKern(k2, X, withWhite);
    CSvgp s2(&k2, &X, &y, Z, beta, 0);
    double prev = s2.elbo();
    bool monotone = true;
    for(int it = 0; it < 60; it++)
      {
        s2.naturalGradientStep(0.3);
        const double cur = s2.elbo();
        if(cur < prev - 1e-9 * max(1.0, fabs(prev)))
          monotone = false;
        prev = cur;
      }
    expect(monotone, "gamma=0.3 steps increase the ELBO monotonically" + tag);
    cout << "  info: after 60 steps of gamma=0.3, bound - ELBO = " << (bound - prev) << endl;
    expect(bound - prev < 1e-6 && bound - prev > -1e-8 * max(1.0, fabs(bound)),
           "gamma=0.3 converges to the DTCVAR bound" + tag);
  }

  // 3. Minibatch estimates average to full batch over a partition.
  {
    CCmpndKern k3(X);
    makeKern(k3, X, withWhite);
    CSvgp s3(&k3, &X, &y, Z, beta, 0);
    s3.naturalGradientStep(0.4);  // move off the prior
    const unsigned int B = 5;
    vector<unsigned int> perm(N);
    for(unsigned int i = 0; i < N; i++)
      perm[i] = i;
    shuffle(perm.begin(), perm.end(), rng);
    vector<vector<unsigned int> > batches(B);
    for(unsigned int i = 0; i < N; i++)
      batches[i % B].push_back(perm[i]);

    double meanElbo = 0.0;
    for(unsigned int b = 0; b < B; b++)
      meanElbo += s3.elbo(batches[b]) / (double)B;
    const double full = s3.elbo();
    expect(fabs(meanElbo - full) < 1e-9 * max(1.0, fabs(full)),
           "minibatch ELBO estimates average to full batch" + tag);

    // Whitened natural parameters after one minibatch step average to the full-batch step.
    const double gamma = 0.5;
    const CMatrix m0(s3.getWhitenedMean());
    vector<CMatrix> S0;
    for(unsigned int j = 0; j < D; j++)
      S0.push_back(s3.getWhitenedCov(j));
    double maxP = 0.0, maxT = 0.0;
    vector<CMatrix> Pavg(D, CMatrix(M, M, 0.0)), Tavg(D, CMatrix(M, 1, 0.0));
    for(unsigned int b = 0; b < B; b++)
      {
        s3.setWhitenedQ(m0, S0);
        s3.naturalGradientStep(gamma, batches[b]);
        for(unsigned int j = 0; j < D; j++)
          {
            CMatrix P = precisionOf(s3.getWhitenedCov(j));
            CMatrix t(M, 1);
            t.gemvColCol(0, P, s3.getWhitenedMean(), j, 1.0, 0.0, "n");
            Pavg[j].axpy(P, 1.0 / (double)B);
            Tavg[j].axpy(t, 1.0 / (double)B);
          }
      }
    s3.setWhitenedQ(m0, S0);
    s3.naturalGradientStep(gamma);
    for(unsigned int j = 0; j < D; j++)
      {
        CMatrix P = precisionOf(s3.getWhitenedCov(j));
        CMatrix t(M, 1);
        t.gemvColCol(0, P, s3.getWhitenedMean(), j, 1.0, 0.0, "n");
        maxP = max(maxP, maxAbsDiff(P, Pavg[j]) / max(1.0, fabs(P.getVal(0, 0))));
        maxT = max(maxT, maxAbsDiff(t, Tavg[j]) / max(1.0, fabs(t.getVal(0, 0))));
      }
    cout << "  info: minibatch natural-param relative diffs: precision=" << maxP
         << " theta1=" << maxT << endl;
    expect(maxP < 1e-7 && maxT < 1e-7,
           "minibatch natural-gradient steps are unbiased (average to full batch)" + tag);

    // Unwhitened setQ / getQ round trip.
    const CMatrix mU(s3.getQMean());
    vector<CMatrix> SU;
    for(unsigned int j = 0; j < D; j++)
      SU.push_back(s3.getQCov(j));
    const double before = s3.elbo();
    s3.setQ(mU, SU);
    expect(fabs(s3.elbo() - before) < 1e-8 * max(1.0, fabs(before)),
           "setQ(getQMean, getQCov) round trip preserves the ELBO" + tag);

    // CIP-002D QW6: B*4 >= N uses gather-from-A (not K_uB); still not full batch.
    vector<unsigned int> large(20);
    for(unsigned int i = 0; i < 20; i++)
      large[i] = i;
    const double elboLarge = s3.elbo(large);
    expect(std::isfinite(elboLarge),
           "large minibatch ELBO finite (gather path)" + tag);
    s3.naturalGradientStep(0.2, large);
    expect(std::isfinite(s3.elbo()),
           "large minibatch natgrad keeps ELBO finite" + tag);
  }
}

int main()
{
  try
    {
      runCase(false);
      runCase(true);

      // 4. M = N, Z = X, no white kernel.
      //    lengthscale 0.3: K_uu well conditioned -> optimum ELBO == exact marginal.
      //    lengthscale 0.8: K_uu numerically singular (min eig ~4e-16) -> must stay
      //    a lower bound and agree with CGp's stable DTCVAR path.  An unwhitened
      //    draft returned +35 here against an exact -11.57.
      for(int c = 0; c < 2; c++)
      {
        const double ell = (c == 0) ? 0.3 : 0.8;
        const unsigned int N = 20;
        const double beta = 10.0;
        mt19937 rng(7);
        normal_distribution<double> nd(0.0, 1.0);
        CMatrix X(N, 1), y(N, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            const double x = -2.0 + 4.0 * (double)i / (double)(N - 1);
            X.setVal(x, i, 0);
            y.setVal(sin(2.0 * x) + 0.3 * nd(rng), i, 0);
          }
        auto build = [&](CCmpndKern& k)
          {
            k.addKern(new CRbfKern(X));
            k.addKern(new CBiasKern(X));
            CMatrix p(1, 3);
            p.setVal(1.0 / (ell * ell), 0);
            p.setVal(1.5, 1);
            p.setVal(0.2, 2);
            k.setParams(p);
          };
        CCmpndKern kern(X); build(kern);
        CSvgp sv(&kern, &X, &y, X, beta, 0);
        sv.naturalGradientStep(1.0);
        CCmpndKern kernE(X); build(kernE);
        const double exact = exactLogMarginal(kernE, X, y, beta);
        CCmpndKern kernG(X); build(kernG);
        CMatrix Xg(X), yg(y);
        CGaussianNoise gnoise(&yg);
        gnoise.setBiasVal(0.0, 0);
        CGp gp(1, 1, &Xg, &yg, &kernG, &gnoise, CGp::DTCVAR, N, 0);
        gp.setActiveSet(X);
        gp.setBiasVal(0.0, 0);
        gp.setScaleVal(1.0, 0);
        gp.setBetaVal(beta);
        gp.invalidateCache();
        const double cgp = gp.logLikelihood();
        const double e = sv.elbo();
        cout << "  info: M=N ell=" << ell << " exact=" << exact << " CGp DTCVAR=" << cgp
             << " SVGP=" << e << endl;
        const string tg = (c == 0) ? " (well conditioned)" : " (K_uu numerically singular)";
        expect(e <= exact + 1e-8 * max(1.0, fabs(exact)), "M=N, Z=X: ELBO <= exact log marginal" + tg);
        if(c == 0)
          expect(fabs(e - exact) < 1e-6 * max(1.0, fabs(exact)),
                 "M=N, Z=X: optimum ELBO == exact log marginal" + tg);
        else
          expect(fabs(e - cgp) < 1e-3 * max(1.0, fabs(cgp)) && exact - e < 1e-2,
                 "M=N, Z=X: ELBO agrees with CGp stable path and is near exact" + tg);
      }

      // 5. Guards.
      {
        CMatrix X(4, 1), y(4, 1), Z(2, 1);
        for(unsigned int i = 0; i < 4; i++)
          {
            X.setVal((double)i, i, 0);
            y.setVal((double)i, i, 0);
          }
        Z.setVal(0.5, 0, 0);
        Z.setVal(2.5, 1, 0);
        CCmpndKern kern(X);
        makeKern(kern, X, false);
        CSvgp sv(&kern, &X, &y, Z, 1.0, 0);
        bool threw = false;
        try { sv.naturalGradientStep(1.5); } catch(ndlexceptions::Error&) { threw = true; }
        expect(threw, "gamma > 1 rejected");
        threw = false;
        try { sv.elbo(vector<unsigned int>()); } catch(ndlexceptions::Error&) { threw = true; }
        expect(threw, "empty batch rejected");
      }

      // 6. Phase 2: finite-difference gradients at fixed whitened q.
      {
        const unsigned int N = 20, M = 5, D = 1;
        const double beta = 10.0;
        mt19937 rng(7);
        normal_distribution<double> nd(0.0, 1.0);
        uniform_real_distribution<double> ud(-2.0, 2.0);
        CMatrix X(N, 1), y(N, D), Z(M, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            X.setVal(ud(rng), i, 0);
            y.setVal(sin(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
          }
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(-1.8 + 3.6 * (double)k / (double)(M - 1), k, 0);
        CCmpndKern kern(X);
        makeKern(kern, X, false);
        CSvgp sv(&kern, &X, &y, Z, beta, 0);
        sv.naturalGradientStep(1.0);  // at Titsias q

        CMatrix g(1, sv.getOptNumParams());
        vector<unsigned int> all(N);
        for(unsigned int i = 0; i < N; i++) all[i] = i;
        const double L = sv.objectiveGradBatch(all, g);
        CMatrix p0(1, sv.getOptNumParams());
        sv.getOptParams(p0);
        const double eps = 1e-6;
        double maxRel = 0.0;
        for(unsigned int i = 0; i < sv.getOptNumParams(); i++)
          {
            CMatrix pp(p0), pm(p0);
            pp.setVal(p0.getVal(i) + eps, i);
            pm.setVal(p0.getVal(i) - eps, i);
            sv.setOptParams(pp);
            // keep whitened q fixed across param perturbations: restore after setOptParams
            // setOptParams does not touch q — good.
            const double Lp = sv.elbo();
            sv.setOptParams(pm);
            const double Lm = sv.elbo();
            sv.setOptParams(p0);
            const double fd = (Lp - Lm) / (2.0 * eps);
            const double an = g.getVal(i);
            const double scale = max(1.0, max(fabs(fd), fabs(an)));
            maxRel = max(maxRel, fabs(fd - an) / scale);
          }
        cout << "  info: FD vs analytic max relative diff=" << maxRel << " ELBO=" << L << endl;
        expect(maxRel < 2e-4, "Phase 2: analytic gradients match finite differences");
      }

      // 7. Phase 3: Adam seed reproducibility + interleaved natgrad improves ELBO.
      {
        const unsigned int N = 40, M = 8, D = 1;
        const double beta = 15.0;
        mt19937 rng(11);
        normal_distribution<double> nd(0.0, 1.0);
        uniform_real_distribution<double> ud(-2.5, 2.5);
        CMatrix X(N, 1), y(N, D), Z(M, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            X.setVal(ud(rng), i, 0);
            y.setVal(0.5 * X.getVal(i, 0) + 0.15 * nd(rng), i, 0);
          }
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(-2.0 + 4.0 * (double)k / (double)(M - 1), k, 0);

        auto runAdam = [&](uint64_t seed, unsigned int batch) {
          CCmpndKern kern(X);
          makeKern(kern, X, false);
          CMatrix kp(1, kern.getNumParams());
          kern.getParams(kp);
          kp.setVal(kp.getVal(0) * 0.5, 0);
          kern.setParams(kp);
          CSvgp sv(&kern, &X, &y, Z, beta * 0.5, 0);
          const double Linit = sv.elbo();
          sv.setNatGradGamma(0.5);
          CAdam adam(0.08, 0.9, 0.999, 1e-8);
          adam.setBatchSize(batch);
          adam.setMaxIters(25);
          adam.setSeed(seed);
          unsigned int nEvents = 0;
          sv.setProgressObserver([&](const StochasticOptimiserEvent& ev) {
            nEvents++;
            (void)ev;
          });
          adam.maximise(sv);
          CMatrix p(1, sv.getOptNumParams());
          sv.getOptParams(p);
          return make_tuple(sv.elbo(), p, Linit, nEvents);
        };

        auto a = runAdam(123, 8);
        auto b = runAdam(123, 8);
        auto c = runAdam(999, 8);
        expect(fabs(get<0>(a) - get<0>(b)) < 1e-12, "Phase 3: same seed => identical ELBO");
        expect(maxAbsDiff(get<1>(a), get<1>(b)) < 1e-12, "Phase 3: same seed => identical params");
        expect(get<3>(a) == 25, "Phase 3: progress observer fires each iteration");
        expect(get<0>(a) > get<2>(a), "Phase 3: Adam+natgrad improves ELBO from initial");
        expect(maxAbsDiff(get<1>(a), get<1>(c)) > 1e-8,
               "Phase 3: different minibatch seeds yield different params");
      }

      // 8. Phase 4: Gaussian expectedLogLikelihood matches closed form; probit GH + natgrad.
      {
        CMatrix yDummy(1, 1);
        yDummy.setVal(0.0, 0, 0);
        CGaussianNoise gnoise(&yDummy);
        gnoise.setParam(0.04, gnoise.getNumParams() - 1);  // sigma2
        double gMu = 0.0, gVar = 0.0;
        const double ell = gnoise.expectedLogLikelihood(1.0, 0.5, 0.3, gMu, gVar);
        const double betaLik = 1.0 / 0.04;
        const double r = 1.0 - 0.5;
        const double ellExact = -ndlutil::HALFLOGTWOPI + 0.5 * log(betaLik)
          - 0.5 * betaLik * (r * r + 0.3);
        expect(fabs(ell - ellExact) < 1e-12, "Phase 4: Gaussian expectedLogLikelihood closed form");
        expect(fabs(gMu - betaLik * r) < 1e-12 && fabs(gVar + 0.5 * betaLik) < 1e-12,
               "Phase 4: Gaussian expectedLogLikelihood grads");

        const unsigned int N = 30, M = 8;
        mt19937 rng(21);
        uniform_real_distribution<double> ud(-2.0, 2.0);
        CMatrix X(N, 1), y(N, 1), Z(M, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            const double x = ud(rng);
            X.setVal(x, i, 0);
            y.setVal(x > 0.0 ? 1.0 : -1.0, i, 0);
          }
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(-1.8 + 3.6 * (double)k / (double)(M - 1), k, 0);
        CCmpndKern kern(X);
        makeKern(kern, X, false);
        CProbitNoise pnoise(&y);
        CSvgp sv(&kern, &X, &y, Z, 1.0, 0);
        sv.setLikelihood(&pnoise);
        const double L0 = sv.elbo();
        for(unsigned int t = 0; t < 15; t++)
          sv.naturalGradientStep(0.2);
        const double L1 = sv.elbo();
        cout << "  info: probit ELBO prior=" << L0 << " after natgrad=" << L1 << endl;
        expect(L1 > L0, "Phase 4: probit Gauss-Hermite natgrad increases ELBO");
        expect(isfinite(L1), "Phase 4: probit ELBO is finite");

        // Error / default paths for patch coverage of ndlutil + CNoise base.
        {
          vector<double> x, w;
          bool threw = false;
          try { ndlutil::gaussHermite(10, x, w); } catch(ndlexceptions::Error&) { threw = true; }
          expect(threw, "Phase 4: gaussHermite rejects n != 20");
          threw = false;
          double dgMu = 0.0, dgVar = 0.0;
          try {
            ndlutil::gaussHermiteExpect(-1.0, -0.5, 20,
              [](double, void*) { return 0.0; },
              [](double, void*) { return 0.0; },
              nullptr, dgMu, dgVar);
          } catch(ndlexceptions::Error&) { threw = true; }
          expect(threw, "Phase 4: gaussHermiteExpect rejects negative var");
          COrderedNoise ono(&y, 3);
          const double oell = ono.expectedLogLikelihood(0.0, 0.0, 1.0, dgMu, dgVar);
          expect(isfinite(oell), "Phase 4: ordered expectedLogLikelihood finite (GH)");
          threw = false;
          try { CAdam bad(-1.0); } catch(ndlexceptions::Error&) { threw = true; }
          expect(threw, "Phase 3: CAdam rejects non-positive learning rate");
        }

        // CIP-0030 Phase 0a: CSvgp via CInducing — white-noise smoke.
        {
          const unsigned int Nn = 40, Mm = 10;
          mt19937 rngWn(11);
          normal_distribution<double> ndWn(0.0, 1.0);
          CMatrix Xn(Nn, 1), yn(Nn, 1), Zn(Mm, 1);
          for(unsigned int i = 0; i < Nn; i++)
            {
              const double x = -2.0 + 4.0 * (double)i / (double)(Nn - 1);
              Xn.setVal(x, i, 0);
              yn.setVal(sin(x) + 0.1 * ndWn(rngWn), i, 0);
            }
          for(unsigned int k = 0; k < Mm; k++)
            Zn.setVal(-1.8 + 3.6 * (double)k / (double)(Mm - 1), k, 0);
          // White-noise inducing rejects Bias; RBF (+ optional White) only.
          CCmpndKern kernWn(Xn);
          CRbfKern* prbfWn = new CRbfKern(Xn);
          prbfWn->setParam(1.0, 0);
          prbfWn->setParam(1.0, 1);
          kernWn.addKern(prbfWn);
          CWhiteKern* pwhWn = new CWhiteKern(Xn);
          pwhWn->setParam(1e-4, 0);
          kernWn.addKern(pwhWn);
          CSvgp svWn(&kernWn, &Xn, &yn, Zn, 10.0, 0);
          expect(svWn.getInducing()->typeName() == "points",
                 "Phase 0a: default inducing type is points");
          const double Lpts = svWn.elbo();
          svWn.setInducingType("whitenoise");
          expect(svWn.getInducing()->typeName() == "whitenoise",
                 "Phase 0a: setInducingType whitenoise");
          const double L0 = svWn.elbo();
          expect(isfinite(L0), "Phase 0a: white-noise ELBO finite at prior");
          for(unsigned int t = 0; t < 20; t++)
            svWn.naturalGradientStep(0.5);
          const double L1 = svWn.elbo();
          cout << "  info: Phase 0a WN ELBO prior=" << L0 << " after natgrad=" << L1
               << " (points prior was " << Lpts << ")" << endl;
          expect(isfinite(L1) && L1 > L0,
                 "Phase 0a: white-noise natgrad increases finite ELBO");
          CMatrix gK, gZ;
          double gB = 0.0;
          const double Lg = svWn.elboGrad(gK, gZ, gB);
          expect(isfinite(Lg) && gK.getCols() == kernWn.getNumParams()
                 && gZ.getRows() == Mm && gZ.getCols() == 1,
                 "Phase 0a: white-noise elboGrad shapes");
        }

        // CIP-0030 Phase 0b: explicit-input core + ∂ELBO/∂X.
        {
          auto fdGX = [](CSvgp& sv, CMatrix& XB, const vector<unsigned int>& rows,
                         double c, CMatrix& gXfd, double eps = 1e-5) {
            const unsigned int B = XB.getRows();
            const unsigned int q = XB.getCols();
            gXfd.resize(B, q);
            for(unsigned int i = 0; i < B; i++)
              for(unsigned int d = 0; d < q; d++)
                {
                  const double x0 = XB.getVal(i, d);
                  XB.setVal(x0 + eps, i, d);
                  const double Lp = sv.dataTermAt(XB, rows, c);
                  XB.setVal(x0 - eps, i, d);
                  const double Lm = sv.dataTermAt(XB, rows, c);
                  XB.setVal(x0, i, d);
                  gXfd.setVal((Lp - Lm) / (2.0 * eps), i, d);
                }
          };

          // --- points / RBF: FD ∂ELBO/∂X and batch-partition exactness ---
          {
            const unsigned int N = 24, M = 6, q = 2;
            mt19937 rng(3);
            normal_distribution<double> nd(0.0, 1.0);
            CMatrix X(N, q), y(N, 1), Z(M, q);
            for(unsigned int i = 0; i < N; i++)
              {
                for(unsigned int d = 0; d < q; d++)
                  X.setVal(nd(rng), i, d);
                y.setVal(X.getVal(i, 0) - 0.5 * X.getVal(i, 1) + 0.05 * nd(rng), i, 0);
              }
            for(unsigned int k = 0; k < M; k++)
              for(unsigned int d = 0; d < q; d++)
                Z.setVal(X.getVal(k * (N / M), d), k, d);
            CCmpndKern kern(X);
            CRbfKern* prbf = new CRbfKern(X);
            prbf->setParam(1.0, 0);
            prbf->setParam(1.0, 1);
            kern.addKern(prbf);
            CSvgp sv(&kern, &X, &y, Z, 5.0, 0);
            sv.naturalGradientStep(1.0);

            vector<unsigned int> all(N);
            for(unsigned int i = 0; i < N; i++)
              all[i] = i;
            CMatrix XB(X);
            CMatrix gK, gZ, gX;
            double gB = 0.0;
            const double c = 1.0;
            const double L = sv.elboGradAt(gK, gZ, gB, &gX, XB, all, c);
            expect(isfinite(L) && gX.getRows() == N && gX.getCols() == q,
                   "Phase 0b: points elboGradAt fills gX");
            CMatrix gXfd;
            fdGX(sv, XB, all, c, gXfd);
            const double err = maxAbsDiff(gX, gXfd);
            cout << "  info: Phase 0b points max|gX-FD|=" << err << endl;
            expect(err < 2e-4, "Phase 0b: points gX matches FD");

            // Batch-partition: mean of dataTermAt over a partition equals full.
            const unsigned int nPart = 4;
            vector<unsigned int> perm(N);
            for(unsigned int i = 0; i < N; i++)
              perm[i] = i;
            shuffle(perm.begin(), perm.end(), rng);
            double meanL = 0.0;
            CMatrix gXmean(N, q);
            gXmean.zeros();
            for(unsigned int p = 0; p < nPart; p++)
              {
                vector<unsigned int> rows;
                for(unsigned int i = 0; i < N; i++)
                  if(i % nPart == p)
                    rows.push_back(perm[i]);
                CMatrix XBp(rows.size(), q);
                for(unsigned int k = 0; k < rows.size(); k++)
                  for(unsigned int d = 0; d < q; d++)
                    XBp.setVal(X.getVal(rows[k], d), k, d);
                const double cp = (double)N / (double)rows.size();
                meanL += sv.dataTermAt(XBp, rows, cp) / (double)nPart;
                CMatrix gKp, gZp, gXp;
                double gBp = 0.0;
                sv.elboGradAt(gKp, gZp, gBp, &gXp, XBp, rows, cp);
                for(unsigned int k = 0; k < rows.size(); k++)
                  for(unsigned int d = 0; d < q; d++)
                    gXmean.addVal(gXp.getVal(k, d) / (double)nPart, rows[k], d);
              }
            const double Lfull = sv.dataTermAt(XB, all, 1.0);
            expect(fabs(meanL - Lfull) < 1e-10,
                   "Phase 0b: partition dataTermAt averages to full");
            // gX under c=N/|B| averages to full-batch gX (same site scaling).
            CMatrix gXfull;
            sv.elboGradAt(gK, gZ, gB, &gXfull, XB, all, 1.0);
            const double gerr = maxAbsDiff(gXmean, gXfull);
            cout << "  info: Phase 0b partition max|gXmean-gXfull|=" << gerr << endl;
            expect(gerr < 1e-8, "Phase 0b: partition gX averages to full");
          }

          // --- white-noise RBF ---
          {
            const unsigned int N = 20, M = 5;
            CMatrix X(N, 1), y(N, 1), Z(M, 1);
            for(unsigned int i = 0; i < N; i++)
              {
                const double x = -1.5 + 3.0 * (double)i / (double)(N - 1);
                X.setVal(x, i, 0);
                y.setVal(sin(2.0 * x), i, 0);
              }
            for(unsigned int k = 0; k < M; k++)
              Z.setVal(-1.2 + 2.4 * (double)k / (double)(M - 1), k, 0);
            CCmpndKern kern(X);
            CRbfKern* prbf = new CRbfKern(X);
            prbf->setParam(1.5, 0);
            prbf->setParam(1.0, 1);
            kern.addKern(prbf);
            CSvgp sv(&kern, &X, &y, Z, 8.0, 0);
            sv.setInducingType("whitenoise");
            sv.naturalGradientStep(0.8);
            vector<unsigned int> all(N);
            for(unsigned int i = 0; i < N; i++)
              all[i] = i;
            CMatrix XB(X), gK, gZ, gX, gXfd;
            double gB = 0.0;
            sv.elboGradAt(gK, gZ, gB, &gX, XB, all, 1.0);
            fdGX(sv, XB, all, 1.0, gXfd);
            const double err = maxAbsDiff(gX, gXfd);
            cout << "  info: Phase 0b WN max|gX-FD|=" << err << endl;
            expect(err < 5e-4, "Phase 0b: white-noise gX matches FD");
          }

          // --- weight-space MLP ridge ---
          {
            const unsigned int N = 16, M = 8, q = 2;
            mt19937 rng(5);
            normal_distribution<double> nd(0.0, 1.0);
            CMatrix X(N, q), y(N, 1), Z(M, q);
            for(unsigned int i = 0; i < N; i++)
              {
                for(unsigned int d = 0; d < q; d++)
                  X.setVal(nd(rng), i, d);
                y.setVal(tanh(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
              }
            for(unsigned int k = 0; k < M; k++)
              for(unsigned int d = 0; d < q; d++)
                Z.setVal(nd(rng) * 0.5, k, d);
            CMlpKern kern(X);
            kern.setParam(1.0, 0);  // weight variance
            kern.setParam(1.0, 1);  // bias variance
            kern.setParam(1.0, 2);  // variance
            CSvgp sv(&kern, &X, &y, Z, 4.0, 0);
            sv.setInducingType("weightspace");
            for(unsigned int t = 0; t < 5; t++)
              sv.naturalGradientStep(0.3);
            vector<unsigned int> all(N);
            for(unsigned int i = 0; i < N; i++)
              all[i] = i;
            CMatrix XB(X), gK, gZ, gX, gXfd;
            double gB = 0.0;
            sv.elboGradAt(gK, gZ, gB, &gX, XB, all, 1.0);
            fdGX(sv, XB, all, 1.0, gXfd, 1e-5);
            const double err = maxAbsDiff(gX, gXfd);
            cout << "  info: Phase 0b weightspace max|gX-FD|=" << err << endl;
            expect(err < 5e-3, "Phase 0b: weight-space gX matches FD");
          }

          // --- Green's / Brownian 1-D ---
          {
            const unsigned int N = 18, M = 6;
            CMatrix X(N, 1), y(N, 1), Z(M, 1);
            for(unsigned int i = 0; i < N; i++)
              {
                const double x = 0.05 + 0.9 * (double)i / (double)(N - 1);
                X.setVal(x, i, 0);
                y.setVal(sin(4.0 * x), i, 0);
              }
            for(unsigned int k = 0; k < M; k++)
              Z.setVal(0.1 + 0.8 * (double)k / (double)(M - 1), k, 0);
            CBrownianKern kern(X);
            kern.setParam(1.0, 0);
            CSvgp sv(&kern, &X, &y, Z, 6.0, 0);
            sv.setInducingType("greens");
            sv.naturalGradientStep(0.5);
            vector<unsigned int> all(N);
            for(unsigned int i = 0; i < N; i++)
              all[i] = i;
            CMatrix XB(X), gK, gZ, gX, gXfd;
            double gB = 0.0;
            sv.elboGradAt(gK, gZ, gB, &gX, XB, all, 1.0);
            fdGX(sv, XB, all, 1.0, gXfd);
            const double err = maxAbsDiff(gX, gXfd);
            cout << "  info: Phase 0b greens max|gX-FD|=" << err << endl;
            expect(err < 5e-4, "Phase 0b: Green's gX matches FD");
          }
        }
      }

      // =====================================================================
      // Numerics soak (backlog 2026-11-19_cip002A-numerics-soak)
      // =====================================================================
      cout << "---- CIP-002A numerics soak ----" << endl;

      // 9. Hyperparameter gradients at q*: envelope theorem vs CGp DTCVAR.
      // CGp packing (inducing free, no scale learn): Z, kern, beta.
      // CSvgp packing: kern, Z, beta. Compare after remapping.
      {
        const unsigned int N = 24, M = 6, D = 1;
        const double beta = 12.0;
        mt19937 rng(41);
        normal_distribution<double> nd(0.0, 1.0);
        uniform_real_distribution<double> ud(-2.0, 2.0);
        CMatrix X(N, 1), y(N, D), Z(M, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            X.setVal(ud(rng), i, 0);
            y.setVal(sin(X.getVal(i, 0)) + 0.15 * nd(rng), i, 0);
          }
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(-1.8 + 3.6 * (double)k / (double)(M - 1), k, 0);

        CCmpndKern kernG(X);
        makeKern(kernG, X, false);
        CMatrix Xg(X), yg(y);
        CGaussianNoise gnoise(&yg);
        gnoise.setBiasVal(0.0, 0);
        CGp gp(1, D, &Xg, &yg, &kernG, &gnoise, CGp::DTCVAR, M, 0);
        gp.setActiveSet(Z);
        gp.setBiasVal(0.0, 0);
        gp.setScaleVal(1.0, 0);
        gp.setInducingFixed(false);
        gp.setBetaVal(beta);
        gp.invalidateCache();

        CCmpndKern kern(X);
        makeKern(kern, X, false);
        CSvgp sv(&kern, &X, &y, Z, beta, 0);
        sv.naturalGradientStep(1.0);
        expect(fabs(sv.elbo() - gp.logLikelihood()) < 1e-8 * max(1.0, fabs(gp.logLikelihood())),
               "Soak: ELBO at q* matches CGp DTCVAR");

        CMatrix gSv(1, sv.getOptNumParams());
        vector<unsigned int> all(N);
        for(unsigned int i = 0; i < N; i++) all[i] = i;
        sv.objectiveGradBatch(all, gSv);

        CMatrix gGp(1, gp.getOptNumParams());
        gp.logLikelihoodGradient(gGp);
        expect(gSv.getCols() == gGp.getCols(),
               "Soak: SVGP and CGp opt-param counts match (kern+Z+beta)");

        const unsigned int nK = kern.getNumParams();
        const unsigned int nZ = M; // 1-D centres
        // Remap: CGp [Z | kern | beta] vs CSvgp [kern | Z | beta]
        double maxRel = 0.0;
        for(unsigned int i = 0; i < nK; i++)
          {
            const double a = gSv.getVal(i);
            const double b = gGp.getVal(nZ + i);
            const double scale = max(1.0, max(fabs(a), fabs(b)));
            maxRel = max(maxRel, fabs(a - b) / scale);
          }
        for(unsigned int i = 0; i < nZ; i++)
          {
            const double a = gSv.getVal(nK + i);
            const double b = gGp.getVal(i);
            const double scale = max(1.0, max(fabs(a), fabs(b)));
            maxRel = max(maxRel, fabs(a - b) / scale);
          }
        {
          const double a = gSv.getVal(nK + nZ);
          const double b = gGp.getVal(nZ + nK);
          const double scale = max(1.0, max(fabs(a), fabs(b)));
          maxRel = max(maxRel, fabs(a - b) / scale);
        }
        cout << "  info: soak q* max rel grad SVGP vs CGp=" << maxRel << endl;
        expect(maxRel < 5e-5, "Soak: grads at q* match CGp DTCVAR (envelope)");
      }

      // 10. Probit natgrad: θ* − θ equals FD ∇_η ELBO on (m, S+mm^T).
      {
        const unsigned int N = 16, M = 4;
        mt19937 rng(55);
        uniform_real_distribution<double> ud(-1.5, 1.5);
        CMatrix X(N, 1), y(N, 1), Z(M, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            const double x = ud(rng);
            X.setVal(x, i, 0);
            y.setVal(x > 0.0 ? 1.0 : -1.0, i, 0);
          }
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(-1.4 + 2.8 * (double)k / (double)(M - 1), k, 0);
        CCmpndKern kern(X);
        makeKern(kern, X, false);
        CProbitNoise pnoise(&y);
        CSvgp sv(&kern, &X, &y, Z, 1.0, 0);
        sv.setLikelihood(&pnoise);
        // Warm off the prior so λ_lik is nontrivial, then freeze q for FD.
        for(unsigned int t = 0; t < 8; t++)
          sv.naturalGradientStep(0.25);

        CMatrix m0 = sv.getQMean();
        CMatrix S0 = sv.getQCov(0);
        auto natFromQS = [](const CMatrix& m, const CMatrix& S) {
          CMatrix Sc(S);
          Sc.setSymmetric(true);
          CMatrix U(Sc);
          U.chol();
          CMatrix P(S.getRows(), S.getCols());
          P.setSymmetric(true);
          P.pdinv(U);
          CMatrix theta1(S.getRows(), 1);
          theta1.gemvColCol(0, P, m, 0, 1.0, 0.0, "n");
          return make_pair(theta1, P);
        };
        auto cur = natFromQS(m0, S0);
        CMatrix mStar(m0), Sstar(S0);
        {
          // gamma=1 step yields θ* then restore.
          sv.naturalGradientStep(1.0);
          mStar = sv.getQMean();
          Sstar = sv.getQCov(0);
          vector<CMatrix> covs(1, S0);
          sv.setQ(m0, covs);
        }
        auto star = natFromQS(mStar, Sstar);
        CMatrix dTheta1(M, 1);
        for(unsigned int i = 0; i < M; i++)
          dTheta1.setVal(star.first.getVal(i) - cur.first.getVal(i), i);
        CMatrix dP(M, M);
        for(unsigned int i = 0; i < M; i++)
          for(unsigned int j = 0; j < M; j++)
            dP.setVal(star.second.getVal(i, j) - cur.second.getVal(i, j), i, j);

        // Expectation params η1=m, η2=S+mm^T. FD ∇_η L.
        CMatrix eta2(M, M);
        eta2.deepCopy(S0);
        eta2.setSymmetric(true);
        for(unsigned int i = 0; i < M; i++)
          for(unsigned int j = 0; j < M; j++)
            eta2.setVal(eta2.getVal(i, j) + m0.getVal(i) * m0.getVal(j), i, j);

        const double eps = 1e-5;
        CMatrix gEta1(M, 1);
        for(unsigned int i = 0; i < M; i++)
          {
            CMatrix mp(m0), mm(m0);
            mp.setVal(m0.getVal(i) + eps, i);
            mm.setVal(m0.getVal(i) - eps, i);
            // Hold η2 fixed: S = η2 − mm^T
            auto Sfrom = [&](const CMatrix& m) {
              CMatrix S(eta2);
              for(unsigned int a = 0; a < M; a++)
                for(unsigned int b = 0; b < M; b++)
                  S.setVal(eta2.getVal(a, b) - m.getVal(a) * m.getVal(b), a, b);
              S.setSymmetric(true);
              return S;
            };
            vector<CMatrix> cp(1, Sfrom(mp)), cm(1, Sfrom(mm));
            sv.setQ(mp, cp);
            const double Lp = sv.elbo();
            sv.setQ(mm, cm);
            const double Lm = sv.elbo();
            gEta1.setVal((Lp - Lm) / (2.0 * eps), i);
          }
        CMatrix gEta2(M, M, 0.0);
        for(unsigned int i = 0; i < M; i++)
          for(unsigned int j = i; j < M; j++)
            {
              CMatrix ep(eta2), em(eta2);
              ep.setVal(eta2.getVal(i, j) + eps, i, j);
              ep.setVal(eta2.getVal(i, j) + eps, j, i);
              em.setVal(eta2.getVal(i, j) - eps, i, j);
              em.setVal(eta2.getVal(i, j) - eps, j, i);
              auto SfromEta = [&](const CMatrix& e2) {
                CMatrix S(e2);
                for(unsigned int a = 0; a < M; a++)
                  for(unsigned int b = 0; b < M; b++)
                    S.setVal(e2.getVal(a, b) - m0.getVal(a) * m0.getVal(b), a, b);
                S.setSymmetric(true);
                return S;
              };
              vector<CMatrix> cp(1, SfromEta(ep)), cm(1, SfromEta(em));
              sv.setQ(m0, cp);
              const double Lp = sv.elbo();
              sv.setQ(m0, cm);
              const double Lm = sv.elbo();
              const double g = (Lp - Lm) / (2.0 * eps);
              gEta2.setVal(g, i, j);
              gEta2.setVal(g, j, i);
            }
        vector<CMatrix> c0(1, S0);
        sv.setQ(m0, c0);

        // Duality for packing (θ1, P=S^{-1}): ∇_η1 L = θ1*−θ1, ∇_η2 L = −½(P*−P)
        // (symmetric-matrix convention with off-diagonals counted once in the
        // Frobenius pairing). Compare θ1 closely; precision via −2 ∇_η2.
        double maxRel1 = 0.0;
        for(unsigned int i = 0; i < M; i++)
          {
            const double a = dTheta1.getVal(i);
            const double b = gEta1.getVal(i);
            const double scale = max(1.0, max(fabs(a), fabs(b)));
            maxRel1 = max(maxRel1, fabs(a - b) / scale);
          }
        double maxRel2 = 0.0;
        for(unsigned int i = 0; i < M; i++)
          for(unsigned int j = i; j < M; j++)
            {
              const double a = -0.5 * dP.getVal(i, j);
              const double b = gEta2.getVal(i, j);
              // Off-diagonal FD of a symmetric matrix element is ∂/∂η_ij with
              // η_ji linked; match the −½ P convention on the upper triangle.
              const double scale = max(1.0, max(fabs(a), fabs(b)));
              maxRel2 = max(maxRel2, fabs(a - b) / scale);
            }
        cout << "  info: soak probit natgrad maxRel θ1=" << maxRel1
             << " maxRel (−½ΔP vs ∇η2)=" << maxRel2 << endl;
        expect(maxRel1 < 5e-3, "Soak: probit natgrad θ1*−θ1 matches FD ∇_η1");
        expect(maxRel2 < 5e-2, "Soak: probit natgrad −½(P*−P) matches FD ∇_η2");
      }

      // 11. Exact probit orthant (N=2) + SVGP vs sparse PPA note on shared Gaussian.
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
        CCmpndKern kB(Xb);
        makeKern(kB, Xb, false);
        kB.setParam(1.2, 0);
        kB.setParam(1.0, 1);
        CProbitNoise pnoise(&Yb);
        CSvgp sv(&kB, &Xb, &Yb, Zb, 1.0, 0);
        sv.setLikelihood(&pnoise);
        for(unsigned int t = 0; t < 30; t++)
          sv.naturalGradientStep(0.35);
        CMatrix K(2, 2);
        kB.compute(K, Xb);
        K.setSymmetric(true);
        const double Lexact = logProbitOrthant2(K, Yb);
        const double Lsv = sv.elbo();
        cout << "  info: soak orthant ELBO=" << Lsv << " exact=" << Lexact
             << " gap=" << (Lexact - Lsv) << endl;
        expect(isfinite(Lsv) && isfinite(Lexact), "Soak: probit orthant quantities finite");
        expect(Lsv <= Lexact + 1e-5, "Soak: SVGP-probit ELBO ≤ exact orthant log marginal");

        // Shared Gaussian problem: SVGP at q* vs CSparsePpa DTCVAR ppaBound.
        const unsigned int Ng = 20, Mg = 6;
        mt19937 rng(63);
        normal_distribution<double> nd(0.0, 1.0);
        uniform_real_distribution<double> ud(-2.0, 2.0);
        CMatrix Xg(Ng, 1), yg(Ng, 1), Zg(Mg, 1);
        for(unsigned int i = 0; i < Ng; i++)
          {
            Xg.setVal(ud(rng), i, 0);
            yg.setVal(0.8 * sin(Xg.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
          }
        for(unsigned int k = 0; k < Mg; k++)
          Zg.setVal(-1.8 + 3.6 * (double)k / (double)(Mg - 1), k, 0);
        CCmpndKern kG(Xg);
        makeKern(kG, Xg, false);
        const double betaG = 20.0;
        CSvgp svg(&kG, &Xg, &yg, Zg, betaG, 0);
        svg.naturalGradientStep(1.0);
        const double LsvG = svg.elbo();
        CCmpndKern kP(Xg);
        makeKern(kP, Xg, false);
        CGaussianNoise nP(&yg);
        nP.setBiasVal(0.0, 0);
        CSparsePpa sp(Xg, yg, kP, nP, CGp::DTCVAR, Mg, betaG, 0);
        const double Lppa = sp.ppaBound();
        cout << "  info: soak shared Gaussian SVGP@q* ELBO=" << LsvG
             << " sparsePPA DTCVAR ppaBound=" << Lppa
             << " (probit orthant is SVGP-only; PPA sites are Gaussian)" << endl;
        expect(isfinite(LsvG) && isfinite(Lppa),
               "Soak: SVGP vs sparse PPA compare quantities finite");
      }

      // 12. Nested inducing sets: Z ⊂ Z′ ⇒ ELBO(Z′) ≥ ELBO(Z) at q*.
      {
        const unsigned int N = 30;
        const double beta = 15.0;
        mt19937 rng(71);
        normal_distribution<double> nd(0.0, 1.0);
        uniform_real_distribution<double> ud(-2.5, 2.5);
        CMatrix X(N, 1), y(N, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            X.setVal(ud(rng), i, 0);
            y.setVal(0.6 * X.getVal(i, 0) + 0.12 * nd(rng), i, 0);
          }
        CMatrix Z(4, 1);
        Z.setVal(-2.0, 0, 0);
        Z.setVal(-0.7, 1, 0);
        Z.setVal(0.7, 2, 0);
        Z.setVal(2.0, 3, 0);
        CMatrix Zp(7, 1);
        Zp.setVal(-2.0, 0, 0);
        Zp.setVal(-1.35, 1, 0);
        Zp.setVal(-0.7, 2, 0);
        Zp.setVal(0.0, 3, 0);
        Zp.setVal(0.7, 4, 0);
        Zp.setVal(1.35, 5, 0);
        Zp.setVal(2.0, 6, 0);
        CCmpndKern k1(X), k2(X);
        makeKern(k1, X, false);
        makeKern(k2, X, false);
        CSvgp s1(&k1, &X, &y, Z, beta, 0);
        CSvgp s2(&k2, &X, &y, Zp, beta, 0);
        s1.naturalGradientStep(1.0);
        s2.naturalGradientStep(1.0);
        const double L1 = s1.elbo();
        const double L2 = s2.elbo();
        cout << "  info: soak nested ELBO(Z)=" << L1 << " ELBO(Z')=" << L2
             << " gap=" << (L2 - L1) << endl;
        expect(L2 + 1e-6 >= L1, "Soak: nested inducing ELBO(Z') ≥ ELBO(Z)");
      }

      // 13. marginal() clamp: crowded Z still finite under near-nonneg assert.
      {
        const unsigned int N = 40, M = 8;
        const double beta = 1e5;
        CMatrix X(N, 1), y(N, 1), Z(M, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            X.setVal(-2.0 + 4.0 * (double)i / (double)(N - 1), i, 0);
            y.setVal(sin(X.getVal(i, 0)), i, 0);
          }
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(1e-8 * (double)(k + 1), k, 0); // collapsed inducing
        CCmpndKern kern(X);
        makeKern(kern, X, false);
        CSvgp sv(&kern, &X, &y, Z, beta, 0);
        bool ok = true;
        try
          {
            sv.naturalGradientStep(1.0);
            CMatrix mu, var;
            sv.latentMeanVar(mu, var, X);
            for(unsigned int i = 0; i < N; i++)
              if(!(isfinite(mu.getVal(i, 0)) && isfinite(var.getVal(i, 0))
                   && var.getVal(i, 0) >= 0.0))
                ok = false;
          }
        catch(ndlexceptions::Error&)
          {
            ok = false;
          }
          catch(std::exception&)
          {
            ok = false;
          }
        expect(ok, "Soak: crowded-Z marginal clamp keeps vars finite/nonneg");
      }

      // CIP-0035 Phase 0a: Cholesky-mode q(v) — FD, KL, Adam reaches CVI.
      {
        const unsigned int N = 24, M = 5, D = 1;
        const double beta = 20.0;
        mt19937 rng(35);
        normal_distribution<double> nd(0.0, 1.0);
        uniform_real_distribution<double> ud(-1.5, 1.5);
        CMatrix X(N, 1), y(N, D), Z(M, 1);
        for(unsigned int i = 0; i < N; i++)
          {
            X.setVal(ud(rng), i, 0);
            y.setVal(0.8 * X.getVal(i, 0) + 0.1 * nd(rng), i, 0);
          }
        for(unsigned int k = 0; k < M; k++)
          Z.setVal(-1.4 + 2.8 * (double)k / (double)(M - 1), k, 0);
        CCmpndKern kern(X);
        makeKern(kern, X, false);
        CSvgp sv(&kern, &X, &y, Z, beta, 0);
        sv.setQParamMode(CSvgp::QModeCholesky);
        expect(sv.getQParamMode() == CSvgp::QModeCholesky,
               "Phase 0a: setQParamMode(Cholesky)");
        expect(sv.getStochasticNumParams()
               == sv.getOptNumParams() + sv.getQParamNum(),
               "Phase 0a: stochastic params append q");

        // Round-trip packed q params.
        CMatrix q0;
        sv.getQParams(q0);
        CMatrix qPert(q0);
        qPert.setVal(q0.getVal(0) + 0.01, 0);
        sv.setQParams(qPert);
        CMatrix q1;
        sv.getQParams(q1);
        expect(fabs(q1.getVal(0) - qPert.getVal(0)) < 1e-14,
               "Phase 0a: setQParams/getQParams round-trip");
        sv.setQParams(q0);

        // FD on q params at fixed hypers.
        vector<unsigned int> all(N);
        for(unsigned int i = 0; i < N; i++) all[i] = i;
        CMatrix gQ;
        sv.qParamsGrad(gQ, all);
        const double eps = 1e-5;
        double maxRel = 0.0;
        for(unsigned int i = 0; i < sv.getQParamNum(); i++)
          {
            CMatrix qp(q0), qm(q0);
            qp.setVal(q0.getVal(i) + eps, i);
            qm.setVal(q0.getVal(i) - eps, i);
            sv.setQParams(qp);
            const double Lp = sv.elbo();
            sv.setQParams(qm);
            const double Lm = sv.elbo();
            sv.setQParams(q0);
            const double fd = (Lp - Lm) / (2.0 * eps);
            const double an = gQ.getVal(i);
            const double scale = max(1.0, max(fabs(fd), fabs(an)));
            maxRel = max(maxRel, fabs(fd - an) / scale);
          }
        cout << "  info: Phase 0a qParams FD max relative diff=" << maxRel << endl;
        expect(maxRel < 5e-4, "Phase 0a: qParamsGrad matches FD");

        // Adam on packed (m~, L) alone (hypers fixed) reaches near CVI ELBO.
        CSvgp ref(&kern, &X, &y, Z, beta, 0);
        ref.naturalGradientStep(1.0);
        const double Lcvi = ref.elbo();

        CSvgp adamSv(&kern, &X, &y, Z, beta, 0);
        adamSv.setQParamMode(CSvgp::QModeCholesky);
        const unsigned int nQ = adamSv.getQParamNum();
        CMatrix q;
        adamSv.getQParams(q);
        CMatrix mMom(1, nQ), vMom(1, nQ);
        mMom.zeros();
        vMom.zeros();
        const double lr = 0.1, b1 = 0.9, b2 = 0.999, epsA = 1e-8;
        double Ladam = adamSv.elbo();
        for(unsigned int t = 1; t <= 800; t++)
          {
            CMatrix gQ;
            adamSv.qParamsGrad(gQ, all);
            const double b1t = 1.0 - pow(b1, (double)t);
            const double b2t = 1.0 - pow(b2, (double)t);
            for(unsigned int i = 0; i < nQ; i++)
              {
                const double g = gQ.getVal(i);
                const double mi = b1 * mMom.getVal(i) + (1.0 - b1) * g;
                const double vi = b2 * vMom.getVal(i) + (1.0 - b2) * g * g;
                mMom.setVal(mi, i);
                vMom.setVal(vi, i);
                const double mhat = mi / b1t;
                const double vhat = vi / b2t;
                q.setVal(q.getVal(i) + lr * mhat / (sqrt(vhat) + epsA), i);
              }
            adamSv.setQParams(q);
            Ladam = adamSv.elbo();
          }
        cout << "  info: Phase 0a Adam-on-q ELBO=" << Ladam << " CVI ELBO=" << Lcvi
             << " gap=" << (Lcvi - Ladam) << endl;
        expect(Ladam > Lcvi - 1e-3,
               "Phase 0a: Adam on Cholesky q reaches near CVI ELBO");
        expect(isfinite(Ladam), "Phase 0a: Adam ELBO is finite");
      }

      // CIP-0035 Phase 0b: marginalsAt / marginalsGradAt / klQ.
      {
        auto checkLayerApi = [&](CSvgp& sv, const CMatrix& XB, const CMatrix& yIn,
                                 double betaIn, const string& kind, double tolX) {
          const unsigned int B = XB.getRows();
          const unsigned int q = XB.getCols();
          const unsigned int dOut = yIn.getCols();
          sv.setQParamMode(CSvgp::QModeCholesky);
          sv.naturalGradientStep(0.4);

          CMatrix mu, var;
          sv.marginalsAt(XB, mu, var);
          expect(mu.getRows() == B && mu.getCols() == dOut,
                 "Phase 0b: marginalsAt mean shape (" + kind + ")");
          expect(var.getRows() == B && var.getCols() == dOut,
                 "Phase 0b: marginalsAt var shape (" + kind + ")");
          bool okMarg = true;
          for(unsigned int i = 0; i < B; i++)
            for(unsigned int j = 0; j < dOut; j++)
              if(!(var.getVal(i, j) >= 0.0 && isfinite(mu.getVal(i, j))))
                okMarg = false;
          expect(okMarg, "Phase 0b: marginalsAt finite/nonneg (" + kind + ")");

          CMatrix gMean(B, dOut), gVar(B, dOut);
          gMean.ones();
          gVar.ones();
          CMatrix gKern, gZ, gQ, gX;
          sv.marginalsGradAt(XB, gMean, gVar, gKern, gZ, gQ, &gX);
          expect(gQ.getCols() == sv.getQParamNum(),
                 "Phase 0b: marginalsGradAt fills gQ (" + kind + ")");
          expect(gX.getRows() == B && gX.getCols() == q,
                 "Phase 0b: marginalsGradAt fills gX (" + kind + ")");

          const double eps = 1e-6;
          double maxRelX = 0.0;
          for(unsigned int i = 0; i < B; i++)
            for(unsigned int d = 0; d < q; d++)
              {
                CMatrix Xp(XB), Xm(XB);
                Xp.setVal(XB.getVal(i, d) + eps, i, d);
                Xm.setVal(XB.getVal(i, d) - eps, i, d);
                CMatrix mup, varp, mum, varm;
                sv.marginalsAt(Xp, mup, varp);
                sv.marginalsAt(Xm, mum, varm);
                double Lp = 0.0, Lm = 0.0;
                for(unsigned int r = 0; r < B; r++)
                  for(unsigned int j = 0; j < dOut; j++)
                    {
                      Lp += mup.getVal(r, j) + varp.getVal(r, j);
                      Lm += mum.getVal(r, j) + varm.getVal(r, j);
                    }
                const double fd = (Lp - Lm) / (2.0 * eps);
                const double an = gX.getVal(i, d);
                const double scale = max(1.0, max(fabs(fd), fabs(an)));
                maxRelX = max(maxRelX, fabs(fd - an) / scale);
              }
          cout << "  info: Phase 0b " << kind << " max|gX-FD| rel=" << maxRelX << endl;
          expect(maxRelX < tolX,
                 "Phase 0b: marginalsGradAt gX matches FD (" + kind + ")");

          vector<unsigned int> all(B);
          for(unsigned int i = 0; i < B; i++) all[i] = i;
          const double c = 1.0;
          CMatrix mu0, var0;
          sv.marginalsAt(XB, mu0, var0);
          CMatrix gMeanL(B, dOut), gVarL(B, dOut);
          gMeanL.zeros();
          gVarL.zeros();
          for(unsigned int i = 0; i < B; i++)
            for(unsigned int j = 0; j < dOut; j++)
              {
                const double yv = yIn.getVal(i, j);
                const double m = mu0.getVal(i, j);
                gMeanL.setVal(c * betaIn * (yv - m), i, j);
                gVarL.setVal(c * (-0.5 * betaIn), i, j);
              }
          CMatrix gKernM, gZM, gQM, gXM;
          sv.marginalsGradAt(XB, gMeanL, gVarL, gKernM, gZM, gQM, &gXM);
          CMatrix gKernE, gZE, gXE;
          double gBetaE = 0.0;
          sv.elboGradAt(gKernE, gZE, gBetaE, &gXE, XB, all, c);
          double maxDX = 0.0;
          for(unsigned int i = 0; i < B; i++)
            for(unsigned int d = 0; d < q; d++)
              maxDX = max(maxDX, fabs(gXM.getVal(i, d) - gXE.getVal(i, d)));
          cout << "  info: Phase 0b " << kind << " max|gX chain-elbo|=" << maxDX << endl;
          expect(maxDX < 1e-8,
                 "Phase 0b: elboGradAt gX = chained marginalsGradAt (" + kind + ")");
          double maxDK = 0.0;
          for(unsigned int i = 0; i < gKernM.getCols(); i++)
            maxDK = max(maxDK, fabs(gKernM.getVal(i) - gKernE.getVal(i)));
          cout << "  info: Phase 0b " << kind << " max|gKern chain-elbo|=" << maxDK << endl;
          expect(maxDK < 1e-6,
                 "Phase 0b: elboGradAt gKern ≈ chained marginalsGradAt (" + kind + ")");
        };

        // points: RBF+bias
        {
          const unsigned int N = 20, M = 6;
          const double beta = 12.0;
          mt19937 rng(41);
          normal_distribution<double> nd(0.0, 1.0);
          CMatrix X(N, 1), y(N, 1), Z(M, 1);
          for(unsigned int i = 0; i < N; i++)
            {
              X.setVal(-2.0 + 4.0 * (double)i / (double)(N - 1), i, 0);
              y.setVal(sin(X.getVal(i, 0)) + 0.08 * nd(rng), i, 0);
            }
          for(unsigned int k = 0; k < M; k++)
            Z.setVal(-1.6 + 3.2 * (double)k / (double)(M - 1), k, 0);
          CCmpndKern kern(X);
          makeKern(kern, X, false);
          CSvgp sv(&kern, &X, &y, Z, beta, 0);
          checkLayerApi(sv, X, y, beta, "points", 5e-4);

          // klQ value and gradient FD
          sv.setQParamMode(CSvgp::QModeCholesky);
          double klSum = 0.0;
          for(unsigned int j = 0; j < sv.getOutputDim(); j++)
            klSum += sv.klDivergence(j);
          CMatrix gKl;
          const double kl = sv.klQ(&gKl);
          expect(fabs(kl - klSum) < 1e-14, "Phase 0b: klQ equals sum klDivergence");
          expect(gKl.getCols() == sv.getQParamNum(), "Phase 0b: klQ fills gQ");
          CMatrix q0;
          sv.getQParams(q0);
          const double eps = 1e-5;
          double maxRel = 0.0;
          for(unsigned int i = 0; i < sv.getQParamNum(); i++)
            {
              CMatrix qp(q0), qm(q0);
              qp.setVal(q0.getVal(i) + eps, i);
              qm.setVal(q0.getVal(i) - eps, i);
              sv.setQParams(qp);
              const double kp = sv.klQ();
              sv.setQParams(qm);
              const double km = sv.klQ();
              sv.setQParams(q0);
              const double fd = (kp - km) / (2.0 * eps);
              const double an = gKl.getVal(i);
              const double scale = max(1.0, max(fabs(fd), fabs(an)));
              maxRel = max(maxRel, fabs(fd - an) / scale);
            }
          cout << "  info: Phase 0b klQ FD max relative diff=" << maxRel << endl;
          expect(maxRel < 5e-4, "Phase 0b: klQ gradient matches FD");
        }

        // white-noise: RBF only (no Bias)
        {
          const unsigned int N = 18, M = 5;
          const double beta = 8.0;
          CMatrix X(N, 1), y(N, 1), Z(M, 1);
          for(unsigned int i = 0; i < N; i++)
            {
              const double x = -1.5 + 3.0 * (double)i / (double)(N - 1);
              X.setVal(x, i, 0);
              y.setVal(sin(2.0 * x), i, 0);
            }
          for(unsigned int k = 0; k < M; k++)
            Z.setVal(-1.2 + 2.4 * (double)k / (double)(M - 1), k, 0);
          CCmpndKern kern(X);
          CRbfKern* prbf = new CRbfKern(X);
          prbf->setParam(1.5, 0);
          prbf->setParam(1.0, 1);
          kern.addKern(prbf);
          CSvgp sv(&kern, &X, &y, Z, beta, 0);
          sv.setInducingType("whitenoise");
          checkLayerApi(sv, X, y, beta, "whitenoise", 5e-4);
        }

        // weight-space: MLP
        {
          const unsigned int N = 14, M = 8, q = 2;
          const double beta = 4.0;
          mt19937 rng(5);
          normal_distribution<double> nd(0.0, 1.0);
          CMatrix X(N, q), y(N, 1), Z(M, q);
          for(unsigned int i = 0; i < N; i++)
            {
              for(unsigned int d = 0; d < q; d++)
                X.setVal(nd(rng), i, d);
              y.setVal(tanh(X.getVal(i, 0)) + 0.1 * nd(rng), i, 0);
            }
          for(unsigned int k = 0; k < M; k++)
            for(unsigned int d = 0; d < q; d++)
              Z.setVal(nd(rng) * 0.5, k, d);
          CMlpKern kern(X);
          kern.setParam(1.0, 0);
          kern.setParam(1.0, 1);
          kern.setParam(1.0, 2);
          CSvgp sv(&kern, &X, &y, Z, beta, 0);
          sv.setInducingType("weightspace");
          checkLayerApi(sv, X, y, beta, "weightspace", 5e-3);
        }

        // Green's: Brownian 1-D
        {
          const unsigned int N = 16, M = 6;
          const double beta = 6.0;
          CMatrix X(N, 1), y(N, 1), Z(M, 1);
          for(unsigned int i = 0; i < N; i++)
            {
              const double x = 0.05 + 0.9 * (double)i / (double)(N - 1);
              X.setVal(x, i, 0);
              y.setVal(sin(4.0 * x), i, 0);
            }
          for(unsigned int k = 0; k < M; k++)
            Z.setVal(0.1 + 0.8 * (double)k / (double)(M - 1), k, 0);
          CBrownianKern kern(X);
          kern.setParam(1.0, 0);
          CSvgp sv(&kern, &X, &y, Z, beta, 0);
          sv.setInducingType("greens");
          checkLayerApi(sv, X, y, beta, "greens", 5e-4);
        }
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
