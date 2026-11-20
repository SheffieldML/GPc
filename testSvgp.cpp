/* CIP-002A Phase 1: SVGP q(u) natural gradients, full batch, Gaussian noise.

   Checks:
   - one gamma = 1 natural-gradient step from the prior reaches the Titsias
     optimum: ELBO == CGp DTCVAR logLikelihood, predictions agree;
   - gamma < 1 steps increase the ELBO monotonically and converge to the bound;
   - M = N, Z = X (no white kernel): optimum ELBO == exact log N(y|0, K + I/beta)
     when K_uu is well conditioned, and stays a lower bound (agreeing with CGp's
     stable path) when K_uu is numerically singular;
   - minibatch estimates of the ELBO and of the natural-parameter step average
     to the full-batch values over a partition of the data.
   Each check runs with and without CWhiteKern (zero cross-covariance) and d = 2. */

#include "CAdam.h"
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CSvgp.h"
#include "ndlexceptions.h"
#include <cmath>
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
  gp.posteriorMeanVar(muG, varG, Xs);
  sv.posteriorMeanVar(muS, varS, Xs);
  // CGp::posteriorMeanVar adds observation noise 1/beta (_posteriorVar); SVGP returns latent variance.
  // Backlog 2026-11-19_latent-vs-predictive-meanvar splits these into latentMeanVar / predictiveMeanVar.
  for(unsigned int i = 0; i < Ns; i++)
    for(unsigned int j = 0; j < D; j++)
      varG.setVal(varG.getVal(i, j) - 1.0 / beta, i, j);
  const double dMu = maxAbsDiff(muG, muS);
  const double dVar = maxAbsDiff(varG, varS);
  cout << "  info: max|mean diff|=" << dMu << " max|var diff|=" << dVar << endl;
  expect(dMu < 1e-8, "predictive mean matches CGp DTCVAR" + tag);
  expect(dVar < 1e-8, "predictive latent variance matches CGp DTCVAR" + tag);

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
          threw = false;
          COrderedNoise ono(&y, 3);
          try { ono.expectedLogLikelihood(0.0, 0.0, 1.0, dgMu, dgVar); }
          catch(ndlexceptions::Error&) { threw = true; }
          expect(threw, "Phase 4: base expectedLogLikelihood throws for ordered");
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
