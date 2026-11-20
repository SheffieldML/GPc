/* CIP-0030 Phase 1–2: CSviBayesGplvm gates V1–V3, V5–V7 (points) and V4/V8
   (inducing families: points / whitenoise / weightspace erf+soft-ReLU / greens). */

#include "CAdam.h"
#include "CBayesGplvm.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CSviBayesGplvm.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
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

static vector<CMatrix> whitenedCovs(const CSvgp& sv, unsigned int D)
{
  vector<CMatrix> cov;
  for(unsigned int j = 0; j < D; j++)
    cov.push_back(sv.getWhitenedCov(j));
  return cov;
}

// Exact uncollapsed ELBO (Gaussian, point inducing) from psi statistics.
static double exactUncollapsed(CKern& kern, const CMatrix& Y, const CMatrix& mu,
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

  double klU = 0.0;
  for(unsigned int j = 0; j < D; j++)
  {
    const CMatrix& Sc = qCov[j];
    CMatrix cholS(Sc);
    cholS.setSymmetric(true);
    cholS.chol();
    klU += 0.5 * (Sc.trace() + qMean.dotColCol(j, qMean, j) - (double)M - logDet(cholS));
  }

  double klX = 0.0;
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < mu.getCols(); q++)
    {
      const double m = mu.getVal(n, q);
      const double s = S.getVal(n, q);
      klX += 0.5 * (m * m + s - log(s) - 1.0);
    }

  return data - klU - klX + kern.priorLogProb();
}

static void setCollapsedOptimum(CSvgp& sv, CKern& kern, const CMatrix& mu,
                                const CMatrix& S, const CMatrix& Z, double beta,
                                const CMatrix& Y)
{
  const unsigned int N = Y.getRows();
  const unsigned int D = Y.getCols();
  const unsigned int M = Z.getRows();
  CMatrix Kuu(M, M);
  kern.compute(Kuu, Z);
  Kuu.setSymmetric(true);
  CMatrix U(Kuu);
  U.jitChol(Kuu);
  CMatrix Psi1(N, M);
  kern.psi1(Psi1, mu, S, Z);
  CMatrix Psi2(M, M);
  kern.psi2(Psi2, mu, S, Z);

  CMatrix A(Psi2);
  A.scale(beta);
  A.trsm(U, 1.0, "L", "U", "T", "N");
  A.trsm(U, 1.0, "R", "U", "N", "N");
  A.setSymmetric(true);
  CMatrix P(M, M);
  P.zeros();
  P.eye();
  P.axpy(A, 1.0);
  P.setSymmetric(true);
  CMatrix cholP(P);
  cholP.chol();
  CMatrix Sstar(M, M);
  Sstar.setSymmetric(true);
  Sstar.pdinv(cholP);

  CMatrix mstar(M, D);
  for(unsigned int j = 0; j < D; j++)
  {
    CMatrix Psy(M, 1);
    Psy.zeros();
    for(unsigned int n = 0; n < N; n++)
    {
      const double yn = Y.getVal(n, j);
      for(unsigned int m = 0; m < M; m++)
        Psy.addVal(Psi1.getVal(n, m) * yn, m, 0);
    }
    Psy.trsm(U, 1.0, "L", "U", "T", "N");
    Psy.scale(beta);
    CMatrix mj(M, 1);
    mj.symv(Sstar, Psy, 1.0, 0.0, "u");
    mstar.copyColCol(j, mj, 0);
  }
  vector<CMatrix> cov(D, Sstar);
  sv.setWhitenedQ(mstar, cov);
}

static void makeRbfKern(CCmpndKern& kern, const CMatrix& Xref)
{
  CRbfKern* prbf = new CRbfKern(Xref);
  prbf->setParam(1.0, 0);
  prbf->setParam(1.0, 1);
  kern.addKern(prbf);
}

int main()
{
  try
  {
    const unsigned int N = 20, D = 3, Q = 2, M = 5;
    mt19937 rngY(7);
    normal_distribution<double> ndY(0.0, 1.0);
    CMatrix Y(N, D);
    for(unsigned int i = 0; i < N; i++)
      for(unsigned int j = 0; j < D; j++)
        Y.setVal(ndY(rngY), i, j);

    CMatrix Xdummy(N, Q);
    Xdummy.zeros();
    CCmpndKern kern(Xdummy);
    makeRbfKern(kern, Xdummy);

    CSviBayesGplvm model(&kern, Y, Q, M, 2.0, 0);
    model.setSeed(11);
    model.setNumSamples(1);
    model.setNatGradGamma(0.2);
    model.setLocalLearningRate(0.0);

    // ---- V1: S → 0 reduces to CSvgp at μ ----
    {
      CMatrix tinyS(N, Q);
      tinyS.setVals(1e-16);
      model.setS(tinyS);
      model.getSvgp().updateKernelMatrices();
      model.getSvgp().naturalGradientStep(1.0);

      CMatrix eps(N, Q);
      eps.zeros();
      model.setFrozenEps(eps);
      const double Lsvi = model.elboEstimate();
      const double Lsvgp = model.getSvgp().elbo();
      const double diff = fabs((Lsvi + model.klX()) - Lsvgp);
      cout << "  info: V1 Lsvi+KL(X)=" << (Lsvi + model.klX())
           << " Lsvgp=" << Lsvgp << " diff=" << diff << endl;
      expect(diff < 1e-6, "V1: S→0 => ELBO+KL(X) matches CSvgp at μ");
    }

    CMatrix S05(N, Q);
    S05.setVals(0.5);
    model.setS(S05);
    model.setFreezeNoise(false);
    model.getSvgp().naturalGradientStep(0.5);

    // ---- V2: MC unbiasedness vs exact psi ELBO ----
    {
      const double Lexact = exactUncollapsed(kern, Y, model.getMu(), model.getS(),
                                             model.getSvgp().getInducingCentres(),
                                             model.getSvgp().getBeta(),
                                             model.getSvgp().getWhitenedMean(),
                                             whitenedCovs(model.getSvgp(), D));
      model.setSeed(99);
      model.setNumSamples(2);
      model.setAntithetic(true);
      const unsigned int nMc = 2000;
      double sum = 0.0, sum2 = 0.0;
      for(unsigned int t = 0; t < nMc; t++)
      {
        const double Li = model.elboEstimate();
        sum += Li;
        sum2 += Li * Li;
      }
      const double mean = sum / (double)nMc;
      const double var = sum2 / (double)nMc - mean * mean;
      const double se = sqrt(max(0.0, var) / (double)nMc);
      cout << "  info: V2 exact=" << Lexact << " MC=" << mean
           << " se=" << se << " |diff|/se="
           << fabs(mean - Lexact) / max(se, 1e-12) << endl;
      expect(fabs(mean - Lexact) < 4.0 * se + 1e-4,
             "V2: MC ELBO unbiased vs psi exact (within 4 se)");
    }

    // ---- V3: collapsed q(u) optimum matches CBayesGplvm ----
    {
      CCmpndKern kernB(model.getMu());
      makeRbfKern(kernB, model.getMu());
      // Match lengthscale / variance to model kern.
      kernB.setParam(kern.getParam(0), 0);
      kernB.setParam(kern.getParam(1), 1);
      CBayesGplvm bgplvm(&kernB, Y, Q, M, 0);
      bgplvm.setLatents(model.getMu(), model.getS(), model.getSvgp().getInducingCentres());
      bgplvm.setBeta(model.getSvgp().getBeta());
      const double Lcoll = bgplvm.logLikelihood();

      setCollapsedOptimum(model.getSvgp(), kern, model.getMu(), model.getS(),
                          model.getSvgp().getInducingCentres(),
                          model.getSvgp().getBeta(), Y);
      const double Lexact = exactUncollapsed(kern, Y, model.getMu(), model.getS(),
                                             model.getSvgp().getInducingCentres(),
                                             model.getSvgp().getBeta(),
                                             model.getSvgp().getWhitenedMean(),
                                             whitenedCovs(model.getSvgp(), D));
      cout << "  info: V3 collapsed=" << Lcoll << " exact uncollapsed=" << Lexact
           << " diff=" << fabs(Lexact - Lcoll) << endl;
      expect(fabs(Lexact - Lcoll) < 1e-5,
             "V3: exact uncollapsed at q(u)* equals CBayesGplvm bound");

      model.setSeed(123);
      model.setNumSamples(4);
      model.setAntithetic(true);
      const unsigned int nMc = 1500;
      double sum = 0.0, sum2 = 0.0;
      for(unsigned int t = 0; t < nMc; t++)
      {
        const double Li = model.elboEstimate();
        sum += Li;
        sum2 += Li * Li;
      }
      const double mean = sum / (double)nMc;
      const double var = sum2 / (double)nMc - mean * mean;
      const double se = sqrt(max(0.0, var) / (double)nMc);
      cout << "  info: V3 MC at q(u)* mean=" << mean << " se=" << se << endl;
      expect(fabs(mean - Lcoll) < 4.0 * se + 1e-3,
             "V3: MC ELBO at q(u)* matches collapsed (within 4 se)");
    }

    // ---- V6: minibatch partition at fixed ε ----
    {
      model.setNumSamples(1);
      model.setAntithetic(false);
      model.setLocalLearningRate(0.0);
      CMatrix epsFull(N, Q);
      mt19937_64 rng(5);
      normal_distribution<double> nd(0.0, 1.0);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int q = 0; q < Q; q++)
          epsFull.setVal(nd(rng), i, q);
      model.setFrozenEps(epsFull);
      const double Lfull = model.elboEstimate();

      const unsigned int nPart = 4;
      double meanL = 0.0;
      for(unsigned int p = 0; p < nPart; p++)
      {
        vector<unsigned int> batch;
        for(unsigned int i = 0; i < N; i++)
          if(i % nPart == p)
            batch.push_back(i);
        CMatrix epsB(batch.size(), Q);
        for(unsigned int b = 0; b < batch.size(); b++)
          for(unsigned int q = 0; q < Q; q++)
            epsB.setVal(epsFull.getVal(batch[b], q), b, q);
        model.setFrozenEps(epsB);
        meanL += model.elboEstimate(batch) / (double)nPart;
      }
      cout << "  info: V6 full=" << Lfull << " partition mean=" << meanL
           << " diff=" << fabs(meanL - Lfull) << endl;
      expect(fabs(meanL - Lfull) < 1e-10,
             "V6: partition ELBO averages to full batch at fixed ε");
    }

    // ---- V7: seed reproducibility ----
    {
      model.setFreezeNoise(false);
      auto run = [&](uint64_t seed) {
        CCmpndKern k2(Xdummy);
        makeRbfKern(k2, Xdummy);
        CSviBayesGplvm m2(&k2, Y, Q, M, 2.0, 0);
        m2.setSeed(seed);
        m2.setNumSamples(1);
        m2.setNatGradGamma(0.15);
        m2.setLocalLearningRate(1e-2);
        CAdam adam(1e-2);
        adam.setBatchSize(N);
        adam.setMaxIters(5);
        adam.setSeed(seed);
        adam.maximise(m2);
        return m2.elboEstimate();
      };
      const double a = run(42);
      const double b = run(42);
      const double c = run(43);
      cout << "  info: V7 L(42)=" << a << " L(42)=" << b << " L(43)=" << c << endl;
      expect(fabs(a - b) < 1e-12, "V7: same seed => identical ELBO trajectory");
      expect(fabs(a - c) > 1e-8, "V7: different seed => different ELBO");
    }

    // ---- V5: FD on μ, logS at fixed ε ----
    {
      CCmpndKern k3(Xdummy);
      makeRbfKern(k3, Xdummy);
      CSviBayesGplvm m3(&k3, Y, Q, M, 2.0, 0);
      m3.setLocalLearningRate(0.0);
      m3.setNatGradGamma(0.0);
      m3.setNumSamples(1);
      CMatrix eps(N, Q);
      mt19937 rng(19);
      normal_distribution<double> nd(0.0, 1.0);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int q = 0; q < Q; q++)
          eps.setVal(nd(rng), i, q);
      m3.setFrozenEps(eps);

      const double epsFD = 1e-5;
      CMatrix gMuFd(N, Q), gLogSFd(N, Q);
      CMatrix mu0 = m3.getMu();
      CMatrix logS0 = m3.getLogS();
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int q = 0; q < Q; q++)
        {
          CMatrix muP(mu0), muM(mu0);
          muP.setVal(mu0.getVal(i, q) + epsFD, i, q);
          muM.setVal(mu0.getVal(i, q) - epsFD, i, q);
          m3.setMu(muP);
          const double Lp = m3.elboEstimate();
          m3.setMu(muM);
          const double Lm = m3.elboEstimate();
          m3.setMu(mu0);
          gMuFd.setVal((Lp - Lm) / (2.0 * epsFD), i, q);

          CMatrix lsP(logS0), lsM(logS0);
          lsP.setVal(logS0.getVal(i, q) + epsFD, i, q);
          lsM.setVal(logS0.getVal(i, q) - epsFD, i, q);
          m3.setLogS(lsP);
          const double Lp2 = m3.elboEstimate();
          m3.setLogS(lsM);
          const double Lm2 = m3.elboEstimate();
          m3.setLogS(logS0);
          gLogSFd.setVal((Lp2 - Lm2) / (2.0 * epsFD), i, q);
        }

      CMatrix XB(N, Q);
      vector<unsigned int> yRows(N);
      for(unsigned int i = 0; i < N; i++)
      {
        yRows[i] = i;
        for(unsigned int q = 0; q < Q; q++)
        {
          const double s = exp(0.5 * logS0.getVal(i, q));
          XB.setVal(mu0.getVal(i, q) + s * eps.getVal(i, q), i, q);
        }
      }
      CMatrix gK, gZ, gX;
      double gB = 0.0;
      m3.getSvgp().elboGradAt(gK, gZ, gB, &gX, XB, yRows, 1.0);
      CMatrix gMu(N, Q), gLogS(N, Q);
      gMu.zeros();
      gLogS.zeros();
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int q = 0; q < Q; q++)
        {
          const double gx = gX.getVal(i, q);
          const double s = exp(0.5 * logS0.getVal(i, q));
          const double ep = eps.getVal(i, q);
          const double S = exp(logS0.getVal(i, q));
          gMu.setVal(gx - mu0.getVal(i, q), i, q);
          gLogS.setVal(0.5 * s * gx * ep + 0.5 * (1.0 - S), i, q);
        }
      const double errMu = maxAbsDiff(gMu, gMuFd);
      const double errLS = maxAbsDiff(gLogS, gLogSFd);
      cout << "  info: V5 max|gMu-FD|=" << errMu << " max|gLogS-FD|=" << errLS << endl;
      expect(errMu < 5e-4, "V5: analytic ∂ELBO/∂μ matches FD at fixed ε");
      expect(errLS < 5e-4, "V5: analytic ∂ELBO/∂logS matches FD at fixed ε");
    }

    // ---- Phase 2: V4 FD across inducing families ----
    auto runV4 = [&](CSviBayesGplvm& m, const string& tag,
                     double tolLocal, double tolGlobal) {
      m.setLocalLearningRate(0.0);
      m.setNatGradGamma(0.0);
      m.setNumSamples(1);
      m.setAntithetic(false);
      const unsigned int Nn = m.getNumData();
      const unsigned int Qq = m.getLatentDim();
      CMatrix eps(Nn, Qq);
      mt19937 rngFd(31);
      normal_distribution<double> ndFd(0.0, 1.0);
      for(unsigned int i = 0; i < Nn; i++)
        for(unsigned int q = 0; q < Qq; q++)
          eps.setVal(ndFd(rngFd), i, q);
      m.setFrozenEps(eps);

      // Warm q(v) a little so grads are non-trivial.
      for(unsigned int t = 0; t < 3; t++)
      {
        CMatrix XB(Nn, Qq);
        vector<unsigned int> yRows(Nn);
        for(unsigned int i = 0; i < Nn; i++)
        {
          yRows[i] = i;
          for(unsigned int q = 0; q < Qq; q++)
          {
            const double s = exp(0.5 * m.getLogS().getVal(i, q));
            XB.setVal(m.getMu().getVal(i, q) + s * eps.getVal(i, q), i, q);
          }
        }
        m.getSvgp().naturalGradientStepAt(0.4, XB, yRows, 1.0);
      }

      // Locals: μ, logS via elboEstimate FD.
      const double epsFD = 1e-5;
      CMatrix mu0 = m.getMu();
      CMatrix logS0 = m.getLogS();
      CMatrix gMuFd(Nn, Qq), gLogSFd(Nn, Qq);
      for(unsigned int i = 0; i < Nn; i++)
        for(unsigned int q = 0; q < Qq; q++)
        {
          CMatrix muP(mu0), muM(mu0);
          muP.setVal(mu0.getVal(i, q) + epsFD, i, q);
          muM.setVal(mu0.getVal(i, q) - epsFD, i, q);
          m.setMu(muP);
          const double Lp = m.elboEstimate();
          m.setMu(muM);
          const double Lm = m.elboEstimate();
          m.setMu(mu0);
          gMuFd.setVal((Lp - Lm) / (2.0 * epsFD), i, q);

          CMatrix lsP(logS0), lsM(logS0);
          lsP.setVal(logS0.getVal(i, q) + epsFD, i, q);
          lsM.setVal(logS0.getVal(i, q) - epsFD, i, q);
          m.setLogS(lsP);
          const double Lp2 = m.elboEstimate();
          m.setLogS(lsM);
          const double Lm2 = m.elboEstimate();
          m.setLogS(logS0);
          gLogSFd.setVal((Lp2 - Lm2) / (2.0 * epsFD), i, q);
        }
      CMatrix XB(Nn, Qq);
      vector<unsigned int> yRows(Nn);
      for(unsigned int i = 0; i < Nn; i++)
      {
        yRows[i] = i;
        for(unsigned int q = 0; q < Qq; q++)
        {
          const double s = exp(0.5 * logS0.getVal(i, q));
          XB.setVal(mu0.getVal(i, q) + s * eps.getVal(i, q), i, q);
        }
      }
      CMatrix gK, gZ, gX;
      double gB = 0.0;
      m.getSvgp().elboGradAt(gK, gZ, gB, &gX, XB, yRows, 1.0);
      CMatrix gMu(Nn, Qq), gLogS(Nn, Qq);
      gMu.zeros();
      gLogS.zeros();
      for(unsigned int i = 0; i < Nn; i++)
        for(unsigned int q = 0; q < Qq; q++)
        {
          const double gx = gX.getVal(i, q);
          const double s = exp(0.5 * logS0.getVal(i, q));
          const double ep = eps.getVal(i, q);
          const double S = exp(logS0.getVal(i, q));
          gMu.setVal(gx - mu0.getVal(i, q), i, q);
          gLogS.setVal(0.5 * s * gx * ep + 0.5 * (1.0 - S), i, q);
        }
      const double errMu = maxAbsDiff(gMu, gMuFd);
      const double errLS = maxAbsDiff(gLogS, gLogSFd);
      cout << "  info: V4 " << tag << " max|gMu-FD|=" << errMu
           << " max|gLogS-FD|=" << errLS << endl;
      expect(errMu < tolLocal, "V4 " + tag + ": ∂ELBO/∂μ matches FD");
      expect(errLS < tolLocal, "V4 " + tag + ": ∂ELBO/∂logS matches FD");

      // Globals: kern / inducing / beta via objectiveGradBatch.
      vector<unsigned int> all(Nn);
      for(unsigned int i = 0; i < Nn; i++)
        all[i] = i;
      CMatrix g(1, m.getStochasticNumParams());
      m.objectiveGradBatch(all, g);
      CMatrix p0(1, m.getStochasticNumParams());
      m.getStochasticParams(p0);
      const double epsG = 1e-6;
      double maxRel = 0.0;
      for(unsigned int i = 0; i < m.getStochasticNumParams(); i++)
      {
        CMatrix pp(p0), pm(p0);
        pp.setVal(p0.getVal(i) + epsG, i);
        pm.setVal(p0.getVal(i) - epsG, i);
        m.setStochasticParams(pp);
        const double Lp = m.elboEstimate();
        m.setStochasticParams(pm);
        const double Lm = m.elboEstimate();
        m.setStochasticParams(p0);
        const double fd = (Lp - Lm) / (2.0 * epsG);
        const double an = g.getVal(i);
        const double scale = max(1.0, max(fabs(fd), fabs(an)));
        maxRel = max(maxRel, fabs(fd - an) / scale);
      }
      cout << "  info: V4 " << tag << " global maxRel=" << maxRel << endl;
      expect(maxRel < tolGlobal, "V4 " + tag + ": global grads match FD");
    };

    {
      // points / RBF
      CCmpndKern kPts(Xdummy);
      makeRbfKern(kPts, Xdummy);
      CSviBayesGplvm mPts(&kPts, Y, Q, M, 2.0, 0);
      mPts.setInducingType("points");
      runV4(mPts, "points", 5e-4, 2e-3);
    }
    {
      // white-noise RBF (Q=2 ok)
      CCmpndKern kWn(Xdummy);
      makeRbfKern(kWn, Xdummy);
      CSviBayesGplvm mWn(&kWn, Y, Q, M, 2.0, 0);
      mWn.setInducingType("whitenoise");
      expect(mWn.getSvgp().getInducing()->typeName() == "whitenoise",
             "V4: setInducingType whitenoise");
      runV4(mWn, "whitenoise", 5e-4, 5e-3);
    }
    {
      // weight-space MLP ridge (erf)
      CMlpKern kMlp(Xdummy);
      kMlp.setParam(1.0, 0);
      kMlp.setParam(1.0, 1);
      kMlp.setParam(1.0, 2);
      CSviBayesGplvm mWs(&kMlp, Y, Q, M, 2.0, 0);
      mWs.setInducingType("weightspace");
      expect(mWs.getSvgp().getInducing()->typeName() == "weightspace",
             "V4: setInducingType weightspace (erf)");
      runV4(mWs, "weightspace-erf", 5e-3, 1e-2);
    }
    {
      // weight-space arc-cosine order-1 (soft-ReLU)
      CArcCosineKern kArc(Xdummy);
      kArc.setOrder(1);
      kArc.setInitParam();
      CSviBayesGplvm mArc(&kArc, Y, Q, M, 2.0, 0);
      mArc.setInducingType("weightspace");
      runV4(mArc, "weightspace-softrelu", 5e-3, 1e-2);
    }
    {
      // Green's / Brownian, latent dim 1
      const unsigned int Q1 = 1;
      CMatrix X1(N, Q1);
      X1.zeros();
      CBrownianKern kGr(X1);
      kGr.setParam(1.0, 0);
      CSviBayesGplvm mGr(&kGr, Y, Q1, M, 2.0, 0);
      mGr.setInducingType("greens");
      expect(mGr.getSvgp().getInducing()->typeName() == "greens",
             "V4: setInducingType greens");
      runV4(mGr, "greens", 5e-4, 5e-3);
    }

    // ---- Phase 2: V8 end-to-end smoke (demo_bayes_gplvm-sized data) ----
    {
      const unsigned int Nd = 20, Dd = 4, Qd = 2, Md = 5;
      CMatrix Yd(Nd, Dd);
      for(unsigned int n = 0; n < Nd; n++)
        for(unsigned int d = 0; d < Dd; d++)
          Yd.setVal(sin(0.4 * n + 0.9 * d) + 0.05 * n, n, d);

      // Collapsed RBF reference.
      CMatrix Xd(Nd, Qd);
      Xd.zeros();
      CCmpndKern kColl(Xd);
      makeRbfKern(kColl, Xd);
      CBayesGplvm collapsed(&kColl, Yd, Qd, Md, 0);
      collapsed.setBeta(1.0);
      collapsed.optimise(25);
      const double Lcoll = collapsed.logLikelihood();

      // SVI at collapsed latents / inducing / beta / kern with q(u)*.
      CCmpndKern kSvi(Xd);
      makeRbfKern(kSvi, Xd);
      {
        CMatrix kp(1, kColl.getNumParams());
        kColl.getParams(kp);
        kSvi.setParams(kp);
      }
      CSviBayesGplvm svi(&kSvi, Yd, Qd, Md, collapsed.getBeta(), 0);
      svi.setMu(collapsed.getMu());
      svi.setS(collapsed.getS());
      svi.getSvgp().setInducingCentres(collapsed.getZ());
      svi.getSvgp().setBeta(collapsed.getBeta());
      setCollapsedOptimum(svi.getSvgp(), kSvi, svi.getMu(), svi.getS(),
                          svi.getSvgp().getInducingCentres(), svi.getSvgp().getBeta(),
                          Yd);
      const double Lexact = exactUncollapsed(kSvi, Yd, svi.getMu(), svi.getS(),
                                             svi.getSvgp().getInducingCentres(),
                                             svi.getSvgp().getBeta(),
                                             svi.getSvgp().getWhitenedMean(),
                                             whitenedCovs(svi.getSvgp(), Dd));
      cout << "  info: V8 exact uncollapsed=" << Lexact
           << " collapsed=" << Lcoll << " diff=" << fabs(Lexact - Lcoll) << endl;
      expect(fabs(Lexact - Lcoll) < 1e-4,
             "V8: exact uncollapsed at q(u)* equals collapsed on demo data");
      svi.setLocalLearningRate(0.0);
      svi.setNatGradGamma(0.0);
      svi.setNumSamples(4);
      svi.setAntithetic(true);
      svi.setSeed(17);
      const unsigned int nMc = 400;
      double sum = 0.0, sum2 = 0.0;
      for(unsigned int t = 0; t < nMc; t++)
      {
        const double Li = svi.elboEstimate();
        sum += Li;
        sum2 += Li * Li;
      }
      const double mean = sum / (double)nMc;
      const double var = sum2 / (double)nMc - mean * mean;
      const double se = sqrt(max(0.0, var) / (double)nMc);
      cout << "  info: V8 points MC=" << mean << " collapsed=" << Lcoll
           << " se=" << se << endl;
      expect(fabs(mean - Lcoll) < 4.0 * se + 1e-2,
             "V8: SVI RBF/points MC ELBO matches collapsed (within 4 se)");

      // Weight-space MLP: short Adam, then large-K ELBO estimate.
      CMlpKern kMlp(Xd);
      kMlp.setParam(1.0, 0);
      kMlp.setParam(1.0, 1);
      kMlp.setParam(1.0, 2);
      CSviBayesGplvm sviWs(&kMlp, Yd, Qd, Md, 1.0, 0);
      sviWs.setInducingType("weightspace");
      sviWs.setSeed(23);
      sviWs.setNumSamples(1);
      sviWs.setNatGradGamma(0.15);
      sviWs.setLocalLearningRate(1e-2);
      const double L0 = sviWs.elboEstimate();
      CAdam adam(5e-3);
      adam.setBatchSize(Nd);
      adam.setMaxIters(15);
      adam.setSeed(23);
      adam.maximise(sviWs);
      sviWs.setNumSamples(1000);
      sviWs.setSeed(101);
      const double L1 = sviWs.elboEstimate();
      cout << "  info: V8 weightspace MLP L0=" << L0 << " L_K1000=" << L1 << endl;
      expect(isfinite(L0) && isfinite(L1),
             "V8: weight-space MLP ELBO finite (K=1000 estimate)");
      expect(L1 > L0 - 5.0,
             "V8: weight-space MLP short train does not collapse ELBO");
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
