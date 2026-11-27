/* CIP-003E Phase 3: SVGP data term / CVI natural gradient / elboGrad through
   the block oracle.

   - A blockSize-2 stand-in whose block factorises must reproduce the scalar
     path exactly (ELBO, gradient, natural-gradient trajectory).
   - A genuinely coupled block (Gaussian with a full 2x2 precision, analytic
     E_q[log p] under independent q(f_1), q(f_2)) must have analytic gradients
     that match finite differences of the ELBO, and CVI steps must increase
     the ELBO.
   - A block size that does not tile the outputs is refused. */

#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CSvgp.h"
#include "ndlexceptions.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

static int fails = 0;

static void expect(bool cond, const string& msg)
{
  if(cond) cout << "PASS: " << msg << endl;
  else { cout << "FAIL: " << msg << endl; fails++; }
}

static double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
      m = max(m, fabs(A.getVal(i, j) - B.getVal(i, j)));
  return m;
}

static void makeKern(CCmpndKern& kern, const CMatrix& X)
{
  kern.addKern(new CRbfKern(X));
  kern.addKern(new CBiasKern(X));
  CMatrix p(1, kern.getNumParams());
  p.setVal(1.0 / (0.8 * 0.8), 0);
  p.setVal(1.5, 1);
  p.setVal(0.2, 2);
  kern.setParams(p);
}

// Block of 2 whose likelihood factorises: must equal the scalar path.
class CTwoBlockGaussian : public CGaussianNoise
{
 public:
  CTwoBlockGaussian(CMatrix* y) : CGaussianNoise(y) {}
  unsigned int blockSize() const { return 2; }
};

// Coupled Gaussian: log p(y|f) = -1/2 (y-f)' P (y-f) + 1/2 log|P| - log 2pi.
class CCoupledGaussian : public CGaussianNoise
{
 public:
  CCoupledGaussian(CMatrix* y, double p00, double p01, double p11)
    : CGaussianNoise(y), P00(p00), P01(p01), P11(p11) {}
  unsigned int blockSize() const { return 2; }
  double expectedLogLikelihoodBlock(const double* y, const double* mu, const double* var,
                                    unsigned int /*j0*/, double* gMu, double* gVar) const
  {
    const double r0 = y[0] - mu[0], r1 = y[1] - mu[1];
    gMu[0] = P00 * r0 + P01 * r1;
    gMu[1] = P01 * r0 + P11 * r1;
    gVar[0] = -0.5 * P00;
    gVar[1] = -0.5 * P11;
    const double quad = P00 * r0 * r0 + 2.0 * P01 * r0 * r1 + P11 * r1 * r1;
    const double logdet = log(P00 * P11 - P01 * P01);
    return -0.5 * (quad + P00 * var[0] + P11 * var[1]) + 0.5 * logdet - log(2.0 * M_PI);
  }
 private:
  double P00, P01, P11;
};

int main()
{
  try
  {
    const unsigned int N = 24, M = 6, D = 2;
    mt19937 rng(17);
    normal_distribution<double> nd(0.0, 1.0);
    CMatrix X(N, 1), Y(N, D), Z(M, 1);
    for(unsigned int i = 0; i < N; i++)
    {
      const double x = -2.0 + 4.0 * (double)i / (double)(N - 1);
      X.setVal(x, i, 0);
      Y.setVal(sin(x) + 0.2 * nd(rng), i, 0);
      Y.setVal(cos(1.3 * x) + 0.2 * nd(rng), i, 1);
    }
    for(unsigned int k = 0; k < M; k++)
      Z.setVal(-1.8 + 3.6 * (double)k / (double)(M - 1), k, 0);

    // --- Factorised block == scalar path ---
    {
      CCmpndKern k1(X), k2(X);
      makeKern(k1, X); makeKern(k2, X);
      CGaussianNoise plain(&Y);
      CTwoBlockGaussian blk(&Y);
      plain.setSigma2(0.3); blk.setSigma2(0.3);
      CSvgp a(&k1, &X, &Y, Z, 1.0, 0), b(&k2, &X, &Y, Z, 1.0, 0);
      a.setLikelihood(&plain);
      b.setLikelihood(&blk);
      expect(fabs(a.elbo() - b.elbo()) < 1e-12, "factorised block: ELBO at the prior == scalar");
      for(unsigned int t = 0; t < 5; t++)
      {
        a.naturalGradientStep(0.3);
        b.naturalGradientStep(0.3);
      }
      expect(fabs(a.elbo() - b.elbo()) < 1e-10 && maxAbsDiff(a.getQMean(), b.getQMean()) < 1e-10,
             "factorised block: natural-gradient trajectory == scalar");
      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      CMatrix ga(1, a.getOptNumParams()), gb(1, b.getOptNumParams());
      a.objectiveGradBatch(all, ga);
      b.objectiveGradBatch(all, gb);
      expect(maxAbsDiff(ga, gb) < 1e-10, "factorised block: full gradient == scalar");
      vector<unsigned int> mini(all.begin(), all.begin() + 7);
      a.objectiveGradBatch(mini, ga);
      b.objectiveGradBatch(mini, gb);
      expect(maxAbsDiff(ga, gb) < 1e-10, "factorised block: minibatch gradient == scalar");
    }

    // --- Coupled block: FD gradient check and CVI ascent ---
    {
      CCmpndKern k(X);
      makeKern(k, X);
      CCoupledGaussian cg(&Y, 3.0, 1.2, 2.5);
      CSvgp sv(&k, &X, &Y, Z, 1.0, 0);
      sv.setLikelihood(&cg);
      for(unsigned int t = 0; t < 4; t++)
        sv.naturalGradientStep(0.3);  // move q off the prior

      vector<unsigned int> all(N);
      for(unsigned int i = 0; i < N; i++) all[i] = i;
      for(int pass = 0; pass < 2; pass++)
      {
        vector<unsigned int> rows = pass == 0 ? all : vector<unsigned int>(all.begin(), all.begin() + 9);
        CMatrix g(1, sv.getOptNumParams());
        sv.objectiveGradBatch(rows, g);
        CMatrix p0(1, sv.getOptNumParams());
        sv.getOptParams(p0);
        // The minibatch objective is the c-scaled batch ELBO; the full pass is
        // elbo(). For the minibatch compare against the same objective by FD
        // of objectiveGradBatch's own value.
        const double eps = 1e-6;
        double maxRel = 0.0;
        for(unsigned int i = 0; i < sv.getOptNumParams(); i++)
        {
          CMatrix pp(p0), pm(p0), gtmp(1, sv.getOptNumParams());
          pp.setVal(p0.getVal(i) + eps, i);
          pm.setVal(p0.getVal(i) - eps, i);
          sv.setOptParams(pp);
          const double Lp = pass == 0 ? sv.elbo() : sv.objectiveGradBatch(rows, gtmp);
          sv.setOptParams(pm);
          const double Lm = pass == 0 ? sv.elbo() : sv.objectiveGradBatch(rows, gtmp);
          sv.setOptParams(p0);
          const double fd = (Lp - Lm) / (2.0 * eps);
          const double an = g.getVal(i);
          maxRel = max(maxRel, fabs(fd - an) / max(1.0, max(fabs(fd), fabs(an))));
        }
        cout << "  info: coupled block FD max relative diff (pass " << pass << ")=" << maxRel << endl;
        expect(maxRel < 5e-4, pass == 0 ? "coupled block: full-batch gradient matches finite differences"
                                         : "coupled block: minibatch gradient matches finite differences");
      }

      CCmpndKern k2(X);
      makeKern(k2, X);
      CSvgp s2(&k2, &X, &Y, Z, 1.0, 0);
      s2.setLikelihood(&cg);
      double prev = s2.elbo();
      bool monotone = true;
      for(int it = 0; it < 40; it++)
      {
        s2.naturalGradientStep(0.3);
        const double cur = s2.elbo();
        if(cur < prev - 1e-9 * max(1.0, fabs(prev))) monotone = false;
        prev = cur;
      }
      expect(monotone, "coupled block: CVI natural-gradient steps increase the ELBO");
      const double Lbefore = prev;
      s2.naturalGradientStep(0.3);
      expect(fabs(s2.elbo() - Lbefore) < 1e-5, "coupled block: CVI approaches a fixed point");
    }

    // --- Multi-class likelihoods end to end (CIP-003E Phase 4/5 coverage) ---
    {
      const unsigned int Nc = 30, C = 3;
      CMatrix Xc(Nc, 1), Yc(Nc, C), Zc(M, 1);
      Yc.zeros();
      for(unsigned int i = 0; i < Nc; i++)
      {
        const double x = -2.5 + 5.0 * (double)i / (double)(Nc - 1);
        Xc.setVal(x, i, 0);
        Yc.setVal(1.0, i, x < -0.8 ? 0 : (x < 0.9 ? 1 : 2));
      }
      for(unsigned int k = 0; k < M; k++)
        Zc.setVal(-2.2 + 4.4 * (double)k / (double)(M - 1), k, 0);
      for(int which = 0; which < 2; which++)
      {
        const string name = which == 0 ? "softmax" : "robust-max";
        CCmpndKern k(Xc);
        makeKern(k, Xc);
        CSoftmaxNoise sm(&Yc, C, 256, 4);
        CRobustMaxNoise rm(&Yc, C, 1e-2);
        CNoise* nz = which == 0 ? static_cast<CNoise*>(&sm) : static_cast<CNoise*>(&rm);
        CSvgp sv(&k, &Xc, &Yc, Zc, 1.0, 0);
        sv.setLikelihood(nz);
        for(unsigned int t = 0; t < 4; t++)
          sv.naturalGradientStep(0.3);
        vector<unsigned int> all(Nc);
        for(unsigned int i = 0; i < Nc; i++) all[i] = i;
        for(int pass = 0; pass < 2; pass++)
        {
          vector<unsigned int> rows = pass == 0 ? all : vector<unsigned int>(all.begin(), all.begin() + 11);
          CMatrix g(1, sv.getOptNumParams());
          sv.objectiveGradBatch(rows, g);
          CMatrix p0(1, sv.getOptNumParams());
          sv.getOptParams(p0);
          const double eps = 1e-6;
          double maxRel = 0.0;
          for(unsigned int i = 0; i < sv.getOptNumParams(); i++)
          {
            CMatrix pp(p0), pm(p0), gtmp(1, sv.getOptNumParams());
            pp.setVal(p0.getVal(i) + eps, i);
            pm.setVal(p0.getVal(i) - eps, i);
            sv.setOptParams(pp);
            const double Lp = pass == 0 ? sv.elbo() : sv.objectiveGradBatch(rows, gtmp);
            sv.setOptParams(pm);
            const double Lm = pass == 0 ? sv.elbo() : sv.objectiveGradBatch(rows, gtmp);
            sv.setOptParams(p0);
            const double fd = (Lp - Lm) / (2.0 * eps);
            maxRel = max(maxRel, fabs(fd - g.getVal(i)) / max(1.0, max(fabs(fd), fabs(g.getVal(i)))));
          }
          cout << "  info: " << name << " SVGP FD max relative diff (pass " << pass << ")=" << maxRel << endl;
          expect(maxRel < 5e-4, "SVGP " + name + (pass == 0 ? ": full-batch" : ": minibatch") + " gradient matches finite differences");
        }
        CCmpndKern k2(Xc);
        makeKern(k2, Xc);
        CSvgp s2(&k2, &Xc, &Yc, Zc, 1.0, 0);
        s2.setLikelihood(nz);
        const double L0 = s2.elbo();
        double prev = L0;
        bool monotone = true;
        double worstDrop = 0.0;
        for(int it = 0; it < 40; it++)
        {
          s2.naturalGradientStep(0.3);
          const double cur = s2.elbo();
          if(cur < prev - 1e-9 * max(1.0, fabs(prev))) monotone = false;
          worstDrop = max(worstDrop, prev - cur);
          prev = cur;
        }
        // Softmax is log-concave in f, so CVI ascends monotonically. Robust-max is
        // not (sites can have negative precision and the step is halved to keep q
        // proper), so only the overall improvement is required of it.
        cout << "  info: " << name << " CVI worst single-step ELBO drop " << worstDrop << endl;
        expect((which == 1 || monotone) && prev > L0,
               "SVGP " + name + ": CVI natural-gradient steps improve the ELBO (" + to_string(L0) + " -> " + to_string(prev) + ")");
      }
    }

    // --- Refusal: block size must tile the outputs ---
    {
      CMatrix Y3(N, 3);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < 3; j++) Y3.setVal(0.1 * i + j, i, j);
      CCmpndKern k(X);
      makeKern(k, X);
      CTwoBlockGaussian bad(&Y3);
      CSvgp sv(&k, &X, &Y3, Z, 1.0, 0);
      bool threw = false;
      try { sv.setLikelihood(&bad); }
      catch(ndlexceptions::Error&) { threw = true; }
      expect(threw, "setLikelihood refuses a block size that does not divide the outputs");
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
