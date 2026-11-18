/* CIP-001C pilot: full SCG comparison — standard DTCVAR (point inducing)
 * vs VIK / white-noise inducing features (Álvarez et al. 2010 style).
 *
 * Same random data, same init, same free parameters (Z, γ, σ², β), s fixed.
 * Point path: CGp DTCVAR + LQ/StableFactor default.
 * VIK path: collapsed DTCVAR bound with closed-form RBF×Gaussian windows;
 *           SCG with central finite-difference gradients (pilot; analytic later).
 *
 * Build: make diagVikCompare && ./diagVikCompare [seed]
 */
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "COptimisable.h"
#include "CTransform.h"
#include "ndlexceptions.h"
#include "ndlutil.h"

using namespace std;

namespace {

const double PI = 3.14159265358979323846;

double rbfK(double x, double z, double ell, double var)
{
  const double r = x - z;
  return var * exp(-0.5 * r * r / (ell * ell));
}

void fillKuuWn(CMatrix& Kuu, const CMatrix& Z, double s)
{
  const unsigned int M = Z.getRows();
  const double s2 = s * s;
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < M; j++)
    {
      const double dz = Z.getVal(i, 0) - Z.getVal(j, 0);
      Kuu.setVal(exp(-dz * dz / (4.0 * s2)), i, j);
    }
  Kuu.setSymmetric(true);
}

void fillKfuWn(CMatrix& Kfu, const CMatrix& X, const CMatrix& Z,
               double ell, double var, double s)
{
  const unsigned int N = X.getRows();
  const unsigned int M = Z.getRows();
  const double sigma = sqrt(var);
  const double v = ell * ell + 2.0 * s * s;
  const double amp = sigma * sqrt(2.0 * sqrt(2.0) * ell * s / v);
  for(unsigned int n = 0; n < N; n++)
  {
    const double x = X.getVal(n, 0);
    for(unsigned int m = 0; m < M; m++)
    {
      const double dz = x - Z.getVal(m, 0);
      Kfu.setVal(amp * exp(-dz * dz / v), n, m);
    }
  }
}

void kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu)
{
  const unsigned int N = Kfu.getRows();
  const unsigned int M = Kfu.getCols();
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < N; j++)
      Kuf.setVal(Kfu.getVal(j, i), i, j);
}

double kappaSym(CMatrix spd)
{
  CMatrix evals(1, spd.getRows());
  spd.setSymmetric(true);
  spd.syev(evals, "n", "u");
  double mx = 0.0, mn = numeric_limits<double>::infinity();
  for(unsigned int i = 0; i < evals.getCols(); i++)
  {
    const double d = fabs(evals.getVal(i));
    mx = max(mx, d);
    if(d > 0.0)
      mn = min(mn, d);
  }
  if(!isfinite(mn) || mn <= 0.0)
    return numeric_limits<double>::infinity();
  return mx / mn;
}

struct BoundStats
{
  double ll;
  double kappaKuu;
  double meff;
  double jitter;
  bool ok;
};

BoundStats collapsedDtcvar(const CMatrix& KuuIn, const CMatrix& Kuf,
                         const CMatrix& y, double var, double beta)
{
  BoundStats r{};
  r.ll = nan("");
  r.kappaKuu = numeric_limits<double>::infinity();
  r.meff = nan("");
  r.jitter = 0.0;
  r.ok = false;

  const unsigned int M = KuuIn.getRows();
  const unsigned int N = y.getRows();
  CMatrix Kuu(KuuIn);
  r.kappaKuu = kappaSym(Kuu);

  CMatrix U(M, M);
  try
  {
    U.deepCopy(Kuu);
    U.chol();
    U.setTriangular(true);
  }
  catch(ndlexceptions::MatrixNonPosDef&)
  {
    try
    {
      CMatrix Kj(Kuu);
      r.jitter = U.jitChol(Kj);
      U.setTriangular(true);
    }
    catch(...)
    {
      return r;
    }
  }

  CMatrix B(Kuf);
  B.trsm(U, 1.0, "l", "u", "t", "n");

  CMatrix C(M, M);
  C.eye();
  C.syrk(B, beta, 1.0, "u", "n");
  C.setSymmetric(true);
  CMatrix Lc(M, M);
  try
  {
    Lc.deepCopy(C);
    Lc.chol();
    Lc.setTriangular(true);
  }
  catch(...)
  {
    return r;
  }

  CMatrix By(M, 1);
  By.gemm(B, y, beta, 0.0, "n", "n");
  By.trsm(Lc, 1.0, "l", "u", "t", "n");
  double quad = 0.0;
  for(unsigned int i = 0; i < M; i++)
  {
    const double v = By.getVal(i, 0);
    quad += v * v;
  }
  double y2 = 0.0;
  for(unsigned int i = 0; i < N; i++)
  {
    const double yi = y.getVal(i, 0);
    y2 += yi * yi;
  }

  // tr(Kff - Qff): diag Kff = var; Qff_nn = ||B[:,n]||^2
  double trGap = 0.0;
  for(unsigned int n = 0; n < N; n++)
  {
    double q = 0.0;
    for(unsigned int m = 0; m < M; m++)
    {
      const double b = B.getVal(m, n);
      q += b * b;
    }
    trGap += var - q;
  }

  // logDet(Lc) already returns log|C| = 2 Σ log L_ii (CMatrix::logDet).
  const double logDetC = logDet(Lc);
  r.ll = 0.5 * (-(double)N * log(2.0 * PI) + (double)N * log(beta)
                - logDetC - beta * y2 + quad - beta * trGap);

  CMatrix Inv(M, M);
  Inv.eye();
  Inv.trsm(Lc, 1.0, "l", "u", "t", "n");
  Inv.trsm(Lc, 1.0, "l", "u", "n", "n");
  double trInv = 0.0;
  for(unsigned int i = 0; i < M; i++)
    trInv += Inv.getVal(i, i);
  r.meff = (double)M - trInv;
  r.ok = isfinite(r.ll);
  return r;
}

BoundStats evalPointBound(const CMatrix& X, const CMatrix& y, const CMatrix& Z,
                        double gamma, double var, double beta)
{
  const double ell = 1.0 / sqrt(gamma);
  const unsigned int M = Z.getRows();
  const unsigned int N = X.getRows();
  CMatrix Kuu(M, M), Kuf(M, N);
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
      Kuu.setVal(rbfK(Z.getVal(i, 0), Z.getVal(j, 0), ell, var), i, j);
    for(unsigned int j = 0; j < N; j++)
      Kuf.setVal(rbfK(Z.getVal(i, 0), X.getVal(j, 0), ell, var), i, j);
  }
  Kuu.setSymmetric(true);
  return collapsedDtcvar(Kuu, Kuf, y, var, beta);
}

BoundStats evalVikBound(const CMatrix& X, const CMatrix& y, const CMatrix& Z,
                      double gamma, double var, double beta, double s)
{
  const double ell = 1.0 / sqrt(gamma);
  const unsigned int M = Z.getRows();
  const unsigned int N = X.getRows();
  CMatrix Kuu(M, M), Kfu(N, M), Kuf(M, N);
  fillKuuWn(Kuu, Z, s);
  fillKfuWn(Kfu, X, Z, ell, var, s);
  kufFromKfu(Kuf, Kfu);
  return collapsedDtcvar(Kuu, Kuf, y, var, beta);
}

// VIK model: free params match CGp sparse order with inducing free, no scale:
//   Z (M), γ_trans, var_trans, β_trans
class CVikDtcvar : public COptimisable
{
public:
  CVikDtcvar(const CMatrix& XIn, const CMatrix& yIn, const CMatrix& Z0,
             double sIn, double gamma0, double var0, double beta0)
    : X(XIn), y(yIn), Z(Z0), s(sIn),
      gamma(gamma0), var(var0), beta(beta0)
  {
    kernXform = new CExpTransform;
    betaXform = CTransform::defaultPositive();
    setVerbosity(1);
  }
  ~CVikDtcvar()
  {
    delete kernXform;
    delete betaXform;
  }

  unsigned int getOptNumParams() const override
  {
    return Z.getRows() + 3; // Z, γ, var, β
  }

  void getOptParams(CMatrix& param) const override
  {
    unsigned int c = 0;
    for(unsigned int i = 0; i < Z.getRows(); i++)
      param.setVal(Z.getVal(i, 0), c++);
    param.setVal(kernXform->xtoa(gamma), c++);
    param.setVal(kernXform->xtoa(var), c++);
    param.setVal(betaXform->xtoa(beta), c++);
  }

  void setOptParams(const CMatrix& param) override
  {
    unsigned int c = 0;
    for(unsigned int i = 0; i < Z.getRows(); i++)
      Z.setVal(param.getVal(c++), i, 0);
    gamma = kernXform->atox(param.getVal(c++));
    var = kernXform->atox(param.getVal(c++));
    beta = betaXform->atox(param.getVal(c++));
  }

  double computeObjectiveVal() const override
  {
    return -evalVikBound(X, y, Z, gamma, var, beta, s).ll;
  }

  double computeObjectiveGradParams(CMatrix& g) const override
  {
    // Central FD in transformed parameter space (pilot).
    const unsigned int P = getOptNumParams();
    CMatrix p(1, P);
    getOptParams(p);
    const double f0 = computeObjectiveVal();
    const double eps = 1e-5;
    for(unsigned int j = 0; j < P; j++)
    {
      CMatrix pp(p), pm(p);
      pp.setVal(p.getVal(j) + eps, j);
      pm.setVal(p.getVal(j) - eps, j);
      CVikDtcvar* self = const_cast<CVikDtcvar*>(this);
      self->setOptParams(pp);
      const double fp = self->computeObjectiveVal();
      self->setOptParams(pm);
      const double fm = self->computeObjectiveVal();
      self->setOptParams(p);
      g.setVal((fp - fm) / (2.0 * eps), j);
    }
    return f0;
  }

  BoundStats stats() const
  {
    return evalVikBound(X, y, Z, gamma, var, beta, s);
  }

  const CMatrix& inducing() const { return Z; }
  double getGamma() const { return gamma; }
  double getVar() const { return var; }
  double getBeta() const { return beta; }
  double getS() const { return s; }

private:
  CMatrix X;
  CMatrix y;
  CMatrix Z;
  double s;
  double gamma;
  double var;
  double beta;
  CTransform* kernXform;
  CTransform* betaXform;
};

void makeRandomData(CMatrix& X, CMatrix& y, CMatrix& Z,
                    unsigned int N, unsigned int M, long seed)
{
  ndlutil::init_genrand((unsigned long)seed);
  X.resize(N, 1);
  y.resize(N, 1);
  Z.resize(M, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    const double x = -2.0 + 4.0 * ndlutil::rand();
    X.setVal(x, i, 0);
    const double f = sin(2.0 * x) + 0.3 * cos(5.0 * x);
    y.setVal(f + 0.1 * ndlutil::randn(), i, 0);
  }
  // Inducing init: evenly spaced over data range (standard sparse recipe).
  double xmin = X.getVal(0, 0), xmax = X.getVal(0, 0);
  for(unsigned int i = 1; i < N; i++)
  {
    xmin = min(xmin, X.getVal(i, 0));
    xmax = max(xmax, X.getVal(i, 0));
  }
  for(unsigned int i = 0; i < M; i++)
  {
    const double z = xmin + (xmax - xmin) * (double)i / (double)(M - 1);
    Z.setVal(z, i, 0);
  }
}

double medianSpacing(const CMatrix& Z)
{
  const unsigned int M = Z.getRows();
  vector<double> zs(M);
  for(unsigned int i = 0; i < M; i++)
    zs[i] = Z.getVal(i, 0);
  sort(zs.begin(), zs.end());
  vector<double> gaps;
  for(unsigned int i = 1; i < M; i++)
    gaps.push_back(zs[i] - zs[i - 1]);
  sort(gaps.begin(), gaps.end());
  return gaps[gaps.size() / 2];
}

void printStats(const string& tag, const BoundStats& b,
                double gamma, double var, double beta, double s)
{
  cout << left << setw(8) << tag
       << " LL=" << setw(12) << setprecision(6) << b.ll
       << " γ=" << setw(10) << gamma
       << " σ²=" << setw(10) << var
       << " β=" << setw(10) << beta
       << " s=" << setw(10) << s
       << " κ(Kuu)=" << scientific << setprecision(3) << b.kappaKuu
       << " Meff=" << fixed << setprecision(2) << b.meff
       << " jit=" << b.jitter
       << (b.ok ? "" : " FAIL") << endl;
}

} // namespace

int main(int argc, char** argv)
{
  const long seed = (argc > 1) ? atol(argv[1]) : 42;
  const unsigned int N = 80;
  const unsigned int M = 15;
  const unsigned int iters = 200;
  const double gamma0 = 1.0;
  const double var0 = 1.0;
  const double beta0 = 25.0;

  CMatrix X, y, Z0;
  makeRandomData(X, y, Z0, N, M, seed);
  const double s = medianSpacing(Z0);

  cout << "==== CIP-001C: DTCVAR point vs VIK (white-noise) ====" << endl;
  cout << "N=" << N << " M=" << M << " seed=" << seed
       << " s_fixed=" << s << " SCG iters=" << iters << endl;
  cout << "Free params: Z (" << M << "), γ, σ², β.  Bias fixed at 0.\n" << endl;

  BoundStats initPt = evalPointBound(X, y, Z0, gamma0, var0, beta0);
  BoundStats initVn = evalVikBound(X, y, Z0, gamma0, var0, beta0, s);
  printStats("init/pt", initPt, gamma0, var0, beta0, 1.0 / sqrt(2.0 * gamma0));
  printStats("init/vik", initVn, gamma0, var0, beta0, s);

  // --- Point DTCVAR via CGp ---
  cout << "\n---- optimise point DTCVAR (CGp) ----" << endl;
  CRbfKern kernPt(X);
  kernPt.setParam(gamma0, 0);
  kernPt.setParam(var0, 1);
  CGaussianNoise noisePt(&y);
  noisePt.setBiasVal(0.0, 0);
  CGp gp(&kernPt, &noisePt, &X, CGp::DTCVAR, M, 0);
  gp.setOutputBiasLearnt(false);
  gp.setOutputScaleLearnt(false);
  gp.setBiasVal(0.0, 0);
  gp.setInducingFixed(false);
  gp.setActiveSet(Z0); // clears K cache; initVals may have randomised X_u
  gp.setBetaVal(beta0);
  gp.updateM();
  gp.setVerbosity(1);
  gp.setMaxIters(iters);
  gp.setObjectiveTol(1e-8);
  gp.setParamTol(1e-8);
  const double llPt0 = gp.logLikelihood();
  BoundStats initPtCgp = evalPointBound(X, y, Z0, gamma0, var0, beta0);
  cout << "  CGp vs hand LL at init: " << llPt0 << " vs " << initPtCgp.ll
       << "  (Δ=" << (llPt0 - initPtCgp.ll) << ")" << endl;
  gp.optimise(iters);
  const double llPt1 = gp.logLikelihood();

  CMatrix Zpt(gp.X_u);
  const double gammaPt = kernPt.getParam(0);
  const double varPt = kernPt.getParam(1);
  const double betaPt = gp.getBetaVal();
  BoundStats endPt = evalPointBound(X, y, Zpt, gammaPt, varPt, betaPt);
  printStats("opt/pt", endPt, gammaPt, varPt, betaPt, 1.0 / sqrt(2.0 * gammaPt));
  cout << "  CGp LL init→final: " << llPt0 << " → " << llPt1
       << "  (hand bound " << endPt.ll << ", Δ=" << (llPt1 - endPt.ll) << ")" << endl;

  // --- VIK / white-noise DTCVAR via CGp (CIP-001C Phase 2) ---
  cout << "\n---- optimise VIK / white-noise DTCVAR (CGp) ----" << endl;
  CRbfKern kernWn(X);
  kernWn.setParam(gamma0, 0);
  kernWn.setParam(var0, 1);
  CGaussianNoise noiseWn(&y);
  noiseWn.setBiasVal(0.0, 0);
  CGp gpWn(&kernWn, &noiseWn, &X, CGp::DTCVAR, M, 0);
  gpWn.setOutputBiasLearnt(false);
  gpWn.setOutputScaleLearnt(false);
  gpWn.setBiasVal(0.0, 0);
  gpWn.setInducingFixed(false);
  gpWn.setActiveSet(Z0);
  gpWn.setBetaVal(beta0);
  gpWn.setInducingType("whitenoise");
  gpWn.getInducing()->setWindowWidth(s);
  gpWn.updateM();
  gpWn.setVerbosity(1);
  gpWn.setMaxIters(iters);
  gpWn.setObjectiveTol(1e-8);
  gpWn.setParamTol(1e-8);
  const double llWn0 = gpWn.logLikelihood();
  BoundStats initVnCgp = evalVikBound(X, y, Z0, gamma0, var0, beta0, s);
  cout << "  CGp-WN vs hand LL at init: " << llWn0 << " vs " << initVnCgp.ll
       << "  (Δ=" << (llWn0 - initVnCgp.ll) << ")" << endl;
  gpWn.optimise(iters);
  const double llWn1 = gpWn.logLikelihood();

  CMatrix Zwn(gpWn.X_u);
  const double gammaWn = kernWn.getParam(0);
  const double varWn = kernWn.getParam(1);
  const double betaWn = gpWn.getBetaVal();
  const double sWn = gpWn.getInducing()->getWindowWidth();
  BoundStats endVn = evalVikBound(X, y, Zwn, gammaWn, varWn, betaWn, sWn);
  printStats("opt/vik", endVn, gammaWn, varWn, betaWn, sWn);
  cout << "  CGp-WN LL init→final: " << llWn0 << " → " << llWn1
       << "  (hand bound " << endVn.ll << ", Δ=" << (llWn1 - endVn.ll) << ")" << endl;
  cout << "  Meff (CGp)=" << gpWn.getEffectiveInducingCount() << endl;

  // Equivalence gate at s = ell/√2 on a small well-conditioned toy.
  // (Large M with moderate γ makes point Kuu ≈ singular; jitter then breaks
  // the exact algebraic identity even when Kuu_wn ≈ Kuu_pt/σ² elementwise.)
  {
    const unsigned int Ne = 8, Me = 3;
    CMatrix Xe(Ne, 1), ye(Ne, 1), Ze(Me, 1);
    for(unsigned int i = 0; i < Ne; i++)
    {
      const double xv = -1.0 + 2.0 * (double)i / (double)(Ne - 1);
      Xe.setVal(xv, i, 0);
      ye.setVal(sin(xv), i, 0);
    }
    for(unsigned int i = 0; i < Me; i++)
      Ze.setVal(Xe.getVal(i, 0), i, 0);
    CRbfKern kernEq(Xe);
    kernEq.setInverseWidth(2.0);
    kernEq.setVariance(1.5);
    CGaussianNoise noiseEqPt(&ye);
    noiseEqPt.setBiasVal(0.0, 0);
    CGaussianNoise noiseEqWn(&ye);
    noiseEqWn.setBiasVal(0.0, 0);
    CGp gpEqPt(&kernEq, &noiseEqPt, &Xe, CGp::DTCVAR, Me, 0);
    gpEqPt.setInducingVals(Ze);
    gpEqPt.setInducingFixed(true);
    gpEqPt.setBetaVal(10.0);
    gpEqPt.setInducingType("points");
    CGp gpEqWn(&kernEq, &noiseEqWn, &Xe, CGp::DTCVAR, Me, 0);
    gpEqWn.setInducingVals(Ze);
    gpEqWn.setInducingFixed(true);
    gpEqWn.setBetaVal(10.0);
    gpEqWn.setInducingType("whitenoise");
    gpEqWn.getInducing()->setWindowWidth(kernEq.getLengthScale() / sqrt(2.0));
    const double llPtEq = gpEqPt.logLikelihood();
    const double llWnEq = gpEqWn.logLikelihood();
    const double dEq = fabs(llPtEq - llWnEq);
    cout << "\n---- equivalence s=ell/√2 (N=8,M=3 toy) ----" << endl;
    cout << "  LL_pt=" << llPtEq << " LL_wn=" << llWnEq
         << " |Δ|=" << scientific << setprecision(3) << dEq << endl;
  }

  // Cross-evaluate: point bound at WN params (with point Kuu) and vice versa.
  BoundStats crossPt = evalPointBound(X, y, Zwn, gammaWn, varWn, betaWn);
  BoundStats crossVn = evalVikBound(X, y, Zpt, gammaPt, varPt, betaPt, s);
  cout << "\n---- cross-eval (same data; swap inducing family) ----" << endl;
  printStats("vik@Zpt", crossVn, gammaPt, varPt, betaPt, s);
  printStats("pt@Zvik", crossPt, gammaWn, varWn, betaWn,
             1.0 / sqrt(2.0 * gammaWn));

  cout << "\n==== summary ====" << endl;
  cout << fixed << setprecision(4);
  cout << "Point DTCVAR:  LL " << initPt.ll << " → " << endPt.ll
       << "   Δ=" << (endPt.ll - initPt.ll)
       << "   κ " << scientific << initPt.kappaKuu << " → " << endPt.kappaKuu << endl;
  cout << "VIK DTCVAR:    LL " << fixed << initVn.ll << " → " << endVn.ll
       << "   Δ=" << (endVn.ll - initVn.ll)
       << "   κ " << scientific << initVn.kappaKuu << " → " << endVn.kappaKuu << endl;
  cout << fixed << setprecision(4);
  cout << "Final LL gap (vik - point): " << (endVn.ll - endPt.ll) << endl;
  cout << "Final Meff point=" << endPt.meff << "  vik=" << endVn.meff << endl;

  return (endPt.ok && endVn.ok) ? 0 : 1;
}
