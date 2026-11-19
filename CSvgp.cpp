#include "CSvgp.h"
#include "CCholAdjoint.h"
#include "CInducing.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <cmath>
#include <iostream>

CSvgp::CSvgp(CKern* kernel, const CMatrix* X, const CMatrix* y, const CMatrix& Zin,
             double betaVal, int verbos)
  : pkern(kernel), pX(X), py(y), pnoise(0), Z(Zin), beta(betaVal), verbosity(verbos),
    betaTransform(CTransform::defaultPositive()), natGradGamma(0.0)
{
  if(X->getRows() != y->getRows())
    throw ndlexceptions::Error("CSvgp: X and y must have the same number of rows");
  if(Zin.getCols() != X->getCols())
    throw ndlexceptions::Error("CSvgp: Z and X must have the same number of columns");
  if(betaVal <= 0.0)
    throw ndlexceptions::Error("CSvgp: beta must be positive");
  nData = X->getRows();
  numInducing = Zin.getRows();
  outputDim = y->getCols();
  updateKernelMatrices();
  // q(u) starts at the prior: q(v) = N(0, I).
  qMean.resize(numInducing, outputDim);
  qMean.zeros();
  CMatrix I(numInducing, numInducing, 0.0);
  I.addDiag(1.0);
  I.setSymmetric(true);
  qCov.assign(outputDim, I);
}

CSvgp::~CSvgp()
{
  delete betaTransform;
}

void CSvgp::setBeta(double val)
{
  if(val <= 0.0)
    throw ndlexceptions::Error("CSvgp::setBeta: beta must be positive");
  beta = val;
}

void CSvgp::updateKernelMatrices()
{
  const unsigned int M = numInducing;
  const unsigned int N = nData;
  Kuu.resize(M, M);
  pkern->compute(Kuu, Z);
  Kuu.setSymmetric(true);
  cholKuu.resize(M, M);
  const double jit = cholKuu.jitChol(Kuu);  // as CGp::_updateInvK: modifies Kuu on jitter
  if(jit > 1e-2 && verbosity > 0)
    cout << "Warning: jitter of " << jit << " added to K_uu in CSvgp." << endl;

  Kuf.resize(M, N);
  pkern->compute(Kuf, Z, *pX);
  A.deepCopy(Kuf);
  A.trsm(cholKuu, 1.0, "L", "U", "T", "N");  // A = U^{-T} K_uf = L^{-1} K_uf

  diagK.resize(N, 1);
  diagQ.resize(N, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    diagK.setVal(pkern->diagComputeElement(*pX, i), i);
    diagQ.setVal(A.dotColCol(i, A, i), i);
  }
}

void CSvgp::setInducing(const CMatrix& Znew)
{
  if(Znew.getRows() != numInducing || Znew.getCols() != Z.getCols())
    throw ndlexceptions::Error("CSvgp::setInducing: dimension mismatch");
  Z.deepCopy(Znew);
  updateKernelMatrices();
}

void CSvgp::setWhitenedQ(const CMatrix& mean, const vector<CMatrix>& cov)
{
  if(mean.getRows() != numInducing || mean.getCols() != outputDim || cov.size() != outputDim)
    throw ndlexceptions::Error("CSvgp::setWhitenedQ: dimension mismatch");
  qMean.deepCopy(mean);
  qCov = cov;
  for(unsigned int j = 0; j < outputDim; j++)
    qCov[j].setSymmetric(true);
}

void CSvgp::setQ(const CMatrix& mean, const vector<CMatrix>& cov)
{
  if(mean.getRows() != numInducing || mean.getCols() != outputDim || cov.size() != outputDim)
    throw ndlexceptions::Error("CSvgp::setQ: dimension mismatch");
  // m~ = L^{-1} m, S~ = L^{-1} S L^{-T}.
  CMatrix mw(mean);
  mw.trsm(cholKuu, 1.0, "L", "U", "T", "N");
  vector<CMatrix> Sw(outputDim);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    CMatrix T(cov[j]);
    T.setSymmetric(false);
    T.trsm(cholKuu, 1.0, "L", "U", "T", "N");  // L^{-1} S
    T.trsm(cholKuu, 1.0, "R", "U", "N", "N");  // (L^{-1} S) U^{-1} = L^{-1} S L^{-T}
    const unsigned int M = numInducing;
    for(unsigned int r = 0; r < M; r++)
      for(unsigned int s = 0; s < r; s++)
      {
        const double v = 0.5 * (T.getVal(r, s) + T.getVal(s, r));
        T.setVal(v, r, s);
        T.setVal(v, s, r);
      }
    T.setSymmetric(true);
    Sw[j] = T;
  }
  setWhitenedQ(mw, Sw);
}

CMatrix CSvgp::getQMean() const
{
  CMatrix m(numInducing, outputDim);
  m.gemm(cholKuu, qMean, 1.0, 0.0, "t", "n");  // L m~
  return m;
}

CMatrix CSvgp::getQCov(unsigned int j) const
{
  const unsigned int M = numInducing;
  CMatrix T(M, M);
  T.gemm(cholKuu, qCov[j], 1.0, 0.0, "t", "n");  // L S~
  CMatrix S(M, M);
  S.gemm(T, cholKuu, 1.0, 0.0, "n", "n");        // L S~ L^T
  S.setSymmetric(true);
  return S;
}

double CSvgp::expectedLogLik(double yv, double mu, double var,
                             double& gMu, double& gVar, unsigned int j) const
{
  if(pnoise != 0)
  {
    if(CProbitNoise* pn = dynamic_cast<CProbitNoise*>(pnoise))
      pn->setExpectedLogLikBias(pn->getBiasVal(j));
    return pnoise->expectedLogLikelihood(yv, mu, var, gMu, gVar);
  }
  const double r = yv - mu;
  gMu = beta * r;
  gVar = -0.5 * beta;
  return -ndlutil::HALFLOGTWOPI + 0.5 * log(beta) - 0.5 * beta * (r * r + var);
}

double CSvgp::expectedLogLikBetaDeriv(double yv, double mu, double var) const
{
  if(pnoise != 0)
    return 0.0;  // β is not a free parameter under a CNoise likelihood
  const double r = yv - mu;
  return 0.5 / beta - 0.5 * (r * r + var);
}

void CSvgp::marginal(unsigned int i, unsigned int j, double& mu, double& var) const
{
  const unsigned int M = numInducing;
  CMatrix a(M, 1);
  a.copyColCol(0, A, i);
  mu = a.dotColCol(0, qMean, j);
  CMatrix Sa(M, 1);
  Sa.gemv(qCov[j], a, 1.0, 0.0, "n");
  var = diagK.getVal(i) - diagQ.getVal(i) + a.dotColCol(0, Sa, 0);
  if(var < 0.0)
    var = 0.0;  // round-off when k_ii ~ ||a_i||^2
}

vector<unsigned int> CSvgp::allIndices() const
{
  vector<unsigned int> idx(nData);
  for(unsigned int i = 0; i < nData; i++)
    idx[i] = i;
  return idx;
}

double CSvgp::klDivergence(unsigned int j) const
{
  // KL(N(m~, S~) || N(0, I)); equals KL(q(u) || p(u)) under u = L v.
  const unsigned int M = numInducing;
  const CMatrix& S = qCov[j];
  CMatrix cholS(S);
  cholS.setSymmetric(true);
  cholS.chol();
  const double logDetS = logDet(cholS);
  return 0.5 * (S.trace() + qMean.dotColCol(j, qMean, j) - (double)M - logDetS);
}

double CSvgp::dataTerm(const vector<unsigned int>& batch) const
{
  if(batch.empty())
    throw ndlexceptions::Error("CSvgp: empty batch");
  const double c = (double)nData / (double)batch.size();
  double L = 0.0;
  double gMu = 0.0, gVar = 0.0, mu = 0.0, var = 0.0;
  for(unsigned int j = 0; j < outputDim; j++)
    for(unsigned int k = 0; k < batch.size(); k++)
    {
      const unsigned int i = batch[k];
      marginal(i, j, mu, var);
      L += c * expectedLogLik(py->getVal(i, j), mu, var, gMu, gVar, j);
    }
  return L;
}

double CSvgp::elbo() const
{
  return elbo(allIndices());
}

double CSvgp::elbo(const vector<unsigned int>& batch) const
{
  double L = dataTerm(batch);
  for(unsigned int j = 0; j < outputDim; j++)
    L -= klDivergence(j);
  L += pkern->priorLogProb();
  return L;
}

void CSvgp::naturalGradientStep(double gamma)
{
  naturalGradientStep(gamma, allIndices());
}

// CVI-form natural-gradient step.  With natural parameters
// theta = (S^{-1} m, S^{-1}) the ELBO's natural gradient is
// (prior + likelihood natural parameters) - theta, so a step of size gamma is
//   theta <- (1 - gamma) theta + gamma (theta_prior + lambda_lik),
//   lambda_lik.mean      = c sum_i a_i (gMu_i - 2 gVar_i mu_i),
//   lambda_lik.precision = -2 c sum_i gVar_i a_i a_i^T,
// where (gMu, gVar) are derivatives of E_q[log p(y_i|f_i)] w.r.t. the
// marginal mean / variance of q(f_i), c = N/|batch|.  In whitened coordinates
// theta_prior = (0, I) and a_i = L^{-1} k_u(x_i).
void CSvgp::naturalGradientStep(double gamma, const vector<unsigned int>& batch)
{
  if(!(gamma > 0.0 && gamma <= 1.0))
    throw ndlexceptions::Error("CSvgp::naturalGradientStep: gamma must be in (0, 1]");
  if(batch.empty())
    throw ndlexceptions::Error("CSvgp: empty batch");
  const unsigned int M = numInducing;
  const double c = (double)nData / (double)batch.size();
  CMatrix a(M, 1);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    // Current natural parameters.
    CMatrix Scopy(qCov[j]);
    Scopy.setSymmetric(true);
    CMatrix cholS(Scopy);
    cholS.chol();
    CMatrix P(M, M);
    P.setSymmetric(true);
    P.pdinv(cholS);
    CMatrix theta1(M, 1);
    theta1.gemvColCol(0, P, qMean, j, 1.0, 0.0, "n");

    // Target natural parameters: prior plus likelihood sites.
    CMatrix Pstar(M, M, 0.0);
    Pstar.addDiag(1.0);
    CMatrix theta1star(M, 1);
    theta1star.zeros();
    double mu = 0.0, var = 0.0, gMu = 0.0, gVar = 0.0;
    for(unsigned int k = 0; k < batch.size(); k++)
    {
      const unsigned int i = batch[k];
      marginal(i, j, mu, var);
      expectedLogLik(py->getVal(i, j), mu, var, gMu, gVar, j);
      a.copyColCol(0, A, i);
      theta1star.axpy(a, c * (gMu - 2.0 * gVar * mu));
      Pstar.ger(a, a, -2.0 * c * gVar);
    }

    // Interpolate in natural-parameter space.
    P.scale(1.0 - gamma);
    P.axpy(Pstar, gamma);
    theta1.scale(1.0 - gamma);
    theta1.axpy(theta1star, gamma);
    // Symmetrise against round-off before factorising.
    for(unsigned int r = 0; r < M; r++)
      for(unsigned int s = 0; s < r; s++)
      {
        const double v = 0.5 * (P.getVal(r, s) + P.getVal(s, r));
        P.setVal(v, r, s);
        P.setVal(v, s, r);
      }
    P.setSymmetric(true);
    CMatrix cholP(P);
    cholP.chol();
    CMatrix Snew(M, M);
    Snew.setSymmetric(true);
    Snew.pdinv(cholP);
    CMatrix mNew(M, 1);
    mNew.gemv(Snew, theta1, 1.0, 0.0, "n");
    qCov[j] = Snew;
    qMean.copyColCol(j, mNew, 0);
  }
}

void CSvgp::posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xs) const
{
  const unsigned int M = numInducing;
  const unsigned int Ns = Xs.getRows();
  CMatrix Kus(M, Ns);
  pkern->compute(Kus, Z, Xs);
  CMatrix As(Kus);
  As.trsm(cholKuu, 1.0, "L", "U", "T", "N");  // L^{-1} K_us
  mu.resize(Ns, outputDim);
  varSigma.resize(Ns, outputDim);
  mu.gemm(As, qMean, 1.0, 0.0, "t", "n");
  CMatrix a(M, 1), Sa(M, 1);
  for(unsigned int i = 0; i < Ns; i++)
  {
    a.copyColCol(0, As, i);
    const double kss = pkern->diagComputeElement(Xs, i);
    const double q = As.dotColCol(i, As, i);
    for(unsigned int j = 0; j < outputDim; j++)
    {
      Sa.gemv(qCov[j], a, 1.0, 0.0, "n");
      varSigma.setVal(kss - q + a.dotColCol(0, Sa, 0), i, j);
    }
  }
}

double CSvgp::elboGrad(CMatrix& gKern, CMatrix& gZ, double& gBeta) const
{
  return elboGrad(gKern, gZ, gBeta, allIndices());
}

// Gradients of the ELBO w.r.t. kernel params, Z and beta at fixed whitened q(v).
// A = U^{-T} K_uf; mu = a^T m~; var = k_ii - ||a||^2 + a^T S~ a.
double CSvgp::elboGrad(CMatrix& gKern, CMatrix& gZ, double& gBeta,
                       const vector<unsigned int>& batch) const
{
  if(batch.empty())
    throw ndlexceptions::Error("CSvgp: empty batch");
  const unsigned int M = numInducing;
  const unsigned int N = nData;
  const unsigned int nKern = pkern->getNumParams();
  const double c = (double)N / (double)batch.size();

  CMatrix Abar(M, N);
  Abar.zeros();
  CMatrix gLambda(N, 1);
  gLambda.zeros();
  gBeta = 0.0;

  CMatrix a(M, 1), Sa(M, 1), mj(M, 1);
  double mu = 0.0, var = 0.0, gMu = 0.0, gVar = 0.0;
  for(unsigned int j = 0; j < outputDim; j++)
  {
    mj.copyColCol(0, qMean, j);
    for(unsigned int k = 0; k < batch.size(); k++)
    {
      const unsigned int i = batch[k];
      const double yv = py->getVal(i, j);
      marginal(i, j, mu, var);
      expectedLogLik(yv, mu, var, gMu, gVar, j);
      gBeta += c * expectedLogLikBetaDeriv(yv, mu, var);
      gLambda.addVal(c * gVar, i);
      a.copyColCol(0, A, i);
      Sa.gemv(qCov[j], a, 1.0, 0.0, "n");
      // ∂L/∂a = c gMu m~ + 2 c gVar (S~ a - a)
      for(unsigned int r = 0; r < M; r++)
      {
        const double da = c * gMu * mj.getVal(r)
          + 2.0 * c * gVar * (Sa.getVal(r) - a.getVal(r));
        Abar.setVal(Abar.getVal(r, i) + da, r, i);
      }
    }
  }

  CMatrix KufBar, Ubar;
  trsmAdjointUT(KufBar, Ubar, cholKuu, A, Abar);
  CMatrix KuuBar;
  cholAdjointUpperBlocked(KuuBar, cholKuu, Ubar);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < i; j++)
    {
      const double v = 0.5 * KuuBar.getVal(i, j);
      KuuBar.setVal(v, i, j);
      KuuBar.setVal(v, j, i);
    }
  KuuBar.setSymmetric(true);

  gKern.resize(1, nKern);
  gKern.zeros();
  // Point inducing routing (same as CGp sparse updateG).
  CInducingPoints inducing(const_cast<CMatrix*>(&Z));
  inducing.gradFromKuuKuf(gKern, gZ, 0, *pkern, *pX, KuuBar, KufBar, false, true);

  CMatrix tempG(1, nKern);
  pkern->getDiagGradTransParams(tempG, *pX, gLambda, false);
  for(unsigned int i = 0; i < nKern; i++)
    gKern.addVal(tempG.getVal(i), i);

  // gKern already in unconstrained space via getGradTransParams / getDiagGradTransParams.
  gBeta *= betaTransform->gradfact(beta);

  return elbo(batch);
}

unsigned int CSvgp::getOptNumParams() const
{
  return pkern->getNumParams() + numInducing * getInputDim() + 1;
}

void CSvgp::getOptParams(CMatrix& param) const
{
  const unsigned int P = getOptNumParams();
  param.resize(1, P);
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    param.setVal(pkern->getTransParam(i), counter++);
  for(unsigned int j = 0; j < getInputDim(); j++)
    for(unsigned int i = 0; i < numInducing; i++)
      param.setVal(Z.getVal(i, j), counter++);
  param.setVal(betaTransform->xtoa(beta), counter++);
}

void CSvgp::setOptParams(const CMatrix& param)
{
  DIMENSIONMATCH(param.getRows() == 1 && param.getCols() == getOptNumParams());
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    pkern->setTransParam(param.getVal(counter++), i);
  for(unsigned int j = 0; j < getInputDim(); j++)
    for(unsigned int i = 0; i < numInducing; i++)
      Z.setVal(param.getVal(counter++), i, j);
  beta = betaTransform->atox(param.getVal(counter++));
  updateKernelMatrices();
}

double CSvgp::objectiveGradBatch(const vector<unsigned int>& batch, CMatrix& g)
{
  CMatrix gKern(1, pkern->getNumParams());
  CMatrix gZ(numInducing, getInputDim());
  double gBeta = 0.0;
  const double L = elboGrad(gKern, gZ, gBeta, batch);
  g.resize(1, getOptNumParams());
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    g.setVal(gKern.getVal(i), counter++);
  for(unsigned int j = 0; j < getInputDim(); j++)
    for(unsigned int i = 0; i < numInducing; i++)
      g.setVal(gZ.getVal(i, j), counter++);
  g.setVal(gBeta, counter++);
  return L;
}

void CSvgp::afterBatchStep(const vector<unsigned int>& batch)
{
  if(natGradGamma > 0.0)
    naturalGradientStep(natGradGamma, batch);
}
