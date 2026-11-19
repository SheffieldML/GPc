#include "CSvgp.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <cmath>
#include <iostream>

CSvgp::CSvgp(CKern* kernel, const CMatrix* X, const CMatrix* y, const CMatrix& Zin,
             double betaVal, int verbos)
  : pkern(kernel), pX(X), py(y), Z(Zin), beta(betaVal), verbosity(verbos)
{
  if(X->getRows() != y->getRows())
    throw ndlexceptions::Error("CSvgp: X and y must have the same number of rows");
  if(Zin.getCols() != X->getCols())
    throw ndlexceptions::Error("CSvgp: Z and X must have the same number of columns");
  if(betaVal <= 0.0)
    throw ndlexceptions::Error("CSvgp: beta must be positive");
  numData = X->getRows();
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

void CSvgp::updateKernelMatrices()
{
  const unsigned int M = numInducing;
  const unsigned int N = numData;
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
                             double& gMu, double& gVar) const
{
  const double r = yv - mu;
  gMu = beta * r;
  gVar = -0.5 * beta;
  return -ndlutil::HALFLOGTWOPI + 0.5 * log(beta) - 0.5 * beta * (r * r + var);
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
  vector<unsigned int> idx(numData);
  for(unsigned int i = 0; i < numData; i++)
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
  const double c = (double)numData / (double)batch.size();
  double L = 0.0;
  double gMu = 0.0, gVar = 0.0, mu = 0.0, var = 0.0;
  for(unsigned int j = 0; j < outputDim; j++)
    for(unsigned int k = 0; k < batch.size(); k++)
    {
      const unsigned int i = batch[k];
      marginal(i, j, mu, var);
      L += c * expectedLogLik(py->getVal(i, j), mu, var, gMu, gVar);
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
  const double c = (double)numData / (double)batch.size();
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
      expectedLogLik(py->getVal(i, j), mu, var, gMu, gVar);
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
