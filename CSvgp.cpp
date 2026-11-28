#include "CSvgp.h"
#include "CCholAdjoint.h"
#include "CInducing.h"
#include "ndlassert.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include "ndlstrutil.h"
#include <cmath>
#include <iostream>

// CIP-002A numerics soak: round-off may make latent var slightly negative; clamp
// only after asserting the dip is tiny relative to k(x,x).
static void clampLatentVar(double& v, double kxx)
{
  if(v < 0.0)
  {
    const double scale = (kxx > 0.0) ? kxx : 1.0;
    SANITYCHECK(v >= -1e-10 * scale);
    v = 0.0;
  }
}

CSvgp::CSvgp(CKern* kernel, const CMatrix* X, const CMatrix* y, const CMatrix& Zin,
             double betaVal, int verbos)
  : pkern(kernel), pX(X), py(y), pnoise(0), Z(Zin), pInducing(0), beta(betaVal),
    verbosity(verbos), betaTransform(CTransform::defaultPositive()), natGradGamma(0.0),
    dropResidual(false), qParamMode(QModeCvi), lazyFullData(true), fullDataValid(false), legacyPaths(false)
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
  pInducing = new CInducingPoints(&Z);
  updateKernelMatrices();
  initVariationalPrior();
}

CSvgp::~CSvgp()
{
  delete pInducing;
  delete betaTransform;
}

void CSvgp::initVariationalPrior()
{
  qMean.resize(numInducing, outputDim);
  qMean.zeros();
  CMatrix I(numInducing, numInducing, 0.0);
  I.addDiag(1.0);
  I.setSymmetric(true);
  qCov.assign(outputDim, I);
  if(qParamMode == QModeCholesky)
    syncCholFromCov();
}

void CSvgp::setInducing(CInducing* pInd)
{
  if(pInd == 0)
    throw ndlexceptions::Error("CSvgp::setInducing requires a non-null inducing object");
  delete pInducing;
  pInducing = pInd;
  const unsigned int Mnew = pInducing->numInducing();
  if(Mnew != numInducing)
  {
    numInducing = Mnew;
    initVariationalPrior();
  }
  updateKernelMatrices();
}

vector<string> CSvgp::hermiteWarnings() const
{
  const CInducingHermite* ph = dynamic_cast<const CInducingHermite*>(pInducing);
  return ph ? ph->warnings(*pkern, *pX) : vector<string>();
}

string CSvgp::hermiteLayoutDescription() const
{
  const CInducingHermite* ph = dynamic_cast<const CInducingHermite*>(pInducing);
  return ph ? ph->describeLayout(*pkern, *pX) : string();
}

void CSvgp::setInducingType(const string& typeName)
{
  if(typeName == "points")
  {
    setInducing(new CInducingPoints(&Z));
  }
  else if(typeName == "whitenoise")
  {
    double s0 = CInducingWhiteNoise::medianSpacing(Z);
    if(!(s0 > 0.0) || !isfinite(s0))
      s0 = 1.0;
    setInducing(new CInducingWhiteNoise(&Z, s0, false));
  }
  else if(typeName == "greens")
  {
    double s0 = CInducingWhiteNoise::medianSpacing(Z);
    if(!(s0 > 0.0) || !isfinite(s0))
      s0 = 1.0;
    setInducing(new CInducingGreens(&Z, s0, false));
  }
  else if(typeName == "patches")
  {
    // CIP-0040: M patches drawn evenly from the training images.
    const CConvKern* ck = dynamic_cast<const CConvKern*>(pkern);
    if(ck == 0)
      throw ndlexceptions::Error("CSvgp::setInducingType(\"patches\") requires a convolutional kernel.");
    CMatrix Zp;
    CInducingPatches::initFromData(Zp, *ck, *pX, numInducing);
    setInducing(new CInducingPatches(Zp));
  }
  else if(typeName == "weightspace")
  {
    CMatrix W;
    const unsigned int d = pkern->wsDomainDim();
    const double s0 = CInducingWeightSpace::makeRidgeInit(W, numInducing, d, 7u);
    setInducing(new CInducingWeightSpace(W, s0, false, true));
  }
  else if(typeName == "hermite")
  {
    // CIP-003D: fixed Hermite ridge windows. E2 / E1 / E5 raise before anything changes.
    const unsigned int d = pkern->wsDomainDim();
    CInducingHermite::checkKernelSupport(*pkern, d);
    CMatrix V;
    vector<unsigned int> ks;
    CInducingHermite::makeLayout(V, ks, numInducing, d);
    setInducing(new CInducingHermite(V, ks));
  }
  else
    throw ndlexceptions::Error("CSvgp::setInducingType: unknown type \"" + typeName + "\"");
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
  Kuu.resize(M, M);
  pInducing->computeKuu(Kuu, *pkern);
  Kuu.setSymmetric(true);
  cholKuu.resize(M, M);
  const double jit = cholKuu.jitChol(Kuu);  // as CGp::_updateInvK: modifies Kuu on jitter
  if(jit > 1e-2 && verbosity > 0)
    cout << "Warning: jitter of " << jit << " added to K_uu in CSvgp." << endl;

  fullDataValid = false;
  if(!lazyFullData)
    ensureFullData();
}

void CSvgp::setLazyFullData(bool val)
{
  lazyFullData = val;
  if(!val)
    ensureFullData();
}

void CSvgp::ensureFullData() const
{
  if(fullDataValid)
    return;
  const unsigned int M = numInducing;
  const unsigned int N = nData;
  Kuf.resize(M, N);
  pInducing->computeKuf(Kuf, *pkern, *pX);
  A.deepCopy(Kuf);
  A.trsm(cholKuu, 1.0, "L", "U", "T", "N");  // A = U^{-T} K_uf = L^{-1} K_uf

  diagK.resize(N, 1);
  diagQ.resize(N, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    diagK.setVal(pkern->diagComputeElement(*pX, i), i);
    diagQ.setVal(A.dotColCol(i, A, i), i);
  }
  fullDataValid = true;
}

void CSvgp::setInducingCentres(const CMatrix& Znew)
{
  pInducing->setCentres(Znew);
  if(pInducing->typeName() != "weightspace" && pInducing->typeName() != "patches")
  {
    // Points / WN / greens share member Z via non-owned pointer.
    if(Znew.getRows() != Z.getRows() || Znew.getCols() != Z.getCols())
      Z.resize(Znew.getRows(), Znew.getCols());
    Z.deepCopy(Znew);
  }
  if(pInducing->numInducing() != numInducing)
  {
    numInducing = pInducing->numInducing();
    initVariationalPrior();
  }
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
  if(qParamMode == QModeCholesky)
    syncCholFromCov();
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

void CSvgp::setLikelihood(CNoise* noise)
{
  if(noise != 0)
  {
    // CIP-003E: refuse a likelihood that cannot supply E_q[log p] gradients,
    // or whose blocks do not tile the output columns.
    noise->requireCapability(CNoise::CapExpectedLogLik, "CSvgp::setLikelihood");
    if(outputDim % noise->blockSize() != 0)
      throw ndlexceptions::Error("CSvgp::setLikelihood: likelihood couples blocks of "
                                 + ndlstrutil::itoa(noise->blockSize())
                                 + " outputs but the model has "
                                 + ndlstrutil::itoa(outputDim) + ".");
  }
  pnoise = noise;
}

double CSvgp::expectedLogLik(double yv, double mu, double var,
                             double& gMu, double& gVar, unsigned int j) const
{
  if(pnoise != 0)
  {
    if(CProbitNoise* pn = dynamic_cast<CProbitNoise*>(pnoise))
      pn->setExpectedLogLikBias(pn->getBiasVal(j));
    else if(COrderedNoise* on = dynamic_cast<COrderedNoise*>(pnoise))
      on->setExpectedLogLikBias(on->getBiasVal(j));
    return pnoise->expectedLogLikelihood(yv, mu, var, gMu, gVar);
  }
  const double r = yv - mu;
  gMu = beta * r;
  gVar = -0.5 * beta;
  return -ndlutil::HALFLOGTWOPI + 0.5 * log(beta) - 0.5 * beta * (r * r + var);
}

// CIP-003E: E_q[log p(y_i | f_i)] for every row of a batch, with the gradients
// w.r.t. the marginal means/variances of all latents. Scalar-block likelihoods
// (and the Gaussian default) keep the per-element path in its original order;
// blockSize() > 1 goes through the block oracle, the K latents of a datum being
// independent under q. Returns sum over rows/latents of c * E[log p].
double CSvgp::expectedLogLikBatch(const vector<unsigned int>& rows,
                                  const CMatrix& mu, const CMatrix& var, double c,
                                  CMatrix& gMuB, CMatrix& gVarB) const
{
  const unsigned int B = rows.size();
  gMuB.resize(B, outputDim);
  gVarB.resize(B, outputDim);
  double L = 0.0;
  double gMu = 0.0, gVar = 0.0;
  const unsigned int bs = (pnoise != 0) ? pnoise->blockSize() : 1;
  if(bs == 1)
  {
    for(unsigned int j = 0; j < outputDim; j++)
      for(unsigned int k = 0; k < B; k++)
      {
        L += c * expectedLogLik(py->getVal(rows[k], j), mu.getVal(k, j),
                                var.getVal(k, j), gMu, gVar, j);
        gMuB.setVal(gMu, k, j);
        gVarB.setVal(gVar, k, j);
      }
    return L;
  }
  vector<double> by(bs), bm(bs), bv(bs), bgm(bs), bgv(bs);
  for(unsigned int k = 0; k < B; k++)
    for(unsigned int j0 = 0; j0 < outputDim; j0 += bs)
    {
      for(unsigned int b = 0; b < bs; b++)
      {
        by[b] = py->getVal(rows[k], j0 + b);
        bm[b] = mu.getVal(k, j0 + b);
        bv[b] = var.getVal(k, j0 + b);
      }
      L += c * pnoise->expectedLogLikelihoodBlock(by.data(), bm.data(), bv.data(), j0,
                                                  bgm.data(), bgv.data());
      for(unsigned int b = 0; b < bs; b++)
      {
        gMuB.setVal(bgm[b], k, j0 + b);
        gVarB.setVal(bgv[b], k, j0 + b);
      }
    }
  return L;
}

double CSvgp::expectedLogLikBetaDeriv(double yv, double mu, double var) const
{
  if(pnoise != 0)
    return 0.0;  // β is not a free parameter under a CNoise likelihood
  const double r = yv - mu;
  return 0.5 / beta - 0.5 * (r * r + var);
}

bool CSvgp::isFullOrderedBatch(const vector<unsigned int>& batch) const
{
  if(batch.size() != nData)
    return false;
  for(unsigned int k = 0; k < nData; k++)
    if(batch[k] != k)
      return false;
  return true;
}

void CSvgp::gatherXB(const vector<unsigned int>& batch, CMatrix& XB) const
{
  const unsigned int B = batch.size();
  const unsigned int q = getInputDim();
  XB.resize(B, q);
  for(unsigned int k = 0; k < B; k++)
  {
    const unsigned int i = batch[k];
    for(unsigned int d = 0; d < q; d++)
      XB.setVal(pX->getVal(i, d), k, d);
  }
}

void CSvgp::formABFromX(const CMatrix& XB, CMatrix& AB, CMatrix& dK,
                        CMatrix& dQ) const
{
  const unsigned int M = numInducing;
  const unsigned int B = XB.getRows();
  AB.resize(M, B);
  dK.resize(B, 1);
  dQ.resize(B, 1);
  pInducing->computeKuf(AB, *pkern, XB);
  AB.trsm(cholKuu, 1.0, "L", "U", "T", "N");
  for(unsigned int k = 0; k < B; k++)
  {
    dK.setVal(pkern->diagComputeElement(XB, k), k);
    dQ.setVal(AB.dotColCol(k, AB, k), k);
  }
}

void CSvgp::formAB(const vector<unsigned int>& batch, CMatrix& AB,
                   CMatrix& dK, CMatrix& dQ) const
{
  const unsigned int M = numInducing;
  const unsigned int B = batch.size();
  AB.resize(M, B);
  dK.resize(B, 1);
  dQ.resize(B, 1);

  // Small minibatch: form K_uB and trsm rather than gathering from full A (CIP-002D QW6).
  if(B * 4 < nData)
  {
    gatherXB(batch, wsXB);
    formABFromX(wsXB, AB, dK, dQ);
    return;
  }

  ensureFullData();
  for(unsigned int k = 0; k < B; k++)
  {
    const unsigned int i = batch[k];
    AB.copyColCol(k, A, i);
    dK.setVal(diagK.getVal(i), k);
    dQ.setVal(diagQ.getVal(i), k);
  }
}

void CSvgp::fillMarginalsFromX(const CMatrix& XB, CMatrix& muOut,
                               CMatrix& varOut) const
{
  const unsigned int M = numInducing;
  const unsigned int B = XB.getRows();
  formABFromX(XB, wsAB, wsDK, wsDQ);
  muOut.resize(B, outputDim);
  varOut.resize(B, outputDim);
  muOut.gemm(wsAB, qMean, 1.0, 0.0, "t", "n");
  for(unsigned int j = 0; j < outputDim; j++)
  {
    wsSAB.resize(M, B);
    wsSAB.symm(qCov[j], wsAB, 1.0, 0.0, "u", "l");
    for(unsigned int k = 0; k < B; k++)
    {
      double v = wsAB.dotColCol(k, wsSAB, k);
      if(!dropResidual)
        v += wsDK.getVal(k) - wsDQ.getVal(k);
      if(!pkern->diagIsEstimated())
        clampLatentVar(v, wsDK.getVal(k));
      varOut.setVal(v, k, j);
    }
  }
}

void CSvgp::applyWhitenedAt(const CMatrix& XB, const CMatrix& vWhitened,
                            CMatrix& fOut) const
{
  if(vWhitened.getRows() != numInducing || vWhitened.getCols() != outputDim)
    throw ndlexceptions::Error(
        "CSvgp::applyWhitenedAt: vWhitened must be M x outputDim");
  formABFromX(XB, wsAB, wsDK, wsDQ);
  fOut.resize(XB.getRows(), outputDim);
  fOut.gemm(wsAB, vWhitened, 1.0, 0.0, "t", "n");
}

void CSvgp::fillMarginals(const vector<unsigned int>& batch,
                          CMatrix& muOut, CMatrix& varOut) const
{
  const unsigned int M = numInducing;
  const unsigned int B = batch.size();
  muOut.resize(B, outputDim);
  varOut.resize(B, outputDim);

  const CMatrix* pA = &A;
  const CMatrix* pDK = &diagK;
  const CMatrix* pDQ = &diagQ;
  if(isFullOrderedBatch(batch))
    ensureFullData();
  else
  {
    formAB(batch, wsAB, wsDK, wsDQ);
    pA = &wsAB;
    pDK = &wsDK;
    pDQ = &wsDQ;
  }

  // mu = A_B^T m~   (B x d)
  muOut.gemm(*pA, qMean, 1.0, 0.0, "t", "n");

  for(unsigned int j = 0; j < outputDim; j++)
  {
    // S~ A_B  then column dots with A_B
    wsSAB.resize(M, B);
    wsSAB.symm(qCov[j], *pA, 1.0, 0.0, "u", "l");
    for(unsigned int k = 0; k < B; k++)
    {
      double v = pA->dotColCol(k, wsSAB, k);
      if(!dropResidual)
        v += pDK->getVal(k) - pDQ->getVal(k);
      if(!pkern->diagIsEstimated())
        clampLatentVar(v, pDK->getVal(k));
      varOut.setVal(v, k, j);
    }
  }
}

void CSvgp::marginal(unsigned int i, unsigned int j, double& mu, double& var) const
{
  vector<unsigned int> one(1, i);
  fillMarginals(one, wsMu, wsVar);
  mu = wsMu.getVal(0, j);
  var = wsVar.getVal(0, j);
}

vector<unsigned int> CSvgp::allIndices() const
{
  vector<unsigned int> idx(nData);
  for(unsigned int i = 0; i < nData; i++)
    idx[i] = i;
  return idx;
}

// Upper factor U with S~_j = U^T U. In Cholesky mode this is qChol[j]^T exactly;
// re-factorising S~ = L L^T can fail numerically when L has a small diagonal
// and larger off-diagonals (it happens within a few Adam steps on hidden layers).
void CSvgp::upperCholOfS(unsigned int j, CMatrix& U) const
{
  const unsigned int M = numInducing;
  if(qParamMode == QModeCholesky && j < qChol.size() && qChol[j].getRows() == M)
  {
    U.resize(M, M);
    U.zeros();
    for(unsigned int r = 0; r < M; r++)
      for(unsigned int c = 0; c <= r; c++)
        U.setVal(qChol[j].getVal(r, c), c, r);
    // pdinv / logDet require the triangular property flag (CHECKS=full).
    U.setTriangular(true);
    return;
  }
  U.deepCopy(qCov[j]);
  U.setSymmetric(true);
  U.chol();
}

double CSvgp::klDivergence(unsigned int j) const
{
  // KL(N(m~, S~) || N(0, I)); equals KL(q(u) || p(u)) under u = L v.
  const unsigned int M = numInducing;
  const CMatrix& S = qCov[j];
  CMatrix cholS;
  upperCholOfS(j, cholS);
  const double logDetS = logDet(cholS);
  return 0.5 * (S.trace() + qMean.dotColCol(j, qMean, j) - (double)M - logDetS);
}

double CSvgp::dataTermAt(const CMatrix& XB, const vector<unsigned int>& yRows,
                         double c) const
{
  if(yRows.empty() || XB.getRows() != yRows.size())
    throw ndlexceptions::Error("CSvgp::dataTermAt: XB rows must match yRows");
  if(!(c > 0.0))
    throw ndlexceptions::Error("CSvgp::dataTermAt: c must be positive");
  fillMarginalsFromX(XB, wsMu, wsVar);
  return expectedLogLikBatch(yRows, wsMu, wsVar, c, wsGMu, wsGVar);
}

double CSvgp::dataTerm(const vector<unsigned int>& batch) const
{
  if(batch.empty())
    throw ndlexceptions::Error("CSvgp: empty batch");
  const double c = (double)nData / (double)batch.size();
  // Fast path via cached A when full batch; otherwise equivalent to dataTermAt.
  if(isFullOrderedBatch(batch))
  {
    fillMarginals(batch, wsMu, wsVar);
    return expectedLogLikBatch(batch, wsMu, wsVar, c, wsGMu, wsGVar);
  }
  gatherXB(batch, wsXB);
  return dataTermAt(wsXB, batch, c);
}

double CSvgp::klPlusPrior() const
{
  double L = 0.0;
  for(unsigned int j = 0; j < outputDim; j++)
    L -= klDivergence(j);
  L += pkern->priorLogProb();
  return L;
}

double CSvgp::elbo() const
{
  return elbo(allIndices());
}

double CSvgp::elbo(const vector<unsigned int>& batch) const
{
  return dataTerm(batch) + klPlusPrior();
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
  const unsigned int B = batch.size();
  const double c = (double)nData / (double)B;

  fillMarginals(batch, wsMu, wsVar);
  const CMatrix* pA = &A;
  if(!isFullOrderedBatch(batch))
  {
    formAB(batch, wsAB, wsDK, wsDQ);
    pA = &wsAB;
  }

  expectedLogLikBatch(batch, wsMu, wsVar, 1.0, wsGMu, wsGVar);
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

    // Target: prior I + sum_k (-2 c gVar) a a^T; theta1 via gemv.
    // Prefer scaled syrk when all site weights are non-negative (Gaussian);
    // otherwise fall back to per-column ger (probit sites can flip sign).
    CMatrix Pstar(M, M, 0.0);
    Pstar.setSymmetric(true);
    wsCoeffs.resize(B, 1);
    CMatrix alphas(B, 1);
    double gMu = 0.0, gVar = 0.0;
    bool allNonNeg = true;
    for(unsigned int k = 0; k < B; k++)
    {
      const unsigned int i = batch[k];
      const double mu = wsMu.getVal(k, j);
      gMu = wsGMu.getVal(k, j); gVar = wsGVar.getVal(k, j);
      wsCoeffs.setVal(c * (gMu - 2.0 * gVar * mu), k);
      const double alpha = -2.0 * c * gVar;
      alphas.setVal(alpha, k);
      if(alpha < 0.0)
        allNonNeg = false;
    }
    if(allNonNeg)
    {
      wsABS.deepCopy(*pA);
      for(unsigned int k = 0; k < B; k++)
        wsABS.scaleCol(k, sqrt(alphas.getVal(k)));
      // beta=0 so syrk may start from a non-flagged matrix; then add I.
      Pstar.syrk(wsABS, 1.0, 0.0, "u", "n");
      for(unsigned int r = 0; r < M; r++)
        for(unsigned int s = 0; s < r; s++)
          Pstar.setVal(Pstar.getVal(s, r), r, s);
      Pstar.setSymmetric(true);
      Pstar.addDiag(1.0);
    }
    else
    {
      Pstar.addDiag(1.0);
      CMatrix a(M, 1);
      for(unsigned int k = 0; k < B; k++)
      {
        a.copyColCol(0, *pA, k);
        Pstar.ger(a, a, alphas.getVal(k));
      }
    }
    CMatrix theta1star(M, 1);
    theta1star.gemv(*pA, wsCoeffs, 1.0, 0.0, "n");

    // Interpolate in natural-parameter space. With sites of either sign (probit,
    // robust-max) the full target can be indefinite, so the step is halved until
    // the interpolated precision is positive definite (the first attempt is the
    // plain step, so log-concave likelihoods are unchanged). If no step works q
    // is left as it is.
    CMatrix P0(P), theta0(theta1), cholP(M, M);
    bool pd = false;
    double stepG = gamma;
    for(unsigned int tries = 0; tries < 40 && !pd; tries++)
    {
      P.deepCopy(P0);
      theta1.deepCopy(theta0);
      P.scale(1.0 - stepG);
      P.axpy(Pstar, stepG);
      theta1.scale(1.0 - stepG);
      theta1.axpy(theta1star, stepG);
      // Symmetrise against round-off before factorising.
      for(unsigned int r = 0; r < M; r++)
        for(unsigned int s = 0; s < r; s++)
        {
          const double v = 0.5 * (P.getVal(r, s) + P.getVal(s, r));
          P.setVal(v, r, s);
          P.setVal(v, s, r);
        }
      P.setSymmetric(true);
      cholP.deepCopy(P);
      try { cholP.chol(); pd = true; }
      catch(ndlexceptions::Error&) { stepG *= 0.5; }
    }
    if(!pd)
      continue;
    CMatrix Snew(M, M);
    Snew.setSymmetric(true);
    Snew.pdinv(cholP);
    CMatrix mNew(M, 1);
    mNew.gemv(Snew, theta1, 1.0, 0.0, "n");
    qCov[j] = Snew;
    qMean.copyColCol(j, mNew, 0);
  }
  if(qParamMode == QModeCholesky)
    syncCholFromCov();
}

void CSvgp::latentMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xs) const
{
  const unsigned int M = numInducing;
  const unsigned int Ns = Xs.getRows();
  CMatrix Kus(M, Ns);
  pInducing->computeKux(Kus, *pkern, Xs);
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

void CSvgp::predictiveMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xs) const
{
  latentMeanVar(mu, varSigma, Xs);
  const double noise = 1.0 / beta;
  for(unsigned int i = 0; i < varSigma.getRows(); i++)
    for(unsigned int j = 0; j < varSigma.getCols(); j++)
      varSigma.setVal(varSigma.getVal(i, j) + noise, i, j);
}

void CSvgp::posteriorMeanVar(CMatrix& mu, CMatrix& varSigma, const CMatrix& Xs) const
{
  // CIP-0031: deprecated alias — SVGP historically returned latent variance.
  latentMeanVar(mu, varSigma, Xs);
}

double CSvgp::elboGrad(CMatrix& gKern, CMatrix& gZ, double& gBeta) const
{
  return elboGrad(gKern, gZ, gBeta, allIndices(), 0);
}

// CIP-0030 Phase 0b: gradients at explicit inputs XB (B x q).
double CSvgp::elboGradAt(CMatrix& gKern, CMatrix& gZ, double& gBeta,
                         CMatrix* gXOrNull, const CMatrix& XB,
                         const vector<unsigned int>& yRows, double c) const
{
  if(yRows.empty() || XB.getRows() != yRows.size())
    throw ndlexceptions::Error("CSvgp::elboGradAt: XB rows must match yRows");
  if(!(c > 0.0))
    throw ndlexceptions::Error("CSvgp::elboGradAt: c must be positive");
  const unsigned int M = numInducing;
  const unsigned int B = yRows.size();
  const unsigned int q = getInputDim();
  const unsigned int nKern = pkern->getNumParams();
  DIMENSIONMATCH(XB.getCols() == q);

  fillMarginalsFromX(XB, wsMu, wsVar);
  // A_B already in wsAB from fillMarginalsFromX.
  CMatrix Abar(M, B);
  Abar.zeros();
  CMatrix gLambda(B, 1);
  gLambda.zeros();
  gBeta = 0.0;

  CMatrix mj(M, 1);
  double gMu = 0.0, gVar = 0.0;
  expectedLogLikBatch(yRows, wsMu, wsVar, 1.0, wsGMu, wsGVar);
  CMatrix coefMu(B, 1);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    mj.copyColCol(0, qMean, j);
    wsSAB.resize(M, B);
    wsSAB.symm(qCov[j], wsAB, 1.0, 0.0, "u", "l");
    if(legacyPaths)
    {
      for(unsigned int k = 0; k < B; k++)
      {
        const unsigned int i = yRows[k];
        const double yv = py->getVal(i, j);
        const double mu = wsMu.getVal(k, j);
        const double var = wsVar.getVal(k, j);
        gMu = wsGMu.getVal(k, j); gVar = wsGVar.getVal(k, j);
        gBeta += c * expectedLogLikBetaDeriv(yv, mu, var);
        gLambda.addVal(c * gVar, k);
        for(unsigned int r = 0; r < M; r++)
        {
          const double da = c * gMu * mj.getVal(r)
            + 2.0 * c * gVar * (wsSAB.getVal(r, k) - wsAB.getVal(r, k));
          Abar.setVal(Abar.getVal(r, k) + da, r, k);
        }
      }
      continue;
    }
    // CIP-003F A4: site adjoint as a rank-one update plus a column-scaled
    // difference, i.e. Abar += m_j (c gMu)^T + (S_j A_B - A_B) diag(2 c gVar).
    wsABS.deepCopy(wsSAB);
    wsABS.axpy(wsAB, -1.0);
    for(unsigned int k = 0; k < B; k++)
    {
      const unsigned int i = yRows[k];
      const double yv = py->getVal(i, j);
      const double mu = wsMu.getVal(k, j);
      const double var = wsVar.getVal(k, j);
      gMu = wsGMu.getVal(k, j); gVar = wsGVar.getVal(k, j);
      gBeta += c * expectedLogLikBetaDeriv(yv, mu, var);
      gLambda.addVal(c * gVar, k);
      coefMu.setVal(c * gMu, k);
      wsABS.scaleCol(k, 2.0 * c * gVar);
    }
    Abar.axpy(wsABS, 1.0);
    Abar.ger(mj, coefMu, 1.0);
  }

  CMatrix KufBar, Ubar;
  trsmAdjointUT(KufBar, Ubar, cholKuu, wsAB, Abar);
  CMatrix KuuBar;
  cholAdjointUpperBlocked(KuuBar, cholKuu, Ubar);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int jj = 0; jj < i; jj++)
    {
      const double v = 0.5 * KuuBar.getVal(i, jj);
      KuuBar.setVal(v, i, jj);
      KuuBar.setVal(v, jj, i);
    }
  KuuBar.setSymmetric(true);

  gKern.resize(1, nKern);
  gKern.zeros();
  gZ.resize(pInducing->centres().getRows(), pInducing->centres().getCols());
  gZ.zeros();
  const bool wantX = (gXOrNull != 0);
  if(wantX)
  {
    gXOrNull->resize(B, q);
    gXOrNull->zeros();
  }
  pInducing->gradFromKuuKuf(gKern, gZ, gXOrNull, *pkern, XB, KuuBar, KufBar,
                            wantX, true);

  CMatrix tempG(1, nKern);
  pkern->getDiagGradTransParams(tempG, XB, gLambda, false);
  for(unsigned int i = 0; i < nKern; i++)
    gKern.addVal(tempG.getVal(i), i);

  if(wantX)
  {
    // ∂k(x,x)/∂x contribution (zero for stationary kernels; needed for MLP).
    CMatrix dgX(B, q);
    pkern->getDiagGradX(dgX, XB, false);
    for(unsigned int k = 0; k < B; k++)
      for(unsigned int d = 0; d < q; d++)
        gXOrNull->addVal(gLambda.getVal(k) * dgX.getVal(k, d), k, d);
  }

  gBeta *= betaTransform->gradfact(beta);
  return dataTermAt(XB, yRows, c) + klPlusPrior();
}

// Gradients of the ELBO w.r.t. kernel params, Z and beta at fixed whitened q(v).
// When gXOrNull is non-null, routes through elboGradAt (Phase 0b). Otherwise
// keeps the cached full-A path for bit-stable point-inducing fixtures.
double CSvgp::elboGrad(CMatrix& gKern, CMatrix& gZ, double& gBeta,
                       const vector<unsigned int>& batch,
                       CMatrix* gXOrNull) const
{
  if(batch.empty())
    throw ndlexceptions::Error("CSvgp: empty batch");
  const double c = (double)nData / (double)batch.size();
  if(gXOrNull != 0)
  {
    gatherXB(batch, wsXB);
    return elboGradAt(gKern, gZ, gBeta, gXOrNull, wsXB, batch, c);
  }
  // CIP-003F A1: small minibatches cost O(B), not O(N): route through the
  // explicit-input path (no full-data matrices, no M x N adjoint).
  if(!legacyPaths && batch.size() * 4 < nData)
  {
    gatherXB(batch, wsXB);
    return elboGradAt(gKern, gZ, gBeta, 0, wsXB, batch, c);
  }

  const unsigned int M = numInducing;
  const unsigned int N = nData;
  const unsigned int B = batch.size();
  const unsigned int nKern = pkern->getNumParams();

  fillMarginals(batch, wsMu, wsVar);
  const CMatrix* pA = &A;
  if(!isFullOrderedBatch(batch))
  {
    formAB(batch, wsAB, wsDK, wsDQ);
    pA = &wsAB;
  }

  // Scatter site adjoints into full M x N Abar (trsmAdjoint expects full A).
  CMatrix Abar(M, N);
  Abar.zeros();
  CMatrix gLambda(N, 1);
  gLambda.zeros();
  gBeta = 0.0;

  CMatrix mj(M, 1);
  double gMu = 0.0, gVar = 0.0;
  expectedLogLikBatch(batch, wsMu, wsVar, 1.0, wsGMu, wsGVar);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    mj.copyColCol(0, qMean, j);
    wsSAB.resize(M, B);
    wsSAB.symm(qCov[j], *pA, 1.0, 0.0, "u", "l");
    for(unsigned int k = 0; k < B; k++)
    {
      const unsigned int i = batch[k];
      const double yv = py->getVal(i, j);
      const double mu = wsMu.getVal(k, j);
      const double var = wsVar.getVal(k, j);
      gMu = wsGMu.getVal(k, j); gVar = wsGVar.getVal(k, j);
      gBeta += c * expectedLogLikBetaDeriv(yv, mu, var);
      gLambda.addVal(c * gVar, i);
      // ∂L/∂a = c gMu m~ + 2 c gVar (S~ a - a)
      for(unsigned int r = 0; r < M; r++)
      {
        const double da = c * gMu * mj.getVal(r)
          + 2.0 * c * gVar * (wsSAB.getVal(r, k) - pA->getVal(r, k));
        Abar.setVal(Abar.getVal(r, i) + da, r, i);
      }
    }
  }

  CMatrix KufBar, Ubar;
  ensureFullData();
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
  gZ.resize(pInducing->centres().getRows(), pInducing->centres().getCols());
  gZ.zeros();
  pInducing->gradFromKuuKuf(gKern, gZ, 0, *pkern, *pX, KuuBar, KufBar, false, true);

  CMatrix tempG(1, nKern);
  pkern->getDiagGradTransParams(tempG, *pX, gLambda, false);
  for(unsigned int i = 0; i < nKern; i++)
    gKern.addVal(tempG.getVal(i), i);

  // gKern already in unconstrained space via getGradTransParams / getDiagGradTransParams.
  gBeta *= betaTransform->gradfact(beta);

  return elbo(batch);
}

void CSvgp::naturalGradientStepAt(double gamma, const CMatrix& XB,
                                  const vector<unsigned int>& yRows, double c)
{
  if(!(gamma > 0.0 && gamma <= 1.0))
    throw ndlexceptions::Error("CSvgp::naturalGradientStepAt: gamma must be in (0, 1]");
  if(yRows.empty() || XB.getRows() != yRows.size())
    throw ndlexceptions::Error("CSvgp::naturalGradientStepAt: XB rows must match yRows");
  if(!(c > 0.0))
    throw ndlexceptions::Error("CSvgp::naturalGradientStepAt: c must be positive");
  const unsigned int M = numInducing;
  const unsigned int B = yRows.size();

  fillMarginalsFromX(XB, wsMu, wsVar);
  const CMatrix& AB = wsAB;

  expectedLogLikBatch(yRows, wsMu, wsVar, 1.0, wsGMu, wsGVar);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    CMatrix Scopy(qCov[j]);
    Scopy.setSymmetric(true);
    CMatrix cholS(Scopy);
    cholS.chol();
    CMatrix P(M, M);
    P.setSymmetric(true);
    P.pdinv(cholS);
    CMatrix theta1(M, 1);
    theta1.gemvColCol(0, P, qMean, j, 1.0, 0.0, "n");

    CMatrix Pstar(M, M, 0.0);
    Pstar.setSymmetric(true);
    wsCoeffs.resize(B, 1);
    CMatrix alphas(B, 1);
    double gMu = 0.0, gVar = 0.0;
    bool allNonNeg = true;
    for(unsigned int k = 0; k < B; k++)
    {
      const unsigned int i = yRows[k];
      const double mu = wsMu.getVal(k, j);
      gMu = wsGMu.getVal(k, j); gVar = wsGVar.getVal(k, j);
      wsCoeffs.setVal(c * (gMu - 2.0 * gVar * mu), k);
      const double alpha = -2.0 * c * gVar;
      alphas.setVal(alpha, k);
      if(alpha < 0.0)
        allNonNeg = false;
    }
    if(allNonNeg)
    {
      wsABS.deepCopy(AB);
      for(unsigned int k = 0; k < B; k++)
        wsABS.scaleCol(k, sqrt(alphas.getVal(k)));
      Pstar.syrk(wsABS, 1.0, 0.0, "u", "n");
      for(unsigned int r = 0; r < M; r++)
        for(unsigned int s = 0; s < r; s++)
          Pstar.setVal(Pstar.getVal(s, r), r, s);
      Pstar.setSymmetric(true);
      Pstar.addDiag(1.0);
    }
    else
    {
      Pstar.addDiag(1.0);
      CMatrix a(M, 1);
      for(unsigned int k = 0; k < B; k++)
      {
        a.copyColCol(0, AB, k);
        Pstar.ger(a, a, alphas.getVal(k));
      }
    }
    CMatrix theta1star(M, 1);
    theta1star.gemv(AB, wsCoeffs, 1.0, 0.0, "n");

    // Step halving as in naturalGradientStep (indefinite targets).
    CMatrix P0(P), theta0(theta1), cholP(M, M);
    bool pd = false;
    double stepG = gamma;
    for(unsigned int tries = 0; tries < 40 && !pd; tries++)
    {
      P.deepCopy(P0);
      theta1.deepCopy(theta0);
      P.scale(1.0 - stepG);
      P.axpy(Pstar, stepG);
      theta1.scale(1.0 - stepG);
      theta1.axpy(theta1star, stepG);
      // Symmetrise against round-off before factorising.
      for(unsigned int r = 0; r < M; r++)
        for(unsigned int s = 0; s < r; s++)
        {
          const double v = 0.5 * (P.getVal(r, s) + P.getVal(s, r));
          P.setVal(v, r, s);
          P.setVal(v, s, r);
        }
      P.setSymmetric(true);
      cholP.deepCopy(P);
      try { cholP.chol(); pd = true; }
      catch(ndlexceptions::Error&) { stepG *= 0.5; }
    }
    if(!pd)
      continue;
    CMatrix Snew(M, M);
    Snew.setSymmetric(true);
    Snew.pdinv(cholP);
    CMatrix mNew(M, 1);
    mNew.gemv(Snew, theta1, 1.0, 0.0, "n");
    qCov[j] = Snew;
    qMean.copyColCol(j, mNew, 0);
  }
  if(qParamMode == QModeCholesky)
    syncCholFromCov();
}

unsigned int CSvgp::getOptNumParams() const
{
  return pkern->getNumParams() + pInducing->getNumParams() + 1;
}

void CSvgp::getOptParams(CMatrix& param) const
{
  const unsigned int P = getOptNumParams();
  param.resize(1, P);
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    param.setVal(pkern->getTransParam(i), counter++);
  pInducing->getParams(param, counter);
  counter += pInducing->getNumParams();
  param.setVal(betaTransform->xtoa(beta), counter++);
}

void CSvgp::setOptParams(const CMatrix& param)
{
  DIMENSIONMATCH(param.getRows() == 1 && param.getCols() == getOptNumParams());
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    pkern->setTransParam(param.getVal(counter++), i);
  pInducing->setParams(param, counter);
  counter += pInducing->getNumParams();
  beta = betaTransform->atox(param.getVal(counter++));
  updateKernelMatrices();
}

void CSvgp::packInducingGrad(CMatrix& gFlat, unsigned int offset,
                             const CMatrix& gZ) const
{
  // Match CInducing::getParams layout: column-major over centres(), then optional s.
  unsigned int counter = offset;
  const unsigned int M = gZ.getRows();
  const unsigned int d = gZ.getCols();
  // Families with fixed windows (e.g. Hermite) have no learnable inducing parameters;
  // never pack gZ past what the optimiser vector reserves for them.
  const unsigned int nInd = pInducing->getNumParams();
  for(unsigned int j = 0; j < d; j++)
    for(unsigned int i = 0; i < M; i++)
      if(counter < offset + nInd)
        gFlat.setVal(gZ.getVal(i, j), counter++);
  while(counter < offset + nInd)
    gFlat.setVal(0.0, counter++);
  // Learned window widths occupy the last numLearnedWindowWidths() slots of the
  // inducing block (CInducing::getParams: centres, then s). Their gradient is the
  // one gradFromKuuKuf left in the inducing object, already chained through the
  // unconstrained parametrisation. Zero here made learn_window_width a no-op.
  if(pInducing->learnsWindowWidth())
  {
    const unsigned int nS = pInducing->numLearnedWindowWidths();
    DIMENSIONMATCH(nS <= nInd);
    for(unsigned int i = 0; i < nS; i++)
      gFlat.setVal(pInducing->windowWidthGradAt(i), offset + nInd - nS + i);
  }
}

double CSvgp::objectiveGradBatch(const vector<unsigned int>& batch, CMatrix& g)
{
  CMatrix gKern(1, pkern->getNumParams());
  CMatrix gZ;
  double gBeta = 0.0;
  const double L = elboGrad(gKern, gZ, gBeta, batch);
  const unsigned int nStoch = getStochasticNumParams();
  g.resize(1, nStoch);
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    g.setVal(gKern.getVal(i), counter++);
  packInducingGrad(g, counter, gZ);
  counter += pInducing->getNumParams();
  g.setVal(gBeta, counter++);
  if(qParamMode == QModeCholesky)
  {
    CMatrix gQ(1, getQParamNum());
    qParamsGrad(gQ, batch);
    for(unsigned int i = 0; i < gQ.getCols(); i++)
      g.setVal(gQ.getVal(i), counter++);
  }
  return L;
}

void CSvgp::getStochasticParams(CMatrix& param) const
{
  const unsigned int nOpt = getOptNumParams();
  const unsigned int nStoch = getStochasticNumParams();
  param.resize(1, nStoch);
  CMatrix opt(1, nOpt);
  getOptParams(opt);
  for(unsigned int i = 0; i < nOpt; i++)
    param.setVal(opt.getVal(i), i);
  if(qParamMode == QModeCholesky)
    getQParams(param, nOpt);
}

void CSvgp::setStochasticParams(const CMatrix& param)
{
  DIMENSIONMATCH(param.getRows() == 1 && param.getCols() == getStochasticNumParams());
  const unsigned int nOpt = getOptNumParams();
  CMatrix opt(1, nOpt);
  for(unsigned int i = 0; i < nOpt; i++)
    opt.setVal(param.getVal(i), i);
  setOptParams(opt);
  if(qParamMode == QModeCholesky)
    setQParams(param, nOpt);
}

unsigned int CSvgp::getStochasticNumParams() const
{
  unsigned int n = getOptNumParams();
  if(qParamMode == QModeCholesky)
    n += getQParamNum();
  return n;
}

void CSvgp::ensureCholStorage()
{
  const unsigned int M = numInducing;
  if(qChol.size() != outputDim)
    qChol.assign(outputDim, CMatrix(M, M));
  for(unsigned int j = 0; j < outputDim; j++)
  {
    if(qChol[j].getRows() != M || qChol[j].getCols() != M)
      qChol[j].resize(M, M);
  }
}

void CSvgp::syncCovFromChol()
{
  ensureCholStorage();
  const unsigned int M = numInducing;
  for(unsigned int j = 0; j < outputDim; j++)
  {
    CMatrix& S = qCov[j];
    S.resize(M, M);
    S.zeros();
    // S = L L^T via syrk on lower L (trans = "n" => A A^T).
    S.syrk(qChol[j], 1.0, 0.0, "l", "n");
    for(unsigned int r = 0; r < M; r++)
      for(unsigned int c = 0; c < r; c++)
        S.setVal(S.getVal(r, c), c, r);
    S.setSymmetric(true);
  }
}

void CSvgp::syncCholFromCov()
{
  ensureCholStorage();
  const unsigned int M = numInducing;
  for(unsigned int j = 0; j < outputDim; j++)
  {
    CMatrix Scopy(qCov[j]);
    Scopy.setSymmetric(true);
    // chol() returns upper by default; request lower.
    Scopy.chol("L");
    qChol[j].resize(M, M);
    qChol[j].zeros();
    for(unsigned int r = 0; r < M; r++)
      for(unsigned int c = 0; c <= r; c++)
        qChol[j].setVal(Scopy.getVal(r, c), r, c);
  }
}

void CSvgp::setQParamMode(QParamMode mode)
{
  if(mode == qParamMode)
    return;
  qParamMode = mode;
  if(mode == QModeCholesky)
    syncCholFromCov();
}

unsigned int CSvgp::getQParamNum() const
{
  const unsigned int M = numInducing;
  const unsigned int tril = M * (M + 1) / 2;
  return outputDim * M + outputDim * tril;
}

void CSvgp::getQParams(CMatrix& param, unsigned int offset) const
{
  if(qParamMode != QModeCholesky)
    throw ndlexceptions::Error("CSvgp::getQParams requires QModeCholesky");
  const unsigned int nQ = getQParamNum();
  if(offset == 0)
    param.resize(1, nQ);
  else
    DIMENSIONMATCH(param.getRows() == 1 && param.getCols() >= offset + nQ);
  const unsigned int M = numInducing;
  unsigned int counter = offset;
  for(unsigned int j = 0; j < outputDim; j++)
  {
    for(unsigned int r = 0; r < M; r++)
      param.setVal(qMean.getVal(r, j), counter++);
    for(unsigned int c = 0; c < M; c++)
      for(unsigned int r = c; r < M; r++)
      {
        const double Lrc = qChol[j].getVal(r, c);
        if(r == c)
          param.setVal(log(max(Lrc, 1e-300)), counter++);
        else
          param.setVal(Lrc, counter++);
      }
  }
}

void CSvgp::setQParams(const CMatrix& param, unsigned int offset)
{
  if(qParamMode != QModeCholesky)
    throw ndlexceptions::Error("CSvgp::setQParams requires QModeCholesky");
  DIMENSIONMATCH(param.getCols() >= offset + getQParamNum());
  ensureCholStorage();
  const unsigned int M = numInducing;
  unsigned int counter = offset;
  for(unsigned int j = 0; j < outputDim; j++)
  {
    for(unsigned int r = 0; r < M; r++)
      qMean.setVal(param.getVal(counter++), r, j);
    qChol[j].zeros();
    for(unsigned int c = 0; c < M; c++)
      for(unsigned int r = c; r < M; r++)
      {
        const double v = param.getVal(counter++);
        if(r == c)
          qChol[j].setVal(exp(v), r, c);
        else
          qChol[j].setVal(v, r, c);
      }
  }
  syncCovFromChol();
}

// d(data term)/d m~ and d(data term)/d S~_j at sites A_B (M x B), using the site
// derivatives already in wsGMu / wsGVar.  CIP-003F A4: one gemv and one gemm per
// output instead of B x M^2 element updates.  legacyPaths keeps the element loops.
void CSvgp::accumulateDataTermQGrads(const CMatrix& Ab, double c, CMatrix& gM,
                                     vector<CMatrix>& gS) const
{
  const unsigned int M = numInducing;
  const unsigned int B = Ab.getCols();
  if(legacyPaths)
  {
    for(unsigned int j = 0; j < outputDim; j++)
      for(unsigned int k = 0; k < B; k++)
      {
        const double gMu = wsGMu.getVal(k, j), gVar = wsGVar.getVal(k, j);
        for(unsigned int r = 0; r < M; r++)
          gM.setVal(gM.getVal(r, j) + c * gMu * Ab.getVal(r, k), r, j);
        for(unsigned int r = 0; r < M; r++)
          for(unsigned int s = 0; s <= r; s++)
          {
            const double v = c * gVar * Ab.getVal(r, k) * Ab.getVal(s, k);
            gS[j].setVal(gS[j].getVal(r, s) + v, r, s);
            if(s != r)
              gS[j].setVal(gS[j].getVal(s, r) + v, s, r);
          }
      }
    return;
  }
  CMatrix coef(B, 1);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    for(unsigned int k = 0; k < B; k++)
      coef.setVal(c * wsGMu.getVal(k, j), k);
    gM.gemvColCol(j, Ab, coef, 0, 1.0, 1.0, "n");
    wsABS.deepCopy(Ab);
    for(unsigned int k = 0; k < B; k++)
      wsABS.scaleCol(k, c * wsGVar.getVal(k, j));
    CMatrix G(M, M);
    G.gemm(wsABS, Ab, 1.0, 0.0, "n", "t");
    // Mirror the upper triangle so the result is exactly symmetric.
    for(unsigned int r = 0; r < M; r++)
      for(unsigned int s = 0; s < r; s++)
        G.setVal(G.getVal(s, r), r, s);
    gS[j].axpy(G, 1.0);
    gS[j].setSymmetric(true);
  }
}

void CSvgp::dataTermQGrads(const vector<unsigned int>& batch,
                           CMatrix& gM, vector<CMatrix>& gS) const
{
  if(batch.empty())
    throw ndlexceptions::Error("CSvgp::dataTermQGrads: empty batch");
  const unsigned int M = numInducing;
  const unsigned int B = batch.size();
  const double c = (double)nData / (double)B;
  gM.resize(M, outputDim);
  gM.zeros();
  gS.assign(outputDim, CMatrix(M, M));
  for(unsigned int j = 0; j < outputDim; j++)
  {
    gS[j].zeros();
    gS[j].setSymmetric(true);
  }

  fillMarginals(batch, wsMu, wsVar);
  const CMatrix* pA = &A;
  if(!isFullOrderedBatch(batch))
  {
    formAB(batch, wsAB, wsDK, wsDQ);
    pA = &wsAB;
  }

  expectedLogLikBatch(batch, wsMu, wsVar, 1.0, wsGMu, wsGVar);
  accumulateDataTermQGrads(*pA, c, gM, gS);
}

void CSvgp::dataTermQGradsAt(const CMatrix& XB, const vector<unsigned int>& yRows,
                             double c, CMatrix& gM, vector<CMatrix>& gS) const
{
  if(yRows.empty() || XB.getRows() != yRows.size())
    throw ndlexceptions::Error("CSvgp::dataTermQGradsAt: XB rows must match yRows");
  if(!(c > 0.0))
    throw ndlexceptions::Error("CSvgp::dataTermQGradsAt: c must be positive");
  const unsigned int M = numInducing;
  const unsigned int B = yRows.size();
  gM.resize(M, outputDim);
  gM.zeros();
  gS.assign(outputDim, CMatrix(M, M));
  for(unsigned int j = 0; j < outputDim; j++)
  {
    gS[j].zeros();
    gS[j].setSymmetric(true);
  }
  fillMarginalsFromX(XB, wsMu, wsVar);
  expectedLogLikBatch(yRows, wsMu, wsVar, 1.0, wsGMu, wsGVar);
  accumulateDataTermQGrads(wsAB, c, gM, gS);
}

void CSvgp::qParamsGradAt(CMatrix& gQ, const CMatrix& XB,
                          const vector<unsigned int>& yRows, double c) const
{
  if(qParamMode != QModeCholesky)
    throw ndlexceptions::Error("CSvgp::qParamsGradAt requires QModeCholesky");
  CMatrix gM;
  vector<CMatrix> gS;
  dataTermQGradsAt(XB, yRows, c, gM, gS);
  packQGradsFromMS(gQ, gM, gS);
}

void CSvgp::qParamsGrad(CMatrix& gQ) const
{
  qParamsGrad(gQ, allIndices());
}

void CSvgp::qParamsGrad(CMatrix& gQ, const vector<unsigned int>& batch) const
{
  if(qParamMode != QModeCholesky)
    throw ndlexceptions::Error("CSvgp::qParamsGrad requires QModeCholesky");
  const unsigned int M = numInducing;
  const unsigned int nQ = getQParamNum();
  gQ.resize(1, nQ);
  gQ.zeros();

  CMatrix gM;
  vector<CMatrix> gS;
  dataTermQGrads(batch, gM, gS);

  packQGradsFromMS(gQ, gM, gS);
  // Subtract the KL gradient analytically (see klGradPackedQ).
  CMatrix gKl;
  klGradPackedQ(gKl);
  for(unsigned int i = 0; i < nQ; i++)
    gQ.setVal(gQ.getVal(i) - gKl.getVal(i), i);
}

void CSvgp::packQGradsFromMS(CMatrix& gQ, const CMatrix& gM,
                             const vector<CMatrix>& gS) const
{
  if(qParamMode != QModeCholesky)
    throw ndlexceptions::Error("CSvgp::packQGradsFromMS requires QModeCholesky");
  const unsigned int M = numInducing;
  gQ.resize(1, getQParamNum());
  unsigned int counter = 0;
  for(unsigned int j = 0; j < outputDim; j++)
  {
    CMatrix Gsym(M, M);
    for(unsigned int r = 0; r < M; r++)
      for(unsigned int s = 0; s < M; s++)
        Gsym.setVal(0.5 * (gS[j].getVal(r, s) + gS[j].getVal(s, r)), r, s);
    CMatrix gL(M, M);
    gL.zeros();
    gL.gemm(Gsym, qChol[j], 2.0, 0.0, "n", "n");

    for(unsigned int r = 0; r < M; r++)
      gQ.setVal(gM.getVal(r, j), counter++);
    for(unsigned int c = 0; c < M; c++)
      for(unsigned int r = c; r < M; r++)
      {
        double g = gL.getVal(r, c);
        if(r == c)
          g *= qChol[j].getVal(r, c);
        gQ.setVal(g, counter++);
      }
  }
}

void CSvgp::marginalsAt(const CMatrix& XB, CMatrix& mean, CMatrix& var) const
{
  DIMENSIONMATCH(XB.getCols() == getInputDim());
  fillMarginalsFromX(XB, mean, var);
}

void CSvgp::marginalsGradAt(const CMatrix& XB, const CMatrix& gMean,
                            const CMatrix& gVar, CMatrix& gKern,
                            CMatrix& gInducing, CMatrix& gQ,
                            CMatrix* gXOrNull) const
{
  const unsigned int M = numInducing;
  const unsigned int B = XB.getRows();
  const unsigned int q = getInputDim();
  const unsigned int nKern = pkern->getNumParams();
  DIMENSIONMATCH(XB.getCols() == q);
  DIMENSIONMATCH(gMean.getRows() == B && gMean.getCols() == outputDim);
  DIMENSIONMATCH(gVar.getRows() == B && gVar.getCols() == outputDim);

  // Ensure wsAB matches XB (same as fillMarginalsFromX).
  formABFromX(XB, wsAB, wsDK, wsDQ);

  CMatrix Abar(M, B);
  Abar.zeros();
  CMatrix gLambda(B, 1);
  gLambda.zeros();
  CMatrix gM(M, outputDim);
  gM.zeros();
  vector<CMatrix> gS(outputDim, CMatrix(M, M));
  for(unsigned int j = 0; j < outputDim; j++)
  {
    gS[j].zeros();
    gS[j].setSymmetric(true);
  }

  CMatrix mj(M, 1);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    mj.copyColCol(0, qMean, j);
    wsSAB.resize(M, B);
    wsSAB.symm(qCov[j], wsAB, 1.0, 0.0, "u", "l");
    for(unsigned int k = 0; k < B; k++)
    {
      const double gMu = gMean.getVal(k, j);
      const double gV = gVar.getVal(k, j);
      gLambda.addVal(gV, k);
      for(unsigned int r = 0; r < M; r++)
      {
        const double da = gMu * mj.getVal(r)
          + 2.0 * gV * (wsSAB.getVal(r, k) - wsAB.getVal(r, k));
        Abar.setVal(Abar.getVal(r, k) + da, r, k);
        gM.setVal(gM.getVal(r, j) + gMu * wsAB.getVal(r, k), r, j);
      }
    }
    // gS_j += A diag(gVar_{:,j}) A^T as one BLAS-3 product (was O(B M^2)
    // scalar accessor calls per output).
    CMatrix AD(wsAB);
    for(unsigned int k = 0; k < B; k++)
    {
      const double gV = gVar.getVal(k, j);
      for(unsigned int r = 0; r < M; r++)
        AD.setVal(AD.getVal(r, k) * gV, r, k);
    }
    gS[j].gemm(AD, wsAB, 1.0, 1.0, "n", "t");
  }

  CMatrix KufBar, Ubar;
  trsmAdjointUT(KufBar, Ubar, cholKuu, wsAB, Abar);
  CMatrix KuuBar;
  cholAdjointUpperBlocked(KuuBar, cholKuu, Ubar);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int jj = 0; jj < i; jj++)
    {
      const double v = 0.5 * KuuBar.getVal(i, jj);
      KuuBar.setVal(v, i, jj);
      KuuBar.setVal(v, jj, i);
    }
  KuuBar.setSymmetric(true);

  gKern.resize(1, nKern);
  gKern.zeros();
  gInducing.resize(pInducing->centres().getRows(), pInducing->centres().getCols());
  gInducing.zeros();
  const bool wantX = (gXOrNull != 0);
  if(wantX)
  {
    gXOrNull->resize(B, q);
    gXOrNull->zeros();
  }
  pInducing->gradFromKuuKuf(gKern, gInducing, gXOrNull, *pkern, XB, KuuBar,
                            KufBar, wantX, true);

  CMatrix tempG(1, nKern);
  pkern->getDiagGradTransParams(tempG, XB, gLambda, false);
  for(unsigned int i = 0; i < nKern; i++)
    gKern.addVal(tempG.getVal(i), i);

  if(wantX)
  {
    CMatrix dgX(B, q);
    pkern->getDiagGradX(dgX, XB, false);
    for(unsigned int k = 0; k < B; k++)
      for(unsigned int d = 0; d < q; d++)
        gXOrNull->addVal(gLambda.getVal(k) * dgX.getVal(k, d), k, d);
  }

  if(qParamMode == QModeCholesky)
    packQGradsFromMS(gQ, gM, gS);
  else
  {
    gQ.resize(1, 0);
  }
}

// Packed gradient of sum_j KL(q(v_j) || N(0, I)) in the Cholesky parametrisation, in the
// layout of packQGradsFromMS: per output the means, then the lower factor column by column
// with the diagonal as log(L_ii). With KL_j = 0.5 (|L|_F^2 - M - 2 sum log L_ii + |m|^2) the
// gradients are m, L_rc (r > c) and L_ii^2 - 1. Going through S^{-1} instead gives
// (I - S^{-1}) L = L - L^{-T}, whose below-diagonal part is zero in exact arithmetic; with
// S = L L^T the explicit inverse loses that to cancellation (condition number squared) when
// L has a small diagonal, which is how q is initialised in hidden layers.
void CSvgp::klGradPackedQ(CMatrix& gQ) const
{
  if(qParamMode != QModeCholesky)
    throw ndlexceptions::Error("CSvgp::klGradPackedQ requires QModeCholesky");
  const unsigned int M = numInducing;
  gQ.resize(1, getQParamNum());
  unsigned int counter = 0;
  for(unsigned int j = 0; j < outputDim; j++)
  {
    for(unsigned int r = 0; r < M; r++)
      gQ.setVal(qMean.getVal(r, j), counter++);
    for(unsigned int c = 0; c < M; c++)
      for(unsigned int r = c; r < M; r++)
      {
        const double l = qChol[j].getVal(r, c);
        gQ.setVal(r == c ? l * l - 1.0 : l, counter++);
      }
  }
}

double CSvgp::klQ(CMatrix* gQOrNull) const
{
  double kl = 0.0;
  for(unsigned int j = 0; j < outputDim; j++)
    kl += klDivergence(j);

  if(gQOrNull == 0)
    return kl;
  if(qParamMode != QModeCholesky)
  {
    gQOrNull->resize(1, 0);
    return kl;
  }

  klGradPackedQ(*gQOrNull);
  return kl;
}

void CSvgp::afterBatchStep(const vector<unsigned int>& batch)
{
  if(natGradGamma > 0.0)
    naturalGradientStep(natGradGamma, batch);
}
