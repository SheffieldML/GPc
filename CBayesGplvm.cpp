/* Bayesian GP-LVM — CIP-0017 Phase 2. Bound follows GPy VarDTC + KL. */
#include "CBayesGplvm.h"
#include "ndlutil.h"
#include <cmath>

using namespace std;

const double CBayesGplvm::JITTER = 1e-8; // match GPy VarDTC.const_jitter

namespace {

void backsubBothSidesUpper(CMatrix& out, const CMatrix& U, const CMatrix& X)
{
  // Match GPy backsub_both_sides on lower L=U^T: L^{-T} X L^{-1} = U^{-1} X U^{-T}.
  // (Upper Cholesky: K = U^T U.)
  out.deepCopy(X);
  out.trsm(U, 1.0, "l", "u", "n", "n");
  out.trsm(U, 1.0, "r", "u", "t", "n");
}

double frobeniusSq(const CMatrix& A)
{
  double s = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
    {
      const double v = A.getVal(i, j);
      s += v * v;
    }
  return s;
}

} // namespace

CBayesGplvm::CBayesGplvm(CKern* kernel, const CMatrix& Yin, unsigned int latDim,
                         unsigned int numInducing, int verbosity)
  : pkern(kernel), ownKern(false), Y(Yin), beta(1.0),
    numData(Yin.getRows()), outputDim(Yin.getCols()),
    latentDim(latDim), numActive(numInducing),
    KupToDate(false)
{
  setVerbosity(verbosity);
  betaTransform = CTransform::defaultPositive();
  sTransform = CTransform::defaultPositive();
  initStoreage();
  initVals();
}

CBayesGplvm::~CBayesGplvm()
{
  if(ownKern)
    delete pkern;
  // transforms are shared singletons via defaultPositive — do not delete.
}

void CBayesGplvm::initStoreage()
{
  mu.resize(numData, latentDim);
  S.resize(numData, latentDim);
  Z.resize(numActive, latentDim);
  Kmm.resize(numActive, numActive);
  LcholKmm.resize(numActive, numActive);
  Psi1.resize(numData, numActive);
  Psi2.resize(numActive, numActive);
  A.resize(numActive, numActive);
  B.resize(numActive, numActive);
  LcholB.resize(numActive, numActive);
  LmiPsi1.resize(numActive, numData);
  LBiLmiPsi1.resize(numActive, numData);
  LBiLmiPsi1Vf.resize(numActive, outputDim);
  dL_dKmm.resize(numActive, numActive);
  dL_dpsi0.resize(numData, 1);
  dL_dpsi1.resize(numData, numActive);
  dL_dpsi2.resize(numActive, numActive);
}

void CBayesGplvm::initVals()
{
  initXpca();
  S.setVals(0.5);
  initZFromMu();
  beta = 1.0;
  setKupToDate(false);
}

void CBayesGplvm::initXpca()
{
  // Mean-subtracted PCA of Y into mu (same spirit as CGplvm::initXpca).
  CMatrix ymean = meanCol(Y);
  CMatrix covm(Y.getCols(), Y.getCols());
  covm.setSymmetric(true);
  covm.syrk(Y, 1.0 / (double)Y.getRows(), 0.0, "u", "t");
  ymean.trans();
  covm.syr(ymean, -1.0, "u");
  CMatrix eigVals(1, Y.getCols());
  covm.syev(eigVals, "v", "u");
  const unsigned int qUse = latentDim < Y.getCols() ? latentDim : Y.getCols();
  CMatrix Winv(Y.getCols(), latentDim);
  Winv.zeros();
  for(unsigned int i = 0; i < qUse; i++)
  {
    Winv.copyColCol(i, covm, Y.getCols() - 1 - i);
    const double ev = eigVals.getVal(Y.getCols() - 1 - i);
    if(ev > 0.0)
      Winv.scaleCol(i, 1.0 / sqrt(ev));
  }
  mu.gemm(Y, Winv, 1.0, 0.0, "n", "n");
  CMatrix meanX = meanCol(mu);
  for(unsigned int i = 0; i < mu.getRows(); i++)
    mu.axpyRowRow(i, meanX, 0, -1.0);
}

void CBayesGplvm::initZFromMu()
{
  // Evenly spaced subset of mu rows (wrap if M > N).
  for(unsigned int m = 0; m < numActive; m++)
  {
    const unsigned int n = (numData == 0) ? 0 : (m * numData) / numActive;
    for(unsigned int q = 0; q < latentDim; q++)
      Z.setVal(mu.getVal(n % numData, q), m, q);
  }
}

double CBayesGplvm::klDivergence() const
{
  double kl = 0.0;
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
    {
      const double m = mu.getVal(n, q);
      const double s = S.getVal(n, q);
      kl += m * m + s - log(s) - 1.0;
    }
  return 0.5 * kl;
}

void CBayesGplvm::klGradient(CMatrix& gMu, CMatrix& gS) const
{
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
    {
      gMu.setVal(mu.getVal(n, q), n, q);
      gS.setVal(0.5 * (1.0 - 1.0 / S.getVal(n, q)), n, q);
    }
}

void CBayesGplvm::updateCaches() const
{
  if(KupToDate)
    return;

  pkern->compute(Kmm, Z);
  for(unsigned int i = 0; i < numActive; i++)
    Kmm.addVal(JITTER, i, i);
  Kmm.setSymmetric(true);
  LcholKmm.deepCopy(Kmm);
  const double jit = LcholKmm.jitChol(LcholKmm);
  if(jit > 1e-2 && getVerbosity() > 2)
    cout << "Warning: jitter " << jit << " on Kmm in CBayesGplvm." << endl;

  psi0Sum = pkern->psi0(mu, S);
  pkern->psi1(Psi1, mu, S, Z);
  pkern->psi2(Psi2, mu, S, Z);

  // A = U^{-T} (beta Psi2) U^{-1}
  CMatrix psi2Beta(Psi2);
  psi2Beta.scale(beta);
  A.deepCopy(psi2Beta);
  A.trsm(LcholKmm, 1.0, "l", "u", "t", "n");
  A.trsm(LcholKmm, 1.0, "r", "u", "n", "n");
  A.setSymmetric(true);

  B.deepCopy(A);
  for(unsigned int i = 0; i < numActive; i++)
    B.addVal(1.0, i, i);
  B.setSymmetric(true);
  LcholB.deepCopy(B);
  const double jitB = LcholB.jitChol(LcholB);
  if(jitB > 1e-2 && getVerbosity() > 2)
    cout << "Warning: jitter " << jitB << " on B in CBayesGplvm." << endl;
  logDetB = logDet(LcholB);

  // LmiPsi1 = U^{-T} Psi1^T  (M x N)
  for(unsigned int m = 0; m < numActive; m++)
    for(unsigned int n = 0; n < numData; n++)
      LmiPsi1.setVal(Psi1.getVal(n, m), m, n);
  LmiPsi1.trsm(LcholKmm, 1.0, "l", "u", "t", "n");

  LBiLmiPsi1.deepCopy(LmiPsi1);
  LBiLmiPsi1.trsm(LcholB, 1.0, "l", "u", "t", "n");

  LBiLmiPsi1Vf.gemm(LBiLmiPsi1, Y, beta, 0.0, "n", "n");
  dataFit = frobeniusSq(LBiLmiPsi1Vf);

  // Gradients of the VarDTC bound (before KL).
  const double D = (double)outputDim;
  dL_dpsi0.setVals(-0.5 * D * beta);

  // Cpsi1Vf: solve U^T w = solve(UB^T, Vf) i.e. LB^T then Lm^T in GPy
  CMatrix tmp(LBiLmiPsi1Vf);
  tmp.trsm(LcholB, 1.0, "l", "u", "n", "n"); // U_B^{-1} Vf  (GPy: LB^T solve = trans=1 on lower)
  // GPy: dtrtrs(LB, Vf, lower=1, trans=1) solves LB^T x = Vf.
  // Lower LB, trans=1: LB^T x = Vf. Our LcholB is upper UB with B = UB^T UB, lower = UB^T.
  // LB^T = UB, so solve UB x = Vf → trsm L U N. Good.
  tmp.trsm(LcholKmm, 1.0, "l", "u", "n", "n"); // U^{-1} ... → Cpsi1Vf (M x D)
  // dL_dpsi1 = (beta Y) @ Cpsi1Vf^T
  CMatrix betaY(Y);
  betaY.scale(beta);
  dL_dpsi1.gemm(betaY, tmp, 1.0, 0.0, "n", "t");

  // DBi_plus_BiPBi = backsub(LB, D I + Vf Vf^T)
  CMatrix VfVfT(numActive, numActive);
  VfVfT.gemm(LBiLmiPsi1Vf, LBiLmiPsi1Vf, 1.0, 0.0, "n", "t");
  for(unsigned int i = 0; i < numActive; i++)
    VfVfT.addVal(D, i, i);
  CMatrix DBi(numActive, numActive);
  backsubBothSidesUpper(DBi, LcholB, VfVfT);

  // dL_dpsi2 = beta * 0.5 * backsub(Lm, D I - DBi)
  CMatrix mid(numActive, numActive);
  mid.zeros();
  for(unsigned int i = 0; i < numActive; i++)
    mid.setVal(D, i, i);
  mid.axpy(DBi, -1.0);
  CMatrix dPsi2Beta(numActive, numActive);
  backsubBothSidesUpper(dPsi2Beta, LcholKmm, mid);
  dL_dpsi2.deepCopy(dPsi2Beta);
  dL_dpsi2.scale(0.5 * beta);
  dL_dpsi2.setSymmetric(true);

  // dL_dKmm
  CMatrix delit(DBi);
  delit.scale(-0.5);
  // -0.5 * B * D
  for(unsigned int i = 0; i < numActive; i++)
    for(unsigned int j = 0; j < numActive; j++)
      delit.addVal(-0.5 * D * B.getVal(i, j), i, j);
  for(unsigned int i = 0; i < numActive; i++)
    delit.addVal(D, i, i);
  backsubBothSidesUpper(dL_dKmm, LcholKmm, delit);
  dL_dKmm.setSymmetric(true);

  // GPy _compute_dL_dR returns ∂L/∂σ² with σ²=1/β; convert to ∂L/∂β.
  const double trYYT = frobeniusSq(Y);
  double dL_dR = -0.5 * (double)numData * D * beta + 0.5 * trYYT * beta * beta;
  dL_dR += 0.5 * D * (psi0Sum * beta * beta - A.trace() * beta);
  double sumADBi = 0.0;
  for(unsigned int i = 0; i < numActive; i++)
    for(unsigned int j = 0; j < numActive; j++)
      sumADBi += A.getVal(i, j) * DBi.getVal(i, j);
  dL_dR += beta * (0.5 * sumADBi - dataFit);
  dL_dbeta = dL_dR * (-1.0 / (beta * beta));

  KupToDate = true;
}

double CBayesGplvm::logLikelihood() const
{
  updateCaches();
  const double D = (double)outputDim;
  const double N = (double)numData;
  const double trYYT = frobeniusSq(Y);
  // GPy _compute_log_marginal_likelihood (homoscedastic)
  const double lik1 = -0.5 * N * D * (ndlutil::LOGTWOPI - log(beta)) - 0.5 * beta * trYYT;
  const double lik2 = -0.5 * D * (beta * psi0Sum - A.trace());
  // logDet(U) = 2 sum log diag(U) = log|B|; GPy uses -D * sum(log diag(LB)).
  const double lik3 = -0.5 * D * logDetB;
  const double lik4 = 0.5 * dataFit;
  return lik1 + lik2 + lik3 + lik4 - klDivergence();
}

double CBayesGplvm::logLikelihoodGradient(CMatrix& g) const
{
  updateCaches();
  DIMENSIONMATCH(g.getRows() == 1);
  DIMENSIONMATCH(g.getCols() == getOptNumParams());
  g.zeros();

  CMatrix gParams(1, pkern->getNumParams());
  CMatrix gZ(numActive, latentDim);
  CMatrix gMu(numData, latentDim);
  CMatrix gS(numData, latentDim);
  pkern->getGradPsi(gParams, gZ, gMu, gS, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z, false);

  CMatrix gKmmParams(1, pkern->getNumParams());
  pkern->getGradParams(gKmmParams, Z, dL_dKmm, false);
  gParams.axpy(gKmmParams, 1.0);

  // Z from Kmm (CGp inducing pattern).
  CMatrix dgKX(numActive, latentDim);
  pkern->getDiagGradX(dgKX, Z);
  for(unsigned int i = 0; i < numActive; i++)
  {
    CMatrix gKX(numActive, latentDim);
    pkern->getGradX(gKX, Z, i, Z, false);
    gKX.scale(2.0);
    gKX.copyRowRow(i, dgKX, i);
    for(unsigned int j = 0; j < latentDim; j++)
      gZ.addVal(gKX.dotColCol(j, dL_dKmm, i), i, j);
  }

  // ELBO = L_var - KL
  CMatrix gMuKl(numData, latentDim);
  CMatrix gSKl(numData, latentDim);
  klGradient(gMuKl, gSKl);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
    {
      gMu.addVal(-gMuKl.getVal(n, q), n, q);
      gS.addVal(-gSKl.getVal(n, q), n, q);
    }

  for(size_t t = 0; t < pkern->getNumTransforms(); t++)
  {
    const unsigned int ind = pkern->getTransformIndex(t);
    const double x = pkern->getParam(ind);
    gParams.setVal(gParams.getVal(ind) * pkern->getTransform(t)->gradfact(x), ind);
  }

  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    g.setVal(gParams.getVal(i), counter++);

  g.setVal(dL_dbeta * betaTransform->gradfact(beta), counter++);

  for(unsigned int m = 0; m < numActive; m++)
    for(unsigned int q = 0; q < latentDim; q++)
      g.setVal(gZ.getVal(m, q), counter++);

  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      g.setVal(gMu.getVal(n, q), counter++);

  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      g.setVal(gS.getVal(n, q) * sTransform->gradfact(S.getVal(n, q)), counter++);

  DIMENSIONMATCH(counter == getOptNumParams());
  return logLikelihood();
}

unsigned int CBayesGplvm::getOptNumParams() const
{
  return pkern->getNumParams() + 1
    + numActive * latentDim
    + numData * latentDim
    + numData * latentDim;
}

void CBayesGplvm::getOptParams(CMatrix& param) const
{
  DIMENSIONMATCH(param.getRows() == 1);
  DIMENSIONMATCH(param.getCols() == getOptNumParams());
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
  {
    double val = pkern->getParam(i);
    // unconstrained
    bool done = false;
    for(size_t t = 0; t < pkern->getNumTransforms(); t++)
    {
      if(pkern->getTransformIndex(t) == i)
      {
        val = pkern->getTransform(t)->xtoa(val);
        done = true;
        break;
      }
    }
    (void)done;
    param.setVal(val, counter++);
  }
  param.setVal(betaTransform->xtoa(beta), counter++);
  for(unsigned int m = 0; m < numActive; m++)
    for(unsigned int q = 0; q < latentDim; q++)
      param.setVal(Z.getVal(m, q), counter++);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      param.setVal(mu.getVal(n, q), counter++);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      param.setVal(sTransform->xtoa(S.getVal(n, q)), counter++);
}

void CBayesGplvm::setOptParams(const CMatrix& param)
{
  DIMENSIONMATCH(param.getRows() == 1);
  DIMENSIONMATCH(param.getCols() == getOptNumParams());
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
  {
    double val = param.getVal(counter++);
    for(size_t t = 0; t < pkern->getNumTransforms(); t++)
    {
      if(pkern->getTransformIndex(t) == i)
      {
        val = pkern->getTransform(t)->atox(val);
        break;
      }
    }
    pkern->setParam(val, i);
  }
  beta = betaTransform->atox(param.getVal(counter++));
  for(unsigned int m = 0; m < numActive; m++)
    for(unsigned int q = 0; q < latentDim; q++)
      Z.setVal(param.getVal(counter++), m, q);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      mu.setVal(param.getVal(counter++), n, q);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      S.setVal(sTransform->atox(param.getVal(counter++)), n, q);
  setKupToDate(false);
}

void CBayesGplvm::optimise(unsigned int iters)
{
  if(getVerbosity() > 2 && getOptNumParams() < 40)
    checkGradients();
  setMaxIters(iters);
  runDefaultOptimiser();
}

void CBayesGplvm::display(ostream& os) const
{
  os << "Bayesian GPLVM:" << endl;
  os << "  N=" << numData << " D=" << outputDim << " q=" << latentDim
     << " M=" << numActive << " beta=" << beta << endl;
  os << "  ELBO=" << logLikelihood() << endl;
  pkern->display(os);
}

void CBayesGplvm::writeParamsToStream(ostream& os) const
{
  writeToStream(os, "baseType", "bayesgplvm");
  writeToStream(os, "type", "bayesgplvm");
  writeToStream(os, "numData", numData);
  writeToStream(os, "outputDim", outputDim);
  writeToStream(os, "latentDim", latentDim);
  writeToStream(os, "numActive", numActive);
  writeToStream(os, "beta", beta);
  Y.toStream(os);
  mu.toStream(os);
  S.toStream(os);
  Z.toStream(os);
  pkern->toStream(os);
}

void CBayesGplvm::readParamsFromStream(istream& is)
{
  string tbase = getBaseTypeStream(is);
  if(tbase != "bayesgplvm")
    throw ndlexceptions::StreamFormatError("baseType", "Expected bayesgplvm");
  string t = getTypeStream(is);
  if(t != "bayesgplvm")
    throw ndlexceptions::StreamFormatError("type", "Expected bayesgplvm");
  numData = readIntFromStream(is, "numData");
  outputDim = readIntFromStream(is, "outputDim");
  latentDim = readIntFromStream(is, "latentDim");
  numActive = readIntFromStream(is, "numActive");
  beta = readDoubleFromStream(is, "beta");
  initStoreage();
  Y.fromStream(is);
  mu.fromStream(is);
  S.fromStream(is);
  Z.fromStream(is);
  CKern* loaded = readKernFromStream(is);
  if(ownKern)
    delete pkern;
  pkern = loaded;
  ownKern = true;
  setKupToDate(false);
}
