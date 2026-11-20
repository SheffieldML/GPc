#include "CSviBayesGplvm.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <cmath>

using namespace std;

CSviBayesGplvm::CSviBayesGplvm(CKern* kernel, const CMatrix& Yin,
                               unsigned int latDim, unsigned int numInd,
                               double beta, int verb)
  : pkern(kernel), ownKern(false), Y(Yin), psvgp(0), verbosity(verb),
    nData(Yin.getRows()), outputDim(Yin.getCols()), latentDim(latDim),
    numInducing(numInd), numSamples(1), antithetic(false), freezeNoise(false),
    rng(0), natGradGamma(0.1), localAlpha(1e-2), localBeta1(0.9),
    localBeta2(0.999), localEps(1e-8), localT(0)
{
  if(kernel == 0)
    throw ndlexceptions::Error("CSviBayesGplvm: null kernel");
  if(latDim == 0 || numInd == 0)
    throw ndlexceptions::Error("CSviBayesGplvm: latentDim and numInducing must be > 0");
  if(nData == 0 || outputDim == 0)
    throw ndlexceptions::Error("CSviBayesGplvm: empty Y");
  if(!(beta > 0.0))
    throw ndlexceptions::Error("CSviBayesGplvm: beta must be positive");

  mu.resize(nData, latentDim);
  logS.resize(nData, latentDim);
  Z.resize(numInducing, latentDim);
  initXpca();
  logS.setVals(log(0.5));
  initZFromMu();

  psvgp = new CSvgp(pkern, &mu, &Y, Z, beta, verbosity);
  psvgp->setNatGradGamma(natGradGamma);

  mMu.resize(nData, latentDim);
  vMu.resize(nData, latentDim);
  mLogS.resize(nData, latentDim);
  vLogS.resize(nData, latentDim);
  mMu.zeros();
  vMu.zeros();
  mLogS.zeros();
  vLogS.zeros();
}

CSviBayesGplvm::~CSviBayesGplvm()
{
  delete psvgp;
  if(ownKern)
    delete pkern;
}

void CSviBayesGplvm::initXpca()
{
  // Mean-subtracted PCA of Y into mu (same as CBayesGplvm::initXpca).
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

void CSviBayesGplvm::initZFromMu()
{
  for(unsigned int m = 0; m < numInducing; m++)
  {
    const unsigned int n = (nData == 0) ? 0 : (m * nData) / numInducing;
    for(unsigned int q = 0; q < latentDim; q++)
      Z.setVal(mu.getVal(n % nData, q), m, q);
  }
  if(psvgp != 0)
    psvgp->setInducingCentres(Z);
}

void CSviBayesGplvm::setInducingType(const string& typeName)
{
  if(psvgp == 0)
    throw ndlexceptions::Error("CSviBayesGplvm::setInducingType: no SVGP core");
  // Keep Z in sync for families that read centres storage (points / WN / greens).
  if(typeName == "points" || typeName == "whitenoise" || typeName == "greens")
    initZFromMu();
  psvgp->setInducingType(typeName);
}

string CSviBayesGplvm::getInducingType() const
{
  if(psvgp == 0 || psvgp->getInducing() == 0)
    return "points";
  return psvgp->getInducing()->typeName();
}

void CSviBayesGplvm::setLikelihood(CNoise* noise)
{
  if(psvgp == 0)
    throw ndlexceptions::Error("CSviBayesGplvm::setLikelihood: no SVGP core");
  psvgp->setLikelihood(noise);
}

CNoise* CSviBayesGplvm::getLikelihood() const
{
  return psvgp == 0 ? 0 : psvgp->getLikelihood();
}

void CSviBayesGplvm::setNumSamples(unsigned int K)
{
  if(K == 0)
    throw ndlexceptions::Error("CSviBayesGplvm::setNumSamples: K must be > 0");
  if(antithetic && (K % 2 != 0))
    antithetic = false; // odd K cannot pair; drop antithetic quietly
  numSamples = K;
}

void CSviBayesGplvm::setAntithetic(bool val)
{
  if(val && (numSamples % 2 != 0))
    throw ndlexceptions::Error("CSviBayesGplvm: antithetic requires even K");
  antithetic = val;
}

void CSviBayesGplvm::setSeed(uint64_t s)
{
  rng.seed(s);
}

void CSviBayesGplvm::setNatGradGamma(double g)
{
  if(!(g >= 0.0 && g <= 1.0))
    throw ndlexceptions::Error("CSviBayesGplvm::setNatGradGamma: gamma in [0, 1]");
  natGradGamma = g; // 0 disables CVI natgrad (FD / oracle tests)
  if(psvgp != 0 && g > 0.0)
    psvgp->setNatGradGamma(g);
}

void CSviBayesGplvm::setFrozenEps(const CMatrix& eps)
{
  frozenEps.deepCopy(eps);
  freezeNoise = true;
}

void CSviBayesGplvm::setLocalLearningRate(double lr)
{
  if(lr < 0.0)
    throw ndlexceptions::Error("CSviBayesGplvm::setLocalLearningRate: lr must be >= 0");
  localAlpha = lr; // 0 disables row-sparse Adam (oracle / FD tests)
}

CMatrix CSviBayesGplvm::getS() const
{
  CMatrix S(logS.getRows(), logS.getCols());
  for(unsigned int i = 0; i < S.getRows(); i++)
    for(unsigned int j = 0; j < S.getCols(); j++)
      S.setVal(exp(logS.getVal(i, j)), i, j);
  return S;
}

void CSviBayesGplvm::setMu(const CMatrix& muIn)
{
  DIMENSIONMATCH(muIn.getRows() == nData && muIn.getCols() == latentDim);
  mu.deepCopy(muIn);
  psvgp->updateKernelMatrices();
}

void CSviBayesGplvm::setLogS(const CMatrix& logSIn)
{
  DIMENSIONMATCH(logSIn.getRows() == nData && logSIn.getCols() == latentDim);
  logS.deepCopy(logSIn);
}

void CSviBayesGplvm::setS(const CMatrix& SIn)
{
  DIMENSIONMATCH(SIn.getRows() == nData && SIn.getCols() == latentDim);
  for(unsigned int i = 0; i < nData; i++)
    for(unsigned int j = 0; j < latentDim; j++)
    {
      const double s = SIn.getVal(i, j);
      if(!(s > 0.0))
        throw ndlexceptions::Error("CSviBayesGplvm::setS: variances must be positive");
      logS.setVal(log(s), i, j);
    }
}

double CSviBayesGplvm::klX() const
{
  double kl = 0.0;
  for(unsigned int n = 0; n < nData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
    {
      const double m = mu.getVal(n, q);
      const double s = exp(logS.getVal(n, q));
      kl += m * m + s - log(s) - 1.0;
    }
  return 0.5 * kl;
}

void CSviBayesGplvm::drawEps(CMatrix& eps, unsigned int nRows)
{
  eps.resize(nRows, latentDim);
  if(freezeNoise)
  {
    if(frozenEps.getRows() != nRows || frozenEps.getCols() != latentDim)
      throw ndlexceptions::Error("CSviBayesGplvm: frozenEps shape mismatch");
    eps.deepCopy(frozenEps);
    return;
  }
  normal_distribution<double> nd(0.0, 1.0);
  for(unsigned int i = 0; i < nRows; i++)
    for(unsigned int q = 0; q < latentDim; q++)
      eps.setVal(nd(rng), i, q);
}

void CSviBayesGplvm::formSamples(const vector<unsigned int>& batch, CMatrix& XB,
                                 vector<unsigned int>& yRows, CMatrix& epsOut)
{
  const unsigned int B = batch.size();
  const unsigned int K = numSamples;
  const unsigned int BK = B * K;
  yRows.resize(BK);
  XB.resize(BK, latentDim);
  drawEps(epsOut, BK);

  if(antithetic)
  {
    // Pair (ε, −ε) within each point's K samples.
    for(unsigned int b = 0; b < B; b++)
      for(unsigned int k = 0; k < K; k += 2)
      {
        const unsigned int r0 = b * K + k;
        const unsigned int r1 = b * K + k + 1;
        for(unsigned int q = 0; q < latentDim; q++)
          epsOut.setVal(-epsOut.getVal(r0, q), r1, q);
      }
  }

  for(unsigned int b = 0; b < B; b++)
  {
    const unsigned int n = batch[b];
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int r = b * K + k;
      yRows[r] = n;
      for(unsigned int q = 0; q < latentDim; q++)
      {
        const double s = exp(0.5 * logS.getVal(n, q)); // s = sqrt(S)
        const double x = mu.getVal(n, q) + s * epsOut.getVal(r, q);
        XB.setVal(x, r, q);
      }
    }
  }
}

void CSviBayesGplvm::mapLocalGrads(const vector<unsigned int>& batch,
                                   const CMatrix& gX, const CMatrix& eps,
                                   CMatrix& gMu, CMatrix& gLogS) const
{
  const unsigned int B = batch.size();
  const unsigned int K = numSamples;
  gMu.resize(nData, latentDim);
  gLogS.resize(nData, latentDim);
  gMu.zeros();
  gLogS.zeros();
  for(unsigned int b = 0; b < B; b++)
  {
    const unsigned int n = batch[b];
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int r = b * K + k;
      for(unsigned int q = 0; q < latentDim; q++)
      {
        const double gx = gX.getVal(r, q);
        const double s = exp(0.5 * logS.getVal(n, q));
        const double ep = eps.getVal(r, q);
        gMu.addVal(gx, n, q);
        // ∂L/∂s = gx * ε; ∂L/∂logS = (1/2) s * ∂L/∂s = (1/2) s * gx * ε
        gLogS.addVal(0.5 * s * gx * ep, n, q);
      }
    }
  }
}

void CSviBayesGplvm::addKlXGrad(const vector<unsigned int>& batch, double scale,
                                CMatrix& gMu, CMatrix& gLogS) const
{
  // ∂(−KL)/∂μ = −μ; ∂(−KL)/∂logS = −½(S − 1) = ½(1 − S)
  for(unsigned int b = 0; b < batch.size(); b++)
  {
    const unsigned int n = batch[b];
    for(unsigned int q = 0; q < latentDim; q++)
    {
      const double m = mu.getVal(n, q);
      const double S = exp(logS.getVal(n, q));
      gMu.addVal(-scale * m, n, q);
      gLogS.addVal(scale * 0.5 * (1.0 - S), n, q);
    }
  }
}

void CSviBayesGplvm::localAdamStep(const vector<unsigned int>& batch,
                                   const CMatrix& gMu, const CMatrix& gLogS)
{
  localT++;
  const double bc1 = 1.0 - pow(localBeta1, (double)localT);
  const double bc2 = 1.0 - pow(localBeta2, (double)localT);
  for(unsigned int b = 0; b < batch.size(); b++)
  {
    const unsigned int n = batch[b];
    for(unsigned int q = 0; q < latentDim; q++)
    {
      // μ
      {
        const double gi = gMu.getVal(n, q);
        const double mi = localBeta1 * mMu.getVal(n, q) + (1.0 - localBeta1) * gi;
        const double vi = localBeta2 * vMu.getVal(n, q) + (1.0 - localBeta2) * gi * gi;
        mMu.setVal(mi, n, q);
        vMu.setVal(vi, n, q);
        const double mhat = mi / bc1;
        const double vhat = vi / bc2;
        mu.setVal(mu.getVal(n, q) + localAlpha * mhat / (sqrt(vhat) + localEps), n, q);
      }
      // logS
      {
        const double gi = gLogS.getVal(n, q);
        const double mi = localBeta1 * mLogS.getVal(n, q) + (1.0 - localBeta1) * gi;
        const double vi = localBeta2 * vLogS.getVal(n, q) + (1.0 - localBeta2) * gi * gi;
        mLogS.setVal(mi, n, q);
        vLogS.setVal(vi, n, q);
        const double mhat = mi / bc1;
        const double vhat = vi / bc2;
        logS.setVal(logS.getVal(n, q) + localAlpha * mhat / (sqrt(vhat) + localEps), n, q);
      }
    }
  }
}

double CSviBayesGplvm::elboEstimate(const vector<unsigned int>& batch)
{
  if(batch.empty())
    throw ndlexceptions::Error("CSviBayesGplvm::elboEstimate: empty batch");
  CMatrix XB, eps;
  vector<unsigned int> yRows;
  formSamples(batch, XB, yRows, eps);
  const double c = (double)nData / (double)(batch.size() * numSamples);
  const double Ldata = psvgp->dataTermAt(XB, yRows, c);
  double LklU = 0.0;
  for(unsigned int j = 0; j < outputDim; j++)
    LklU -= psvgp->klDivergence(j);
  LklU += pkern->priorLogProb();
  // −KL(q(X)) on batch, scaled.
  double LklX = 0.0;
  const double scale = (double)nData / (double)batch.size();
  for(unsigned int b = 0; b < batch.size(); b++)
  {
    const unsigned int n = batch[b];
    for(unsigned int q = 0; q < latentDim; q++)
    {
      const double m = mu.getVal(n, q);
      const double s = exp(logS.getVal(n, q));
      LklX += 0.5 * (m * m + s - log(s) - 1.0);
    }
  }
  return Ldata + LklU - scale * LklX;
}

double CSviBayesGplvm::elboEstimate()
{
  vector<unsigned int> all(nData);
  for(unsigned int i = 0; i < nData; i++)
    all[i] = i;
  return elboEstimate(all);
}

unsigned int CSviBayesGplvm::getStochasticNumParams() const
{
  return psvgp->getOptNumParams();
}

void CSviBayesGplvm::getStochasticParams(CMatrix& param) const
{
  psvgp->getOptParams(param);
}

void CSviBayesGplvm::setStochasticParams(const CMatrix& param)
{
  psvgp->setOptParams(param);
  // Keep local Z mirror in sync for initZFromMu callers.
  Z.deepCopy(psvgp->getInducingCentres());
}

double CSviBayesGplvm::objectiveGradBatch(const vector<unsigned int>& batch,
                                          CMatrix& g)
{
  if(batch.empty())
    throw ndlexceptions::Error("CSviBayesGplvm::objectiveGradBatch: empty batch");

  CMatrix XB, eps;
  vector<unsigned int> yRows;
  formSamples(batch, XB, yRows, eps);
  const double c = (double)nData / (double)(batch.size() * numSamples);

  CMatrix gKern, gZ, gX;
  double gBeta = 0.0;
  const double Lcore = psvgp->elboGradAt(gKern, gZ, gBeta, &gX, XB, yRows, c);

  CMatrix gMu, gLogS;
  mapLocalGrads(batch, gX, eps, gMu, gLogS);
  const double scale = (double)nData / (double)batch.size();
  addKlXGrad(batch, scale, gMu, gLogS);

  // −KL(q(X)) contribution to objective (Lcore already has data + KL(q(u)) + prior).
  double LklX = 0.0;
  for(unsigned int b = 0; b < batch.size(); b++)
  {
    const unsigned int n = batch[b];
    for(unsigned int q = 0; q < latentDim; q++)
    {
      const double m = mu.getVal(n, q);
      const double s = exp(logS.getVal(n, q));
      LklX += 0.5 * (m * m + s - log(s) - 1.0);
    }
  }
  const double L = Lcore - scale * LklX;

  // Pack global grads (same layout as CSvgp).
  g.resize(1, getStochasticNumParams());
  unsigned int counter = 0;
  for(unsigned int i = 0; i < pkern->getNumParams(); i++)
    g.setVal(gKern.getVal(i), counter++);
  const unsigned int indStart = counter;
  const unsigned int Mz = gZ.getRows();
  const unsigned int dz = gZ.getCols();
  for(unsigned int j = 0; j < dz; j++)
    for(unsigned int i = 0; i < Mz; i++)
      g.setVal(gZ.getVal(i, j), counter++);
  while(counter < indStart + psvgp->getInducing()->getNumParams())
    g.setVal(0.0, counter++);
  g.setVal(gBeta, counter++);

  // Row-sparse Adam on local params (ascend). lr=0 disables (FD / oracles).
  if(localAlpha > 0.0)
  {
    localAdamStep(batch, gMu, gLogS);
    psvgp->updateKernelMatrices();
  }

  return L;
}

void CSviBayesGplvm::afterBatchStep(const vector<unsigned int>& batch)
{
  if(!(natGradGamma > 0.0))
    return;
  // Rebuild samples for sites (same freeze / seed stream as objective if frozen).
  CMatrix XB, eps;
  vector<unsigned int> yRows;
  formSamples(batch, XB, yRows, eps);
  const double c = (double)nData / (double)(batch.size() * numSamples);
  psvgp->naturalGradientStepAt(natGradGamma, XB, yRows, c);
}

void CSviBayesGplvm::display(ostream& os) const
{
  os << "SVI Bayesian GPLVM (uncollapsed, sampled q(X)):" << endl;
  os << "  N=" << nData << " D=" << outputDim << " q=" << latentDim
     << " M=" << numInducing
     << " beta=" << (psvgp == 0 ? 0.0 : psvgp->getBeta())
     << " inducing=" << getInducingType() << endl;
  os << "  samples=" << numSamples
     << " antithetic=" << (antithetic ? 1 : 0)
     << " natgrad_gamma=" << natGradGamma
     << " local_lr=" << localAlpha << endl;
  os << "  KL(q(X)||p(X))=" << klX() << endl;
  if(pkern != 0)
    pkern->display(os);
}

void CSviBayesGplvm::writeParamsToStream(ostream& os) const
{
  if(psvgp == 0 || pkern == 0)
    throw ndlexceptions::Error("CSviBayesGplvm::writeParamsToStream: uninitialised model");
  writeToStream(os, "baseType", "svibayesgplvm");
  writeToStream(os, "type", "svibayesgplvm");
  writeToStream(os, "numData", nData);
  writeToStream(os, "outputDim", outputDim);
  writeToStream(os, "latentDim", latentDim);
  writeToStream(os, "numInducing", numInducing);
  writeToStream(os, "beta", psvgp->getBeta());
  writeToStream(os, "numSamples", numSamples);
  writeToStream(os, "antithetic", antithetic);
  writeToStream(os, "natGradGamma", natGradGamma);
  writeToStream(os, "localLr", localAlpha);
  writeToStream(os, "inducingType", getInducingType());
  writeToStream(os, "windowWidth", psvgp->getInducing()->getWindowWidth());
  Y.toStream(os);
  mu.toStream(os);
  logS.toStream(os);
  const unsigned int nInd = psvgp->getInducing()->getNumParams();
  writeToStream(os, "numInducingParams", nInd);
  CMatrix indP(1, nInd > 0 ? nInd : 1);
  indP.zeros();
  if(nInd > 0)
    psvgp->getInducing()->getParams(indP, 0);
  indP.toStream(os);
  psvgp->getWhitenedMean().toStream(os);
  for(unsigned int j = 0; j < outputDim; j++)
    psvgp->getWhitenedCov(j).toStream(os);
  pkern->toStream(os);
}

void CSviBayesGplvm::readParamsFromStream(istream& is)
{
  string tbase = getBaseTypeStream(is);
  if(tbase != "svibayesgplvm")
    throw ndlexceptions::StreamFormatError("baseType", "Expected svibayesgplvm");
  string t = getTypeStream(is);
  if(t != "svibayesgplvm")
    throw ndlexceptions::StreamFormatError("type", "Expected svibayesgplvm");
  nData = (unsigned int)readIntFromStream(is, "numData");
  outputDim = (unsigned int)readIntFromStream(is, "outputDim");
  latentDim = (unsigned int)readIntFromStream(is, "latentDim");
  numInducing = (unsigned int)readIntFromStream(is, "numInducing");
  const double beta = readDoubleFromStream(is, "beta");
  numSamples = (unsigned int)readIntFromStream(is, "numSamples");
  antithetic = readBoolFromStream(is, "antithetic");
  natGradGamma = readDoubleFromStream(is, "natGradGamma");
  localAlpha = readDoubleFromStream(is, "localLr");
  const string indType = readStringFromStream(is, "inducingType");
  const double winW = readDoubleFromStream(is, "windowWidth");

  if(nData == 0 || outputDim == 0 || latentDim == 0 || numInducing == 0)
    throw ndlexceptions::StreamFormatError("dims", "SVI Bayesian GPLVM dims must be > 0");
  if(!(beta > 0.0))
    throw ndlexceptions::StreamFormatError("beta", "beta must be positive");

  Y.resize(nData, outputDim);
  mu.resize(nData, latentDim);
  logS.resize(nData, latentDim);
  Z.resize(numInducing, latentDim);
  Y.fromStream(is);
  mu.fromStream(is);
  logS.fromStream(is);

  const unsigned int nInd = (unsigned int)readIntFromStream(is, "numInducingParams");
  CMatrix indP(1, nInd > 0 ? nInd : 1);
  indP.fromStream(is);

  CMatrix qM(numInducing, outputDim);
  qM.fromStream(is);
  vector<CMatrix> qC(outputDim);
  for(unsigned int j = 0; j < outputDim; j++)
  {
    qC[j].resize(numInducing, numInducing);
    qC[j].fromStream(is);
  }

  CKern* loaded = readKernFromStream(is);
  if(ownKern)
    delete pkern;
  pkern = loaded;
  ownKern = true;

  delete psvgp;
  Z.zeros();
  psvgp = new CSvgp(pkern, &mu, &Y, Z, beta, verbosity);
  if(indType != "points")
    psvgp->setInducingType(indType);
  if(winW > 0.0)
    psvgp->getInducing()->setWindowWidth(winW);
  if(nInd > 0)
  {
    if(nInd != psvgp->getInducing()->getNumParams())
      throw ndlexceptions::StreamFormatError(
        "numInducingParams", "mismatch with inducing type");
    psvgp->getInducing()->setParams(indP, 0);
  }
  psvgp->updateKernelMatrices();
  psvgp->setWhitenedQ(qM, qC);
  if(natGradGamma > 0.0)
    psvgp->setNatGradGamma(natGradGamma);
  Z.deepCopy(psvgp->getInducingCentres());

  mMu.resize(nData, latentDim);
  vMu.resize(nData, latentDim);
  mLogS.resize(nData, latentDim);
  vLogS.resize(nData, latentDim);
  mMu.zeros();
  vMu.zeros();
  mLogS.zeros();
  vLogS.zeros();
  localT = 0;
  freezeNoise = false;
}
