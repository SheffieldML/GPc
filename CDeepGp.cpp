#include "CDeepGp.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;

namespace {

void pcaTop(const CMatrix& X, unsigned int D, CMatrix& P)
{
  const unsigned int d = X.getCols();
  P.resize(d, D);
  if(D > d)
    throw ndlexceptions::Error("CDeepGp: PCA width cannot exceed input dim");
  if(D == d)
  {
    P.zeros();
    P.addDiag(1.0);
    return;
  }
  CMatrix C(d, d);
  C.gemm(X, X, 1.0, 0.0, "t", "n");
  C.setSymmetric(true);
  CMatrix residual(C);
  for(unsigned int k = 0; k < D; k++)
  {
    CMatrix v(d, 1);
    for(unsigned int i = 0; i < d; i++)
      v.setVal((i == k % d) ? 1.0 : 0.01 * (double)(i + 1), i, 0);
    for(unsigned int it = 0; it < 40; it++)
    {
      CMatrix w(d, 1);
      w.gemv(residual, v, 1.0, 0.0, "n");
      double nrm = 0.0;
      for(unsigned int i = 0; i < d; i++)
        nrm += w.getVal(i) * w.getVal(i);
      nrm = sqrt(max(nrm, 1e-30));
      for(unsigned int i = 0; i < d; i++)
        v.setVal(w.getVal(i) / nrm, i, 0);
    }
    for(unsigned int i = 0; i < d; i++)
      P.setVal(v.getVal(i), i, k);
    CMatrix Cv(d, 1);
    Cv.gemv(residual, v, 1.0, 0.0, "n");
    double lam = 0.0;
    for(unsigned int i = 0; i < d; i++)
      lam += v.getVal(i) * Cv.getVal(i);
    for(unsigned int i = 0; i < d; i++)
      for(unsigned int j = 0; j < d; j++)
        residual.setVal(residual.getVal(i, j) - lam * v.getVal(i) * v.getVal(j),
                        i, j);
    residual.setSymmetric(true);
  }
}

void packOptIntoStochastic(CSvgp& layer, const CMatrix& gKern, const CMatrix& gZ,
                           double gBeta, const CMatrix& gQ, CMatrix& gOut)
{
  const unsigned int nStoch = layer.getStochasticNumParams();
  const unsigned int nOpt = layer.getOptNumParams();
  gOut.resize(1, nStoch);
  gOut.zeros();
  unsigned int counter = 0;
  for(unsigned int i = 0; i < gKern.getCols(); i++)
    gOut.setVal(gKern.getVal(i), counter++);
  for(unsigned int j = 0; j < gZ.getCols(); j++)
    for(unsigned int i = 0; i < gZ.getRows(); i++)
      gOut.setVal(gZ.getVal(i, j), counter++);
  while(counter < nOpt - 1)
    gOut.setVal(0.0, counter++);
  if(nOpt > 0 && counter == nOpt - 1)
    gOut.setVal(gBeta, counter++);
  for(unsigned int i = 0; i < gQ.getCols(); i++)
    gOut.setVal(gQ.getVal(i), counter++);
}

} // namespace

CDeepGp::CDeepGp(const CMatrix& Xin, const CMatrix& Yin,
                 const vector<CKern*>& kerns,
                 const vector<unsigned int>& dims,
                 const vector<unsigned int>& numInducing,
                 double beta, double chi, int verbos)
  : X(Xin), Y(Yin), nData(Xin.getRows()), inputDim(Xin.getCols()),
    outputDim(Yin.getCols()), numLayers(kerns.size()), verbosity(verbos),
    pkerns(kerns), layerDims(dims), M(numInducing), numSamples(1), seed(0),
    rng(0), freezeNoise(false), natGradGamma(0.0)
{
  if(Xin.getRows() != Yin.getRows())
    throw ndlexceptions::Error("CDeepGp: X and Y must have the same number of rows");
  if(numLayers == 0)
    throw ndlexceptions::Error("CDeepGp: need at least one layer");
  if(layerDims.size() != numLayers + 1)
    throw ndlexceptions::Error("CDeepGp: dims must have L+1 entries");
  if(M.size() != numLayers)
    throw ndlexceptions::Error("CDeepGp: numInducing must have L entries");
  if(layerDims[0] != inputDim || layerDims[numLayers] != outputDim)
    throw ndlexceptions::Error("CDeepGp: dims[0]/dims[L] must match X/Y columns");
  if(beta <= 0.0)
    throw ndlexceptions::Error("CDeepGp: beta must be positive");
  for(unsigned int ell = 0; ell < numLayers; ell++)
    if(M[ell] == 0 || M[ell] > nData)
      throw ndlexceptions::Error("CDeepGp: numInducing must be in 1..N");

  meanP.resize(numLayers);
  Zstore.resize(numLayers);
  hiddenY.resize(numLayers);
  layers.resize(numLayers);

  initMeanMaps();
  initInducing();
  initKernVariances(chi);

  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    if(pkerns[ell] == 0)
      throw ndlexceptions::Error("CDeepGp: null kernel");
    if(pkerns[ell]->getInputDim() != layerDims[ell])
      throw ndlexceptions::Error("CDeepGp: kernel input dim mismatch at layer");

    const CMatrix* pY = &Y;
    if(ell + 1 < numLayers)
    {
      hiddenY[ell].resize(nData, layerDims[ell + 1]);
      hiddenY[ell].zeros();
      pY = &hiddenY[ell];
    }
    layers[ell].reset(new CSvgp(pkerns[ell], &X, pY, Zstore[ell], beta, verbosity));
    layers[ell]->setQParamMode(CSvgp::QModeCholesky);
    if(ell + 1 < numLayers)
    {
      CMatrix m(layers[ell]->getNumInducing(), layers[ell]->getOutputDim());
      m.zeros();
      vector<CMatrix> S(layers[ell]->getOutputDim());
      for(unsigned int j = 0; j < S.size(); j++)
      {
        S[j].resize(layers[ell]->getNumInducing(), layers[ell]->getNumInducing());
        S[j].zeros();
        S[j].addDiag(1e-4);
        S[j].setSymmetric(true);
      }
      layers[ell]->setWhitenedQ(m, S);
    }
  }
  packLayerOffsets();
}

CDeepGp::~CDeepGp() {}

unsigned int CDeepGp::getHiddenDim(unsigned int ell) const
{
  if(ell > numLayers)
    throw ndlexceptions::Error("CDeepGp::getHiddenDim: layer out of range");
  return layerDims[ell];
}

CSvgp& CDeepGp::layer(unsigned int ell)
{
  if(ell >= numLayers)
    throw ndlexceptions::Error("CDeepGp::layer: index out of range");
  return *layers[ell];
}

const CSvgp& CDeepGp::layer(unsigned int ell) const
{
  if(ell >= numLayers)
    throw ndlexceptions::Error("CDeepGp::layer: index out of range");
  return *layers[ell];
}

void CDeepGp::initMeanMaps()
{
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    const unsigned int Din = layerDims[ell];
    const unsigned int Dout = layerDims[ell + 1];
    meanP[ell].resize(Din, Dout);
    meanP[ell].zeros();
    if(ell + 1 == numLayers)
      continue;
    if(ell == 0 && Din != Dout)
      pcaTop(X, Dout, meanP[ell]);
    else if(Din == Dout)
      meanP[ell].addDiag(1.0);
    else
      throw ndlexceptions::Error(
          "CDeepGp: non-square mean map only supported on first layer (PCA)");
  }
}

void CDeepGp::initInducing()
{
  vector<unsigned long> permL = ndlutil::randperm(nData);
  vector<unsigned int> perm(nData);
  for(unsigned int i = 0; i < nData; i++)
    perm[i] = (unsigned int)permL[i];

  CMatrix H(nData, inputDim);
  H.deepCopy(X);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    const unsigned int Din = layerDims[ell];
    const unsigned int Dout = layerDims[ell + 1];
    Zstore[ell].resize(M[ell], Din);
    for(unsigned int m = 0; m < M[ell]; m++)
      for(unsigned int d = 0; d < Din; d++)
        Zstore[ell].setVal(H.getVal(perm[m], d), m, d);

    if(ell + 1 < numLayers)
    {
      CMatrix Hnext(nData, Dout);
      Hnext.gemm(H, meanP[ell], 1.0, 0.0, "n", "n");
      H.deepCopy(Hnext);
    }
  }
}

void CDeepGp::initKernVariances(double chi)
{
  if(!(chi > 0.0))
    throw ndlexceptions::Error("CDeepGp: chi must be positive");
  if(numLayers == 1)
    return;
  const double hidVar = chi / (double)(numLayers - 1);
  for(unsigned int ell = 0; ell + 1 < numLayers; ell++)
  {
    CKern* k = pkerns[ell];
    if(k->getNumParams() >= 2)
    {
      try { k->setParam(hidVar, 1); }
      catch(...) {}
    }
    else if(k->getNumParams() == 1)
    {
      try { k->setParam(hidVar, 0); }
      catch(...) {}
    }
  }
}

void CDeepGp::setNumSamples(unsigned int K)
{
  if(K == 0)
    throw ndlexceptions::Error("CDeepGp::setNumSamples: K >= 1");
  numSamples = K;
}

void CDeepGp::setSeed(uint64_t s)
{
  seed = s;
  rng.seed(s);
}

void CDeepGp::setFrozenEps(const CMatrix& eps)
{
  frozenEps.deepCopy(eps);
  freezeNoise = true;
}

void CDeepGp::setLayerInducingType(unsigned int ell, const string& typeName)
{
  if(ell >= numLayers)
    throw ndlexceptions::Error("CDeepGp::setLayerInducingType: layer out of range");
  layers[ell]->setInducingType(typeName);
  layers[ell]->setQParamMode(CSvgp::QModeCholesky);
  if(ell + 1 < numLayers)
  {
    CMatrix m(layers[ell]->getNumInducing(), layers[ell]->getOutputDim());
    m.zeros();
    vector<CMatrix> S(layers[ell]->getOutputDim());
    for(unsigned int j = 0; j < S.size(); j++)
    {
      S[j].resize(layers[ell]->getNumInducing(), layers[ell]->getNumInducing());
      S[j].zeros();
      S[j].addDiag(1e-4);
      S[j].setSymmetric(true);
    }
    layers[ell]->setWhitenedQ(m, S);
  }
  packLayerOffsets();
}

void CDeepGp::setLayerLearnWindowWidth(unsigned int ell, bool learn, bool untied)
{
  if(ell >= numLayers)
    throw ndlexceptions::Error("CDeepGp::setLayerLearnWindowWidth: layer out of range");
  CInducingWeightSpace* pWs =
      dynamic_cast<CInducingWeightSpace*>(layers[ell]->getInducing());
  if(pWs == 0)
    throw ndlexceptions::Error(
        "CDeepGp::setLayerLearnWindowWidth: layer is not weight-space");
  pWs->setLearnWindowWidth(learn);
  pWs->setWindowWidthUntied(untied);
  packLayerOffsets();
}

void CDeepGp::setLikelihood(CNoise* noise)
{
  layers.back()->setLikelihood(noise);
}

CNoise* CDeepGp::getLikelihood() const
{
  return layers.back()->getLikelihood();
}

void CDeepGp::packLayerOffsets()
{
  layerParamOffset.resize(numLayers + 1);
  layerParamOffset[0] = 0;
  for(unsigned int ell = 0; ell < numLayers; ell++)
    layerParamOffset[ell + 1] =
      layerParamOffset[ell] + layers[ell]->getStochasticNumParams();
}

void CDeepGp::drawEps(CMatrix& eps, unsigned int nRows)
{
  eps.resize(nRows, 1);
  if(freezeNoise)
  {
    if(frozenEps.getRows() != nRows || frozenEps.getCols() != 1)
      throw ndlexceptions::Error("CDeepGp::drawEps: frozen eps shape mismatch");
    eps.deepCopy(frozenEps);
    return;
  }
  normal_distribution<double> nd(0.0, 1.0);
  for(unsigned int i = 0; i < nRows; i++)
    eps.setVal(nd(rng), i, 0);
}

unsigned int CDeepGp::getStochasticNumParams() const
{
  return layerParamOffset.back();
}

void CDeepGp::getStochasticParams(CMatrix& param) const
{
  const unsigned int P = getStochasticNumParams();
  param.resize(1, P);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    CMatrix pe;
    layers[ell]->getStochasticParams(pe);
    const unsigned int off = layerParamOffset[ell];
    for(unsigned int i = 0; i < pe.getCols(); i++)
      param.setVal(pe.getVal(i), off + i);
  }
}

void CDeepGp::setStochasticParams(const CMatrix& param)
{
  DIMENSIONMATCH(param.getRows() == 1 && param.getCols() == getStochasticNumParams());
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    const unsigned int off = layerParamOffset[ell];
    const unsigned int n = layerParamOffset[ell + 1] - off;
    CMatrix pe(1, n);
    for(unsigned int i = 0; i < n; i++)
      pe.setVal(param.getVal(off + i), i);
    layers[ell]->setStochasticParams(pe);
  }
}

double CDeepGp::elboEstimate()
{
  vector<unsigned int> all(nData);
  for(unsigned int i = 0; i < nData; i++)
    all[i] = i;
  return elboEstimate(all);
}

double CDeepGp::elboEstimate(const vector<unsigned int>& batch)
{
  CMatrix g;
  return objectiveGradBatch(batch, g);
}

double CDeepGp::objectiveGradBatch(const vector<unsigned int>& batch, CMatrix& g)
{
  if(batch.empty())
    throw ndlexceptions::Error("CDeepGp: empty batch");
  const unsigned int B = batch.size();
  const unsigned int K = numSamples;
  const unsigned int Ptot = getStochasticNumParams();
  g.resize(1, Ptot);
  g.zeros();

  if(numLayers == 1)
    return layers[0]->objectiveGradBatch(batch, g);

  unsigned int epsPerPoint = 0;
  for(unsigned int ell = 0; ell + 1 < numLayers; ell++)
    epsPerPoint += layerDims[ell + 1];
  vector<unsigned int> layerEpsOff(numLayers - 1);
  unsigned int offPoint = 0;
  for(unsigned int ell = 0; ell + 1 < numLayers; ell++)
  {
    layerEpsOff[ell] = offPoint;
    offPoint += layerDims[ell + 1];
  }

  CMatrix epsAll;
  drawEps(epsAll, B * K * epsPerPoint);

  const double c = (double)nData / ((double)B * (double)K);
  double Ldata = 0.0;

  vector<CMatrix> gLayer(numLayers);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    gLayer[ell].resize(1, layers[ell]->getStochasticNumParams());
    gLayer[ell].zeros();
  }

  for(unsigned int k = 0; k < K; k++)
  {
    vector<CMatrix> Hs(numLayers);
    vector<CMatrix> mus(numLayers - 1);
    vector<CMatrix> vars(numLayers - 1);
    vector<CMatrix> epss(numLayers - 1);

    Hs[0].resize(B, inputDim);
    for(unsigned int b = 0; b < B; b++)
      for(unsigned int d = 0; d < inputDim; d++)
        Hs[0].setVal(X.getVal(batch[b], d), b, d);

    for(unsigned int ell = 0; ell + 1 < numLayers; ell++)
    {
      const unsigned int Dout = layerDims[ell + 1];
      CMatrix mu, var;
      layers[ell]->marginalsAt(Hs[ell], mu, var);
      mus[ell].deepCopy(mu);
      vars[ell].deepCopy(var);
      epss[ell].resize(B, Dout);
      for(unsigned int b = 0; b < B; b++)
        for(unsigned int j = 0; j < Dout; j++)
          {
            const unsigned int idx =
              (k * B + b) * epsPerPoint + layerEpsOff[ell] + j;
            epss[ell].setVal(epsAll.getVal(idx, 0), b, j);
          }
      CMatrix Hnext(B, Dout);
      Hnext.gemm(Hs[ell], meanP[ell], 1.0, 0.0, "n", "n");
      for(unsigned int b = 0; b < B; b++)
        for(unsigned int j = 0; j < Dout; j++)
          {
            const double s = sqrt(max(vars[ell].getVal(b, j), 0.0));
            Hnext.setVal(Hnext.getVal(b, j) + mus[ell].getVal(b, j)
                         + s * epss[ell].getVal(b, j), b, j);
          }
      Hs[ell + 1].deepCopy(Hnext);
    }

    // c = N/(|B|K) already averages over samples; sum over k (do not /K again).
    const unsigned int Lf = numLayers - 1;
    CMatrix gKern, gZ, gX;
    double gBeta = 0.0;
    layers[Lf]->elboGradAt(gKern, gZ, gBeta, &gX, Hs[Lf], batch, c);
    Ldata += layers[Lf]->dataTermAt(Hs[Lf], batch, c);

    CMatrix gQf;
    layers[Lf]->qParamsGradAt(gQf, Hs[Lf], batch, c);
    CMatrix gFinal;
    packOptIntoStochastic(*layers[Lf], gKern, gZ, gBeta, gQf, gFinal);
    for(unsigned int i = 0; i < gFinal.getCols(); i++)
      gLayer[Lf].setVal(gLayer[Lf].getVal(i) + gFinal.getVal(i), i);

    CMatrix gH(gX);
    for(int ell = (int)numLayers - 2; ell >= 0; --ell)
    {
      const unsigned int Dout = layerDims[ell + 1];
      CMatrix gMean(B, Dout), gVar(B, Dout);
      for(unsigned int b = 0; b < B; b++)
        for(unsigned int j = 0; j < Dout; j++)
          {
            const double gh = gH.getVal(b, j);
            const double s = sqrt(max(vars[ell].getVal(b, j), 1e-300));
            gMean.setVal(gh, b, j);
            gVar.setVal(gh * epss[ell].getVal(b, j) / (2.0 * s), b, j);
          }
      CMatrix gKernH, gZH, gQH, gXH;
      layers[ell]->marginalsGradAt(Hs[ell], gMean, gVar, gKernH, gZH, gQH, &gXH);

      CMatrix gHprev(B, layerDims[ell]);
      gHprev.gemm(gH, meanP[ell], 1.0, 0.0, "n", "t");
      for(unsigned int b = 0; b < B; b++)
        for(unsigned int d = 0; d < layerDims[ell]; d++)
          gHprev.setVal(gHprev.getVal(b, d) + gXH.getVal(b, d), b, d);
      gH.deepCopy(gHprev);

      CMatrix gThis;
      packOptIntoStochastic(*layers[ell], gKernH, gZH, 0.0, gQH, gThis);
      for(unsigned int i = 0; i < gThis.getCols(); i++)
        gLayer[ell].setVal(gLayer[ell].getVal(i) + gThis.getVal(i), i);
    }
  }

  double L = Ldata;
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    CMatrix gKl;
    const double kl = layers[ell]->klQ(&gKl);
    L -= kl;
    L += pkerns[ell]->priorLogProb();
    const unsigned int nOpt = layers[ell]->getOptNumParams();
    for(unsigned int i = 0; i < gKl.getCols(); i++)
      gLayer[ell].setVal(gLayer[ell].getVal(nOpt + i) - gKl.getVal(i), nOpt + i);
  }

  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    const unsigned int off = layerParamOffset[ell];
    for(unsigned int i = 0; i < gLayer[ell].getCols(); i++)
      g.setVal(gLayer[ell].getVal(i), off + i);
  }
  return L;
}

void CDeepGp::afterBatchStep(const vector<unsigned int>& batch)
{
  if(natGradGamma > 0.0)
    layers.back()->naturalGradientStep(natGradGamma, batch);
}
