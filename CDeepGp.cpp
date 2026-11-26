#include "CDeepGp.h"
#include "ndlexceptions.h"
#include "ndlutil.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

using namespace std;

namespace {

void pcaTop(const CMatrix& X, unsigned int D, CMatrix& P)
{
  const unsigned int d = X.getCols();
  if(D > d)
  {
    // CIP-003C: hidden width above the input dimension. Take all d principal
    // directions (an orthonormal basis of R^d) and pad the extra D-d columns
    // with zeros: those hidden units carry no mean-path signal and are driven
    // by the GP layer alone. d orthonormal vectors is the most R^d admits.
    CMatrix Pd;
    pcaTop(X, d, Pd);
    P.resize(d, D);
    P.zeros();
    for(unsigned int i = 0; i < d; i++)
      for(unsigned int j = 0; j < d; j++)
        P.setVal(Pd.getVal(i, j), i, j);
    return;
  }
  P.resize(d, D);
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
  // Fixed-window families (Hermite) reserve no inducing parameters: pack gZ only
  // up to the count the optimiser vector holds.
  const unsigned int zEnd = counter + layer.getInducing()->getNumParams();
  for(unsigned int j = 0; j < gZ.getCols(); j++)
    for(unsigned int i = 0; i < gZ.getRows(); i++)
      if(counter < zEnd)
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
    chiScale(chi), betaStore(beta), pkerns(kerns), ownKerns(false),
    layerDims(dims), M(numInducing), numSamples(1), seed(0),
    rng(0), freezeNoise(false), natGradGamma(0.0), dropResidual(false)
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

  inducingTypes.assign(numLayers, "points");
  rebuildLayers(beta);
  initKernVariances(chi);
  // Rebuild again so kernel variance changes take effect in matrices.
  for(unsigned int ell = 0; ell < numLayers; ell++)
    layers[ell]->updateKernelMatrices();
}

CDeepGp::~CDeepGp() {}

void CDeepGp::rebuildLayers(double beta)
{
  meanP.resize(numLayers);
  Zstore.resize(numLayers);
  hiddenY.resize(numLayers);
  layerX.resize(numLayers);
  layers.resize(numLayers);
  if(inducingTypes.size() != numLayers)
    inducingTypes.assign(numLayers, "points");

  initMeanMaps();
  initInducing();

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
    // Each core needs inputs with D_{ℓ-1} columns (it checks Z against them);
    // use X pushed through the mean maps. Training uses the explicit-input API.
    if(ell == 0)
      layerX[ell].deepCopy(X);
    else
    {
      layerX[ell].resize(nData, layerDims[ell]);
      layerX[ell].gemm(layerX[ell - 1], meanP[ell - 1], 1.0, 0.0, "n", "n");
    }
    layers[ell].reset(new CSvgp(pkerns[ell], &layerX[ell], pY, Zstore[ell], beta, verbosity));
    layers[ell]->setQParamMode(CSvgp::QModeCholesky);
    if(inducingTypes[ell] != "points")
      layers[ell]->setInducingType(inducingTypes[ell]);
    layers[ell]->setDropResidual(dropResidual);
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
  // Set on the layer first: a refused type (E1/E2/E5 for Hermite) must leave the
  // recorded type, and therefore the stream, unchanged.
  layers[ell]->setInducingType(typeName);
  inducingTypes[ell] = typeName;
  layers[ell]->setQParamMode(CSvgp::QModeCholesky);
  layers[ell]->setDropResidual(dropResidual);
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

string CDeepGp::getLayerInducingType(unsigned int ell) const
{
  if(ell >= numLayers)
    throw ndlexceptions::Error("CDeepGp::getLayerInducingType: layer out of range");
  return inducingTypes[ell];
}

void CDeepGp::setDropResidual(bool val)
{
  dropResidual = val;
  for(unsigned int ell = 0; ell < numLayers; ell++)
    layers[ell]->setDropResidual(val);
}

void CDeepGp::setLayerLearnWindowWidth(unsigned int ell, bool learn, bool untied)
{
  if(ell >= numLayers)
    throw ndlexceptions::Error("CDeepGp::setLayerLearnWindowWidth: layer out of range");
  if(layers[ell]->getInducing()->typeName() == "hermite" && learn)
    throw ndlexceptions::Error(
        "Hermite inducing (E3): Hermite windows have no width; learn_window_width cannot be enabled.");
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

void CDeepGp::forwardSample(const CMatrix& Xin, const CMatrix& epsFlat,
                            CMatrix& Hout) const
{
  const unsigned int B = Xin.getRows();
  vector<CMatrix> Hs(numLayers + 1);
  Hs[0].deepCopy(Xin);
  unsigned int epsOff = 0;
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    const unsigned int Dout = layerDims[ell + 1];
    CMatrix mu, var;
    layers[ell]->marginalsAt(Hs[ell], mu, var);
    CMatrix Hnext(B, Dout);
    Hnext.gemm(Hs[ell], meanP[ell], 1.0, 0.0, "n", "n");
    for(unsigned int b = 0; b < B; b++)
      for(unsigned int j = 0; j < Dout; j++)
      {
        const double s = sqrt(max(var.getVal(b, j), 0.0));
        const double ep = (epsFlat.getRows() > epsOff + b * Dout + j)
          ? epsFlat.getVal(epsOff + b * Dout + j, 0) : 0.0;
        Hnext.setVal(Hnext.getVal(b, j) + mu.getVal(b, j) + s * ep, b, j);
      }
    Hs[ell + 1].deepCopy(Hnext);
    epsOff += B * Dout;
  }
  Hout.deepCopy(Hs.back());
}

void CDeepGp::forwardPriorDraw(const CMatrix& Xin,
                               const vector<CMatrix>& vs,
                               CMatrix& Hout) const
{
  if(vs.size() != numLayers)
    throw ndlexceptions::Error(
        "CDeepGp::forwardPriorDraw: need one whitened draw per layer");
  const unsigned int B = Xin.getRows();
  CMatrix H(Xin);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    const unsigned int Dout = layerDims[ell + 1];
    CMatrix f;
    layers[ell]->applyWhitenedAt(H, vs[ell], f);
    CMatrix Hnext(B, Dout);
    Hnext.gemm(H, meanP[ell], 1.0, 0.0, "n", "n");
    for(unsigned int b = 0; b < B; b++)
      for(unsigned int j = 0; j < Dout; j++)
        Hnext.setVal(Hnext.getVal(b, j) + f.getVal(b, j), b, j);
    H.deepCopy(Hnext);
  }
  Hout.deepCopy(H);
}

void CDeepGp::predictMeanVar(const CMatrix& Xstar, CMatrix& mu, CMatrix& var,
                             unsigned int samples, uint64_t sampleSeed,
                             bool addNoise) const
{
  if(Xstar.getCols() != inputDim)
    throw ndlexceptions::Error(
        "CDeepGp::predictMeanVar: Xstar input dim mismatch");
  const unsigned int Ns = Xstar.getRows();
  const unsigned int Dout = outputDim;
  const double noise =
      (addNoise && layers.back()->getLikelihood() == 0
       && layers.back()->getBeta() > 0.0)
          ? (1.0 / layers.back()->getBeta())
          : 0.0;

  if(samples == 0 || numLayers == 1)
  {
    CMatrix H(Xstar);
    for(unsigned int ell = 0; ell + 1 < numLayers; ell++)
    {
      CMatrix m, v;
      layers[ell]->marginalsAt(H, m, v);
      CMatrix Hnext(H.getRows(), layerDims[ell + 1]);
      Hnext.gemm(H, meanP[ell], 1.0, 0.0, "n", "n");
      for(unsigned int i = 0; i < Hnext.getRows(); i++)
        for(unsigned int j = 0; j < Hnext.getCols(); j++)
          Hnext.setVal(Hnext.getVal(i, j) + m.getVal(i, j), i, j);
      H.deepCopy(Hnext);
    }
    layers.back()->marginalsAt(H, mu, var);
    if(noise > 0.0)
      for(unsigned int i = 0; i < var.getRows(); i++)
        for(unsigned int j = 0; j < var.getCols(); j++)
          var.setVal(var.getVal(i, j) + noise, i, j);
    return;
  }

  // Mixture of S Gaussians at the final layer (CIP-0035 prediction).
  unsigned int epsPerPoint = 0;
  for(unsigned int ell = 0; ell + 1 < numLayers; ell++)
    epsPerPoint += layerDims[ell + 1];
  mt19937_64 prng(sampleSeed);
  normal_distribution<double> nd(0.0, 1.0);

  CMatrix sumMu(Ns, Dout);
  CMatrix sumMuSq(Ns, Dout);
  CMatrix sumVar(Ns, Dout);
  sumMu.zeros();
  sumMuSq.zeros();
  sumVar.zeros();

  for(unsigned int s = 0; s < samples; s++)
  {
    CMatrix epsFlat(epsPerPoint > 0 ? Ns * epsPerPoint : 1, 1);
    epsFlat.zeros();
    for(unsigned int i = 0; i < Ns * epsPerPoint; i++)
      epsFlat.setVal(nd(prng), i, 0);

    CMatrix H(Xstar);
    unsigned int epsOff = 0;
    for(unsigned int ell = 0; ell + 1 < numLayers; ell++)
    {
      const unsigned int Dh = layerDims[ell + 1];
      CMatrix m, v;
      layers[ell]->marginalsAt(H, m, v);
      CMatrix Hnext(Ns, Dh);
      Hnext.gemm(H, meanP[ell], 1.0, 0.0, "n", "n");
      for(unsigned int b = 0; b < Ns; b++)
        for(unsigned int j = 0; j < Dh; j++)
        {
          const double sig = sqrt(max(v.getVal(b, j), 0.0));
          const double ep = epsFlat.getVal(epsOff + b * Dh + j, 0);
          Hnext.setVal(Hnext.getVal(b, j) + m.getVal(b, j) + sig * ep, b, j);
        }
      H.deepCopy(Hnext);
      epsOff += Ns * Dh;
    }
    CMatrix muS, varS;
    layers.back()->marginalsAt(H, muS, varS);
    for(unsigned int i = 0; i < Ns; i++)
      for(unsigned int j = 0; j < Dout; j++)
      {
        const double m = muS.getVal(i, j);
        const double vv = varS.getVal(i, j) + noise;
        sumMu.setVal(sumMu.getVal(i, j) + m, i, j);
        sumMuSq.setVal(sumMuSq.getVal(i, j) + m * m, i, j);
        sumVar.setVal(sumVar.getVal(i, j) + vv, i, j);
      }
  }
  const double invS = 1.0 / (double)samples;
  mu.resize(Ns, Dout);
  var.resize(Ns, Dout);
  for(unsigned int i = 0; i < Ns; i++)
    for(unsigned int j = 0; j < Dout; j++)
    {
      const double mbar = sumMu.getVal(i, j) * invS;
      const double eVar = sumVar.getVal(i, j) * invS;
      const double vMu = sumMuSq.getVal(i, j) * invS - mbar * mbar;
      mu.setVal(mbar, i, j);
      var.setVal(max(eVar + vMu, 0.0), i, j);
    }
}

double CDeepGp::jacobianSpread(unsigned int nProbe, uint64_t probeSeed) const
{
  if(nProbe == 0)
    throw ndlexceptions::Error("CDeepGp::jacobianSpread: nProbe >= 1");
  // Rectangular maps (CIP-003C): J is Dout x Din with min(Din,Dout) singular
  // values, taken from the smaller of J^T J (Din x Din) and J J^T.
  const unsigned int Din = inputDim;
  const unsigned int Dout = outputDim;
  const unsigned int k = min(Din, Dout);
  mt19937_64 prng(probeSeed);
  normal_distribution<double> nd(0.0, 1.0);
  vector<double> spreads;
  spreads.reserve(nProbe);
  const double h = 1e-4;

  for(unsigned int t = 0; t < nProbe; t++)
  {
    CMatrix x0(1, Din);
    for(unsigned int d = 0; d < Din; d++)
      x0.setVal(nd(prng), 0, d);

    // One prior whitened draw v~N(0,I) per layer (prototype V7).
    vector<CMatrix> vs(numLayers);
    for(unsigned int ell = 0; ell < numLayers; ell++)
    {
      const unsigned int Mi = layers[ell]->getNumInducing();
      const unsigned int Do = layers[ell]->getOutputDim();
      vs[ell].resize(Mi, Do);
      for(unsigned int r = 0; r < Mi; r++)
        for(unsigned int c = 0; c < Do; c++)
          vs[ell].setVal(nd(prng), r, c);
    }

    CMatrix J(Dout, Din);
    for(unsigned int d = 0; d < Din; d++)
    {
      CMatrix xp(x0), xm(x0);
      xp.setVal(x0.getVal(0, d) + h, 0, d);
      xm.setVal(x0.getVal(0, d) - h, 0, d);
      CMatrix yp, ym;
      forwardPriorDraw(xp, vs, yp);
      forwardPriorDraw(xm, vs, ym);
      for(unsigned int o = 0; o < Dout; o++)
        J.setVal((yp.getVal(0, o) - ym.getVal(0, o)) / (2.0 * h), o, d);
    }
    // Singular values via eigenvalues of the smaller Gram matrix.
    CMatrix G(k, k);
    if(Din <= Dout)
      G.gemm(J, J, 1.0, 0.0, "t", "n");
    else
      G.gemm(J, J, 1.0, 0.0, "n", "t");
    G.setSymmetric(true);
    CMatrix lam(1, k);
    G.syev(lam, "n", "l");
    double lmin = 1e300, lmax = 0.0;
    for(unsigned int i = 0; i < k; i++)
    {
      const double s = sqrt(max(fabs(lam.getVal(i)), 0.0));
      lmin = min(lmin, s);
      lmax = max(lmax, s);
    }
    const double spread = log(max(lmax, 1e-300)) - log(max(lmin, 1e-300));
    spreads.push_back(spread);
  }
  sort(spreads.begin(), spreads.end());
  return spreads[spreads.size() / 2];
}

void CDeepGp::display(ostream& os) const
{
  os << "Deep GP (doubly stochastic VI on CSvgp layers):" << endl;
  os << "  N=" << nData << " Din=" << inputDim << " Dout=" << outputDim
     << " L=" << numLayers << " chi=" << chiScale
     << " beta=" << betaStore
     << " dropResidual=" << (dropResidual ? 1 : 0) << endl;
  os << "  samples=" << numSamples
     << " natgrad_gamma=" << natGradGamma << endl;
  os << "  dims:";
  for(unsigned int ell = 0; ell <= numLayers; ell++)
    os << " " << layerDims[ell];
  os << endl;
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    os << "  layer " << ell << ": M=" << M[ell]
       << " inducing=" << inducingTypes[ell]
       << " kern=" << (pkerns[ell] ? pkerns[ell]->getType() : "?") << endl;
  }
}

void CDeepGp::writeParamsToStream(ostream& os) const
{
  if(layers.empty() || pkerns.empty())
    throw ndlexceptions::Error("CDeepGp::writeParamsToStream: uninitialised model");
  writeToStream(os, "baseType", "deepgp");
  writeToStream(os, "type", "deepgp");
  writeToStream(os, "numData", nData);
  writeToStream(os, "inputDim", inputDim);
  writeToStream(os, "outputDim", outputDim);
  writeToStream(os, "numLayers", numLayers);
  writeToStream(os, "beta", betaStore);
  writeToStream(os, "chi", chiScale);
  writeToStream(os, "numSamples", numSamples);
  writeToStream(os, "natGradGamma", natGradGamma);
  writeToStream(os, "dropResidual", dropResidual);
  for(unsigned int ell = 0; ell <= numLayers; ell++)
    writeToStream(os, "dim", layerDims[ell]);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    writeToStream(os, "numInducing", M[ell]);
    writeToStream(os, "inducingType", inducingTypes[ell]);
    if(inducingTypes[ell] == "hermite")
    {
      // CIP-003D: V and ks stored explicitly (see CGp).
      const CInducingHermite* ph =
          dynamic_cast<const CInducingHermite*>(layers[ell]->getInducing());
      if(ph == 0)
        throw ndlexceptions::Error("CDeepGp: hermite layer without a hermite inducing object");
      ph->directions().toStream(os);
      ph->degreesAsRow().toStream(os);
    }
  }
  X.toStream(os);
  Y.toStream(os);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    meanP[ell].toStream(os);
    pkerns[ell]->toStream(os);
    CMatrix pe;
    layers[ell]->getStochasticParams(pe);
    const unsigned int np = pe.getCols();
    writeToStream(os, "numStochParams", np);
    CMatrix peWrite(1, np > 0 ? np : 1);
    peWrite.zeros();
    if(np > 0)
      peWrite.deepCopy(pe);
    peWrite.toStream(os);
  }
}

void CDeepGp::readParamsFromStream(istream& is)
{
  string tbase = getBaseTypeStream(is);
  if(tbase != "deepgp")
    throw ndlexceptions::StreamFormatError("baseType", "Expected deepgp");
  string t = getTypeStream(is);
  if(t != "deepgp")
    throw ndlexceptions::StreamFormatError("type", "Expected deepgp");
  nData = (unsigned int)readIntFromStream(is, "numData");
  inputDim = (unsigned int)readIntFromStream(is, "inputDim");
  outputDim = (unsigned int)readIntFromStream(is, "outputDim");
  numLayers = (unsigned int)readIntFromStream(is, "numLayers");
  betaStore = readDoubleFromStream(is, "beta");
  chiScale = readDoubleFromStream(is, "chi");
  numSamples = (unsigned int)readIntFromStream(is, "numSamples");
  natGradGamma = readDoubleFromStream(is, "natGradGamma");
  dropResidual = readBoolFromStream(is, "dropResidual");

  layerDims.resize(numLayers + 1);
  for(unsigned int ell = 0; ell <= numLayers; ell++)
    layerDims[ell] = (unsigned int)readIntFromStream(is, "dim");
  M.resize(numLayers);
  inducingTypes.resize(numLayers);
  vector<std::unique_ptr<CInducingHermite> > hermiteLoaded(numLayers);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    M[ell] = (unsigned int)readIntFromStream(is, "numInducing");
    inducingTypes[ell] = readStringFromStream(is, "inducingType");
    if(inducingTypes[ell] == "hermite")
    {
      CMatrix Vh, ksh;
      Vh.fromStream(is);
      ksh.fromStream(is);
      // E6: validated here, before any layer is rebuilt.
      hermiteLoaded[ell].reset(
          CInducingHermite::fromStreamData(Vh, ksh, M[ell], layerDims[ell]));
    }
  }
  X.resize(nData, inputDim);
  Y.resize(nData, outputDim);
  X.fromStream(is);
  Y.fromStream(is);

  ownedKerns.clear();
  pkerns.clear();
  ownKerns = true;
  vector<CMatrix> savedMeanP(numLayers);
  vector<CMatrix> savedParams(numLayers);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    savedMeanP[ell].fromStream(is);
    CKern* k = readKernFromStream(is);
    ownedKerns.emplace_back(k);
    pkerns.push_back(k);
    const unsigned int np = (unsigned int)readIntFromStream(is, "numStochParams");
    savedParams[ell].resize(1, np > 0 ? np : 1);
    savedParams[ell].fromStream(is);
  }

  rebuildLayers(betaStore);
  for(unsigned int ell = 0; ell < numLayers; ell++)
  {
    meanP[ell].deepCopy(savedMeanP[ell]);
    if(hermiteLoaded[ell])
      layers[ell]->setInducing(hermiteLoaded[ell].release()); // stored V, ks
    else if(inducingTypes[ell] != "points")
      layers[ell]->setInducingType(inducingTypes[ell]);
    layers[ell]->setDropResidual(dropResidual);
    layers[ell]->setStochasticParams(savedParams[ell]);
  }
  packLayerOffsets();
}
