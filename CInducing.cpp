#include "CInducing.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace std;

// ---- CInducingPoints ----

CInducingPoints::CInducingPoints(CMatrix* pZIn)
  : pZ(pZIn)
{
  if(pZ == 0)
    throw ndlexceptions::Error("CInducingPoints requires non-null centres.");
}

unsigned int CInducingPoints::numInducing() const
{
  return pZ->getRows();
}

unsigned int CInducingPoints::domainDim() const
{
  return pZ->getCols();
}

void CInducingPoints::computeKuu(CMatrix& Kuu, const CKern& kern) const
{
  kern.compute(Kuu, *pZ);
}

void CInducingPoints::computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const
{
  kern.compute(Kuf, *pZ, X);
}

void CInducingPoints::computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const
{
  kern.compute(Kux, *pZ, Xstar);
}

void CInducingPoints::gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                                    const CKern& kern, const CMatrix& X,
                                    const CMatrix& gKuu, const CMatrix& gKuf,
                                    bool wantX, bool regulariseKern) const
{
  const unsigned int M = numInducing();
  const unsigned int d = domainDim();
  const unsigned int N = X.getRows();
  const unsigned int nParams = kern.getNumParams();
  DIMENSIONMATCH(gKernParams.getCols() == nParams);
  DIMENSIONMATCH(gKuu.getRows() == M && gKuu.getCols() == M);
  DIMENSIONMATCH(gKuf.getRows() == M && gKuf.getCols() == N);

  tempG.resize(1, nParams);
  tempG2.resize(1, nParams);
  // Mirror CGp::updateG sparse block (getGradTransParams on Xu with regularise).
  kern.getGradTransParams(tempG, *pZ, gKuu, regulariseKern);
  kern.getGradTransParams(tempG2, *pZ, X, gKuf, regulariseKern);
  for(unsigned int i = 0; i < nParams; i++)
  {
    gKernParams.addVal(tempG.getVal(i), i);
    gKernParams.addVal(tempG2.getVal(i), i);
  }

  gZ.resize(M, d);
  gZ.zeros();
  dgKX.resize(M, d);
  gKX.resize(M, d);
  gKX_uf.resize(N, d);
  kern.getDiagGradX(dgKX, *pZ);
  for(unsigned int i = 0; i < M; i++)
  {
    kern.getGradX(gKX_uf, *pZ, i, X);
    kern.getGradX(gKX, *pZ, i, *pZ);
    gKX.scale(2.0);
    gKX.copyRowRow(i, dgKX, i);
    for(unsigned int j = 0; j < d; j++)
      gZ.setVal(gKX.dotColCol(j, gKuu, i), i, j);
    for(unsigned int j = 0; j < d; j++)
      gZ.addVal(gKX_uf.dotColRow(j, gKuf, i), i, j);
  }

  if(wantX && gXOrNull != 0)
  {
    gXOrNull->resize(N, d);
    gKX_uf2.resize(M, d);
    for(unsigned int i = 0; i < N; i++)
    {
      kern.getGradX(gKX_uf2, X, i, *pZ);
      for(unsigned int j = 0; j < d; j++)
        gXOrNull->setVal(gKX_uf2.dotColCol(j, gKuf, i), i, j);
    }
  }
}

unsigned int CInducingPoints::getNumParams() const
{
  return numInducing() * domainDim();
}

void CInducingPoints::getParams(CMatrix& param, unsigned int offset) const
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      param.setVal(pZ->getVal(i, j), counter);
      counter++;
    }
  }
}

void CInducingPoints::setParams(const CMatrix& param, unsigned int offset)
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      pZ->setVal(param.getVal(counter), i, j);
      counter++;
    }
  }
}

void CInducingPoints::setCentres(const CMatrix& Z)
{
  DIMENSIONMATCH(Z.getRows() == pZ->getRows());
  DIMENSIONMATCH(Z.getCols() == pZ->getCols());
  pZ->deepCopy(Z);
}

// ---- CInducingWhiteNoise ----

CInducingWhiteNoise::CInducingWhiteNoise(CMatrix* pZIn, double sIn, bool learnSIn)
  : pZ(pZIn), s(sIn), learnS(learnSIn)
{
  if(pZ == 0)
    throw ndlexceptions::Error("CInducingWhiteNoise requires non-null centres.");
  if(!(s > 0.0))
    throw ndlexceptions::Error("CInducingWhiteNoise window width must be positive.");
}

unsigned int CInducingWhiteNoise::numInducing() const
{
  return pZ->getRows();
}

unsigned int CInducingWhiteNoise::domainDim() const
{
  return pZ->getCols();
}

void CInducingWhiteNoise::setWindowWidth(double sIn)
{
  if(!(sIn > 0.0))
    throw ndlexceptions::Error("CInducingWhiteNoise window width must be positive.");
  s = sIn;
}

double CInducingWhiteNoise::medianSpacing(const CMatrix& Z)
{
  const unsigned int M = Z.getRows();
  const unsigned int d = Z.getCols();
  if(M < 2)
    return 1.0;
  vector<double> nn(M);
  for(unsigned int i = 0; i < M; i++)
  {
    double best = numeric_limits<double>::infinity();
    for(unsigned int j = 0; j < M; j++)
    {
      if(i == j)
        continue;
      double r2 = 0.0;
      for(unsigned int k = 0; k < d; k++)
      {
        const double dz = Z.getVal(i, k) - Z.getVal(j, k);
        r2 += dz * dz;
      }
      const double r = sqrt(r2);
      if(r < best)
        best = r;
    }
    nn[i] = best;
  }
  sort(nn.begin(), nn.end());
  if(M % 2 == 1)
    return nn[M / 2];
  return 0.5 * (nn[M / 2 - 1] + nn[M / 2]);
}

void CInducingWhiteNoise::fillKuu(CMatrix& Kuu) const
{
  const unsigned int M = numInducing();
  const unsigned int d = domainDim();
  const double s2 = s * s;
  Kuu.resize(M, M);
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
    {
      double r2 = 0.0;
      for(unsigned int k = 0; k < d; k++)
      {
        const double dz = pZ->getVal(i, k) - pZ->getVal(j, k);
        r2 += dz * dz;
      }
      Kuu.setVal(exp(-r2 / (4.0 * s2)), i, j);
    }
  }
  Kuu.setSymmetric(true);
}

void CInducingWhiteNoise::kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu) const
{
  const unsigned int N = Kfu.getRows();
  const unsigned int M = Kfu.getCols();
  DIMENSIONMATCH(Kuf.getRows() == M && Kuf.getCols() == N);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int n = 0; n < N; n++)
      Kuf.setVal(Kfu.getVal(n, m), m, n);
}

void CInducingWhiteNoise::computeKuu(CMatrix& Kuu, const CKern& /*kern*/) const
{
  fillKuu(Kuu);
}

void CInducingWhiteNoise::computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const
{
  const unsigned int M = numInducing();
  const unsigned int N = X.getRows();
  Kfu.resize(N, M);
  kern.wnCrossCompute(Kfu, X, *pZ, s);
  kufFromKfu(Kuf, Kfu);
}

void CInducingWhiteNoise::computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const
{
  // Same layout as compute(kX, Xu, Xin): M × Nstar.
  const unsigned int M = numInducing();
  const unsigned int Ns = Xstar.getRows();
  Kfu.resize(Ns, M);
  kern.wnCrossCompute(Kfu, Xstar, *pZ, s);
  kufFromKfu(Kux, Kfu);
}

void CInducingWhiteNoise::gradZFromKuu(CMatrix& gZ, const CMatrix& gKuu, const CMatrix& Kuu) const
{
  // Kuu[i,j] = exp(-||zi-zj||^2/(4s^2)); ∂Kij/∂zi = -Kij (zi-zj)/(2s^2).
  // gKuu is stored 0.5-symmetrised (see gpCovGrads), matching CInducingPoints'
  // getGradX(...).scale(2.0) convention: use 2*gKuu off-diagonal.
  const unsigned int M = numInducing();
  const unsigned int d = domainDim();
  const double inv2s2 = 1.0 / (2.0 * s * s);
  gZ.resize(M, d);
  gZ.zeros();
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
    {
      if(i == j)
        continue;
      const double g = 2.0 * gKuu.getVal(i, j);
      const double k = Kuu.getVal(i, j);
      const double coeff = -g * k * inv2s2;
      for(unsigned int c = 0; c < d; c++)
      {
        const double dz = pZ->getVal(i, c) - pZ->getVal(j, c);
        gZ.addVal(coeff * dz, i, c);
      }
    }
  }
}

void CInducingWhiteNoise::gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                                        const CKern& kern, const CMatrix& X,
                                        const CMatrix& gKuu, const CMatrix& gKuf,
                                        bool wantX, bool regulariseKern) const
{
  const unsigned int M = numInducing();
  const unsigned int d = domainDim();
  const unsigned int N = X.getRows();
  const unsigned int nParams = kern.getNumParams();
  DIMENSIONMATCH(gKernParams.getCols() == nParams);
  DIMENSIONMATCH(gKuu.getRows() == M && gKuu.getCols() == M);
  DIMENSIONMATCH(gKuf.getRows() == M && gKuf.getCols() == N);

  // Kernel-free Kuu: centre grads only from gKuu.
  CMatrix Kuu(M, M);
  fillKuu(Kuu);
  gradZFromKuu(gZ, gKuu, Kuu);

  // gKuf (M×N) → gKfu (N×M); recompute Kfu for closed-form grads.
  Kfu.resize(N, M);
  kern.wnCrossCompute(Kfu, X, *pZ, s);
  gKfu.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      gKfu.setVal(gKuf.getVal(m, n), n, m);

  tempG.resize(1, nParams);
  tempG.zeros();
  kern.wnGradParams(tempG, X, *pZ, s, gKfu);
  // Match getGradTransParams: apply transforms then optional prior.
  for(unsigned int i = 0; i < kern.getNumTransforms(); i++)
  {
    const unsigned int idx = kern.getTransformIndex(i);
    const double val = tempG.getVal(idx);
    const double param = kern.getParam(idx);
    tempG.setVal(val * kern.getTransformGradFact(param, i), idx);
  }
  if(regulariseKern)
    kern.addPriorGrad(tempG);
  for(unsigned int i = 0; i < nParams; i++)
    gKernParams.addVal(tempG.getVal(i), i);

  CMatrix gZfromKuf(M, d);
  gZfromKuf.zeros();
  kern.wnGradZ(gZfromKuf, X, *pZ, s, gKfu);
  for(unsigned int i = 0; i < M; i++)
    for(unsigned int j = 0; j < d; j++)
      gZ.addVal(gZfromKuf.getVal(i, j), i, j);

  if(wantX && gXOrNull != 0)
  {
    gXOrNull->resize(N, d);
    gXOrNull->zeros();
    kern.wnGradX(*gXOrNull, X, *pZ, s, gKfu);
  }

  // s learnable is out of default scope; learnS reserved for a later pass.
  (void)learnS;
}

unsigned int CInducingWhiteNoise::getNumParams() const
{
  unsigned int n = numInducing() * domainDim();
  if(learnS)
    n += 1;
  return n;
}

void CInducingWhiteNoise::getParams(CMatrix& param, unsigned int offset) const
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      param.setVal(pZ->getVal(i, j), counter);
      counter++;
    }
  }
  if(learnS)
  {
    param.setVal(s, counter);
    counter++;
  }
}

void CInducingWhiteNoise::setParams(const CMatrix& param, unsigned int offset)
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      pZ->setVal(param.getVal(counter), i, j);
      counter++;
    }
  }
  if(learnS)
  {
    setWindowWidth(param.getVal(counter));
    counter++;
  }
}

void CInducingWhiteNoise::setCentres(const CMatrix& Z)
{
  DIMENSIONMATCH(Z.getRows() == pZ->getRows());
  DIMENSIONMATCH(Z.getCols() == pZ->getCols());
  pZ->deepCopy(Z);
}

// ---- CInducingGreens (CIP-0029 Phase 0) ----

CInducingGreens::CInducingGreens(CMatrix* pZIn, double sIn, bool learnSIn)
  : pZ(pZIn), s(sIn), learnS(learnSIn)
{
  if(pZ == 0)
    throw ndlexceptions::Error("CInducingGreens requires non-null centres.");
  if(!(s > 0.0))
    throw ndlexceptions::Error("CInducingGreens window width must be positive.");
  assertOneD();
}

void CInducingGreens::assertOneD() const
{
  if(pZ->getCols() != 1)
    throw ndlexceptions::Error(
        "CInducingGreens requires 1-D centres (X_u columns == 1); "
        "multi-D causal Green's features are out of scope for CIP-0029.");
}

unsigned int CInducingGreens::numInducing() const
{
  return pZ->getRows();
}

unsigned int CInducingGreens::domainDim() const
{
  return pZ->getCols();
}

void CInducingGreens::setWindowWidth(double sIn)
{
  if(!(sIn > 0.0))
    throw ndlexceptions::Error("CInducingGreens window width must be positive.");
  s = sIn;
}

void CInducingGreens::fillKuu(CMatrix& Kuu) const
{
  // Same Gaussian-window Gram as CInducingWhiteNoise (1-D special case).
  const unsigned int M = numInducing();
  const double s2 = s * s;
  Kuu.resize(M, M);
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
    {
      const double dz = pZ->getVal(i, 0) - pZ->getVal(j, 0);
      Kuu.setVal(exp(-(dz * dz) / (4.0 * s2)), i, j);
    }
  }
  Kuu.setSymmetric(true);
}

void CInducingGreens::computeKuu(CMatrix& Kuu, const CKern& /*kern*/) const
{
  assertOneD();
  fillKuu(Kuu);
}

void CInducingGreens::kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu) const
{
  const unsigned int N = Kfu.getRows();
  const unsigned int M = Kfu.getCols();
  DIMENSIONMATCH(Kuf.getRows() == M && Kuf.getCols() == N);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int n = 0; n < N; n++)
      Kuf.setVal(Kfu.getVal(n, m), m, n);
}

void CInducingGreens::gradZFromKuu(CMatrix& gZ, const CMatrix& gKuu, const CMatrix& Kuu) const
{
  const unsigned int M = numInducing();
  const double inv2s2 = 1.0 / (2.0 * s * s);
  gZ.resize(M, 1);
  gZ.zeros();
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
    {
      if(i == j)
        continue;
      const double g = 2.0 * gKuu.getVal(i, j);
      const double k = Kuu.getVal(i, j);
      const double coeff = -g * k * inv2s2;
      const double dz = pZ->getVal(i, 0) - pZ->getVal(j, 0);
      gZ.addVal(coeff * dz, i, 0);
    }
  }
}

void CInducingGreens::computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const
{
  assertOneD();
  const unsigned int M = numInducing();
  const unsigned int N = X.getRows();
  Kfu.resize(N, M);
  kern.greensCrossCompute(Kfu, X, *pZ, s);
  kufFromKfu(Kuf, Kfu);
}

void CInducingGreens::computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const
{
  assertOneD();
  const unsigned int M = numInducing();
  const unsigned int Ns = Xstar.getRows();
  Kfu.resize(Ns, M);
  kern.greensCrossCompute(Kfu, Xstar, *pZ, s);
  kufFromKfu(Kux, Kfu);
}

void CInducingGreens::gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                                     const CKern& kern, const CMatrix& X,
                                     const CMatrix& gKuu, const CMatrix& gKuf,
                                     bool wantX, bool regulariseKern) const
{
  assertOneD();
  const unsigned int M = numInducing();
  const unsigned int N = X.getRows();
  const unsigned int nParams = kern.getNumParams();
  DIMENSIONMATCH(gKernParams.getCols() == nParams);
  DIMENSIONMATCH(gKuu.getRows() == M && gKuu.getCols() == M);
  DIMENSIONMATCH(gKuf.getRows() == M && gKuf.getCols() == N);

  CMatrix Kuu(M, M);
  fillKuu(Kuu);
  gradZFromKuu(gZ, gKuu, Kuu);

  Kfu.resize(N, M);
  kern.greensCrossCompute(Kfu, X, *pZ, s);
  gKfu.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      gKfu.setVal(gKuf.getVal(m, n), n, m);

  tempG.resize(1, nParams);
  tempG.zeros();
  kern.greensGradParams(tempG, X, *pZ, s, gKfu);
  for(unsigned int i = 0; i < kern.getNumTransforms(); i++)
  {
    const unsigned int idx = kern.getTransformIndex(i);
    const double val = tempG.getVal(idx);
    const double param = kern.getParam(idx);
    tempG.setVal(val * kern.getTransformGradFact(param, i), idx);
  }
  if(regulariseKern)
    kern.addPriorGrad(tempG);
  for(unsigned int i = 0; i < nParams; i++)
    gKernParams.addVal(tempG.getVal(i), i);

  CMatrix gZfromKuf(M, 1);
  gZfromKuf.zeros();
  kern.greensGradZ(gZfromKuf, X, *pZ, s, gKfu);
  for(unsigned int i = 0; i < M; i++)
    gZ.addVal(gZfromKuf.getVal(i, 0), i, 0);

  if(wantX && gXOrNull != 0)
  {
    gXOrNull->resize(N, 1);
    gXOrNull->zeros();
    kern.greensGradX(*gXOrNull, X, *pZ, s, gKfu);
  }

  (void)learnS;
}

unsigned int CInducingGreens::getNumParams() const
{
  unsigned int n = numInducing() * domainDim();
  if(learnS)
    n += 1;
  return n;
}

void CInducingGreens::getParams(CMatrix& param, unsigned int offset) const
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      param.setVal(pZ->getVal(i, j), counter);
      counter++;
    }
  }
  if(learnS)
  {
    param.setVal(s, counter);
    counter++;
  }
}

void CInducingGreens::setParams(const CMatrix& param, unsigned int offset)
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      pZ->setVal(param.getVal(counter), i, j);
      counter++;
    }
  }
  if(learnS)
  {
    setWindowWidth(param.getVal(counter));
    counter++;
  }
}

void CInducingGreens::setCentres(const CMatrix& Z)
{
  DIMENSIONMATCH(Z.getRows() == pZ->getRows());
  DIMENSIONMATCH(Z.getCols() == pZ->getCols());
  if(Z.getCols() != 1)
    throw ndlexceptions::Error("CInducingGreens::setCentres requires 1-D centres.");
  pZ->deepCopy(Z);
}

// ---- CInducingWeightSpace (CIP-0028 Design A; ridge default 2026-11-19) ----

CInducingWeightSpace::CInducingWeightSpace(const CMatrix& WIn, double sIn, bool learnSIn,
                                           bool ridgeIn)
  : W(WIn), s(sIn), learnS(learnSIn), ridge(ridgeIn)
{
  if(W.getRows() == 0 || W.getCols() == 0)
    throw ndlexceptions::Error("CInducingWeightSpace requires non-empty W.");
  if(ridge && W.getCols() < 2)
    throw ndlexceptions::Error(
        "CInducingWeightSpace ridge requires W with cols >= 2 (d+1 direction + offset).");
  if(!ridge && W.getCols() < 1)
    throw ndlexceptions::Error("CInducingWeightSpace isotropic requires W M x (d+1).");
  if(!(s > 0.0))
    throw ndlexceptions::Error("CInducingWeightSpace window width must be positive.");
}

unsigned int CInducingWeightSpace::numInducing() const
{
  return W.getRows();
}

unsigned int CInducingWeightSpace::domainDim() const
{
  return W.getCols();
}

unsigned int CInducingWeightSpace::weightDim() const
{
  return ridge ? (domainDim() - 1) : domainDim();
}

void CInducingWeightSpace::setWindowWidth(double sIn)
{
  if(!(sIn > 0.0))
    throw ndlexceptions::Error("CInducingWeightSpace window width must be positive.");
  s = sIn;
}

void CInducingWeightSpace::ridgeFactors(CMatrix& V, CMatrix& aOut) const
{
  BOUNDCHECK(ridge);
  const unsigned int M = numInducing();
  const unsigned int Dp = weightDim();
  V.resize(M, Dp);
  aOut.resize(M, 1);
  for(unsigned int m = 0; m < M; m++)
  {
    double nrm = 0.0;
    for(unsigned int j = 0; j < Dp; j++)
    {
      const double v = W.getVal(m, j);
      nrm += v * v;
    }
    nrm = sqrt(nrm);
    if(!(nrm > 1e-15))
      throw ndlexceptions::Error("CInducingWeightSpace ridge direction has zero norm.");
    for(unsigned int j = 0; j < Dp; j++)
      V.setVal(W.getVal(m, j) / nrm, m, j);
    aOut.setVal(W.getVal(m, Dp), m, 0);
  }
}

void CInducingWeightSpace::fillKuuIsotropic(CMatrix& Kuu) const
{
  const unsigned int M = numInducing();
  const unsigned int d = domainDim();
  const double s2 = s * s;
  Kuu.resize(M, M);
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
    {
      double r2 = 0.0;
      for(unsigned int k = 0; k < d; k++)
      {
        const double dz = W.getVal(i, k) - W.getVal(j, k);
        r2 += dz * dz;
      }
      Kuu.setVal(exp(-r2 / (4.0 * s2)), i, j);
    }
  }
  Kuu.setSymmetric(true);
}

void CInducingWeightSpace::fillKuuRidge(CMatrix& Kuu) const
{
  // K_uu[m,n] from 2×2 Woodbury in cosines (CIP-0028 Update 2026-11-19 §3).
  ridgeFactors(Vtmp, atmp);
  const unsigned int M = numInducing();
  const unsigned int Dp = weightDim();
  const double s2 = s * s;
  const double alpha = 1.0 / s2 - 0.5;
  const double lognorm2 = -0.5 * log(M_PI * s2) - 0.5 * (double)(Dp - 1) * log(2.0 * M_PI);
  Kuu.resize(M, M);
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = i; j < M; j++)
    {
      double c = 0.0;
      for(unsigned int k = 0; k < Dp; k++)
        c += Vtmp.getVal(i, k) * Vtmp.getVal(j, k);
      if(c > 1.0) c = 1.0;
      if(c < -1.0) c = -1.0;
      const double am = atmp.getVal(i, 0);
      const double an = atmp.getVal(j, 0);
      const double detI = (1.0 + alpha) * (1.0 + alpha) - (alpha * c) * (alpha * c);
      BOUNDCHECK(detI > 0.0);
      const double inv11 = (1.0 + alpha) / detI;
      const double inv12 = -alpha * c / detI;
      const double H11 = inv11 + c * inv12;
      const double H12 = inv12 + c * inv11;
      const double bm = am / s2;
      const double bn = an / s2;
      const double quad = H11 * (bm * bm + bn * bn) + 2.0 * H12 * bm * bn;
      const double logK = lognorm2 + 0.5 * (double)Dp * log(2.0 * M_PI)
                          - 0.5 * log(detI) + 0.5 * quad
                          - (am * am + an * an) / (2.0 * s2);
      const double kij = exp(logK);
      Kuu.setVal(kij, i, j);
      if(j != i)
        Kuu.setVal(kij, j, i);
    }
  }
  Kuu.setSymmetric(true);
}

void CInducingWeightSpace::fillKuu(CMatrix& Kuu) const
{
  if(ridge)
    fillKuuRidge(Kuu);
  else
    fillKuuIsotropic(Kuu);
}

void CInducingWeightSpace::computeKuu(CMatrix& Kuu, const CKern& /*kern*/) const
{
  fillKuu(Kuu);
}

void CInducingWeightSpace::kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu) const
{
  const unsigned int N = Kfu.getRows();
  const unsigned int M = Kfu.getCols();
  DIMENSIONMATCH(Kuf.getRows() == M && Kuf.getCols() == N);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int n = 0; n < N; n++)
      Kuf.setVal(Kfu.getVal(n, m), m, n);
}

void CInducingWeightSpace::gradWFromKuu(CMatrix& gW, const CMatrix& gKuu, const CMatrix& Kuu) const
{
  const unsigned int M = numInducing();
  const unsigned int d = domainDim();
  const double inv2s2 = 1.0 / (2.0 * s * s);
  gW.resize(M, d);
  gW.zeros();
  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = 0; j < M; j++)
    {
      if(i == j)
        continue;
      const double g = 2.0 * gKuu.getVal(i, j);
      const double k = Kuu.getVal(i, j);
      const double coeff = -g * k * inv2s2;
      for(unsigned int c = 0; c < d; c++)
      {
        const double dz = W.getVal(i, c) - W.getVal(j, c);
        gW.addVal(coeff * dz, i, c);
      }
    }
  }
}

void CInducingWeightSpace::computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const
{
  const unsigned int M = numInducing();
  const unsigned int N = X.getRows();
  Kfu.resize(N, M);
  if(ridge)
  {
    ridgeFactors(Vtmp, atmp);
    kern.wsRidgeCrossCompute(Kfu, X, Vtmp, atmp, s);
  }
  else
    kern.wsCrossCompute(Kfu, X, W, s);
  kufFromKfu(Kuf, Kfu);
}

void CInducingWeightSpace::computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const
{
  const unsigned int M = numInducing();
  const unsigned int Ns = Xstar.getRows();
  Kfu.resize(Ns, M);
  if(ridge)
  {
    ridgeFactors(Vtmp, atmp);
    kern.wsRidgeCrossCompute(Kfu, Xstar, Vtmp, atmp, s);
  }
  else
    kern.wsCrossCompute(Kfu, Xstar, W, s);
  kufFromKfu(Kux, Kfu);
}

void CInducingWeightSpace::gradRawFromKuuRidge(CMatrix& gRaw, const CMatrix& gKuu) const
{
  // Analytic ∂⟨gKuu, Kuu⟩ through unit directions / offsets, then chain to raw W.
  ridgeFactors(Vtmp, atmp);
  const unsigned int M = numInducing();
  const unsigned int Dp = weightDim();
  const double s2 = s * s;
  const double alpha = 1.0 / s2 - 0.5;
  CMatrix gV(M, Dp);
  CMatrix ga(M, 1);
  gV.zeros();
  ga.zeros();

  for(unsigned int i = 0; i < M; i++)
  {
    for(unsigned int j = i; j < M; j++)
    {
      double c = 0.0;
      for(unsigned int k = 0; k < Dp; k++)
        c += Vtmp.getVal(i, k) * Vtmp.getVal(j, k);
      if(c > 1.0) c = 1.0;
      if(c < -1.0) c = -1.0;
      const double am = atmp.getVal(i, 0);
      const double an = atmp.getVal(j, 0);
      const double detI = (1.0 + alpha) * (1.0 + alpha) - (alpha * c) * (alpha * c);
      BOUNDCHECK(detI > 0.0);
      const double inv11 = (1.0 + alpha) / detI;
      const double inv12 = -alpha * c / detI;
      const double H11 = inv11 + c * inv12;
      const double H12 = inv12 + c * inv11;
      const double bm = am / s2;
      const double bn = an / s2;
      const double quad = H11 * (bm * bm + bn * bn) + 2.0 * H12 * bm * bn;
      const double lognorm2 = -0.5 * log(M_PI * s2)
                              - 0.5 * (double)(Dp - 1) * log(2.0 * M_PI);
      const double logK = lognorm2 + 0.5 * (double)Dp * log(2.0 * M_PI)
                          - 0.5 * log(detI) + 0.5 * quad
                          - (am * am + an * an) / (2.0 * s2);
      const double Kij = exp(logK);
      const double gsym = (i == j) ? gKuu.getVal(i, j)
                                   : (gKuu.getVal(i, j) + gKuu.getVal(j, i));

      // ∂logK/∂am, ∂logK/∂an
      const double dlog_dam = (H11 * bm + H12 * bn) / s2 - am / s2;
      const double dlog_dan = (H11 * bn + H12 * bm) / s2 - an / s2;
      ga.addVal(gsym * Kij * dlog_dam, i, 0);
      if(i != j)
        ga.addVal(gsym * Kij * dlog_dan, j, 0);

      if(i != j)
      {
        const double ddet = -2.0 * alpha * alpha * c;
        const double dinv11 = -(1.0 + alpha) / (detI * detI) * ddet;
        const double dinv12 = -alpha / detI
                              + alpha * c / (detI * detI) * ddet;
        const double dH11 = dinv11 + inv12 + c * dinv12;
        const double dH12 = dinv12 + inv11 + c * dinv11;
        const double dquad = dH11 * (bm * bm + bn * bn) + 2.0 * dH12 * bm * bn;
        const double dlog_dc = -0.5 / detI * ddet + 0.5 * dquad;
        const double coeff = gsym * Kij * dlog_dc;
        for(unsigned int k = 0; k < Dp; k++)
        {
          gV.addVal(coeff * Vtmp.getVal(j, k), i, k);
          gV.addVal(coeff * Vtmp.getVal(i, k), j, k);
        }
      }
    }
  }

  // Chain through v = w_dir / ||w_dir||: g_w = (I - vv^T) g_v / ||w||.
  gRaw.resize(M, Dp + 1);
  gRaw.zeros();
  for(unsigned int i = 0; i < M; i++)
  {
    double nrm = 0.0;
    for(unsigned int k = 0; k < Dp; k++)
    {
      const double wk = W.getVal(i, k);
      nrm += wk * wk;
    }
    nrm = sqrt(nrm);
    BOUNDCHECK(nrm > 1e-15);
    double gdotv = 0.0;
    for(unsigned int k = 0; k < Dp; k++)
      gdotv += gV.getVal(i, k) * Vtmp.getVal(i, k);
    for(unsigned int k = 0; k < Dp; k++)
    {
      const double gv = gV.getVal(i, k) - gdotv * Vtmp.getVal(i, k);
      gRaw.setVal(gv / nrm, i, k);
    }
    gRaw.setVal(ga.getVal(i, 0), i, Dp);
  }
}

void CInducingWeightSpace::gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                                         const CKern& kern, const CMatrix& X,
                                         const CMatrix& gKuu, const CMatrix& gKuf,
                                         bool wantX, bool regulariseKern) const
{
  const unsigned int M = numInducing();
  const unsigned int d = domainDim();
  const unsigned int N = X.getRows();
  const unsigned int nParams = kern.getNumParams();
  DIMENSIONMATCH(gKernParams.getCols() == nParams);
  DIMENSIONMATCH(gKuu.getRows() == M && gKuu.getCols() == M);
  DIMENSIONMATCH(gKuf.getRows() == M && gKuf.getCols() == N);

  CMatrix Kuu(M, M);
  fillKuu(Kuu);

  Kfu.resize(N, M);
  gKfu.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      gKfu.setVal(gKuf.getVal(m, n), n, m);

  tempG.resize(1, nParams);
  tempG.zeros();

  if(ridge)
  {
    gradRawFromKuuRidge(gZ, gKuu);
    ridgeFactors(Vtmp, atmp);
    kern.wsRidgeCrossCompute(Kfu, X, Vtmp, atmp, s);
    kern.wsRidgeGradParams(tempG, X, Vtmp, atmp, s, gKfu);

    CMatrix gV(M, weightDim());
    CMatrix ga(M, 1);
    kern.wsRidgeGradVA(gV, ga, X, Vtmp, atmp, s, gKfu);
    // Chain Kfu grads through normalisation into raw W.
    for(unsigned int i = 0; i < M; i++)
    {
      double nrm = 0.0;
      for(unsigned int k = 0; k < weightDim(); k++)
        nrm += W.getVal(i, k) * W.getVal(i, k);
      nrm = sqrt(nrm);
      BOUNDCHECK(nrm > 1e-15);
      double gdotv = 0.0;
      for(unsigned int k = 0; k < weightDim(); k++)
        gdotv += gV.getVal(i, k) * Vtmp.getVal(i, k);
      for(unsigned int k = 0; k < weightDim(); k++)
      {
        const double gv = gV.getVal(i, k) - gdotv * Vtmp.getVal(i, k);
        gZ.addVal(gv / nrm, i, k);
      }
      gZ.addVal(ga.getVal(i, 0), i, weightDim());
    }
  }
  else
  {
    gradWFromKuu(gZ, gKuu, Kuu);
    kern.wsCrossCompute(Kfu, X, W, s);
    kern.wsGradParams(tempG, X, W, s, gKfu);
    CMatrix gWfromKuf(M, d);
    gWfromKuf.zeros();
    kern.wsGradW(gWfromKuf, X, W, s, gKfu);
    for(unsigned int i = 0; i < M; i++)
      for(unsigned int j = 0; j < d; j++)
        gZ.addVal(gWfromKuf.getVal(i, j), i, j);
  }

  for(unsigned int i = 0; i < kern.getNumTransforms(); i++)
  {
    const unsigned int idx = kern.getTransformIndex(i);
    const double val = tempG.getVal(idx);
    const double param = kern.getParam(idx);
    tempG.setVal(val * kern.getTransformGradFact(param, i), idx);
  }
  if(regulariseKern)
    kern.addPriorGrad(tempG);
  for(unsigned int i = 0; i < nParams; i++)
    gKernParams.addVal(tempG.getVal(i), i);

  if(wantX && gXOrNull != 0)
  {
    gXOrNull->resize(N, X.getCols());
    gXOrNull->zeros();
    if(ridge)
    {
      ridgeFactors(Vtmp, atmp);
      kern.wsRidgeGradX(*gXOrNull, X, Vtmp, atmp, s, gKfu);
    }
    else
      kern.wsGradX(*gXOrNull, X, W, s, gKfu);
  }

  (void)learnS;
}

unsigned int CInducingWeightSpace::getNumParams() const
{
  unsigned int n = numInducing() * domainDim();
  if(learnS)
    n += 1;
  return n;
}

void CInducingWeightSpace::getParams(CMatrix& param, unsigned int offset) const
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      param.setVal(W.getVal(i, j), counter);
      counter++;
    }
  }
  if(learnS)
  {
    param.setVal(s, counter);
    counter++;
  }
}

void CInducingWeightSpace::setParams(const CMatrix& param, unsigned int offset)
{
  unsigned int counter = offset;
  for(unsigned int j = 0; j < domainDim(); j++)
  {
    for(unsigned int i = 0; i < numInducing(); i++)
    {
      W.setVal(param.getVal(counter), i, j);
      counter++;
    }
  }
  if(learnS)
  {
    setWindowWidth(param.getVal(counter));
    counter++;
  }
}

void CInducingWeightSpace::setCentres(const CMatrix& Z)
{
  DIMENSIONMATCH(Z.getRows() == W.getRows());
  DIMENSIONMATCH(Z.getCols() == W.getCols());
  W.deepCopy(Z);
}
