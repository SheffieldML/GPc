#include "CInducing.h"
#include "ndlutil.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
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
                                    bool wantX, bool regulariseKern,
                                    const CMatrix* KuuCached,
                                    const CMatrix* KufCached) const
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

  // CIP-002D follow-up: batched ∂Z (RBF/Lin GEMM; default per-inducing getGradX).
  // Pass GP-cached K_uu/K_uf when provided (CoE 2026-11-22).
  kern.inducingGradZ(gZ, *pZ, X, gKuu, gKuf, KuuCached, KufCached);

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
                                        bool wantX, bool regulariseKern,
                                        const CMatrix* /*KuuCached*/,
                                        const CMatrix* /*KufCached*/) const
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
                                     bool wantX, bool regulariseKern,
                                     const CMatrix* /*KuuCached*/,
                                     const CMatrix* /*KufCached*/) const
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
  : W(WIn), s(sIn), learnS(learnSIn), untiedS(false), ridge(ridgeIn), gScache(0.0)
{
  if(W.getRows() == 0 || W.getCols() == 0)
    throw ndlexceptions::Error("CInducingWeightSpace requires non-empty W.");
  if(ridge && W.getCols() < 2)
    throw ndlexceptions::Error(
        "CInducingWeightSpace ridge requires W with cols >= 2 (d+1 direction + offset).");
  if(!ridge && W.getCols() < 1)
    throw ndlexceptions::Error("CInducingWeightSpace isotropic requires W M x (d+1).");
  validateWindowWidth(s);
  syncSVecFromScalar();
}

void CInducingWeightSpace::validateWindowWidth(double sIn)
{
  // Whitened ridge windows: κ(K_uu) grows ~exp(c (s/h)²) in offset spacing h.
  // At s=√2 the ridge matches the prior along its direction and K_uu can become
  // singular (especially with an offset at a=0). Cap below that regime.
  if(!(sIn > 0.0))
    throw ndlexceptions::Error(
        "CInducingWeightSpace window width must be positive.");
  if(sIn < windowWidthMin() || sIn > windowWidthMax())
  {
    ostringstream msg;
    msg << "CInducingWeightSpace window width must lie in ["
        << windowWidthMin() << ", " << windowWidthMax()
        << "] whitened units (got " << sIn
        << "). Larger s ill-conditions K_uu; s=√2 is singular.";
    throw ndlexceptions::Error(msg.str());
  }
}

double CInducingWeightSpace::windowWidthFromUnconstrained(double u)
{
  // σ(u) ∈ (0,1) → s ∈ (sLo, sHi)
  double sig = 1.0 / (1.0 + exp(-u));
  if(sig < 1e-12)
    sig = 1e-12;
  if(sig > 1.0 - 1e-12)
    sig = 1.0 - 1e-12;
  return windowWidthMin() + (windowWidthMax() - windowWidthMin()) * sig;
}

double CInducingWeightSpace::windowWidthToUnconstrained(double sIn)
{
  validateWindowWidth(sIn);
  double z = (sIn - windowWidthMin()) / (windowWidthMax() - windowWidthMin());
  if(z < 1e-12)
    z = 1e-12;
  if(z > 1.0 - 1e-12)
    z = 1.0 - 1e-12;
  return log(z / (1.0 - z));
}

double CInducingWeightSpace::windowWidthDsDu(double sIn)
{
  const double span = windowWidthMax() - windowWidthMin();
  double z = (sIn - windowWidthMin()) / span;
  if(z < 0.0)
    z = 0.0;
  if(z > 1.0)
    z = 1.0;
  return span * z * (1.0 - z);
}

void CInducingWeightSpace::syncSVecFromScalar()
{
  const unsigned int M = W.getRows();
  sVec.resize(M, 1);
  for(unsigned int i = 0; i < M; i++)
    sVec.setVal(s, i, 0);
}

double CInducingWeightSpace::windowWidthAt(unsigned int m) const
{
  BOUNDCHECK(m < sVec.getRows());
  return sVec.getVal(m, 0);
}

void CInducingWeightSpace::setWindowWidthUntied(bool untied)
{
  untiedS = untied;
  if(!untiedS)
    syncSVecFromScalar();
}

void CInducingWeightSpace::setWindowWidths(const CMatrix& sIn)
{
  const unsigned int M = numInducing();
  DIMENSIONMATCH(sIn.getRows() == M && sIn.getCols() == 1);
  for(unsigned int i = 0; i < M; i++)
    validateWindowWidth(sIn.getVal(i, 0));
  sVec.deepCopy(sIn);
  untiedS = true;
  s = sVec.getVal(0, 0);
}

void CInducingWeightSpace::setWindowWidthAt(unsigned int m, double sIn)
{
  validateWindowWidth(sIn);
  BOUNDCHECK(m < numInducing());
  if(sVec.getRows() != numInducing())
    syncSVecFromScalar();
  sVec.setVal(sIn, m, 0);
  untiedS = true;
  s = sVec.getVal(0, 0);
}

double CInducingWeightSpace::makeRidgeInit(CMatrix& WOut, unsigned int M,
                                           unsigned int inputDim, unsigned int rngSeed)
{
  // CIP-0037: directions before offsets once D' is moderate; one-sided offsets.
  // If M ≤ 4 D' use n_off=1 (one window per direction, nDir may exceed D').
  // Otherwise n_off = ceil(M/(4 D')) and nDir = ceil(M/n_off). Offsets on [1, 2.5]
  // (a=1.5 when n_off=1). Within each block of ≤ D' directions, Gram–Schmidt;
  // further blocks are independent random frames (prototype orth_dirs).
  if(M == 0)
    throw ndlexceptions::Error("makeRidgeInit requires M > 0.");
  const unsigned int Dp = inputDim + 1;
  unsigned int nOff = 1;
  unsigned int nDir = M;
  if(M > 4 * Dp)
  {
    nOff = (M + 4 * Dp - 1) / (4 * Dp); // ceil(M / (4 Dp))
    if(nOff < 1)
      nOff = 1;
    // Round up so nDir * nOff >= M: every row is assigned a direction
    // (rounding down left trailing rows zero; CIP-003C item 1).
    nDir = (M + nOff - 1) / nOff;
    if(nDir < 1)
      nDir = 1;
  }

  WOut.resize(M, Dp + 1);
  WOut.zeros();
  ndlutil::init_genrand(rngSeed);

  vector<vector<double> > dirs(nDir, vector<double>(Dp, 0.0));
  unsigned int di = 0;
  while(di < nDir)
  {
    const unsigned int batch = min(Dp, nDir - di);
    // Fresh orthonormal frame of size batch in R^{Dp}.
    for(unsigned int b = 0; b < batch; b++)
    {
      for(unsigned int j = 0; j < Dp; j++)
        dirs[di + b][j] = ndlutil::randn();
      for(unsigned int p = 0; p < b; p++)
      {
        double dot = 0.0;
        for(unsigned int j = 0; j < Dp; j++)
          dot += dirs[di + b][j] * dirs[di + p][j];
        for(unsigned int j = 0; j < Dp; j++)
          dirs[di + b][j] -= dot * dirs[di + p][j];
      }
      double nrm = 0.0;
      for(unsigned int j = 0; j < Dp; j++)
        nrm += dirs[di + b][j] * dirs[di + b][j];
      nrm = sqrt(nrm);
      if(!(nrm > 1e-12))
      {
        for(unsigned int j = 0; j < Dp; j++)
          dirs[di + b][j] = (j == ((di + b) % Dp)) ? 1.0 : 0.0;
        for(unsigned int p = 0; p < b; p++)
        {
          double dot = 0.0;
          for(unsigned int j = 0; j < Dp; j++)
            dot += dirs[di + b][j] * dirs[di + p][j];
          for(unsigned int j = 0; j < Dp; j++)
            dirs[di + b][j] -= dot * dirs[di + p][j];
        }
        nrm = 0.0;
        for(unsigned int j = 0; j < Dp; j++)
          nrm += dirs[di + b][j] * dirs[di + b][j];
        nrm = sqrt(max(nrm, 1e-30));
      }
      for(unsigned int j = 0; j < Dp; j++)
        dirs[di + b][j] /= nrm;
    }
    di += batch;
  }

  // Assign rows: up to nOff one-sided offsets per direction.
  unsigned int row = 0;
  for(unsigned int dIdx = 0; dIdx < nDir && row < M; dIdx++)
  {
    const unsigned int remain = M - row;
    const unsigned int dirsLeft = nDir - dIdx;
    const unsigned int nThis = min(nOff, (remain + dirsLeft - 1) / dirsLeft);
    for(unsigned int o = 0; o < nThis && row < M; o++, row++)
    {
      for(unsigned int j = 0; j < Dp; j++)
        WOut.setVal(dirs[dIdx][j], row, j);
      // One-sided offsets on [1, 2.5]; never a=0 (erf odd ⇒ ±a duplicate up to sign).
      double a0 = 1.5;
      if(nThis > 1)
        a0 = 1.0 + 1.5 * (double)o / (double)(nThis - 1);
      WOut.setVal(a0, row, Dp);
    }
  }

  // Offset spacing Δa = 1.5/(n_off-1) on [1, 2.5]. Start near 0.75·Δa (capped),
  // then shrink to κ(K_uu)≤10^4. Single-offset layouts start at s=0.8.
  const double spacing = (nOff <= 1) ? (1.5 / 0.75) : (1.5 / (double)(nOff - 1));
  double s0 = min(0.8, 0.75 * spacing);
  if(s0 < 0.08)
    s0 = 0.08;

  // κ-budget refine (CIP-0028 Phase-1 gate): shrink s until κ(K_uu)≤10^4.
  // K_uu is kernel-free for ridge, so a dummy MLP kernel is enough.
  const double kappaBudget = 1e4;
  const double sFloor = 0.05;
  CMatrix Xdummy(1, max(inputDim, 1u));
  Xdummy.zeros();
  CMlpKern dummyKern(Xdummy);
  for(unsigned int attempt = 0; attempt < 16; attempt++)
  {
    CInducingWeightSpace probe(WOut, s0, false, true);
    CMatrix Kuu(M, M);
    probe.computeKuu(Kuu, dummyKern);
    for(unsigned int i = 0; i < M; i++)
      Kuu.addVal(1e-12, i, i);
    Kuu.setSymmetric(true);
    CMatrix lam(1, M);
    Kuu.syev(lam, "n", "u");
    double lmin = fabs(lam.getVal(0));
    double lmax = lmin;
    for(unsigned int i = 1; i < M; i++)
    {
      const double li = fabs(lam.getVal(i));
      if(li < lmin) lmin = li;
      if(li > lmax) lmax = li;
    }
    const double kappa = lmax / max(lmin, 1e-18);
    if(kappa <= kappaBudget || s0 <= sFloor * 1.01)
      break;
    s0 = max(sFloor, 0.85 * s0);
  }
  return s0;
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
  validateWindowWidth(sIn);
  s = sIn;
  // Broadcast to all units (CIP-0032); leaves untiedS unchanged.
  if(sVec.getRows() != numInducing())
    syncSVecFromScalar();
  else
  {
    for(unsigned int i = 0; i < numInducing(); i++)
      sVec.setVal(s, i, 0);
  }
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
  // K_uu[m,n] from 2×2 Woodbury in cosines.
  // CIP-0028: tied s. CIP-0032: α_m = 1/s_m² - 1/2 (heterogeneous pair).
  ridgeFactors(Vtmp, atmp);
  const unsigned int M = numInducing();
  const unsigned int Dp = weightDim();
  BOUNDCHECK(sVec.getRows() == M);
  Kuu.resize(M, M);
  for(unsigned int i = 0; i < M; i++)
  {
    const double si = sVec.getVal(i, 0);
    const double s2i = si * si;
    const double alphai = 1.0 / s2i - 0.5;
    const double am = atmp.getVal(i, 0);
    for(unsigned int j = i; j < M; j++)
    {
      double c = 0.0;
      for(unsigned int k = 0; k < Dp; k++)
        c += Vtmp.getVal(i, k) * Vtmp.getVal(j, k);
      if(c > 1.0) c = 1.0;
      if(c < -1.0) c = -1.0;
      const double sj = sVec.getVal(j, 0);
      const double s2j = sj * sj;
      const double alphaj = 1.0 / s2j - 0.5;
      const double an = atmp.getVal(j, 0);
      const double detI = (1.0 + alphai) * (1.0 + alphaj) - alphai * alphaj * c * c;
      BOUNDCHECK(detI > 0.0);
      const double minv11 = (1.0 + alphaj) / detI;
      const double minv12 = -alphai * c / detI;
      const double minv21 = -alphaj * c / detI;
      const double minv22 = (1.0 + alphai) / detI;
      const double H11 = minv11 + c * minv21;
      const double H12 = minv12 + c * minv22;
      const double H21 = c * minv11 + minv21;
      const double H22 = c * minv12 + minv22;
      const double bm = am / s2i;
      const double bn = an / s2j;
      const double quad = H11 * bm * bm + H22 * bn * bn + (H12 + H21) * bm * bn;
      const double lognorm = -0.25 * log(M_PI * s2i) - 0.25 * log(M_PI * s2j)
                             - 0.5 * (double)(Dp - 1) * log(2.0 * M_PI);
      const double logK = lognorm + 0.5 * (double)Dp * log(2.0 * M_PI)
                          - 0.5 * log(detI) + 0.5 * quad
                          - am * am / (2.0 * s2i) - an * an / (2.0 * s2j);
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
    if(!untiedS)
      kern.wsRidgeCrossCompute(Kfu, X, Vtmp, atmp, s);
    else
    {
      // CIP-0032: per-unit s_m via single-column calls (scalar kernel API).
      const unsigned int Dp = weightDim();
      CMatrix Vm(1, Dp);
      CMatrix am(1, 1);
      CMatrix Km(N, 1);
      for(unsigned int m = 0; m < M; m++)
      {
        for(unsigned int k = 0; k < Dp; k++)
          Vm.setVal(Vtmp.getVal(m, k), 0, k);
        am.setVal(atmp.getVal(m, 0), 0, 0);
        kern.wsRidgeCrossCompute(Km, X, Vm, am, sVec.getVal(m, 0));
        for(unsigned int n = 0; n < N; n++)
          Kfu.setVal(Km.getVal(n, 0), n, m);
      }
    }
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
    if(!untiedS)
      kern.wsRidgeCrossCompute(Kfu, Xstar, Vtmp, atmp, s);
    else
    {
      const unsigned int Dp = weightDim();
      CMatrix Vm(1, Dp);
      CMatrix am(1, 1);
      CMatrix Km(Ns, 1);
      for(unsigned int m = 0; m < M; m++)
      {
        for(unsigned int k = 0; k < Dp; k++)
          Vm.setVal(Vtmp.getVal(m, k), 0, k);
        am.setVal(atmp.getVal(m, 0), 0, 0);
        kern.wsRidgeCrossCompute(Km, Xstar, Vm, am, sVec.getVal(m, 0));
        for(unsigned int n = 0; n < Ns; n++)
          Kfu.setVal(Km.getVal(n, 0), n, m);
      }
    }
  }
  else
    kern.wsCrossCompute(Kfu, Xstar, W, s);
  kufFromKfu(Kux, Kfu);
}

void CInducingWeightSpace::gradRawFromKuuRidge(CMatrix& gRaw, const CMatrix& gKuu) const
{
  // Analytic ∂⟨gKuu, Kuu⟩ through unit directions / offsets, then chain to raw W.
  // CIP-0032: per-unit s_m (α_i = 1/s_i² - 1/2).
  ridgeFactors(Vtmp, atmp);
  const unsigned int M = numInducing();
  const unsigned int Dp = weightDim();
  BOUNDCHECK(sVec.getRows() == M);
  CMatrix gV(M, Dp);
  CMatrix ga(M, 1);
  gV.zeros();
  ga.zeros();

  for(unsigned int i = 0; i < M; i++)
  {
    const double s2i = sVec.getVal(i, 0) * sVec.getVal(i, 0);
    const double alphai = 1.0 / s2i - 0.5;
    const double am = atmp.getVal(i, 0);
    for(unsigned int j = i; j < M; j++)
    {
      double c = 0.0;
      for(unsigned int k = 0; k < Dp; k++)
        c += Vtmp.getVal(i, k) * Vtmp.getVal(j, k);
      if(c > 1.0) c = 1.0;
      if(c < -1.0) c = -1.0;
      const double s2j = sVec.getVal(j, 0) * sVec.getVal(j, 0);
      const double alphaj = 1.0 / s2j - 0.5;
      const double an = atmp.getVal(j, 0);
      const double detI = (1.0 + alphai) * (1.0 + alphaj) - alphai * alphaj * c * c;
      BOUNDCHECK(detI > 0.0);
      const double minv11 = (1.0 + alphaj) / detI;
      const double minv12 = -alphai * c / detI;
      const double minv21 = -alphaj * c / detI;
      const double minv22 = (1.0 + alphai) / detI;
      const double H11 = minv11 + c * minv21;
      const double H12 = minv12 + c * minv22;
      const double H21 = c * minv11 + minv21;
      const double H22 = c * minv12 + minv22;
      const double bm = am / s2i;
      const double bn = an / s2j;
      const double lognorm = -0.25 * log(M_PI * s2i) - 0.25 * log(M_PI * s2j)
                             - 0.5 * (double)(Dp - 1) * log(2.0 * M_PI);
      const double quad = H11 * bm * bm + H22 * bn * bn + (H12 + H21) * bm * bn;
      const double logK = lognorm + 0.5 * (double)Dp * log(2.0 * M_PI)
                          - 0.5 * log(detI) + 0.5 * quad
                          - am * am / (2.0 * s2i) - an * an / (2.0 * s2j);
      const double Kij = exp(logK);
      const double gsym = (i == j) ? gKuu.getVal(i, j)
                                   : (gKuu.getVal(i, j) + gKuu.getVal(j, i));

      const double dlog_dam = (H11 * bm + 0.5 * (H12 + H21) * bn) / s2i - am / s2i;
      const double dlog_dan = (H22 * bn + 0.5 * (H12 + H21) * bm) / s2j - an / s2j;
      ga.addVal(gsym * Kij * dlog_dam, i, 0);
      if(i != j)
        ga.addVal(gsym * Kij * dlog_dan, j, 0);

      if(i != j)
      {
        const double ddet = -2.0 * alphai * alphaj * c;
        // ∂Minv/∂c through det and numerator (α-fixed).
        const double dminv11 = -(1.0 + alphaj) / (detI * detI) * ddet;
        const double dminv12 = -alphai / detI + alphai * c / (detI * detI) * ddet;
        const double dminv21 = -alphaj / detI + alphaj * c / (detI * detI) * ddet;
        const double dminv22 = -(1.0 + alphai) / (detI * detI) * ddet;
        const double dH11 = dminv11 + minv21 + c * dminv21;
        const double dH12 = dminv12 + minv22 + c * dminv22;
        const double dH21 = minv11 + c * dminv11 + dminv21;
        const double dH22 = minv12 + c * dminv12 + dminv22;
        const double dquad = dH11 * bm * bm + dH22 * bn * bn
                             + (dH12 + dH21) * bm * bn;
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
                                         bool wantX, bool regulariseKern,
                                         const CMatrix* /*KuuCached*/,
                                         const CMatrix* /*KufCached*/) const
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
    const unsigned int Dp = weightDim();
    CMatrix gV(M, Dp);
    CMatrix ga(M, 1);
    gV.zeros();
    ga.zeros();
    if(!untiedS)
    {
      kern.wsRidgeCrossCompute(Kfu, X, Vtmp, atmp, s);
      kern.wsRidgeGradParams(tempG, X, Vtmp, atmp, s, gKfu);
      kern.wsRidgeGradVA(gV, ga, X, Vtmp, atmp, s, gKfu);
    }
    else
    {
      // CIP-0032: accumulate K_fu grads with per-unit s_m.
      CMatrix Vm(1, Dp);
      CMatrix am(1, 1);
      CMatrix Km(N, 1);
      CMatrix gKm(N, 1);
      CMatrix gPiece(1, nParams);
      CMatrix gVm(1, Dp);
      CMatrix gam(1, 1);
      for(unsigned int m = 0; m < M; m++)
      {
        for(unsigned int k = 0; k < Dp; k++)
          Vm.setVal(Vtmp.getVal(m, k), 0, k);
        am.setVal(atmp.getVal(m, 0), 0, 0);
        for(unsigned int n = 0; n < N; n++)
          gKm.setVal(gKfu.getVal(n, m), n, 0);
        const double sm = sVec.getVal(m, 0);
        kern.wsRidgeCrossCompute(Km, X, Vm, am, sm);
        for(unsigned int n = 0; n < N; n++)
          Kfu.setVal(Km.getVal(n, 0), n, m);
        gPiece.zeros();
        kern.wsRidgeGradParams(gPiece, X, Vm, am, sm, gKm);
        for(unsigned int p = 0; p < nParams; p++)
          tempG.addVal(gPiece.getVal(p), p);
        kern.wsRidgeGradVA(gVm, gam, X, Vm, am, sm, gKm);
        for(unsigned int k = 0; k < Dp; k++)
          gV.setVal(gVm.getVal(0, k), m, k);
        ga.setVal(gam.getVal(0, 0), m, 0);
      }
    }
    // Chain Kfu grads through normalisation into raw W.
    for(unsigned int i = 0; i < M; i++)
    {
      double nrm = 0.0;
      for(unsigned int k = 0; k < Dp; k++)
        nrm += W.getVal(i, k) * W.getVal(i, k);
      nrm = sqrt(nrm);
      BOUNDCHECK(nrm > 1e-15);
      double gdotv = 0.0;
      for(unsigned int k = 0; k < Dp; k++)
        gdotv += gV.getVal(i, k) * Vtmp.getVal(i, k);
      for(unsigned int k = 0; k < Dp; k++)
      {
        const double gv = gV.getVal(i, k) - gdotv * Vtmp.getVal(i, k);
        gZ.addVal(gv / nrm, i, k);
      }
      gZ.addVal(ga.getVal(i, 0), i, Dp);
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
      if(!untiedS)
        kern.wsRidgeGradX(*gXOrNull, X, Vtmp, atmp, s, gKfu);
      else
      {
        const unsigned int Dp = weightDim();
        CMatrix Vm(1, Dp);
        CMatrix am(1, 1);
        CMatrix gKm(N, 1);
        CMatrix gXm(N, X.getCols());
        for(unsigned int m = 0; m < M; m++)
        {
          for(unsigned int k = 0; k < Dp; k++)
            Vm.setVal(Vtmp.getVal(m, k), 0, k);
          am.setVal(atmp.getVal(m, 0), 0, 0);
          for(unsigned int n = 0; n < N; n++)
            gKm.setVal(gKfu.getVal(n, m), n, 0);
          kern.wsRidgeGradX(gXm, X, Vm, am, sVec.getVal(m, 0), gKm);
          for(unsigned int n = 0; n < N; n++)
            for(unsigned int j = 0; j < X.getCols(); j++)
              gXOrNull->addVal(gXm.getVal(n, j), n, j);
        }
      }
    }
    else
      kern.wsGradX(*gXOrNull, X, W, s, gKfu);
  }

  gScache = 0.0;
  gSvec.resize(M, 1);
  gSvec.zeros();
  if(learnS)
  {
    // ⟨gKuu, ∂Kuu/∂s⟩ via central FD on fillKuu; K_fu via wsRidgeGradS.
    if(!untiedS)
    {
      const double eps = 1e-7 * ((s > 1.0) ? s : 1.0);
      CMatrix Kup(M, M), Kum(M, M);
      const double s0 = s;
      const_cast<CInducingWeightSpace*>(this)->s = s0 + eps;
      const_cast<CInducingWeightSpace*>(this)->syncSVecFromScalar();
      fillKuu(Kup);
      const_cast<CInducingWeightSpace*>(this)->s = s0 - eps;
      const_cast<CInducingWeightSpace*>(this)->syncSVecFromScalar();
      fillKuu(Kum);
      const_cast<CInducingWeightSpace*>(this)->s = s0;
      const_cast<CInducingWeightSpace*>(this)->syncSVecFromScalar();
      double gSkuu = 0.0;
      for(unsigned int i = 0; i < M; i++)
        for(unsigned int j = 0; j < M; j++)
          gSkuu += gKuu.getVal(i, j) * (Kup.getVal(i, j) - Kum.getVal(i, j)) / (2.0 * eps);

      CMatrix gSkuf(1, 1);
      gSkuf.zeros();
      if(ridge)
      {
        ridgeFactors(Vtmp, atmp);
        kern.wsRidgeGradS(gSkuf, X, Vtmp, atmp, s, gKfu);
      }
      else
        kern.wsGradS(gSkuf, X, W, s, gKfu);
      gScache = gSkuu + gSkuf.getVal(0);
      gSvec.setVal(gScache, 0, 0);
    }
    else
    {
      // CIP-0032: one FD / analytic grad per s_m.
      BOUNDCHECK(ridge);
      ridgeFactors(Vtmp, atmp);
      const unsigned int Dp = weightDim();
      CMatrix Kup(M, M), Kum(M, M);
      CMatrix Vm(1, Dp);
      CMatrix am(1, 1);
      CMatrix gKm(N, 1);
      CMatrix gSkuf(1, 1);
      for(unsigned int m = 0; m < M; m++)
      {
        const double sm = sVec.getVal(m, 0);
        const double eps = 1e-7 * ((sm > 1.0) ? sm : 1.0);
        const_cast<CInducingWeightSpace*>(this)->sVec.setVal(sm + eps, m, 0);
        fillKuu(Kup);
        const_cast<CInducingWeightSpace*>(this)->sVec.setVal(sm - eps, m, 0);
        fillKuu(Kum);
        const_cast<CInducingWeightSpace*>(this)->sVec.setVal(sm, m, 0);
        double gSkuu = 0.0;
        for(unsigned int i = 0; i < M; i++)
          for(unsigned int j = 0; j < M; j++)
            gSkuu += gKuu.getVal(i, j) * (Kup.getVal(i, j) - Kum.getVal(i, j)) / (2.0 * eps);

        for(unsigned int k = 0; k < Dp; k++)
          Vm.setVal(Vtmp.getVal(m, k), 0, k);
        am.setVal(atmp.getVal(m, 0), 0, 0);
        for(unsigned int n = 0; n < N; n++)
          gKm.setVal(gKfu.getVal(n, m), n, 0);
        gSkuf.zeros();
        kern.wsRidgeGradS(gSkuf, X, Vm, am, sm, gKm);
        gSvec.setVal(gSkuu + gSkuf.getVal(0), m, 0);
      }
      gScache = 0.0;
      for(unsigned int m = 0; m < M; m++)
        gScache += gSvec.getVal(m, 0);
    }
  }
}

unsigned int CInducingWeightSpace::getNumParams() const
{
  unsigned int n = numInducing() * domainDim();
  if(learnS)
    n += untiedS ? numInducing() : 1;
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
    // Pack unconstrained u with s = sLo + (sHi-sLo)·σ(u) so SCG cannot
    // drive widths outside the κ-safe interval.
    if(!untiedS)
    {
      param.setVal(windowWidthToUnconstrained(s), counter);
      counter++;
    }
    else
    {
      for(unsigned int i = 0; i < numInducing(); i++)
      {
        param.setVal(windowWidthToUnconstrained(sVec.getVal(i, 0)), counter);
        counter++;
      }
    }
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
    if(!untiedS)
    {
      setWindowWidth(windowWidthFromUnconstrained(param.getVal(counter)));
      counter++;
    }
    else
    {
      CMatrix sIn(numInducing(), 1);
      for(unsigned int i = 0; i < numInducing(); i++)
      {
        sIn.setVal(windowWidthFromUnconstrained(param.getVal(counter)), i, 0);
        counter++;
      }
      setWindowWidths(sIn);
    }
  }
}

void CInducingWeightSpace::setCentres(const CMatrix& Z)
{
  DIMENSIONMATCH(Z.getRows() == W.getRows());
  DIMENSIONMATCH(Z.getCols() == W.getCols());
  W.deepCopy(Z);
}

// ---- CInducingHermite (CIP-003D) ----

namespace {

// Orthonormal frame of n <= Dp random directions in R^Dp (Gram-Schmidt, applied
// twice for stability; a degenerate draw is redrawn, never padded).
void hermiteOrthFrame(vector<vector<double> >& rows, unsigned int n, unsigned int Dp)
{
  rows.assign(n, vector<double>(Dp, 0.0));
  for(unsigned int b = 0; b < n; b++)
  {
    for(unsigned int attempt = 0; attempt < 100; attempt++)
    {
      for(unsigned int j = 0; j < Dp; j++)
        rows[b][j] = ndlutil::randn();
      for(unsigned int pass = 0; pass < 2; pass++)
        for(unsigned int p = 0; p < b; p++)
        {
          double dot = 0.0;
          for(unsigned int j = 0; j < Dp; j++)
            dot += rows[b][j] * rows[p][j];
          for(unsigned int j = 0; j < Dp; j++)
            rows[b][j] -= dot * rows[p][j];
        }
      double nrm = 0.0;
      for(unsigned int j = 0; j < Dp; j++)
        nrm += rows[b][j] * rows[b][j];
      nrm = sqrt(nrm);
      if(nrm > 1e-8)
      {
        for(unsigned int j = 0; j < Dp; j++)
          rows[b][j] /= nrm;
        break;
      }
      if(attempt == 99)
        throw ndlexceptions::Error(
            "CInducingHermite: could not draw an orthonormal degree-1 frame.");
    }
  }
}

void hermiteRandomUnit(vector<double>& v, unsigned int Dp)
{
  v.assign(Dp, 0.0);
  for(unsigned int attempt = 0; attempt < 100; attempt++)
  {
    double nrm = 0.0;
    for(unsigned int j = 0; j < Dp; j++)
    {
      v[j] = ndlutil::randn();
      nrm += v[j] * v[j];
    }
    nrm = sqrt(nrm);
    if(nrm > 1e-8)
    {
      for(unsigned int j = 0; j < Dp; j++)
        v[j] /= nrm;
      return;
    }
  }
  throw ndlexceptions::Error("CInducingHermite: could not draw a random direction.");
}

string hermiteUInt(unsigned int x)
{
  return (x == numeric_limits<unsigned int>::max()) ? string(">=UINT_MAX")
                                                      : to_string(x);
}

} // namespace

vector<string> HermiteLayoutReport::warnings() const
{
  vector<string> out;
  if(warnConditioning)
  {
    ostringstream os;
    os << "Warning (CIP-003D W1): Hermite K_uu condition number " << kappa
       << " exceeds " << CInducingHermite::kappaWarnThreshold()
       << " (M=" << numInducing << ", D'=" << weightDim << ").";
    out.push_back(os.str());
  }
  if(warnNearCapacity)
  {
    ostringstream os;
    os << "Warning (CIP-003D W2): Hermite layout uses " << 100.0 * fillFraction
       << "% of its capacity (M=" << numInducing << " of "
       << hermiteUInt(totalCapacity) << ", D'=" << weightDim
       << "); conditioning degrades near capacity.";
    out.push_back(os.str());
  }
  return out;
}

string HermiteLayoutReport::summary() const
{
  ostringstream os;
  os << "hermite M=" << numInducing << " D'=" << weightDim << " degrees";
  for(size_t i = 0; i < degrees.size(); i++)
    if(count[i] > 0)
      os << " " << degrees[i] << ":" << count[i] << "/" << hermiteUInt(capacity[i]);
  os << " fill=" << 100.0 * fillFraction << "% kappa=" << kappa;
  return os.str();
}

unsigned int CInducingHermite::degreeCapacity(unsigned int weightDim, unsigned int k)
{
  if(k == 0 || (k % 2) == 0)
    throw ndlexceptions::Error("CInducingHermite: degrees must be odd and >= 1.");
  if(k == 1)
    return weightDim;
  // floor(C(D'+k-1, k) / 2), in long double, saturating.
  long double c = 1.0L;
  const unsigned int n = weightDim + k - 1;
  for(unsigned int i = 1; i <= k; i++)
    c = c * (long double)(n - k + i) / (long double)i;
  c = floorl(c / 2.0L);
  const long double cap = (long double)numeric_limits<unsigned int>::max();
  return (c >= cap) ? numeric_limits<unsigned int>::max() : (unsigned int)c;
}

unsigned int CInducingHermite::totalCapacity(unsigned int weightDim, unsigned int kmax)
{
  unsigned long long total = 0;
  const unsigned long long cap = numeric_limits<unsigned int>::max();
  for(unsigned int k = 1; k <= kmax; k += 2)
  {
    total += degreeCapacity(weightDim, k);
    if(total >= cap)
      return numeric_limits<unsigned int>::max();
  }
  return (unsigned int)total;
}

void CInducingHermite::makeLayout(CMatrix& VOut, vector<unsigned int>& ksOut,
                                  unsigned int M, unsigned int inputDim,
                                  unsigned int rngSeed, unsigned int kmax)
{
  if(M == 0)
    throw ndlexceptions::Error("CInducingHermite::makeLayout requires M > 0.");
  if(inputDim == 0)
    throw ndlexceptions::Error("CInducingHermite::makeLayout requires inputDim > 0.");
  if(kmax == 0 || (kmax % 2) == 0)
    throw ndlexceptions::Error("CInducingHermite::makeLayout: kmax must be odd and >= 1.");
  const unsigned int Dp = inputDim + 1;

  // E1: ask more than the rule can hold -> refuse, naming what is available.
  const unsigned int total = totalCapacity(Dp, kmax);
  if(M > total)
  {
    ostringstream os;
    os << "CInducingHermite (E1): requested M=" << M << " windows but capacity is "
       << hermiteUInt(total) << " for input dim d=" << inputDim << " (D'=" << Dp
       << ", odd degrees up to " << kmax << "); per-degree capacities:";
    for(unsigned int k = 1; k <= kmax; k += 2)
      os << " " << k << ":" << hermiteUInt(degreeCapacity(Dp, k));
    os << ". Lower M, or use inducing='points' / 'weightspace'.";
    throw ndlexceptions::Error(os.str());
  }

  vector<unsigned int> degs;
  for(unsigned int k = 1; k <= kmax; k += 2)
    degs.push_back(k);
  vector<unsigned int> alloc(degs.size(), 0);
  alloc[0] = min(Dp, M);
  unsigned int rest = M - alloc[0];
  while(rest > 0)
  {
    bool moved = false;
    for(size_t i = 1; i < degs.size() && rest > 0; i++)
      if(alloc[i] < degreeCapacity(Dp, degs[i]))
      {
        alloc[i]++;
        rest--;
        moved = true;
      }
    if(!moved)
      break;
  }
  if(rest > 0) // unreachable given the total check; assert, never truncate.
    throw ndlexceptions::Error(
        "CInducingHermite (E5): allocation left windows unassigned despite capacity check.");

  ndlutil::init_genrand(rngSeed);
  VOut.resize(M, Dp);
  VOut.zeros();
  ksOut.assign(M, 0);
  unsigned int row = 0;
  for(size_t i = 0; i < degs.size(); i++)
  {
    if(alloc[i] == 0)
      continue;
    vector<vector<double> > frame;
    if(degs[i] == 1)
      hermiteOrthFrame(frame, alloc[i], Dp);
    else
    {
      frame.resize(alloc[i]);
      for(unsigned int b = 0; b < alloc[i]; b++)
        hermiteRandomUnit(frame[b], Dp);
    }
    for(unsigned int b = 0; b < alloc[i]; b++, row++)
    {
      for(unsigned int j = 0; j < Dp; j++)
        VOut.setVal(frame[b][j], row, j);
      ksOut[row] = degs[i];
    }
  }

  // E5: delivered must equal asked, and the layout must pass the full audit.
  if(row != M || VOut.getRows() != M || ksOut.size() != M)
    throw ndlexceptions::Error("CInducingHermite (E5): delivered window count differs from requested M.");
  CInducingHermite audited(VOut, ksOut);
  (void)audited;
}

CInducingHermite::CInducingHermite(const CMatrix& VIn, const vector<unsigned int>& ksIn)
  : V(VIn), ks(ksIn)
{
  audit();
}

void CInducingHermite::audit()
{
  const unsigned int M = V.getRows();
  const unsigned int Dp = V.getCols();
  if(M == 0 || Dp == 0)
    throw ndlexceptions::Error("CInducingHermite (E5): empty windows.");
  if(ks.size() != M)
    throw ndlexceptions::Error(
        "CInducingHermite (E5): " + to_string(ks.size()) + " degrees for " +
        to_string(M) + " direction rows.");
  unsigned int maxK = 1;
  for(unsigned int m = 0; m < M; m++)
  {
    if(ks[m] == 0 || (ks[m] % 2) == 0)
      throw ndlexceptions::Error(
          "CInducingHermite (E5): window " + to_string(m) + " has degree " +
          to_string(ks[m]) + "; degrees must be odd and >= 1.");
    maxK = max(maxK, ks[m]);
    double nrm = 0.0;
    for(unsigned int j = 0; j < Dp; j++)
      nrm += V.getVal(m, j) * V.getVal(m, j);
    if(!(fabs(sqrt(nrm) - 1.0) < 1e-9))
      throw ndlexceptions::Error(
          "CInducingHermite (E5): direction " + to_string(m) +
          " is not a unit vector (norm " + to_string(sqrt(nrm)) + ").");
  }

  // Report: degree histogram against the capacity rule.
  const unsigned int kTop = max(9u, maxK);
  HermiteLayoutReport& r = layoutReport;
  r.numInducing = M;
  r.weightDim = Dp;
  r.degrees.clear();
  r.count.clear();
  r.capacity.clear();
  for(unsigned int k = 1; k <= kTop; k += 2)
  {
    r.degrees.push_back(k);
    r.count.push_back(0);
    r.capacity.push_back(degreeCapacity(Dp, k));
  }
  for(unsigned int m = 0; m < M; m++)
    r.count[(ks[m] - 1) / 2]++;
  for(size_t i = 0; i < r.degrees.size(); i++)
    if(r.count[i] > r.capacity[i])
      throw ndlexceptions::Error(
          "CInducingHermite (E5): degree " + to_string(r.degrees[i]) + " holds " +
          to_string(r.count[i]) + " windows; capacity is " +
          hermiteUInt(r.capacity[i]) + " (D'=" + to_string(Dp) + ").");
  r.totalCapacity = totalCapacity(Dp, kTop);
  r.fillFraction = (double)M / (double)r.totalCapacity;

  // K_uu block: must be positive definite or the layout is unusable.
  CMatrix Kuu(M, M);
  fillKuu(Kuu);
  CMatrix lam(1, M);
  Kuu.syev(lam, "n", "u");
  double lmin = lam.getVal(0), lmax = lam.getVal(0);
  for(unsigned int i = 1; i < M; i++)
  {
    lmin = min(lmin, lam.getVal(i));
    lmax = max(lmax, lam.getVal(i));
  }
  if(!(lmin > kuuMinEigenvalue()))
  {
    ostringstream os;
    os << "CInducingHermite (E5): K_uu block is not positive definite (min eigenvalue "
       << lmin << " <= " << kuuMinEigenvalue()
       << "); duplicate or near-parallel directions of the same degree.";
    throw ndlexceptions::Error(os.str());
  }
  r.kappa = lmax / lmin;
  r.warnConditioning = r.kappa > kappaWarnThreshold();
  r.warnNearCapacity = r.fillFraction > nearCapacityFraction();
}

void CInducingHermite::fillKuu(CMatrix& Kuu) const
{
  const unsigned int M = numInducing();
  const unsigned int Dp = weightDim();
  Kuu.resize(M, M);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int n = 0; n < M; n++)
    {
      double val = 0.0;
      if(ks[m] == ks[n])
      {
        double dot = 0.0;
        for(unsigned int j = 0; j < Dp; j++)
          dot += V.getVal(m, j) * V.getVal(n, j);
        val = pow(dot, (int)ks[m]);
      }
      Kuu.setVal(val, m, n);
    }
  Kuu.setSymmetric(true);
}

void CInducingHermite::computeKuu(CMatrix& Kuu, const CKern& /*kern*/) const
{
  fillKuu(Kuu);
}

void CInducingHermite::computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const
{
  const unsigned int M = numInducing();
  const unsigned int N = X.getRows();
  kern.wsHermiteCrossCompute(Kfu, X, V, ks);
  Kuf.resize(M, N);
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int n = 0; n < N; n++)
      Kuf.setVal(Kfu.getVal(n, m), m, n);
}

void CInducingHermite::computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const
{
  computeKuf(Kux, kern, Xstar);
}

void CInducingHermite::gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                                      const CKern& kern, const CMatrix& X,
                                      const CMatrix& gKuu, const CMatrix& gKuf,
                                      bool wantX, bool regulariseKern,
                                      const CMatrix* /*KuuCached*/,
                                      const CMatrix* /*KufCached*/) const
{
  // K_uu is kernel-free and the windows are fixed, so gKuu contributes nothing;
  // only K_fu carries kernel-parameter and data-X dependence.
  const unsigned int M = numInducing();
  const unsigned int N = X.getRows();
  const unsigned int nParams = kern.getNumParams();
  DIMENSIONMATCH(gKernParams.getCols() == nParams);
  DIMENSIONMATCH(gKuu.getRows() == M && gKuu.getCols() == M);
  DIMENSIONMATCH(gKuf.getRows() == M && gKuf.getCols() == N);
  gZ.resize(M, V.getCols());
  gZ.zeros();

  gKfu.resize(N, M);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int m = 0; m < M; m++)
      gKfu.setVal(gKuf.getVal(m, n), n, m);

  tempG.resize(1, nParams);
  tempG.zeros();
  kern.wsHermiteGradParams(tempG, X, V, ks, gKfu);
  for(unsigned int i = 0; i < kern.getNumTransforms(); i++)
  {
    const unsigned int idx = kern.getTransformIndex(i);
    tempG.setVal(tempG.getVal(idx) * kern.getTransformGradFact(kern.getParam(idx), i), idx);
  }
  if(regulariseKern)
    kern.addPriorGrad(tempG);
  for(unsigned int i = 0; i < nParams; i++)
    gKernParams.addVal(tempG.getVal(i), i);

  if(wantX && gXOrNull != 0)
    kern.wsHermiteGradX(*gXOrNull, X, V, ks, gKfu);
}

void CInducingHermite::setCentres(const CMatrix&)
{
  throw ndlexceptions::Error(
      "CInducingHermite: windows are fixed; setCentres is not supported.");
}
