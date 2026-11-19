#ifndef CINDUCING_H
#define CINDUCING_H

#include <string>
#include "CMatrix.h"
#include "CKern.h"
#include "ndlexceptions.h"

// CIP-001C: inducing-variable contract for sparse CGp.
class CInducing
{
 public:
  virtual ~CInducing() {}

  virtual unsigned int numInducing() const = 0;
  virtual unsigned int domainDim() const = 0;
  virtual bool kuuDependsOnKernel() const = 0;
  virtual std::string typeName() const = 0;

  virtual void computeKuu(CMatrix& Kuu, const CKern& kern) const = 0;
  // Kuf is M×N (GPc layout: rows inducing, cols data).
  virtual void computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const = 0;
  // Prediction cross-covariance; same layout as CKern::compute(kX, Xu, Xin).
  virtual void computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const = 0;

  // Route gKuu/gKuf into kernel-parameter grads, inducing-centre grads, and
  // optional data-X grads. gKernParams is accumulated into; gZ is overwritten.
  virtual void gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                             const CKern& kern, const CMatrix& X,
                             const CMatrix& gKuu, const CMatrix& gKuf,
                             bool wantX, bool regulariseKern) const = 0;

  virtual unsigned int getNumParams() const = 0;
  virtual void getParams(CMatrix& param, unsigned int offset) const = 0;
  virtual void setParams(const CMatrix& param, unsigned int offset) = 0;

  virtual const CMatrix& centres() const = 0;
  virtual void setCentres(const CMatrix& Z) = 0;

  // White-noise window width (scalar s). Points inducing ignore these.
  virtual double getWindowWidth() const { return 0.0; }
  virtual void setWindowWidth(double /*s*/) {}
};

// Point inducing: Kuu/Kuf are ordinary kernel evaluations on Z.
class CInducingPoints : public CInducing
{
 public:
  explicit CInducingPoints(CMatrix* pZIn);
  virtual ~CInducingPoints() {}

  unsigned int numInducing() const;
  unsigned int domainDim() const;
  bool kuuDependsOnKernel() const { return true; }
  std::string typeName() const { return "points"; }

  void computeKuu(CMatrix& Kuu, const CKern& kern) const;
  void computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const;
  void computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const;
  void gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                      const CKern& kern, const CMatrix& X,
                      const CMatrix& gKuu, const CMatrix& gKuf,
                      bool wantX, bool regulariseKern) const;

  unsigned int getNumParams() const;
  void getParams(CMatrix& param, unsigned int offset) const;
  void setParams(const CMatrix& param, unsigned int offset);

  const CMatrix& centres() const { return *pZ; }
  void setCentres(const CMatrix& Z);

 private:
  CMatrix* pZ; // not owned; points at CGp::X_u
  mutable CMatrix dgKX;
  mutable CMatrix gKX;
  mutable CMatrix gKX_uf;
  mutable CMatrix gKX_uf2;
  mutable CMatrix tempG;
  mutable CMatrix tempG2;
};

// White-noise inducing features with fixed Gaussian windows (CIP-001C Phase 2).
class CInducingWhiteNoise : public CInducing
{
 public:
  CInducingWhiteNoise(CMatrix* pZIn, double sIn, bool learnSIn = false);
  virtual ~CInducingWhiteNoise() {}

  unsigned int numInducing() const;
  unsigned int domainDim() const;
  bool kuuDependsOnKernel() const { return false; }
  std::string typeName() const { return "whitenoise"; }

  void computeKuu(CMatrix& Kuu, const CKern& kern) const;
  void computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const;
  void computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const;
  void gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                      const CKern& kern, const CMatrix& X,
                      const CMatrix& gKuu, const CMatrix& gKuf,
                      bool wantX, bool regulariseKern) const;

  unsigned int getNumParams() const;
  void getParams(CMatrix& param, unsigned int offset) const;
  void setParams(const CMatrix& param, unsigned int offset);

  const CMatrix& centres() const { return *pZ; }
  void setCentres(const CMatrix& Z);

  double getWindowWidth() const { return s; }
  void setWindowWidth(double sIn);

  // Median nearest-neighbour spacing of rows of Z (for default s).
  static double medianSpacing(const CMatrix& Z);

 private:
  void fillKuu(CMatrix& Kuu) const;
  void kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu) const;
  void gradZFromKuu(CMatrix& gZ, const CMatrix& gKuu, const CMatrix& Kuu) const;

  CMatrix* pZ; // not owned; points at CGp::X_u
  double s;
  bool learnS;
  mutable CMatrix Kfu;
  mutable CMatrix gKfu;
  mutable CMatrix tempG;
};

// CIP-0029: 1-D Green's / causal-root inducing (Gaussian windows on the line).
// Phase 0: kernel-free Kuu + CLI/stream; Kuf/grads throw until Phase 1 kernels land.
class CInducingGreens : public CInducing
{
 public:
  CInducingGreens(CMatrix* pZIn, double sIn, bool learnSIn = false);
  virtual ~CInducingGreens() {}

  unsigned int numInducing() const;
  unsigned int domainDim() const;
  bool kuuDependsOnKernel() const { return false; }
  std::string typeName() const { return "greens"; }

  void computeKuu(CMatrix& Kuu, const CKern& kern) const;
  void computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const;
  void computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const;
  void gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                      const CKern& kern, const CMatrix& X,
                      const CMatrix& gKuu, const CMatrix& gKuf,
                      bool wantX, bool regulariseKern) const;

  unsigned int getNumParams() const;
  void getParams(CMatrix& param, unsigned int offset) const;
  void setParams(const CMatrix& param, unsigned int offset);

  const CMatrix& centres() const { return *pZ; }
  void setCentres(const CMatrix& Z);

  double getWindowWidth() const { return s; }
  void setWindowWidth(double sIn);

 private:
  void fillKuu(CMatrix& Kuu) const;
  void assertOneD() const;
  void kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu) const;
  void gradZFromKuu(CMatrix& gZ, const CMatrix& gKuu, const CMatrix& Kuu) const;

  CMatrix* pZ; // not owned; points at CGp::X_u (1-D)
  double s;
  bool learnS;
  mutable CMatrix Kfu;
  mutable CMatrix gKfu;
  mutable CMatrix tempG;
};

// CIP-0028 Design A: weight-space inducing owns W (M × (d+1)); X_u unused.
// Phase 0: kernel-free Kuu + CLI/stream; Kuf/grads throw until Phase 1 MLP hooks.
class CInducingWeightSpace : public CInducing
{
 public:
  // Ridge (default): W is M × (d+2) = [direction v ∈ R^{d+1} | offset a].
  // Isotropic oracle: W is M × (d+1) centres; pass ridgeIn=false.
  CInducingWeightSpace(const CMatrix& WIn, double sIn, bool learnSIn = false,
                       bool ridgeIn = true);
  virtual ~CInducingWeightSpace() {}

  unsigned int numInducing() const;
  unsigned int domainDim() const;
  bool kuuDependsOnKernel() const { return false; }
  std::string typeName() const { return "weightspace"; }
  bool isRidge() const { return ridge; }
  unsigned int weightDim() const; // d+1 (direction / isotropic centre dim)

  void computeKuu(CMatrix& Kuu, const CKern& kern) const;
  void computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const;
  void computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const;
  void gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                      const CKern& kern, const CMatrix& X,
                      const CMatrix& gKuu, const CMatrix& gKuf,
                      bool wantX, bool regulariseKern) const;

  unsigned int getNumParams() const;
  void getParams(CMatrix& param, unsigned int offset) const;
  void setParams(const CMatrix& param, unsigned int offset);

  const CMatrix& centres() const { return W; }
  void setCentres(const CMatrix& Z);

  double getWindowWidth() const { return s; }
  void setWindowWidth(double sIn);

  // Extract unit directions V (M × (d+1)) and offsets a (M × 1). Ridge only.
  void ridgeFactors(CMatrix& V, CMatrix& a) const;

 private:
  void fillKuu(CMatrix& Kuu) const;
  void fillKuuIsotropic(CMatrix& Kuu) const;
  void fillKuuRidge(CMatrix& Kuu) const;
  void kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu) const;
  void gradWFromKuu(CMatrix& gW, const CMatrix& gKuu, const CMatrix& Kuu) const;

  CMatrix W; // owned; ridge: M×(d+2); isotropic: M×(d+1)
  double s;
  bool learnS;
  bool ridge;
  mutable CMatrix Kfu;
  mutable CMatrix gKfu;
  mutable CMatrix tempG;
  mutable CMatrix Vtmp;
  mutable CMatrix atmp;
};

#endif /* CINDUCING_H */
