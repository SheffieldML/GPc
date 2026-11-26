#ifndef CINDUCING_H
#define CINDUCING_H

#include <string>
#include <vector>
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
                             bool wantX, bool regulariseKern,
                             const CMatrix* KuuCached = 0,
                             const CMatrix* KufCached = 0) const = 0;

  virtual unsigned int getNumParams() const = 0;
  virtual void getParams(CMatrix& param, unsigned int offset) const = 0;
  virtual void setParams(const CMatrix& param, unsigned int offset) = 0;

  virtual const CMatrix& centres() const = 0;
  virtual void setCentres(const CMatrix& Z) = 0;

  // White-noise / weight-space window width (scalar s). Points ignore these.
  virtual double getWindowWidth() const { return 0.0; }
  virtual void setWindowWidth(double /*s*/) {}
  // Global learnable s (CIP-001C/0028 follow-on). Default off.
  virtual bool learnsWindowWidth() const { return false; }
  virtual void setLearnWindowWidth(bool /*learn*/) {}
  // ∂L/∂s from the last gradFromKuuKuf call when learnsWindowWidth(); else 0.
  virtual double windowWidthGrad() const { return 0.0; }
  // CIP-0032: how many learnable width slots are packed (0, 1, or M).
  virtual unsigned int numLearnedWindowWidths() const
  {
    return learnsWindowWidth() ? 1u : 0u;
  }
  virtual double windowWidthGradAt(unsigned int /*i*/) const
  {
    return windowWidthGrad();
  }
  // CIP-0032: untied per-unit widths (weight-space). Default: tied / absent.
  virtual bool isWindowWidthUntied() const { return false; }
  virtual unsigned int numWindowWidths() const { return 0u; }
  virtual double getWindowWidthAt(unsigned int /*m*/) const { return getWindowWidth(); }
  virtual void setWindowWidthAt(unsigned int /*m*/, double sIn) { setWindowWidth(sIn); }
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
                      bool wantX, bool regulariseKern,
                      const CMatrix* KuuCached = 0,
                      const CMatrix* KufCached = 0) const;

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
                      bool wantX, bool regulariseKern,
                      const CMatrix* KuuCached = 0,
                      const CMatrix* KufCached = 0) const;

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
                      bool wantX, bool regulariseKern,
                      const CMatrix* KuuCached = 0,
                      const CMatrix* KufCached = 0) const;

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
                      bool wantX, bool regulariseKern,
                      const CMatrix* KuuCached = 0,
                      const CMatrix* KufCached = 0) const;

  unsigned int getNumParams() const;
  void getParams(CMatrix& param, unsigned int offset) const;
  void setParams(const CMatrix& param, unsigned int offset);

  const CMatrix& centres() const { return W; }
  void setCentres(const CMatrix& Z);

  // Tied scalar (CIP-0028). Broadcasts to all units; does not clear untied mode.
  double getWindowWidth() const { return s; }
  void setWindowWidth(double sIn);
  // CIP-0032: per-unit widths (M×1). Sets untied mode.
  const CMatrix& getWindowWidths() const { return sVec; }
  void setWindowWidths(const CMatrix& sIn);
  bool isWindowWidthUntied() const { return untiedS; }
  void setWindowWidthUntied(bool untied);
  double windowWidthAt(unsigned int m) const;

  bool learnsWindowWidth() const { return learnS; }
  void setLearnWindowWidth(bool learn) { learnS = learn; }
  // Tied: scalar in gScache. Untied: per-unit grads in gSvec (M×1).
  double windowWidthGrad() const { return gScache; }
  const CMatrix& windowWidthGrads() const { return gSvec; }
  unsigned int numLearnedWindowWidths() const
  {
    if(!learnS)
      return 0u;
    return untiedS ? numInducing() : 1u;
  }
  // Grad w.r.t. unconstrained u (chain ∂L/∂s · ds/du) for opt packing.
  double windowWidthGradAt(unsigned int i) const
  {
    const double gs = untiedS
                          ? ((i < gSvec.getRows()) ? gSvec.getVal(i, 0) : 0.0)
                          : gScache;
    const double sNat = untiedS ? windowWidthAt(i) : s;
    return gs * windowWidthDsDu(sNat);
  }
  unsigned int numWindowWidths() const { return numInducing(); }
  double getWindowWidthAt(unsigned int m) const { return windowWidthAt(m); }
  void setWindowWidthAt(unsigned int m, double sIn);

  // Extract unit directions V (M × (d+1)) and offsets a (M × 1). Ridge only.
  void ridgeFactors(CMatrix& V, CMatrix& a) const;

  // CIP-0037 ridge init: directions first (n_off=1 while M≤4 D'; else
  // n_off=ceil(M/(4 D'))), one-sided offsets on [1, 2.5] (never a=0), and s from
  // offset spacing so κ(K_uu) stays ~≤10^4. WOut is M × (d+2); returns window width.
  static double makeRidgeInit(CMatrix& WOut, unsigned int M, unsigned int inputDim,
                              unsigned int rngSeed = 7u);

  // Whitened-unit bounds for ridge window width (CIP-0028 stability follow-on).
  // Above ~1, κ(K_uu) explodes vs offset spacing; s=√2 is a singular special case.
  static double windowWidthMin() { return 0.05; }
  static double windowWidthMax() { return 1.0; }
  static void validateWindowWidth(double sIn);

 private:
  void fillKuu(CMatrix& Kuu) const;
  void fillKuuIsotropic(CMatrix& Kuu) const;
  void fillKuuRidge(CMatrix& Kuu) const;
  void syncSVecFromScalar();
  // Learnable s packed in unconstrained coordinates: s = sLo + (sHi-sLo)·σ(u).
  static double windowWidthFromUnconstrained(double u);
  static double windowWidthToUnconstrained(double sIn);
  static double windowWidthDsDu(double sIn);
  void kufFromKfu(CMatrix& Kuf, const CMatrix& Kfu) const;
  void gradWFromKuu(CMatrix& gW, const CMatrix& gKuu, const CMatrix& Kuu) const;
  // Analytic ∂⟨gKuu,Kuu⟩/∂(unit V, a); then chain to raw W via normalisation.
  void gradRawFromKuuRidge(CMatrix& gRaw, const CMatrix& gKuu) const;

  CMatrix W; // owned; ridge: M×(d+2); isotropic: M×(d+1)
  double s;          // tied / broadcast mirror (CIP-0028)
  CMatrix sVec;      // M×1 widths (CIP-0032; always populated)
  bool learnS;
  bool untiedS;      // false ⇒ treat as one shared s (default)
  bool ridge;
  mutable double gScache;
  mutable CMatrix gSvec; // M×1 when untied learnS
  mutable CMatrix Kfu;
  mutable CMatrix gKfu;
  mutable CMatrix tempG;
  mutable CMatrix Vtmp;
  mutable CMatrix atmp;
};

// CIP-003D: what a Hermite layout actually delivered, against what was asked.
// Recorded at construction so callers (C++ verbosity, Python warnings.warn,
// describe()) can surface it; nothing here changes the layout.
struct HermiteLayoutReport
{
  unsigned int numInducing;                // delivered M
  unsigned int weightDim;                  // D' = d + 1
  std::vector<unsigned int> degrees;       // 1, 3, 5, ... kmax
  std::vector<unsigned int> count;         // windows allocated per degree
  std::vector<unsigned int> capacity;      // capacity per degree
  unsigned int totalCapacity;              // sum of capacity (saturates at UINT_MAX)
  double fillFraction;                     // numInducing / totalCapacity
  double kappa;                            // condition number of the K_uu block
  bool warnConditioning;                   // W1: kappa > kappaWarnThreshold()
  bool warnNearCapacity;                   // W2: fillFraction > nearCapacityFraction()

  // Human-readable warning lines (empty when none fire). "Warning (CIP-003D) ..."
  std::vector<std::string> warnings() const;
  // Degree histogram, kappa and fill, one line, for describe()/str(model).
  std::string summary() const;
};

// CIP-003D: fixed Hermite ridge windows (odd degrees), no width, no offset, no
// learnable parameters. K_uu = delta_{kl} (v_m . v_n)^k is kernel-free.
// Phases 1-2: layout, capacity, audit, K_uu, K_uf / K_ux (MLP kernels only,
// error E2 otherwise). Gradients throw NotImplementedError until phase 3.
class CInducingHermite : public CInducing
{
 public:
  // V is M x D' with unit rows; ks has M odd degrees >= 1. Audited here
  // (error E5): shapes, odd degrees, unit norm, K_uu block positive definite.
  CInducingHermite(const CMatrix& VIn, const std::vector<unsigned int>& ksIn);
  virtual ~CInducingHermite() {}

  unsigned int numInducing() const { return V.getRows(); }
  unsigned int domainDim() const { return V.getCols(); }
  unsigned int weightDim() const { return V.getCols(); }
  bool kuuDependsOnKernel() const { return false; }
  std::string typeName() const { return "hermite"; }

  const CMatrix& directions() const { return V; }
  const std::vector<unsigned int>& degrees() const { return ks; }
  const HermiteLayoutReport& report() const { return layoutReport; }

  void computeKuu(CMatrix& Kuu, const CKern& kern) const;
  void computeKuf(CMatrix& Kuf, const CKern& kern, const CMatrix& X) const;
  void computeKux(CMatrix& Kux, const CKern& kern, const CMatrix& Xstar) const;
  void gradFromKuuKuf(CMatrix& gKernParams, CMatrix& gZ, CMatrix* gXOrNull,
                      const CKern& kern, const CMatrix& X,
                      const CMatrix& gKuu, const CMatrix& gKuf,
                      bool wantX, bool regulariseKern,
                      const CMatrix* KuuCached = 0,
                      const CMatrix* KufCached = 0) const;

  // Fixed windows: nothing to optimise.
  unsigned int getNumParams() const { return 0; }
  void getParams(CMatrix& /*param*/, unsigned int /*offset*/) const {}
  void setParams(const CMatrix& /*param*/, unsigned int /*offset*/) {}

  const CMatrix& centres() const { return V; }
  // Windows are fixed (error E4 territory): refuse rather than ignore.
  void setCentres(const CMatrix& Z);

  // Capacity rule (CIP-003D, kept fixed): degree 1 holds D' windows; odd
  // degree k >= 3 holds floor(C(D'+k-1, k)/2). Saturates at UINT_MAX.
  static unsigned int degreeCapacity(unsigned int weightDim, unsigned int k);
  static unsigned int totalCapacity(unsigned int weightDim, unsigned int kmax = 9);

  // Layout of M windows for inputDim d (D' = d+1): degree 1 takes min(D', M)
  // with an orthonormal frame; the rest are dealt round-robin to degrees
  // 3, 5, ..., kmax up to capacity, with random unit directions. Throws
  // (error E1) if M exceeds totalCapacity, and (E5) if what was built differs
  // from what was asked. Deterministic for a given seed.
  static void makeLayout(CMatrix& VOut, std::vector<unsigned int>& ksOut,
                         unsigned int M, unsigned int inputDim,
                         unsigned int rngSeed = 7u, unsigned int kmax = 9);

  static double kappaWarnThreshold() { return 1e4; }       // CIP-0028 budget
  static double nearCapacityFraction() { return 0.9; }
  static double kuuMinEigenvalue() { return 1e-12; }       // below: not PD (E5)

 private:
  void fillKuu(CMatrix& Kuu) const;
  void audit();

  CMatrix V;                         // M x D', unit rows
  std::vector<unsigned int> ks;      // M odd degrees
  HermiteLayoutReport layoutReport;
  mutable CMatrix Kfu; // N x M scratch for computeKuf/Kux
};

#endif /* CINDUCING_H */
