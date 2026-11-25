/* HMC over hyperparameters of variational (bound-based) models — CIP-003B.

   CBayesGplvm and CMrd optimise a VarDTC bound over kernel parameters, the
   noise precision and the variational parameters (q(X), Z). The sampled target
   here is the bound as a function of the hyperparameters only, with q(X) and Z
   held at their current values (normally the optimised ones). That is not a
   posterior over q; it is the generalised posterior exp(bound) x prior, and it
   inherits the bound's bias. The bound is deterministic, so HMC is valid.
   Kernels carry their own priors; the noise precision is described by a prior
   on the variance 1/beta that this class owns.

   Stochastic ELBOs (CSvgp, CSviBayesGplvm, CDeepGp) are not supported: their
   mini-batch estimates are noisy, which breaks HMC's accept step. */
#ifndef CVARIATIONALTARGET_H
#define CVARIATIONALTARGET_H

#include <memory>
#include <string>
#include <vector>
#include "COptimisable.h"
#include "CPosteriorTarget.h"
#include "CDist.h"
#include "CKern.h"

class CBayesGplvm;
class CMrd;

class CBoundHyperTarget : public COptimisable, public CPosteriorTarget
{
 public:
  explicit CBoundHyperTarget(COptimisable& base);

  // Kernel parameters live at base optimiser indices offset .. offset+n-1
  // (transformed space). Names are prefix + the kernel's sample names.
  void addKernel(CKern& kern, unsigned int offset, const std::string& prefix);
  // The noise precision beta lives at base index `index` as log(beta). It is
  // sampled as the variance 1/beta, named prefix + "likelihood.variance".
  // `prior` (cloned; may be null) is a prior on that variance.
  void addPrecision(unsigned int index, const std::string& prefix, const CDist* prior);
  // Hold a sampled parameter fixed, by reported name.
  void fix(const std::string& name);
  // Names of the parameters currently sampled.
  std::vector<std::string> names() const;

  // CPosteriorTarget
  void sampledParams(std::vector<SampledParam>& params) const;
  const COptimisable& optimisable() const { return *this; }

  // COptimisable over the free entries only.
  unsigned int getOptNumParams() const;
  void getOptParams(CMatrix& param) const;
  void setOptParams(const CMatrix& param);
  double computeObjectiveGradParams(CMatrix& g) const;
  double computeObjectiveVal() const;

 private:
  struct Entry
  {
    unsigned int baseIndex;
    CKern* kern;              // null for the precision entry
    unsigned int kernIndex;
    unsigned int kernEntry;   // which kernel record owns this entry
    std::string prefix;
    bool free;
  };
  struct KernRecord
  {
    CKern* kern;
    unsigned int offset;
  };
  struct PrecisionRecord
  {
    unsigned int baseIndex;
    std::string prefix;
    std::unique_ptr<CDist> prior;
  };
  void freeIndices(std::vector<unsigned int>& idx) const;
  double priorTerms(const CMatrix& full, CMatrix* gFull) const;

  COptimisable& base;
  std::vector<Entry> entries;
  std::vector<KernRecord> kernels;
  std::vector<PrecisionRecord> precisions;
};

// Convenience constructors matching each model's parameter packing.
// CBayesGplvm packs [kern..., log beta, Z, mu, S]; CMrd packs per view
// [kern..., log beta] then the shared Z, mu, S.
std::unique_ptr<CBoundHyperTarget> makeBoundHyperTarget(CBayesGplvm& model, const CDist* variancePrior=0);
std::unique_ptr<CBoundHyperTarget> makeBoundHyperTarget(CMrd& model, const CDist* variancePrior=0);

#endif
