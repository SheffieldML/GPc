/* CIP-003B: what a sampler needs to know about a model beyond
   COptimisable's objective and gradient.

   COptimisable exposes the negative log posterior E(a) and its gradient in the
   unconstrained space a that the optimiser works in. Sampling in a needs two
   extra things:

   1. The change-of-variables term. A prior placed on the constrained value x
      = atox(a) has density p_x(x(a)) |dx/da| in a, so the target is
      -E(a) + sum log|dx/da|. GPy does not add this term; we include it by
      default and offer setIncludeJacobian(false) to reproduce GPy samples.
   2. A properness check. Every sampled parameter needs either a prior or a
      bounded transform; otherwise the posterior can be improper and a chain
      would wander without converging. GPy does not check this.
*/
#ifndef CPOSTERIORTARGET_H
#define CPOSTERIORTARGET_H

#include <string>
#include <vector>
#include "CMatrix.h"
#include "CTransform.h"
#include "COptimisable.h"
#include "ndlexceptions.h"

struct SampledParam
{
  std::string name;
  const CTransform* transform; // null if sampled directly
  bool hasPrior;
  double constrainedValue;     // x = atox(a)
};

class CPosteriorTarget
{
 public:
  CPosteriorTarget() : includeJacobian(true), allowImproper(false) {}
  virtual ~CPosteriorTarget() {}

  // One entry per optimiser parameter, in getOptParams order. Models that
  // cannot describe every parameter must throw NotImplementedError.
  virtual void sampledParams(std::vector<SampledParam>& params) const=0;
  virtual const COptimisable& optimisable() const=0;

  void setIncludeJacobian(bool val) { includeJacobian = val; }
  bool getIncludeJacobian() const { return includeJacobian; }
  // Explicit override for users who accept an improper target.
  void setAllowImproper(bool val) { allowImproper = val; }

  // Empty if every sampled parameter has a prior or a bounded transform;
  // otherwise a message naming the offending parameters.
  std::string improperReason() const
  {
    std::vector<SampledParam> params;
    sampledParams(params);
    std::string bad;
    for(size_t i=0; i<params.size(); i++)
    {
      const bool bounded = params[i].transform!=0 && params[i].transform->isBounded();
      if(!params[i].hasPrior && !bounded)
        bad += (bad.empty() ? "" : ", ") + params[i].name;
    }
    if(bad.empty())
      return "";
    return "improper posterior: no prior and no bounded transform on " + bad;
  }

  // Throws unless the target is proper (or improper targets were allowed).
  void assertProper() const
  {
    if(allowImproper)
      return;
    const std::string why = improperReason();
    if(!why.empty())
      throw ndlexceptions::Error(why + " (add a prior, or setAllowImproper(true))");
  }

  // log sum |dx/da| over sampled parameters; if g is non-null, adds the
  // gradient d/da of that sum (log-density convention) to g.
  double logJacobian(CMatrix* g=0) const
  {
    std::vector<SampledParam> params;
    sampledParams(params);
    if(g!=0)
    {
      DIMENSIONMATCH(g->getRows()==1);
      DIMENSIONMATCH(g->getCols()==params.size());
    }
    double lj = 0.0;
    for(size_t i=0; i<params.size(); i++)
    {
      if(params[i].transform==0)
        continue;
      lj += params[i].transform->logJacobian(params[i].constrainedValue);
      if(g!=0)
        g->addVal(params[i].transform->gradLogJacobian(params[i].constrainedValue), i);
    }
    return lj;
  }

  // Negative log density of the sampled target and its gradient in a-space:
  //   E(a) - [log Jacobian], with the Jacobian omitted if switched off.
  double computeSamplingObjectiveGrad(CMatrix& g) const
  {
    assertProper();
    double E = optimisable().computeObjectiveGradParams(g);
    if(includeJacobian)
    {
      CMatrix gj(g.getRows(), g.getCols());
      gj.zeros();
      E -= logJacobian(&gj);
      g.axpy(gj, -1.0);
    }
    return E;
  }
  double computeSamplingObjective() const
  {
    assertProper();
    double E = optimisable().computeObjectiveVal();
    if(includeJacobian)
      E -= logJacobian(0);
    return E;
  }

 private:
  bool includeJacobian;
  bool allowImproper;
};

#endif
