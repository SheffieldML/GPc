/* CIP-003B: Hamiltonian Monte Carlo over the parameters of a COptimisable
   model, in the unconstrained space the optimiser already uses.

   The sampler needs only setOptParams / getOptParams on the model and the
   negative log posterior and gradient from CPosteriorTarget (which adds the
   change-of-variables term and refuses improper targets). Each leapfrog step
   costs one objective-and-gradient evaluation, the same as an SCG step; no
   allocations beyond the parameter vectors happen in the trajectory loop.

   Randomness comes from the shared ndlutil Mersenne Twister, which is global:
   constructing a CHmc seeds it. */
#ifndef CHMC_H
#define CHMC_H

#include <iostream>
#include <vector>
#include "CMatrix.h"
#include "COptimisable.h"
#include "CPosteriorTarget.h"

class CHmc
{
 public:
  // model and target must refer to the same object (e.g. a CGp).
  CHmc(COptimisable& model, const CPosteriorTarget& target, unsigned long seed=1);

  void setStepSize(double eps);
  double getStepSize() const { return stepSize; }
  void setNumLeapfrog(unsigned int steps);
  unsigned int getNumLeapfrog() const { return numLeapfrog; }
  // Diagonal mass matrix (1 x P, strictly positive). Default: identity.
  void setMassDiag(const CMatrix& mass);

  // One transition. Returns true if the proposal was accepted. On rejection
  // (or a failed numerical evaluation) the model is restored to its previous
  // parameters.
  bool step();
  // Dual-averaging step-size adaptation (Hoffman & Gelman 2014) over `burnin`
  // transitions; the chain moves, so these draws should be discarded. Fixes
  // the step size afterwards.
  void adaptStepSize(unsigned int burnin, double targetAccept=0.8);
  // Draw n states. samples is n x P (optimiser space), logPost is n x 1 and
  // holds the log density of the sampled target at each state.
  void sample(unsigned int n, CMatrix& samples, CMatrix& logPost);

  double acceptRate() const;
  double lastAcceptProb() const { return lastAlpha; }
  double lastLogPost() const { return -currentE; }
  // Largest |H_end - H_start| seen in the last trajectory (energy error).
  double lastEnergyError() const { return lastEnergyErr; }

 private:
  bool evaluate(CMatrix& g, double& E);
  COptimisable& model;
  const CPosteriorTarget& target;
  unsigned int P;
  double stepSize;
  unsigned int numLeapfrog;
  CMatrix mass;
  CMatrix state;    // current parameters, 1 x P
  CMatrix gradient; // gradient of E at state
  double currentE;
  bool haveState;
  unsigned long nSteps;
  unsigned long nAccepted;
  double lastAlpha;
  double lastEnergyErr;
};

// CIP-003B: write samples as dense CSV (readable by pandas / NumPy / arviz).
// Header: iteration,log_posterior,<parameter names>. Rows hold one draw each.
// log_posterior is the log density of the sampled target (it includes the
// log-Jacobian when that is switched on). If constrained is true the
// parameter columns are the model's own (constrained) values, e.g. positive
// length scales; otherwise they are the unconstrained optimiser values.
void writeSamplesCsv(const CPosteriorTarget& target, const CMatrix& samples,
                     const CMatrix& logPost, std::ostream& out,
                     bool constrained=true);

// CIP-003B: build a prior from a text spec shared by the CLI and the Python
// façade: "gamma:shape,rate" or "gaussian:precision". The caller owns the
// returned distribution unless it hands it to a model (models keep the pointer).
class CDist;
CDist* priorFromSpec(const std::string& spec);

#endif
