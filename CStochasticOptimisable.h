/* CIP-002A Phase 3: stochastic objective interface (minibatch, not COptimisable).

   Deterministic drivers (SCG, BFGS, …) assume a full objective and break under
   minibatch noise.  Models that support seeded batches implement this thin
   contract; CAdam drives it.  Progress is a std::function hook so CIP-001E
   OptimiserEvent observers can wrap it when that CIP lands. */

#ifndef CSTOCHASTICOPTIMISABLE_H
#define CSTOCHASTICOPTIMISABLE_H

#include "CMatrix.h"
#include <functional>
#include <iostream>
#include <string>
#include <vector>

struct StochasticOptimiserEvent {
  unsigned int iteration;
  double objectiveEstimate;
  unsigned int batchSize;
  bool terminal;
};

class CStochasticOptimisable {
 public:
  virtual ~CStochasticOptimisable() {}

  virtual unsigned int numData() const = 0;
  virtual unsigned int getStochasticNumParams() const = 0;
  virtual void getStochasticParams(CMatrix& param) const = 0;
  virtual void setStochasticParams(const CMatrix& param) = 0;
  // Returns minibatch objective estimate; fills g (1 x nParams) in unconstrained space.
  virtual double objectiveGradBatch(const std::vector<unsigned int>& batch,
                                    CMatrix& g) = 0;
  // Optional hook after an Adam step (e.g. interleaved natural-gradient on q).
  virtual void afterBatchStep(const std::vector<unsigned int>& /*batch*/) {}

  void setProgressObserver(std::function<void(const StochasticOptimiserEvent&)> obs)
  {
    progressObserver = obs;
  }
  void clearProgressObserver() { progressObserver = nullptr; }
  void notifyProgress(const StochasticOptimiserEvent& ev) const
  {
    if(progressObserver)
      progressObserver(ev);
  }

  /// CIP-001E / CIP-0030: scrolling Adam progress for CLI (verbosity-gated).
  void installDefaultTextProgress(std::ostream& os, int verbosity,
                                  const std::string& optimiserName = "adam")
  {
    if(verbosity <= 0)
    {
      clearProgressObserver();
      return;
    }
    setProgressObserver(
        [&os, verbosity, optimiserName](const StochasticOptimiserEvent& ev) {
          if(verbosity <= 0)
            return;
          os << "Running " << optimiserName << " · iter " << ev.iteration
             << " · objective " << ev.objectiveEstimate
             << " · batch " << ev.batchSize << std::endl;
          if(ev.terminal)
            os << "Warning: Maximum number of iterations has been exceeded"
               << std::endl;
        });
  }

 private:
  std::function<void(const StochasticOptimiserEvent&)> progressObserver;
};

#endif
