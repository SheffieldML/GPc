/* CIP-002A Phase 3: stochastic objective interface (minibatch, not COptimisable).

   Deterministic drivers (SCG, BFGS, …) assume a full objective and break under
   minibatch noise.  Models that support seeded batches implement this thin
   contract; CAdam drives it.  Progress is a std::function hook so CIP-001E
   OptimiserEvent observers can wrap it when that CIP lands. */

#ifndef CSTOCHASTICOPTIMISABLE_H
#define CSTOCHASTICOPTIMISABLE_H

#include "CMatrix.h"
#include <functional>
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

 private:
  std::function<void(const StochasticOptimiserEvent&)> progressObserver;
};

#endif
