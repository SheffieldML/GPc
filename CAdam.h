/* CIP-002A Phase 3: Adam on seeded minibatches for CStochasticOptimisable. */

#ifndef CADAM_H
#define CADAM_H

#include "CMatrix.h"
#include "CStochasticOptimisable.h"
#include <cstdint>
#include <vector>

class CAdam {
 public:
  CAdam(double learningRate = 1e-2, double beta1 = 0.9, double beta2 = 0.999,
        double epsilon = 1e-8);

  void setLearningRate(double lr) { alpha = lr; }
  void setBatchSize(unsigned int b) { batchSize = b; }
  void setMaxIters(unsigned int n) { maxIters = n; }
  void setSeed(uint64_t s) { seed = s; }
  void setLearningRateSchedule(double decay) { lrDecay = decay; }

  // Maximise objectiveEstimate (Adam on +g). Returns final estimate.
  // Moment estimates and the step count persist across calls on the same
  // CAdam object (CIP-003C), so chunked runs equal one long run. They are
  // discarded automatically if the parameter count changes; call
  // resetState() to discard them explicitly. A freshly constructed CAdam
  // behaves exactly as before.
  double maximise(CStochasticOptimisable& model);

  void resetState();
  unsigned long getStepCount() const { return stepCount; }

 private:
  double alpha;
  double beta1;
  double beta2;
  double eps;
  double lrDecay;
  unsigned int batchSize;
  unsigned int maxIters;
  uint64_t seed;

  CMatrix mState;
  CMatrix vState;
  unsigned int stateParams;
  unsigned long stepCount;
};

#endif
