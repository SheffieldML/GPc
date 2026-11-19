#include "CAdam.h"
#include "ndlexceptions.h"
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

CAdam::CAdam(double learningRate, double b1, double b2, double epsilon)
  : alpha(learningRate), beta1(b1), beta2(b2), eps(epsilon), lrDecay(1.0),
    batchSize(32), maxIters(100), seed(0)
{
  if(!(alpha > 0.0) || !(beta1 >= 0.0 && beta1 < 1.0) || !(beta2 >= 0.0 && beta2 < 1.0)
     || !(eps > 0.0))
    throw ndlexceptions::Error("CAdam: invalid hyperparameters");
}

double CAdam::maximise(CStochasticOptimisable& model)
{
  const unsigned int N = model.numData();
  const unsigned int P = model.getStochasticNumParams();
  if(N == 0 || P == 0)
    throw ndlexceptions::Error("CAdam: empty model");
  const unsigned int B = min(batchSize, N);

  CMatrix param(1, P);
  model.getStochasticParams(param);
  CMatrix m(1, P);
  CMatrix v(1, P);
  m.zeros();
  v.zeros();
  CMatrix g(1, P);

  vector<unsigned int> order(N);
  for(unsigned int i = 0; i < N; i++)
    order[i] = i;
  mt19937_64 rng(seed);
  unsigned int cursor = N;  // force reshuffle
  double lastObj = 0.0;
  double lr = alpha;

  for(unsigned int t = 1; t <= maxIters; t++)
  {
    if(cursor + B > N)
    {
      shuffle(order.begin(), order.end(), rng);
      cursor = 0;
    }
    vector<unsigned int> batch(order.begin() + cursor, order.begin() + cursor + B);
    cursor += B;

    lastObj = model.objectiveGradBatch(batch, g);
    // Maximise: ascend.
    const double bc1 = 1.0 - pow(beta1, (double)t);
    const double bc2 = 1.0 - pow(beta2, (double)t);
    for(unsigned int i = 0; i < P; i++)
    {
      const double gi = g.getVal(i);
      const double mi = beta1 * m.getVal(i) + (1.0 - beta1) * gi;
      const double vi = beta2 * v.getVal(i) + (1.0 - beta2) * gi * gi;
      m.setVal(mi, i);
      v.setVal(vi, i);
      const double mhat = mi / bc1;
      const double vhat = vi / bc2;
      param.setVal(param.getVal(i) + lr * mhat / (sqrt(vhat) + eps), i);
    }
    model.setStochasticParams(param);
    model.afterBatchStep(batch);

    StochasticOptimiserEvent ev;
    ev.iteration = t;
    ev.objectiveEstimate = lastObj;
    ev.batchSize = B;
    ev.terminal = (t == maxIters);
    model.notifyProgress(ev);

    lr *= lrDecay;
  }
  return lastObj;
}
