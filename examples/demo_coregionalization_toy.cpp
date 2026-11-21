/* CIP-0015 Phase B: GPy coregionalization_toy via CCoregionalizedRegression. */
#include <cmath>
#include <iostream>
#include <vector>
#include "CMultioutput.h"
#include "ndlutil.h"

using namespace std;

namespace {

const unsigned int N0 = 50;
const unsigned int N1 = 30;
const unsigned int N_PRED = 40;
const int OPT_ITERS = 25;

void fillSinusoidOutput(CMatrix& X, CMatrix& Y, unsigned int n, double xMax,
                        double yOffset)
{
  X.resize(n, 1);
  Y.resize(n, 1);
  for(unsigned int i = 0; i < n; i++)
  {
    const double x = ndlutil::genrand_real1() * xMax;
    X.setVal(x, i, 0);
    Y.setVal(sin(x) + yOffset + 0.05 * ndlutil::randn(), i, 0);
  }
}

}  // namespace

int main()
{
  ndlutil::init_genrand(1UL);

  CMatrix X0, Y0, X1, Y1;
  fillSinusoidOutput(X0, Y0, N0, 8.0, 0.0);
  fillSinusoidOutput(X1, Y1, N1, 5.0, 2.0);

  vector<CMatrix> Xs;
  Xs.push_back(X0);
  Xs.push_back(X1);
  vector<CMatrix> Ys;
  Ys.push_back(Y0);
  Ys.push_back(Y1);

  CRbfKern spatial(1);
  spatial.setInitParam();
  spatial.setByPath("lengthscale", 1.0);
  spatial.setByPath("variance", 1.0);

  CCoregionalizedRegression coreg(Xs, Ys, spatial, /*rank=*/1, /*withIndexWhite=*/true,
                                  /*verbosity=*/1);
  CGp& model = coreg.getModel();
  cout << "demo_coregionalization_toy: N=" << coreg.getX().getRows()
       << " outputs=" << coreg.getNumOutputs()
       << " initial LL=" << model.logLikelihood() << endl;

  model.optimise(OPT_ITERS);

  cout << "objective=" << model.computeObjectiveVal() << endl;
  const vector<CRowSlice>& slices = coreg.getSlices();
  for(size_t d = 0; d < slices.size(); d++)
    cout << "  slice[" << d << "]=[" << slices[d].start << "," << slices[d].stop << ")"
         << endl;

  // Per-output predictive mean on a shared spatial grid (index stamped last).
  CMatrix Xstar(N_PRED, 2);
  for(unsigned int i = 0; i < N_PRED; i++)
  {
    const double x = 8.0 * static_cast<double>(i) / static_cast<double>(N_PRED - 1);
    Xstar.setVal(x, i, 0);
  }
  CMatrix mu(N_PRED, 1);
  CMatrix var(N_PRED, 1);
  for(unsigned int d = 0; d < coreg.getNumOutputs(); d++)
  {
    stampOutputIndex(Xstar, d);
    model.predictiveMeanVar(mu, var, Xstar);
    cout << "  predict output " << d << ": mu[0]=" << mu.getVal(0)
         << " mu[mid]=" << mu.getVal(N_PRED / 2)
         << " mu[end]=" << mu.getVal(N_PRED - 1) << endl;
  }
  return 0;
}
