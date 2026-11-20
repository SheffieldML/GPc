/* CIP-001D Phase B: native FTC objective reference for python/tests.
 *
 * Mirrors python/src/wrap_gp.cpp construction on a fixed toy (N=8) and prints
 * computeObjectiveVal after the same path sets as test_gp_regression.py.
 * CIP-0027: RBF + likelihood.variance (no auto Bias/White).
 *
 * Build (from python/): make ref_ftc_objective
 * Run: ./ref_ftc_objective
 */
#include <cmath>
#include <iomanip>
#include <iostream>

#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"

using namespace std;

int main()
{
  const unsigned int N = 8;
  CMatrix X(N, 1);
  CMatrix Y(N, 1);
  for(unsigned int i = 0; i < N; ++i)
  {
    const double x = static_cast<double>(i) / static_cast<double>(N - 1);
    X.setVal(x, i, 0);
    Y.setVal(sin(2.0 * M_PI * x) + 0.1, i, 0);
  }

  CCmpndKern kern(X);
  {
    CRbfKern rbf(X);
    kern.addKern(&rbf);
  }
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);

  CGaussianNoise noise(&Y);
  for(unsigned int j = 0; j < Y.getCols(); ++j)
    noise.setBiasVal(0.0, j);

  CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
  model.setLikelihoodVariance(1.0);
  model.setByPath("rbf.lengthscale", 0.5);
  model.setByPath("rbf.variance", 1.0);
  model.setByPath("likelihood.variance", 0.05);

  cout << setprecision(17) << model.computeObjectiveVal() << endl;
  return 0;
}
