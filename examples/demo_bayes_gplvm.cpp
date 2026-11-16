/* CIP-0017 Phase 2: thin Bayesian GP-LVM CLI demo on synthetic Y. */
#include <iostream>
#include <cmath>
#include "CBayesGplvm.h"

using namespace std;

int main(int argc, char** argv)
{
  unsigned int iters = 30;
  if(argc > 1)
    iters = (unsigned int)atoi(argv[1]);

  const unsigned int N = 20;
  const unsigned int D = 4;
  const unsigned int Q = 2;
  const unsigned int M = 5;

  CMatrix Y(N, D);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int d = 0; d < D; d++)
      Y.setVal(sin(0.4 * n + 0.9 * d) + 0.05 * n, n, d);

  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 1);
  model.setBeta(1.0);
  const double L0 = model.logLikelihood();
  model.optimise(iters);
  const double L1 = model.logLikelihood();

  cout << "demo_bayes_gplvm: N=" << N << " D=" << D << " q=" << Q
       << " M=" << M << " iters=" << iters
       << " ELBO0=" << L0 << " ELBO=" << L1 << endl;
  model.display(cout);
  return 0;
}
