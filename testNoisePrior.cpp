/* CIP-003B: priors on likelihood.variance (CNoise is CRegularisable).
   Checks that the exact-GP objective and gradient include the noise prior and
   that the analytic gradient matches finite differences in the optimiser's
   transformed space. No .mat fixtures. */
#include <cmath>
#include <iostream>
#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "CDist.h"
#include "ndlutil.h"

using namespace std;

namespace {

const unsigned int N = 8;

int checkFiniteDifference(CGp& model, const string& label)
{
  const unsigned int P = model.getOptNumParams();
  CMatrix a(1, P);
  model.getOptParams(a);
  CMatrix g(1, P);
  g.zeros();
  model.computeObjectiveGradParams(g);
  int fail = 0;
  const double h = 1e-6;
  for(unsigned int p = 0; p < P; p++)
  {
    CMatrix ap(a), am(a);
    ap.setVal(a.getVal(p) + h, p);
    am.setVal(a.getVal(p) - h, p);
    model.setOptParams(ap);
    const double fp = model.computeObjectiveVal();
    model.setOptParams(am);
    const double fm = model.computeObjectiveVal();
    const double fd = (fp - fm) / (2 * h);
    if(fabs(fd - g.getVal(p)) > 1e-5 * (1.0 + fabs(fd)))
    {
      cout << "FAIL " << label << ": param " << p << " analytic " << g.getVal(p)
           << " finite difference " << fd << endl;
      fail++;
    }
  }
  model.setOptParams(a);
  if(fail == 0)
    cout << "PASS " << label << ": gradient matches finite differences (" << P << " params)" << endl;
  return fail;
}

} // namespace

int main()
{
  int fail = 0;
  CMatrix X(N, 1), y(N, 1);
  for(unsigned int i = 0; i < N; i++)
  {
    const double xv = -1.0 + 2.0 * (double)i / (double)(N - 1);
    X.setVal(xv, i, 0);
    y.setVal(sin(2.0 * xv), i, 0);
  }
  CCmpndKern kern(X);
  kern.addKern(new CRbfKern(X));
  CGaussianNoise noise(&y);
  for(int i = 0; i < noise.getOutputDim(); i++)
    noise.setBiasVal(0.0, i);
  CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
  model.setLikelihoodVariance(0.3);

  // Objective without a noise prior.
  const double base = model.computeObjectiveVal();
  const unsigned int P = model.getOptNumParams();
  cout << "Optimised parameters: " << P << endl;
  fail += checkFiniteDifference(model, "no noise prior");

  // Gamma prior on likelihood.variance (last noise parameter; biases come first).
  CGammaDist* prior = new CGammaDist();
  prior->setParam(2.0, 0); // shape a
  prior->setParam(3.0, 1); // rate  b
  const unsigned int varIndex = noise.getNumParams() - 1;
  noise.addPrior(prior, varIndex);
  const double withPrior = model.computeObjectiveVal();
  const double expectShift = -prior->logProb(0.3);
  if(fabs((withPrior - base) - expectShift) > 1e-9 * (1.0 + fabs(expectShift)))
  {
    cout << "FAIL objective shift " << (withPrior - base) << " expected " << expectShift << endl;
    fail++;
  }
  else
    cout << "PASS objective shifts by -log prior(variance)" << endl;
  fail += checkFiniteDifference(model, "Gamma prior on likelihood.variance");

  // Sparse approximations must refuse rather than silently drop the prior.
  cout << "Number of failures: " << fail << endl;
  return fail > 0 ? 1 : 0;
}
