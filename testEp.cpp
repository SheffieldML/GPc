/* CIP-0033 Phase A: Bernoulli–probit ep_moments (GPy mirror) + dense EP smoke. */
#include "CEpInference.h"
#include "CKern.h"
#include "CNoise.h"
#include "ndlutil.h"
#include <cmath>
#include <iostream>

using namespace std;

static int checkClose(const string& label, double a, double b, double tol = 1e-10)
{
  const double diff = fabs(a - b);
  if(diff <= tol)
  {
    cout << label << " passes." << endl;
    return 0;
  }
  cout << "FAILURE: " << label << ": |a-b|=" << diff
       << " (a=" << a << ", b=" << b << ")" << endl;
  return 1;
}

static int testMomentsMatchGPyFormula()
{
  // Hand evaluation of GPy Bernoulli.moments_match_ep for y=1, tau=2, v=1.
  const double tau = 2.0;
  const double v = 1.0;
  const double sign = 1.0;
  const double den = sqrt(tau * tau + tau);
  const double z = sign * v / den;
  const double phiOverPhi = ndlutil::gradLnCumGaussian(z);
  const double Zref = exp(ndlutil::lnCumGaussian(z));
  const double muRef = v / tau + sign * phiOverPhi / den;
  const double s2Ref = 1.0 / tau
                       - (phiOverPhi / (tau * tau + tau)) * (z + phiOverPhi);

  double Zhat = 0.0, muHat = 0.0, s2Hat = 0.0;
  ndlnoise::bernoulliProbitEpMoments(Zhat, muHat, s2Hat, 1.0, tau, v);

  int fail = 0;
  fail += checkClose("ep_moments Zhat", Zhat, Zref);
  fail += checkClose("ep_moments muHat", muHat, muRef);
  fail += checkClose("ep_moments sigma2Hat", s2Hat, s2Ref);
  fail += checkClose("ep_moments sigma2Hat>0", s2Hat > 0.0 ? 1.0 : 0.0, 1.0);

  // y=0 should match y=-1.
  double Z0, m0, s0, Zm, mm, sm;
  ndlnoise::bernoulliProbitEpMoments(Z0, m0, s0, 0.0, tau, v);
  ndlnoise::bernoulliProbitEpMoments(Zm, mm, sm, -1.0, tau, v);
  fail += checkClose("y=0 vs y=-1 Z", Z0, Zm);
  fail += checkClose("y=0 vs y=-1 mu", m0, mm);
  fail += checkClose("y=0 vs y=-1 s2", s0, sm);
  return fail;
}

static int testEpDenseSmoke()
{
  // Two points, RBF-like Gram, separable labels — EP should converge, sites τ>0.
  const unsigned int n = 2;
  CMatrix X(n, 1);
  X.setVal(-1.0, 0, 0);
  X.setVal(1.0, 1, 0);
  CMatrix Y(n, 1);
  Y.setVal(1.0, 0, 0);
  Y.setVal(0.0, 1, 0);

  CRbfKern kern(X);
  kern.setVariance(1.0);
  kern.setLengthScale(1.0);
  CMatrix K(n, n);
  kern.compute(K, X);
  for(unsigned int i = 0; i < n; i++)
    K.setVal(K.getVal(i, i) + 1e-6, i, i);

  CEpInference ep;
  ep.setMaxIters(50);
  ep.setTol(1e-8);
  CEpSites sites;
  CMatrix mu, varDiag;
  const double logZ = ep.run(K, Y, sites, mu, varDiag);

  int fail = 0;
  fail += checkClose("EP converged", ep.converged() ? 1.0 : 0.0, 1.0, 0.0);
  fail += checkClose("EP iters>0", ep.lastIters() > 0 ? 1.0 : 0.0, 1.0, 0.0);
  fail += checkClose("logZ finite", std::isfinite(logZ) ? 1.0 : 0.0, 1.0, 0.0);
  for(unsigned int i = 0; i < n; i++)
  {
    fail += checkClose("site tau>0", sites.tau.getVal(i, 0) > 0.0 ? 1.0 : 0.0, 1.0, 0.0);
    fail += checkClose("varDiag>0", varDiag.getVal(i, 0) > 0.0 ? 1.0 : 0.0, 1.0, 0.0);
  }
  // Means should pull toward class signs (y0=+1 → mu0>0, y1=0 → mu1<0) on this toy.
  fail += checkClose("mu0 sign", mu.getVal(0, 0) > 0.0 ? 1.0 : 0.0, 1.0, 0.0);
  fail += checkClose("mu1 sign", mu.getVal(1, 0) < 0.0 ? 1.0 : 0.0, 1.0, 0.0);
  cout << "  EP iters=" << ep.lastIters()
       << " tau=[" << sites.tau.getVal(0, 0) << "," << sites.tau.getVal(1, 0) << "]"
       << " mu=[" << mu.getVal(0, 0) << "," << mu.getVal(1, 0) << "]"
       << " logZ=" << logZ << endl;
  return fail;
}

int main()
{
  int fail = 0;
  cout << "==== testEp CIP-0033 (GPy-mirrored EP) ====" << endl;
  fail += testMomentsMatchGPyFormula();
  fail += testEpDenseSmoke();
  cout << "Total failures: " << fail << endl;
  return fail;
}
