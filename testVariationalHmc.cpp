/* CIP-003B: HMC over hyperparameters of variational (CBayesGplvm, CMrd) and
   PPA (CPpa) models. Checks the sampled target's gradient against finite
   differences, the prior terms against their closed form, that the variational
   parameters stay fixed, and the sampler against grid integration. Run from
   the repository root (the PPA fixture is under testdata/ppa). */
#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "CHmc.h"
#include "CVariationalTarget.h"
#include "CBayesGplvm.h"
#include "CMrd.h"
#include "CPpa.h"
#include "CKern.h"
#include "CNoise.h"
#include "CDist.h"
#include "CMatrix.h"
#include "ndlutil.h"
#include "ndlstrutil.h"

using namespace std;

namespace {

int fail = 0;

void report(bool ok, const string& what)
{
  if(ok)
    cout << "PASS " << what << endl;
  else
  {
    cout << "FAIL " << what << endl;
    fail++;
  }
}

string num(double v)
{
  ostringstream o;
  o << v;
  return o.str();
}

CGammaDist* gamma(double a, double b)
{
  CGammaDist* g = new CGammaDist();
  g->setParam(a, 0);
  g->setParam(b, 1);
  return g;
}

// Largest relative gradient error of the sampling objective (Jacobian on).
double gradientError(COptimisable& model, const CPosteriorTarget& target)
{
  const unsigned int P = model.getOptNumParams();
  CMatrix a(1, P), g(1, P);
  model.getOptParams(a);
  target.computeSamplingObjectiveGrad(g);
  double worst = 0.0;
  for(unsigned int i=0; i<P; i++)
  {
    const double a0 = a.getVal(i);
    const double h = 1e-5;
    a.setVal(a0+h, i);
    model.setOptParams(a);
    const double Ep = target.computeSamplingObjective();
    a.setVal(a0-h, i);
    model.setOptParams(a);
    const double Em = target.computeSamplingObjective();
    a.setVal(a0, i);
    model.setOptParams(a);
    const double fd = (Ep-Em)/(2.0*h);
    worst = max(worst, fabs(fd-g.getVal(i))/(1.0+fabs(fd)));
  }
  return worst;
}

CMatrix makeY(unsigned int N, unsigned int D, double phase)
{
  CMatrix Y(N, D);
  for(unsigned int n=0; n<N; n++)
    for(unsigned int d=0; d<D; d++)
      Y.setVal(sin(0.3*n+0.7*d+phase)+0.1*n, n, d);
  return Y;
}

bool sameMatrix(const CMatrix& A, const CMatrix& B)
{
  for(unsigned int i=0; i<A.getRows(); i++)
    for(unsigned int j=0; j<A.getCols(); j++)
      if(A.getVal(i, j)!=B.getVal(i, j))
        return false;
  return true;
}

CMatrix loadDat(const string& path, int rows, int cols)
{
  CMatrix A(rows, cols);
  ifstream in(path.c_str());
  if(!in)
    throw ndlexceptions::FileReadError(path);
  for(int i=0; i<rows; i++)
    for(int j=0; j<cols; j++)
    {
      double v=0.0;
      if(!(in >> v))
        throw ndlexceptions::FileFormatError(path);
      A.setVal(v, i, j);
    }
  return A;
}

}  // namespace

int run()
{
  // 1. CBayesGplvm: hyperparameters under the VarDTC bound, q(X) and Z fixed.
  {
    const unsigned int N=8, D=3, Q=2, M=3;
    CMatrix Y = makeY(N, D, 0.0);
    CRbfKern kern(Q);
    kern.setInitParam();
    kern.setInverseWidth(1.0);
    kern.setVariance(1.0);
    CBayesGplvm model(&kern, Y, Q, M, 0);
    model.setBeta(20.0);
    for(unsigned int n=0; n<N; n++)
      for(unsigned int q=0; q<Q; q++)
        model.getS().setVal(0.3, n, q);
    CGammaDist* pw = gamma(3.0, 2.0);
    CGammaDist* pv = gamma(2.0, 1.0);
    kern.addPrior(pw, 0);
    kern.addPrior(pv, 1);
    unique_ptr<CGammaDist> noisePrior(gamma(2.0, 20.0));
    auto target = makeBoundHyperTarget(model, noisePrior.get());

    const bool proper = target->improperReason().empty();
    const double err = gradientError(*target, *target);
    report(proper && err<1e-6, "BayesGplvm target: proper, gradient (priors + Jacobian) max rel error "+num(err));

    // The prior terms equal their closed forms.
    CMatrix a(1, target->getOptNumParams());
    target->getOptParams(a);
    const double withPriors = target->computeObjectiveVal();
    const double bound = model.computeObjectiveVal();
    const double expected = -(pw->logProb(kern.getParam(0)) + pv->logProb(kern.getParam(1))
                              + noisePrior->logProb(1.0/model.getBeta()));
    report(fabs((withPriors-bound)-expected)<1e-10, "BayesGplvm prior terms equal closed form");

    // Without a prior on the variance the target is flagged improper.
    auto bare = makeBoundHyperTarget(model, 0);
    report(!bare->improperReason().empty(), "BayesGplvm target without a noise prior is improper");

    // Fixing a parameter removes it from the sampled set.
    auto fixedT = makeBoundHyperTarget(model, noisePrior.get());
    fixedT->fix("likelihood.variance");
    report(fixedT->getOptNumParams()==2, "fix() removes the parameter");

    // Sampling leaves q(X) and Z untouched and the sampler accepts.
    const CMatrix mu0(model.getMu()), S0(model.getS()), Z0(model.getZ());
    CHmc hmc(*target, *target, 5);
    hmc.setStepSize(0.1);
    hmc.adaptStepSize(150);
    CMatrix samples, lp;
    hmc.sample(300, samples, lp);
    report(sameMatrix(mu0, model.getMu()) && sameMatrix(S0, model.getS()) && sameMatrix(Z0, model.getZ()),
           "BayesGplvm sampling leaves mu, S and Z unchanged");
    report(hmc.acceptRate()>0.5, "BayesGplvm sampler accept rate "+num(hmc.acceptRate()));
  }

  // 1b. Grid check: the sampled inverse width matches exp(-objective) on a grid.
  {
    const unsigned int N=8, D=3, Q=2, M=3;
    CMatrix Y = makeY(N, D, 0.0);
    CRbfKern kern(Q);
    kern.setInitParam();
    kern.setInverseWidth(1.0);
    kern.setVariance(1.0);
    CBayesGplvm model(&kern, Y, Q, M, 0);
    model.setBeta(20.0);
    for(unsigned int n=0; n<N; n++)
      for(unsigned int q=0; q<Q; q++)
        model.getS().setVal(0.3, n, q);
    kern.addPrior(gamma(3.0, 2.0), 0);
    kern.addPrior(gamma(2.0, 1.0), 1);
    unique_ptr<CGammaDist> noisePrior(gamma(2.0, 20.0));
    auto target = makeBoundHyperTarget(model, noisePrior.get());
    target->fix("variance");
    target->fix("likelihood.variance");
    CHmc hmc(*target, *target, 11);
    hmc.setStepSize(0.3);
    hmc.adaptStepSize(300);
    CMatrix samples, lp;
    const unsigned int n = 8000;
    hmc.sample(n, samples, lp);
    // The reported value for the free parameter is the stored inverse width x = exp(a).
    vector<double> xs(n);
    double mean = 0.0;
    for(unsigned int i=0; i<n; i++) { xs[i] = exp(samples.getVal(i, 0)); mean += xs[i]; }
    mean /= n;
    const unsigned int nb = 40, bl = n/nb;
    vector<double> bm(nb);
    double bmean = 0.0;
    for(unsigned int b=0; b<nb; b++) { double s=0.0; for(unsigned int i=0; i<bl; i++) s += xs[b*bl+i]; bm[b] = s/bl; bmean += bm[b]; }
    bmean /= nb;
    double bvar = 0.0;
    for(unsigned int b=0; b<nb; b++) bvar += (bm[b]-bmean)*(bm[b]-bmean);
    const double se = sqrt(bvar/(nb-1.0)/nb);
    // grid in x: density exp(-E(x)); the target's Jacobian is already accounted for by sampling a=log x.
    const int G = 4000;
    const double lo = 1e-4, hi = 40.0;
    vector<double> logp(G+1), grid(G+1);
    double mx = -1e300;
    for(int i=0; i<=G; i++)
    {
      grid[i] = lo + (hi-lo)*i/G;
      CMatrix a(1, 1);
      a.setVal(log(grid[i]), 0);
      target->setOptParams(a);
      logp[i] = -target->computeObjectiveVal();
      mx = max(mx, logp[i]);
    }
    double Z = 0.0, m1 = 0.0;
    for(int i=0; i<=G; i++)
    {
      const double w = (i==0 || i==G) ? 0.5 : 1.0;
      const double p = exp(logp[i]-mx);
      Z += w*p;
      m1 += w*p*grid[i];
    }
    const double gridMean = m1/Z;
    report(fabs(mean-gridMean) < 4.0*se+1e-3,
           "BayesGplvm inverse width mean "+num(mean)+" vs grid "+num(gridMean)+" (se "+num(se)+")");
  }

  // 2. CMrd: per-view kernels and precisions, shared q(X) and Z fixed.
  {
    const unsigned int N=8, Q=3, M=3;
    vector<CMatrix> Ys;
    Ys.push_back(makeY(N, 2, 0.0));
    Ys.push_back(makeY(N, 3, 1.1));
    CRbfKern* k0 = new CRbfKern(Q);
    k0->setInitParam();
    CRbfKern* k1 = new CRbfKern(Q);
    k1->setInitParam();
    vector<CKern*> ks;
    ks.push_back(k0);
    ks.push_back(k1);
    CMrd model(Ys, ks, Q, M, 0);
    vector<unique_ptr<CGammaDist> > keep;
    for(unsigned int v=0; v<2; v++)
      for(unsigned int i=0; i<2; i++)
      {
        keep.push_back(unique_ptr<CGammaDist>(gamma(2.0, 1.0)));
        ks[v]->addPrior(keep.back().get(), i);
      }
    unique_ptr<CGammaDist> noisePrior(gamma(2.0, 5.0));
    auto target = makeBoundHyperTarget(model, noisePrior.get());
    vector<string> names = target->names();
    const bool named = names.size()==6 && names[0]=="view0.lengthscale" && names[2]=="view0.likelihood.variance"
      && names[3]=="view1.lengthscale";
    const double err = gradientError(*target, *target);
    const CMatrix mu0(model.getMu()), Z0(model.getZ());
    CHmc hmc(*target, *target, 3);
    hmc.setStepSize(0.05);
    hmc.adaptStepSize(100);
    CMatrix samples, lp;
    hmc.sample(100, samples, lp);
    report(target->improperReason().empty() && named && err<1e-6
           && sameMatrix(mu0, model.getMu()) && sameMatrix(Z0, model.getZ()) && hmc.acceptRate()>0.4,
           "Mrd target: proper, named per view, gradient max rel error "+num(err)+", accept "+num(hmc.acceptRate()));
  }

  // 3. CPpa: the M-step objective at fixed sites.
  {
    const string dir = "testdata" + ndlstrutil::dirSep() + "ppa" + ndlstrutil::dirSep();
    const int N = 30, D = 2;
    CMatrix X = loadDat(dir + "phase0_X.txt", N, D);
    CMatrix y = loadDat(dir + "phase0_y.txt", N, 1);
    CCmpndKern kern(X);
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CBiasKern(X));
    kern.addKern(new CWhiteKern(X));
    CProbitNoise noise(&y);
    noise.initParams();
    CPpa model(X, y, kern, noise, 0);
    model.optimise(3, 0, 0);   // E-steps only: sites move, kernel fixed

    bool refused = false;
    try { vector<SampledParam> sp; model.sampledParams(sp); } catch(const ndlexceptions::Error&) { refused = true; }
    report(refused, "CPpa refuses to sample until setKernelObjective(true)");

    model.setKernelObjective(true);
    const bool improperBefore = !model.improperReason().empty();
    for(unsigned int i=0; i<kern.getNumParams(); i++)
      kern.addPriorAt(gamma(2.0, 1.0), i);
    const bool properAfter = model.improperReason().empty();
    const double err = gradientError(model, model);
    // The sampled objective is the kernel M-step objective, prior included.
    const double E = model.computeObjectiveVal();
    const double direct = -model.kernelLogLikelihood();
    CHmc hmc(model, model, 9);
    hmc.setStepSize(0.05);
    hmc.adaptStepSize(100);
    CMatrix samples, lp;
    hmc.sample(200, samples, lp);
    report(improperBefore && properAfter && err<1e-6 && fabs(E-direct)<1e-12 && hmc.acceptRate()>0.4,
           "CPpa kernel objective: improper without priors, gradient max rel error "+num(err)+", accept "+num(hmc.acceptRate()));
  }

  cout << "Number of failures: " << fail << endl;
  return fail;
}

int main()
{
  try
  {
    return run();
  }
  catch(const ndlexceptions::Error& e)
  {
    cout << "EXCEPTION: " << e.getMessage() << endl;
    return 1;
  }
}
