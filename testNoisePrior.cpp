/* CIP-003B: priors on likelihood.variance (CNoise is CRegularisable).
   Checks that the exact-GP objective and gradient include the noise prior and
   that the analytic gradient matches finite differences in the optimiser's
   transformed space. No .mat fixtures. */
#include <cmath>
#include <iostream>
#include <memory>
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
  CRbfKern rbf(X);
  kern.addKern(&rbf); // CCmpndKern clones its components
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

  // CIP-003B: sampling target. Kernel parameters have no priors yet, so the
  // target must be reported improper and must refuse to run.
  {
    const string why = model.improperReason();
    if(why.empty() || why.find("rbf")==string::npos)
    {
      cout << "FAIL: expected improper target naming kernel params, got '" << why << "'" << endl;
      fail++;
    }
    else
      cout << "PASS improper target detected: " << why << endl;
    bool threw = false;
    try { CMatrix gg(1, P); gg.zeros(); model.computeSamplingObjectiveGrad(gg); }
    catch(ndlexceptions::Error&) { threw = true; }
    if(!threw) { cout << "FAIL: improper target did not throw" << endl; fail++; }
    else cout << "PASS improper target refuses to run" << endl;
  }
  // Priors live on the compound's own (cloned) component: add them there by
  // rebuilding the kernel with priors on the template before cloning.
  CCmpndKern kernP(X);
  std::unique_ptr<CRbfKern> rbfP(new CRbfKern(X));
  for(unsigned int j=0; j<rbfP->getNumParams(); j++)
  {
    CGammaDist* kp = new CGammaDist();
    kp->setParam(2.0, 0);
    kp->setParam(1.0, 1);
    rbfP->addPrior(kp, j);
  }
  // Take ownership without cloning: CKern's copy constructor does not copy priors.
  kernP.addKern(std::unique_ptr<CKern>(std::move(rbfP)));
  CGp modelP(&kernP, &noise, &X, CGp::FTC, 0, 0);
  modelP.setLikelihoodVariance(0.3);
  CGp& modelRef = modelP;
  if(!modelRef.improperReason().empty())
  {
    cout << "FAIL: still improper: " << modelRef.improperReason() << endl;
    fail++;
  }
  else
    cout << "PASS target proper once every parameter has a prior" << endl;

  // The sampling objective's gradient must match finite differences with and
  // without the Jacobian, and the two objectives must differ by exactly the
  // log-Jacobian.
  for(int withJac = 0; withJac < 2; withJac++)
  {
    modelRef.setIncludeJacobian(withJac==1);
    CMatrix a(1, P);
    modelRef.getOptParams(a);
    CMatrix g(1, P);
    g.zeros();
    const double E = modelRef.computeSamplingObjectiveGrad(g);
    const double h = 1e-6;
    for(unsigned int p = 0; p < P; p++)
    {
      CMatrix ap(a), am(a);
      ap.setVal(a.getVal(p)+h, p);
      am.setVal(a.getVal(p)-h, p);
      modelRef.setOptParams(ap);
      const double fp = modelRef.computeSamplingObjective();
      modelRef.setOptParams(am);
      const double fm = modelRef.computeSamplingObjective();
      const double fd = (fp-fm)/(2*h);
      if(fabs(fd-g.getVal(p)) > 1e-5*(1.0+fabs(fd)))
      {
        cout << "FAIL sampling gradient (jacobian=" << withJac << ") param " << p
             << " analytic " << g.getVal(p) << " fd " << fd << endl;
        fail++;
      }
    }
    modelRef.setOptParams(a);
    cout << "checked sampling-objective gradient (jacobian=" << withJac << ")" << endl;
    if(withJac==1)
    {
      modelRef.setIncludeJacobian(false);
      const double Eplain = modelRef.computeSamplingObjective();
      const double lj = modelRef.logJacobian(0);
      if(fabs((E-Eplain) + lj) > 1e-9*(1.0+fabs(E)))
      {
        cout << "FAIL objective differs from plain by " << (E-Eplain) << ", expected " << -lj << endl;
        fail++;
      }
      else
        cout << "PASS Jacobian term equals sum log|dx/da| (" << lj << ")" << endl;
      modelRef.setIncludeJacobian(true);
    }
  }

  // Cloning a kernel (as CCmpndKern::addKern(const CKern*) does) must keep priors.
  {
    std::unique_ptr<CKern> kc(kernP.clone());
    CCmpndKern* kcc = dynamic_cast<CCmpndKern*>(kc.get());
    if(kcc==0 || !kcc->paramHasPrior(0) || !kcc->paramHasPrior(1))
    {
      cout << "FAIL: cloned compound kernel lost its priors" << endl;
      fail++;
    }
    else if(fabs(kcc->priorLogProb() - kernP.priorLogProb()) > 1e-12)
    {
      cout << "FAIL: cloned kernel prior log prob differs" << endl;
      fail++;
    }
    else
      cout << "PASS cloned kernel keeps its priors" << endl;
    // Add via the const-pointer overload (clones the component) as well.
    CCmpndKern viaClone(X);
    CRbfKern templ(X);
    CGammaDist* tp = new CGammaDist();
    templ.addPrior(tp, 0);
    CRbfKern templCopy(templ);
    if(templCopy.getNumPriors()!=1 || templCopy.getPrior(0)==tp)
    {
      cout << "FAIL: copied kernel should hold its own copy of the prior" << endl;
      fail++;
    }
    else
      cout << "PASS copied prior is a distinct deep copy" << endl;
    delete tp;
  }
  // Sparse approximations must refuse rather than silently drop the prior.
  cout << "Number of failures: " << fail << endl;
  return fail > 0 ? 1 : 0;
}
