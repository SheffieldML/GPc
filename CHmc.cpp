#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include "CHmc.h"
#include "ndlutil.h"
#include "ndlexceptions.h"

CHmc::CHmc(COptimisable& m, const CPosteriorTarget& t, unsigned long seed)
  : model(m), target(t), P(m.getOptNumParams()), stepSize(0.1), numLeapfrog(10),
    mass(1, m.getOptNumParams()), state(1, m.getOptNumParams()),
    gradient(1, m.getOptNumParams()), currentE(0.0), haveState(false),
    nSteps(0), nAccepted(0), lastAlpha(0.0), lastEnergyErr(0.0)
{
  if(&target.optimisable()!=&model)
    throw ndlexceptions::Error("CHmc: model and target must be the same object.");
  target.assertProper();
  for(unsigned int i=0; i<P; i++)
    mass.setVal(1.0, i);
  ndlutil::init_genrand(seed);
}

void CHmc::setStepSize(double eps)
{
  if(!(eps>0.0))
    throw ndlexceptions::Error("CHmc: step size must be positive.");
  stepSize = eps;
}

void CHmc::setNumLeapfrog(unsigned int steps)
{
  if(steps==0)
    throw ndlexceptions::Error("CHmc: need at least one leapfrog step.");
  numLeapfrog = steps;
}

void CHmc::setMassDiag(const CMatrix& m)
{
  DIMENSIONMATCH(m.getRows()==1);
  DIMENSIONMATCH(m.getCols()==P);
  for(unsigned int i=0; i<P; i++)
    if(!(m.getVal(i)>0.0))
      throw ndlexceptions::Error("CHmc: mass entries must be positive.");
  mass.deepCopy(m);
}

// Evaluate E and its gradient at the model's current parameters. Returns
// false on a numerical failure or non-finite value so the caller can reject.
bool CHmc::evaluate(CMatrix& g, double& E)
{
  try
  {
    g.zeros();
    E = target.computeSamplingObjectiveGrad(g);
  }
  catch(ndlexceptions::Error&)
  {
    return false;
  }
  if(!std::isfinite(E))
    return false;
  for(unsigned int i=0; i<P; i++)
    if(!std::isfinite(g.getVal(i)))
      return false;
  return true;
}

bool CHmc::step()
{
  if(!haveState)
  {
    model.getOptParams(state);
    if(!evaluate(gradient, currentE))
      throw ndlexceptions::Error("CHmc: initial state has a non-finite objective or gradient.");
    haveState = true;
  }
  CMatrix a(state);
  CMatrix grad(gradient);
  CMatrix p(1, P);
  double kinetic0 = 0.0;
  for(unsigned int i=0; i<P; i++)
  {
    const double pi = sqrt(mass.getVal(i))*ndlutil::randn();
    p.setVal(pi, i);
    kinetic0 += 0.5*pi*pi/mass.getVal(i);
  }
  const double H0 = currentE + kinetic0;

  bool ok = true;
  double E = currentE;
  // Leapfrog: half momentum step, alternating full steps, final half step.
  for(unsigned int i=0; i<P; i++)
    p.setVal(p.getVal(i) - 0.5*stepSize*grad.getVal(i), i);
  for(unsigned int l=0; l<numLeapfrog && ok; l++)
  {
    for(unsigned int i=0; i<P; i++)
      a.setVal(a.getVal(i) + stepSize*p.getVal(i)/mass.getVal(i), i);
    try
    {
      model.setOptParams(a);
    }
    catch(ndlexceptions::Error&)
    {
      ok = false;
      break;
    }
    ok = evaluate(grad, E);
    if(!ok)
      break;
    const double scale = (l+1<numLeapfrog) ? stepSize : 0.5*stepSize;
    for(unsigned int i=0; i<P; i++)
      p.setVal(p.getVal(i) - scale*grad.getVal(i), i);
  }

  double alpha = 0.0;
  bool accept = false;
  if(ok)
  {
    double kinetic1 = 0.0;
    for(unsigned int i=0; i<P; i++)
      kinetic1 += 0.5*p.getVal(i)*p.getVal(i)/mass.getVal(i);
    const double H1 = E + kinetic1;
    lastEnergyErr = fabs(H1-H0);
    alpha = std::min(1.0, exp(H0-H1));
    accept = ndlutil::rand() < alpha;
  }
  else
    lastEnergyErr = std::numeric_limits<double>::infinity();
  lastAlpha = alpha;
  nSteps++;
  if(accept)
  {
    state.deepCopy(a);
    gradient.deepCopy(grad);
    currentE = E;
    nAccepted++;
  }
  else
  {
    model.setOptParams(state); // restore: the model never keeps a rejected state
  }
  return accept;
}

void CHmc::adaptStepSize(unsigned int burnin, double targetAccept)
{
  const double mu = log(10.0*stepSize);
  const double gamma = 0.05, t0 = 10.0, kappa = 0.75;
  double Hbar = 0.0, logEpsBar = 0.0;
  for(unsigned int m=1; m<=burnin; m++)
  {
    step();
    const double w = 1.0/((double)m + t0);
    Hbar = (1.0-w)*Hbar + w*(targetAccept - lastAlpha);
    const double logEps = mu - sqrt((double)m)/gamma*Hbar;
    const double eta = pow((double)m, -kappa);
    logEpsBar = eta*logEps + (1.0-eta)*logEpsBar;
    stepSize = exp(logEps);
  }
  if(burnin>0)
    stepSize = exp(logEpsBar);
  nSteps = 0;
  nAccepted = 0;
}

void CHmc::sample(unsigned int n, CMatrix& samples, CMatrix& logPost)
{
  samples.resize(n, P);
  logPost.resize(n, 1);
  for(unsigned int s=0; s<n; s++)
  {
    step();
    for(unsigned int i=0; i<P; i++)
      samples.setVal(state.getVal(i), s, i);
    logPost.setVal(-currentE, s, 0);
  }
}

double CHmc::acceptRate() const
{
  return nSteps==0 ? 0.0 : (double)nAccepted/(double)nSteps;
}

void writeSamplesCsv(const CPosteriorTarget& target, const CMatrix& samples,
                     const CMatrix& logPost, std::ostream& out, bool constrained)
{
  std::vector<SampledParam> params;
  target.sampledParams(params);
  DIMENSIONMATCH(samples.getCols()==params.size());
  DIMENSIONMATCH(logPost.getRows()==samples.getRows());
  const std::streamsize oldPrecision = out.precision();
  out << "iteration,log_posterior";
  for(size_t j=0; j<params.size(); j++)
    out << "," << params[j].name;
  out << "\n";
  out << std::setprecision(17);
  for(unsigned int i=0; i<samples.getRows(); i++)
  {
    out << i << "," << logPost.getVal(i, 0);
    for(size_t j=0; j<params.size(); j++)
    {
      double v = samples.getVal(i, (unsigned int)j);
      if(constrained && params[j].transform!=0)
        v = params[j].transform->atox(v);
      out << "," << v;
    }
    out << "\n";
  }
  out.precision(oldPrecision);
}
