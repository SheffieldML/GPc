/* Dedicated tests for COptimisable drivers (CIP-0010).
   Toy objective: f(x) = 0.5 ||x - target||^2, minimiser at target. */
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include "COptimisable.h"
#include "ndlutil.h"

using namespace std;

namespace {
const double GRADTOL = 1e-6;
const double OPTTOL = 1e-4;
const unsigned int NDIM = 3;

class CQuadOptimisable : public COptimisable
{
public:
  CQuadOptimisable()
  {
    target.resize(1, NDIM);
    target.setVal(1.0, 0);
    target.setVal(-2.0, 1);
    target.setVal(0.5, 2);
    params.resize(1, NDIM);
    params.zeros();
  }

  unsigned int getOptNumParams() const
  {
    return NDIM;
  }
  void getOptParams(CMatrix& param) const
  {
    param.deepCopy(params);
  }
  void setOptParams(const CMatrix& param)
  {
    params.deepCopy(param);
  }
  double computeObjectiveVal() const
  {
    double f = 0.0;
    for(unsigned int i = 0; i < NDIM; i++)
    {
      double d = params.getVal(i) - target.getVal(i);
      f += 0.5 * d * d;
    }
    return f;
  }
  double computeObjectiveGradParams(CMatrix& g) const
  {
    g.resize(1, NDIM);
    for(unsigned int i = 0; i < NDIM; i++)
      g.setVal(params.getVal(i) - target.getVal(i), i);
    return computeObjectiveVal();
  }

  void setParamsTo(double v0, double v1, double v2)
  {
    params.setVal(v0, 0);
    params.setVal(v1, 1);
    params.setVal(v2, 2);
  }
  const CMatrix& getTarget() const { return target; }
  const CMatrix& getParams() const { return params; }

private:
  CMatrix target;
  CMatrix params;
};

int failNear(const CMatrix& got, const CMatrix& want, double tol, const string& label)
{
  if(got.maxAbsDiff(want) > tol)
  {
    cout << "FAILURE: " << label << " (maxAbsDiff=" << got.maxAbsDiff(want) << ")." << endl;
    return 1;
  }
  cout << label << " matches." << endl;
  return 0;
}

int testGradients()
{
  int fail = 0;
  CQuadOptimisable model;
  model.setVerbosity(0);
  model.setParamsTo(0.2, -0.3, 1.1);

  CMatrix analytic(1, NDIM);
  model.computeObjectiveGradParams(analytic);

  CMatrix numerical(1, NDIM);
  CMatrix p(1, NDIM);
  model.getOptParams(p);
  const double h = ndlutil::GRADCHANGE;
  for(unsigned int j = 0; j < NDIM; j++)
  {
    double orig = p.getVal(j);
    p.setVal(orig + h, j);
    model.setOptParams(p);
    double fp = model.computeObjectiveVal();
    p.setVal(orig - h, j);
    model.setOptParams(p);
    double fm = model.computeObjectiveVal();
    numerical.setVal(0.5 * (fp - fm) / h, j);
    p.setVal(orig, j);
  }
  model.setOptParams(p);

  fail += failNear(analytic, numerical, GRADTOL, "finite-difference gradients");

  // Exercise COptimisable::checkGradients for coverage (prints; no fail count).
  // It does not restore setOptParams after FD probes; reset afterwards.
  model.setParamsTo(0.2, -0.3, 1.1);
  model.checkGradients();
  model.setParamsTo(0.2, -0.3, 1.1);
  return fail;
}

int runDriver(CQuadOptimisable& model, const string& name, void (COptimisable::*method)())
{
  model.setParamsTo(0.0, 0.0, 0.0);
  model.setVerbosity(0);
  model.setMaxIters(500);
  model.setObjectiveTol(1e-12);
  model.setParamTol(1e-12);
  if(name == "graddesc")
  {
    model.setLearnRate(0.25);
    model.setMomentum(0.0);
  }
  double f0 = model.computeObjectiveVal();
  (model.*method)();
  double f1 = model.computeObjectiveVal();
  int fail = 0;
  if(!(f1 < f0 - 1e-8))
  {
    cout << "FAILURE: " << name << " did not decrease objective (" << f0 << " -> " << f1 << ")." << endl;
    fail++;
  }
  else
    cout << name << " decreased objective." << endl;
  fail += failNear(model.getParams(), model.getTarget(), OPTTOL, name + " near minimiser");
  return fail;
}

int testLbfgsHonoursMaxIters()
{
  int fail = 0;
  CQuadOptimisable model;
  model.setParamsTo(0.0, 0.0, 0.0);
  model.setVerbosity(0);
  model.setMaxIters(3);
  model.setObjectiveTol(1e-30);
  model.setParamTol(1e-30);
  model.lbfgsOptimise();
  // GPy-aligned: budget caps reverse-comm steps and f-evals; with tiny
  // tolerances the Fortran driver would otherwise continue.
  if(model.getIter() > 3 || model.getFuncEval() > 3)
  {
    cout << "FAILURE: lbfgs exceeded budget (iter=" << model.getIter()
         << " funcEval=" << model.getFuncEval() << ")." << endl;
    fail++;
  }
  else if(model.getIter() < 1 && model.getFuncEval() < 1)
  {
    cout << "FAILURE: lbfgs did not run under budget." << endl;
    fail++;
  }
  else
    cout << "lbfgs honours maxIters as iter/f-eval budget (GPy-aligned)." << endl;
  return fail;
}

int testDrivers()
{
  int fail = 0;
  CQuadOptimisable model;
  fail += runDriver(model, "scg", &COptimisable::scgOptimise);
  fail += runDriver(model, "conjgrad", &COptimisable::cgOptimise);
  fail += runDriver(model, "graddesc", &COptimisable::gdOptimise);
  fail += runDriver(model, "lbfgs", &COptimisable::lbfgsOptimise);
  fail += testLbfgsHonoursMaxIters();
  return fail;
}

int testOptimiserEvents()
{
  int fail = 0;
  CQuadOptimisable model;
  model.setParamsTo(0.0, 0.0, 0.0);
  model.setVerbosity(0);
  model.setDefaultTextProgress(false);
  model.setMaxIters(50);
  model.setObjectiveTol(1e-12);
  model.setParamTol(1e-12);

  vector<OptimiserEvent> events;
  model.addOptimiserObserver([&events](const OptimiserEvent& e) {
    events.push_back(e);
  });
  model.scgOptimise();

  bool sawStart = false;
  bool sawIter = false;
  bool sawTerminal = false;
  int lastIter = -1;
  for(size_t i = 0; i < events.size(); ++i)
  {
    const OptimiserEvent& e = events[i];
    if(e.phase == OptimiserPhase::Start)
      sawStart = true;
    if(e.phase == OptimiserPhase::Iter)
    {
      sawIter = true;
      if(e.iter < lastIter)
      {
        cout << "FAILURE: optimiser event iter went backwards." << endl;
        fail++;
      }
      lastIter = e.iter;
      if(!std::isfinite(e.objective))
      {
        cout << "FAILURE: optimiser Iter event has non-finite objective." << endl;
        fail++;
      }
      if(e.optimiser != COptimisable::SCG)
      {
        cout << "FAILURE: SCG event has wrong optimiser id." << endl;
        fail++;
      }
    }
    if(e.phase == OptimiserPhase::Converged || e.phase == OptimiserPhase::MaxIters)
      sawTerminal = true;
  }
  if(!sawStart)
  {
    cout << "FAILURE: no Start optimiser event." << endl;
    fail++;
  }
  else
    cout << "optimiser Start event observed." << endl;
  if(!sawIter)
  {
    cout << "FAILURE: no Iter optimiser event." << endl;
    fail++;
  }
  else
    cout << "optimiser Iter events observed (" << events.size() << " total)." << endl;
  if(!sawTerminal)
  {
    cout << "FAILURE: no terminal optimiser event." << endl;
    fail++;
  }
  else
    cout << "optimiser terminal event observed." << endl;

  // Silent default text at verbosity 0.
  model.clearOptimiserObservers();
  model.setDefaultTextProgress(true);
  model.setVerbosity(0);
  model.setParamsTo(0.0, 0.0, 0.0);
  std::ostringstream sink;
  model.setOptimiserProgressStream(&sink);
  model.scgOptimise();
  if(!sink.str().empty())
  {
    cout << "FAILURE: verbosity 0 still wrote progress text." << endl;
    fail++;
  }
  else
    cout << "verbosity 0 produces no default progress text." << endl;
  model.setOptimiserProgressStream(&cout);

  // Non-TTY sink: no Iteration: spam; optional log stream records events.
  {
    CQuadOptimisable m2;
    m2.setParamsTo(0.0, 0.0, 0.0);
    m2.setVerbosity(2);
    m2.setDefaultTextProgress(true);
    m2.setInPlaceProgress(0);
    m2.setMaxIters(20);
    m2.setObjectiveTol(1e-12);
    m2.setParamTol(1e-12);
    std::ostringstream progress;
    std::ostringstream log;
    m2.setOptimiserProgressStream(&progress);
    m2.setOptimiserLogStream(&log);
    m2.scgOptimise();
    const string prog = progress.str();
    const string logText = log.str();
    if(prog.find("Iteration:") != string::npos)
    {
      cout << "FAILURE: non-TTY progress still used Iteration: lines." << endl;
      fail++;
    }
    else
      cout << "non-TTY progress has no Iteration: lines." << endl;
    if(prog.find("Running scg") == string::npos)
    {
      cout << "FAILURE: expected Running scg in progress sink." << endl;
      fail++;
    }
    if(logText.find("Start") == string::npos || logText.find("Iter") == string::npos)
    {
      cout << "FAILURE: optimiser log stream missing Start/Iter lines." << endl;
      fail++;
    }
    else
      cout << "optimiser log stream records events." << endl;
    m2.setOptimiserProgressStream(&cout);
    m2.setOptimiserLogStream(0);
  }
  return fail;
}

int testDefaultOptimiserStrings()
{
  int fail = 0;
  CQuadOptimisable model;
  model.setVerbosity(0);
  model.setMaxIters(200);
  model.setObjectiveTol(1e-12);
  model.setParamTol(1e-12);
  model.setLearnRate(0.25);
  model.setMomentum(0.0);

  const char* names[] = {"scg", "conjgrad", "graddesc", "quasinew", "lbfgs"};
  for(unsigned int i = 0; i < sizeof(names) / sizeof(names[0]); i++)
  {
    string name = names[i];
    model.setParamsTo(0.0, 0.0, 0.0);
    try
    {
      model.setDefaultOptimiserStr(name);
    }
    catch(...)
    {
      cout << "FAILURE: setDefaultOptimiserStr(\"" << name << "\") threw." << endl;
      fail++;
      continue;
    }
    string canonical = model.getDefaultOptimiserStr();
    if(name == "lbfgs")
    {
      if(canonical != "quasinew")
      {
        cout << "FAILURE: lbfgs synonym should report quasinew, got " << canonical << endl;
        fail++;
      }
      else
        cout << "lbfgs synonym maps to quasinew." << endl;
    }
    else if(canonical != name)
    {
      cout << "FAILURE: getDefaultOptimiserStr after " << name << " gave " << canonical << endl;
      fail++;
    }
    else
      cout << "optimiser string " << name << " round-trips." << endl;

    double f0 = model.computeObjectiveVal();
    model.runDefaultOptimiser();
    double f1 = model.computeObjectiveVal();
    if(!(f1 < f0 - 1e-8))
    {
      cout << "FAILURE: runDefaultOptimiser(" << name << ") did not decrease objective." << endl;
      fail++;
    }
    else
      cout << "runDefaultOptimiser(" << name << ") decreased objective." << endl;
  }

  try
  {
    model.setDefaultOptimiserStr("not-an-optimiser");
    cout << "FAILURE: unknown optimiser string did not throw." << endl;
    fail++;
  }
  catch(ndlexceptions::NotImplementedError&)
  {
    cout << "unknown optimiser string throws." << endl;
  }
  catch(...)
  {
    cout << "FAILURE: unknown optimiser string threw unexpected type." << endl;
    fail++;
  }
  return fail;
}
} // namespace

int main()
{
  int fail = 0;
  try
  {
    fail += testGradients();
    fail += testDrivers();
    fail += testOptimiserEvents();
    fail += testDefaultOptimiserStrings();
    cout << endl << "Total number of failures: " << fail << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
  catch(std::exception& err)
  {
    cerr << "Unhandled exception: " << err.what() << endl;
    return 1;
  }
}
