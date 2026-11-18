#include "COptimisable.h"
#include <cstdio>
#include <sstream>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define GPC_ISATTY _isatty
#define GPC_FILENO _fileno
#else
#include <unistd.h>
#define GPC_ISATTY isatty
#define GPC_FILENO fileno
#endif

const double COptimisable::phi=1.618033988749895;
const double COptimisable::cphi=-0.6180339887498949;
const double COptimisable::smallNum=1e-11;

OptimiserEvent COptimisable::makeOptimiserEvent(OptimiserPhase phase,
                                                int optimiserId,
                                                double objectiveVal,
                                                double stepOrScale,
                                                const std::string& message) const
{
  OptimiserEvent e;
  e.phase = phase;
  e.optimiser = optimiserId;
  e.iter = static_cast<int>(iter);
  e.funcEval = static_cast<int>(funcEval);
  e.objective = objectiveVal;
  e.stepOrScale = stepOrScale;
  e.message = message;
  return e;
}

const char* COptimisable::optimiserName(int optimiserId)
{
  switch(optimiserId)
  {
  case CG: return "cg";
  case SCG: return "scg";
  case GD: return "gd";
  case BFGS: return "bfgs";
  case LBFGS: return "lbfgs";
  default: return "optimiser";
  }
}

bool COptimisable::shouldUseInPlaceProgress() const
{
  if(inPlaceProgressMode == 0)
    return false;
  if(inPlaceProgressMode > 0)
    return true;
  if(!progressStream)
    return false;
  if(progressStream != &std::cout && progressStream != &std::cerr)
    return false;
  FILE* fp = (progressStream == &std::cerr) ? stderr : stdout;
#ifdef _WIN32
  HANDLE h = GetStdHandle((progressStream == &std::cerr) ? STD_ERROR_HANDLE
                                                         : STD_OUTPUT_HANDLE);
  if(h != INVALID_HANDLE_VALUE)
  {
    DWORD mode = 0;
    if(GetConsoleMode(h, &mode))
      SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
  }
#endif
  return GPC_ISATTY(GPC_FILENO(fp)) != 0;
}

std::string COptimisable::buildProgressBlock(const OptimiserEvent& event) const
{
  std::ostringstream status;
  status << "Running " << optimiserName(event.optimiser)
         << " · iter " << event.iter;
  if(std::isfinite(event.objective))
    status << " · objective " << event.objective;
  if(std::isfinite(event.stepOrScale))
    status << " · scale " << event.stepOrScale;
  status << '\n';

  std::string block = status.str();
  const std::string snap = progressSnapshot(event);
  if(!snap.empty())
    block += snap;
  if(!snap.empty() && snap[snap.size() - 1] != '\n')
    block += '\n';

  switch(event.phase)
  {
  case OptimiserPhase::MaxIters:
    block += "Warning: Maximum number of iterations has been exceeded\n";
    break;
  case OptimiserPhase::Converged:
    if(!event.message.empty())
      block += event.message + "\n";
    else
      block += "Optimization converged.\n";
    break;
  case OptimiserPhase::Failed:
    if(!event.message.empty())
      block += event.message + "\n";
    else
      block += "Optimization failed.\n";
    break;
  default:
    break;
  }
  return block;
}

void COptimisable::writeOptimiserLogLine(const OptimiserEvent& event) const
{
  if(!logStream)
    return;
  std::ostream& out = *logStream;
  const char* phase = "Iter";
  switch(event.phase)
  {
  case OptimiserPhase::Start: phase = "Start"; break;
  case OptimiserPhase::Iter: phase = "Iter"; break;
  case OptimiserPhase::Converged: phase = "Converged"; break;
  case OptimiserPhase::MaxIters: phase = "MaxIters"; break;
  case OptimiserPhase::Failed: phase = "Failed"; break;
  }
  out << phase << '\t' << optimiserName(event.optimiser)
      << '\t' << event.iter
      << '\t' << event.objective;
  if(std::isfinite(event.stepOrScale))
    out << '\t' << event.stepOrScale;
  if(!event.message.empty())
    out << '\t' << event.message;
  out << endl;
}

void COptimisable::writeDefaultOptimiserText(const OptimiserEvent& event) const
{
  writeOptimiserLogLine(event);

  if(!progressStream || getVerbosity() <= 0)
    return;

  std::ostream& out = *progressStream;
  const bool inPlace = shouldUseInPlaceProgress();
  const bool terminal =
      event.phase == OptimiserPhase::Converged
      || event.phase == OptimiserPhase::MaxIters
      || event.phase == OptimiserPhase::Failed;

  // Non-TTY (pipes, logs, test sinks): avoid Iteration spam — Start + terminal only.
  if(!inPlace && event.phase == OptimiserPhase::Iter)
    return;

  const std::string block = buildProgressBlock(event);
  if(inPlace)
  {
    if(progressLinesPrinted > 0)
      out << "\033[" << progressLinesPrinted << "A\r";
    out << block;
    out << "\033[J" << std::flush;
    progressLinesPrinted = 0;
    for(size_t i = 0; i < block.size(); ++i)
    {
      if(block[i] == '\n')
        ++progressLinesPrinted;
    }
    if(terminal)
    {
      out << '\n' << std::flush;
      progressLinesPrinted = 0;
    }
  }
  else
  {
    out << block << std::flush;
  }
}

void COptimisable::notifyOptimiserEvent(const OptimiserEvent& event) const
{
  if(defaultTextProgress)
    writeDefaultOptimiserText(event);
  else
    writeOptimiserLogLine(event);
  for(size_t i = 0; i < optimiserObservers.size(); ++i)
    optimiserObservers[i](event);
}

void COptimisable::checkGradients()
{
  double change = ndlutil::GRADCHANGE;
  double origParam = 0.0;
  double objectivePlus = 0.0;
  double objectiveMinus = 0.0;
  int nParams = getOptNumParams();
  
  CMatrix analyticGrad(1, nParams);
  CMatrix numericalDiff(1, nParams);
  CMatrix diffNumericalAnalytic(1, nParams);
  CMatrix params(1, nParams);
  CMatrix origParams(1, nParams);
  
  getOptParams(params);
  origParams.deepCopy(params);
  computeObjectiveGradParams(analyticGrad);
  for(int j=0; j<nParams; j++) 
  {
    origParam = origParams.getVal(j);
    params.setVal(origParam + change, j);
    setOptParams(params);
    objectivePlus = computeObjectiveVal();
    params.setVal(origParam - change, j);
    setOptParams(params);
    objectiveMinus = computeObjectiveVal();
    numericalDiff.setVal(0.5*(objectivePlus - objectiveMinus)/change, j);
    params.setVal(origParams.getVal(j), j);
  }
  
  cout << "Numerical differences:" << endl << numericalDiff << endl;
  cout << "Analytic gradients:" << endl << analyticGrad << endl;
  diffNumericalAnalytic.deepCopy(analyticGrad);
  diffNumericalAnalytic-=numericalDiff;
  cout << "Differences: " << endl << diffNumericalAnalytic << endl;
}

void COptimisable::gdOptimise()
{
  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Start, GD, 0.0,
                                          std::numeric_limits<double>::quiet_NaN(),
                                          "Gradient Descent Optimisation."));
  int nParams = getOptNumParams();
  double objectiveVal = 0.0;
  double oldObjective = 0.0;
  double diffObjective = 0.0;
  double diffParam = 0.0;
  CMatrix params(1, nParams);
  CMatrix oldParams(1, nParams);
  CMatrix gradParams(1, nParams);
  CMatrix changeParams(1, nParams);
  changeParams.zeros();
  getOptParams(params);
  if(evalFunc)
    objectiveVal = computeObjectiveVal();
  for(iter=0; iter<getMaxIters(); iter++)     
  {
    oldParams.deepCopy(params);
    computeObjectiveGradParams(gradParams);
    if(momentum>0)
	{
	  changeParams.axpy(gradParams, -learnRate/momentum);
	  params.axpy(changeParams, momentum);
	  changeParams.scale(momentum);
	}
      else
	{
	  params.axpy(gradParams, -learnRate);	    
	}
      setOptParams(params);
      if(evalFunc)
	{
	  oldObjective = objectiveVal;
	  objectiveVal = computeObjectiveVal();
	  diffObjective = fabs(objectiveVal-oldObjective);
	}
      notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Iter, GD, objectiveVal));
      diffParam = params.maxAbsDiff(oldParams);
      if(diffObjective<objectiveTol && diffParam<parameterTol)
	{
	  std::ostringstream msg;
	  msg << "Param difference: " << diffParam << "\n"
	      << "Objective difference: " << diffObjective << "\n"
	      << "Converged ..";
	  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Converged, GD, objectiveVal,
	                                          std::numeric_limits<double>::quiet_NaN(),
	                                          msg.str()));
	  break;
	}
      
    } 
  if(getVerbosity()>2 && progressStream)
  {
    *progressStream << "Parameters: " << endl;
    *progressStream << params << endl;
  }
}
void COptimisable::gdPullbackOptimise()
{
  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Start, GD, 0.0,
                                          std::numeric_limits<double>::quiet_NaN(),
                                          "Gradient Descent with pullback Optimisation."));

  int nParams = getOptNumParams();
  double objectiveVal = 0.0;
  double oldObjective = 0.0;
  double diffObjective = 0.0;
  double diffParam = 0.0;
  CMatrix params(1, nParams);
  CMatrix oldParams(1, nParams);
  CMatrix gradParams(1, nParams);
  CMatrix changeParams(1, nParams);
  changeParams.zeros();
  getOptParams(params);
  objectiveVal = computeObjectiveVal();
  for(iter=0; iter<getMaxIters(); iter++)     
  {
    while(1)
    {
      oldParams.deepCopy(params);
      computeObjectiveGradParams(gradParams);
      params.axpy(gradParams, -learnRate);	    
      setOptParams(params);
      oldObjective = objectiveVal;
      objectiveVal = computeObjectiveVal();
      diffObjective = oldObjective-objectiveVal;
      if(diffObjective<0)
      {
	params.deepCopy(oldParams);
	learnRate = learnRate/2;
	objectiveVal=oldObjective;
	setOptParams(params);
	if(getVerbosity()>2 && progressStream)
	  *progressStream << "Back tracking, learning rate: " << learnRate << endl;
      }
      else
      {
	learnRate = learnRate*1.1;
	break;
      }
    }
    notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Iter, GD, objectiveVal));
    diffParam = params.maxAbsDiff(oldParams);
    if(diffObjective<objectiveTol && diffParam<parameterTol)
    {
      std::ostringstream msg;
      msg << "Param difference: " << diffParam << "\n"
          << "Objective difference: " << diffObjective << "\n"
          << "Converged ..";
      notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Converged, GD, objectiveVal,
                                              std::numeric_limits<double>::quiet_NaN(),
                                              msg.str()));
      break;
    }
    
  } 
  if(getVerbosity()>2 && progressStream)
  {
    *progressStream << "Parameters: " << endl;
    *progressStream << params << endl;
  }
}
double COptimisable::oneDObjectiveVal(const double val)
{
  DIMENSIONMATCH(paramStoreOne.getRows()==1);
  DIMENSIONMATCH(paramStoreTwo.getRows()==1);
  DIMENSIONMATCH(paramStoreOne.getCols()==getOptNumParams());
  DIMENSIONMATCH(paramStoreTwo.getCols()==getOptNumParams());
  getOptParams(paramStoreOne);
  paramStoreTwo.deepCopy(paramStoreOne);
  // add val times direction to the parameters.
  paramStoreTwo.axpy(direction, val);
  setOptParams(paramStoreTwo);
  double objective = computeObjectiveVal();
  setOptParams(paramStoreOne);
  return objective;
}
void COptimisable::lbfgsOptimise()
{
  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Start, BFGS, 0.0,
                                          std::numeric_limits<double>::quiet_NaN(),
                                          "Limited Memory BFGS Optimisation."));
  int nParams = getOptNumParams();
  int iflag = 0;
  int memSize = 10;
  // vector<> frees automatically on return or throw
  vector<double> Xvals(nParams);
  vector<double> work(nParams*(2*memSize+1) + 2*memSize);
  vector<double> gvals(nParams);
  vector<double> diagVals(nParams);
  
  CMatrix X(1, nParams);
  CMatrix g(1, nParams);
  int iPrint[2] ={-1, 0};
  if(getVerbosity()>2)
  {
    iPrint[0] = 1;
  }
  double f = 0.0;
  getOptParams(X);
  // Nocedal LBFGS (ndlfortran.f) has no budget args. GPy/paramz opt_lbfgsb passes
  // max_iters as both SciPy maxiter and maxfun; mirror that in this C++ loop.
  // See docs/optimisers.md.
  iter = 0;
  funcEval = 0;
  const unsigned int budget = getMaxIters();
  while(true)
  {
    f = computeObjectiveGradParams(g);
    funcEval++;
    X.toArray(&Xvals[0]);
    g.toArray(&gvals[0]);
    lbfgs_(nParams, memSize, &Xvals[0], f, &gvals[0], 0, &diagVals[0], iPrint, getObjectiveTol(), getParamTol(), &work[0], iflag);
    if(iflag<0)
    {
      if(iflag==-1)
      {
	notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Failed, BFGS, f,
	                                        std::numeric_limits<double>::quiet_NaN(),
	                                        "Warning: lbfgsOptimise: linesearch failed."));
	break;
      }
      else if(iflag == -2)
      {
	throw ndlexceptions::Error("An element of the inverse Hessian provided is not positive.");
      }
      else if(iflag == -3)
      {
	throw ndlexceptions::Error("Inproper input to lbfgs_.");
      }
      else
      {
	throw ndlexceptions::Error("Unhandled negative iflag from lbfgs_.");
      }
    }
    else if(iflag==0)
    {
      X.fromArray(&Xvals[0]);
      setOptParams(X);
      notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Converged, BFGS, f));
      break;
    }
    else if(iflag==1)
    {
      X.fromArray(&Xvals[0]);
      setOptParams(X);
      iter++;
      notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Iter, BFGS, f));
      // GPy-aligned budget: stop on algorithm steps or function evaluations.
      if(iter >= budget || funcEval >= budget)
      {
	std::string msg;
	if(iter >= budget)
	  msg = "Warning: Maximum number of iterations has been exceeded";
	if(funcEval >= budget)
	{
	  if(!msg.empty()) msg += "\n";
	  msg += "Warning: Maximum number of function evaluations has been exceeded";
	}
	notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::MaxIters, BFGS, f,
	                                        std::numeric_limits<double>::quiet_NaN(),
	                                        msg));
	break;
      }
    }
    else
    {
      throw ndlexceptions::Error("Unhandled iflag.");
    }
  }
}
void COptimisable::scgOptimise()
{
  // taken from the paper by Martin Moller: "A scaled conjugate gradient algorithm for fast supervised learning".
  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Start, SCG, 0.0,
                                          std::numeric_limits<double>::quiet_NaN(),
                                          "Scaled Conjugate Gradient Optimisation."));
  int nParams = getOptNumParams();
  //double objectiveVal = 0.0;
  //double oldObjective = 0.0;
  //double diffObjective = 0.0;
  //double diffParam = 0.0;
  
  bool success = true;
  double beta;
  double Delta;
  double lambda;
  double lambdaBar;
  double sigma;
  //double rr;

  CMatrix w(1, nParams);
  CMatrix wPlus(1, nParams);

  CMatrix r(1, nParams);
  CMatrix p(1, nParams);
  CMatrix rp(1, nParams);
  CMatrix s(1, nParams);
  getOptParams(w);

  const double m_step = 1.0e-4;
  const double m_reg = 1.0;
  double oldObj = 0.0;
  //double obj = oldObj;
  double mu = 0.0;
  //double theta = 0.0;
  double sigmaInv = 0.0;
  double delta = 0.0;
  double alpha = 0.0;
  double newObj = 0.0;
  //double gamma = 0.0;
  //int j = 1;					// j counts number of iterations.
  
  // 1
  lambda = m_reg; // lambda is the scale??
  lambdaBar = 0.0;
  oldObj = computeObjectiveGradParams(r);
  r.negate();
  p.deepCopy(r);

  for(iter=1; iter<=getMaxIters(); iter++)
  {
    // should check that p is not infinite above
    double normp = p.normRow(0);
    double normp2 = normp*normp;
    // 2
    if(success)
	{
	  // can get a divide by zero here if pp is too small.
	  
	  sigma = m_step/normp;
	  wPlus.deepCopy(w);
	  wPlus.axpy(p, sigma);
	  setOptParams(wPlus);
	  computeObjectiveGradParams(s);
	  sigmaInv = 1/sigma;
	  s.scale(sigmaInv);
	  s.axpy(r, sigmaInv);
	  delta = s.dotRowRow(0, p, 0);
	}
      
    // 3 Scale s_iter
    double lambdaDiff = lambda-lambdaBar;
    s.axpy(p, lambdaDiff);
    delta += lambdaDiff*normp;
      
    // 4 
    if(delta <= 0.0) // Make Hessian positive definite.
	{
	  double deltaOverNormp2 = delta/normp2;
	  s.axpy(p, (lambda-2.0*deltaOverNormp2));
	  
	  lambdaBar = 2.0*(lambda - deltaOverNormp2);
	  delta = lambda*normp2 - delta; 
	  lambda = lambdaBar;
	}
      
    // 5  Calculate step size.
    mu=p.dotRowRow(0, r, 0);
    alpha = mu/delta;
      
    // 6 Compute the comparison parameter.
    wPlus.deepCopy(w);
    wPlus.axpy(p, alpha);
    setOptParams(wPlus);
    newObj = computeObjectiveVal();
    Delta = 2.0*delta*(oldObj - newObj)/(mu*mu);

    // 7 Check whether a successful error reduction can be made.
    if(Delta >= 0.0)  // update is successful
      {
	w.deepCopy(wPlus); 	  
	oldObj = newObj;
	computeObjectiveGradParams(rp);
	rp.negate();
	lambdaBar = 0; 
	success = true;
	  // 7.a Check for algorithm restart.
	  if(iter % nParams == 0) // restart algorithm
	    p.deepCopy(rp); 
	  else
	    {	      
	      double rpnorm2 = rp.norm2Row(0);
	      double rrp = r.dotRowRow(0, rp, 0); 
	      
	      beta = (rpnorm2 - rrp)/mu;
	      p.scale(beta); 
	      p.axpy(rp, 1.0);
	    }
	  
	  r.deepCopy(rp);
	  
	  // 7.b Reduce the scale parameter
	  if(Delta >= 0.75) lambda *= 0.5;
	  if(lambda<1e-15) lambda = 1e-15;
	}
    else // no reduction in error is possible.
      {
	setOptParams(w);
	lambdaBar = lambda; 
	success = false; 
      }
    
    // 8 Increase the scale parameter
    if(Delta < 0.25) lambda *= 4.0;
    
    // 9 Check for convergence       
    notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Iter, SCG, oldObj, lambda));
    if (success && fabs(p.max()*alpha) < getParamTol() && max(fabs(newObj-oldObj)) < getObjectiveTol())
      {
	std::ostringstream msg;
	msg << "Convergence criterion for parameters and objective met\n"
	    << "Largest tolerance " << fabs(newObj - oldObj);
	notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Converged, SCG, oldObj, lambda, msg.str()));
	return;
      }
  }
  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::MaxIters, SCG, oldObj, lambda,
                                          "Warning: Maximum number of iterations has been exceeded"));
}
void COptimisable::cgOptimise()
{
  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Start, CG, 0.0,
                                          std::numeric_limits<double>::quiet_NaN(),
                                          "Conjugate Gradient Optimisation."));
  
  // a C++ translation of Carl Rasmussen's minimize function
  int nParams = getOptNumParams();
  bool success = true;
  
  const double INT = 0.1; // don't reevaluate within 0.1 of the limit of the current bracket.
  const double EXT = 3.0; // extrapolate maximum 3.0 times the current step-size.
  const unsigned int MAX = 20; // maximum 20 function evaluations per line search
  const double RATIO = 10.0; // maximum allowed slope ratio.
  const double SIG = 0.1; 
  const double RHO = SIG/2.0; // SIG and RHO are the constants controlling the Wolfe-Powell conditions. 
  
  
  CMatrix df0(1, nParams); // gradient direction.
  CMatrix dF0(1, nParams); // gradient direction.
  CMatrix s(1, nParams); // search direction.
  CMatrix X(1, nParams); // parameter vector.
  CMatrix X0(1, nParams); // parameter vector.
  CMatrix wPlus(1, nParams); // parameter vector.
  
  double red = 1.0; 
  iter = 0;  // start with zero iterations.
  funcEval = 0; // start with zero function evaluations.
  bool ls_failed = false;  // no previous line search has failed.
  
  // compute initial gradient and function value.
  double f0 = computeObjectiveGradParams(df0);
  funcEval++;  // add to functional computation tally.
  s.deepCopy(df0);
  s.negate(); // initial search direction (steepest descent)
  double d0 = -s.norm2Row(0);
  
  getOptParams(X);
  double F0 = 0.0;
  double x1 = 0.0;
  double x2 = 0.0;
  double x3 = red/(1-d0); // initial step size is red/(|s|+1)
  double x4 = 0.0;
  double f1 = 0.0;
  double f2 = 0.0;
  double f3 = 0.0;
  double f4 = 0.0;
  double d1 = 0.0;
  double d2 = 0.0;
  double d3 = 0.0;
  double d4 = 0.0;
  double A = 0.0;
  double B = 0.0;
  vector<double> fX;
  
  CMatrix df3(1, nParams);
  int M = 0;	 
  while((isIterTerminate() 
	 && iter<getMaxIters())  
	|| (isFuncEvalTerminate() 
	    && funcEval<getMaxFuncEvals()))
  {
    iter++; //update number if iterations
    // make a copy of current values
    X0.deepCopy(X);
    F0 = f0;
    dF0.deepCopy(df0);
    if(MAX <= getMaxFuncEvals() || isFuncEvalTerminate())
    {
      M = MAX;
    }
    else 
    {
      M = getMaxFuncEvals();
    }
    while(true) // keep doing line search until break.
    {
      x2 = 0.0;
      f2 = f0;
      d2 = d0;
      f3 = f0;
      df3.deepCopy(df0);
      success = false;
      while(!success && M>0)
      {
	try
	{
	  M--;
	  funcEval++;
	  // compute gradient
	  wPlus.deepCopy(X);
	  wPlus.axpy(s, x3);
	  setOptParams(wPlus);  
	  f3 = computeObjectiveGradParams(df3);
	  if(!(isnan(f3) || isinf(f3) || df3.isAnyNan() || df3.isAnyInf()))
	  {
	    // clean computation of gradients, success!
	    success = true;
	  }
	  else
	  {
	    if(getVerbosity()>1)
	      cout << "cgOptimise: Warning gradient or function value was NaN or inf." << endl;
	  }
	}
	catch(ndlexceptions::MatrixNonPosDef err)
	{
	  if(getVerbosity()>1)
	    cout << "cgOptimise: Matrix non-positive definite in gradient of function value computation." << endl;
	}
	catch(ndlexceptions::MatrixConditionError err)
	{
	  if(getVerbosity()>1)
	    cout << "cgOptimise: Matrix conditioning error in gradient of function value computation." << endl;
	}
	catch(ndlexceptions::MatrixSingular err)
	{
	  if(getVerbosity()>1)
	    cout << "cgOptimise: Matrix singularity error in gradient of function value computation." << endl;
	}
	if(!success) // if any of these errors have occured, pull back and retry.
	{
	  notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Failed, CG, f0,
	                                          std::numeric_limits<double>::quiet_NaN(),
	                                          "Pulling back by half."));
	  x3 = (x2 + x3)/2; 
	}
      }
      if(f3<F0) // keep best values.
      {
	X0.deepCopy(X);
	X0.axpy(s, x3);
	F0 = f3;
	dF0.deepCopy(df3);
      }
      d3 = df3.dotRowRow(0, s, 0);
      if(d3>SIG*d0 || f3>f0+x3*RHO*d0 || M==0) // Is line search over?
      {
	break;
      }
      x1 = x2; f1 = f2; d1 = d2;   // move point 2 to point 1.
      x2 = x3; f2 = f3; d2 = d3;   // move point 3 to point 2.
      A = 6.0*(f1-f2)+3.0*(d2+d1)*(x2-x1); // do cubic extrapolation.
      B = 3.0*(f2-f1)-(2.0*d1+d2)*(x2-x1);
      x3 = x1-d1*((x2-x1)*(x2-x1))/(B + sqrt(B*B-A*d1*(x2-x1)));  // new extrapolation point.
      // checks on new extropolation point.
      if(isnan(x3) || isinf(x3) || x3 < 0.0)
	x3=x2*EXT;   // extrapolate maximum ammount.
      else if(x3>x2*EXT) // new point beyond extrapolation limit?
	x3=x2*EXT;   // extrapolate maximum ammount.
      else if(x3<x2+INT*(x2-x1))  // new point too close to previous point?
	x3=x2+INT*(x2-x1);  // extrapolate minimum amount
    }
    while((abs(d3)>-SIG*d0 || f3>f0+x3*RHO*d0) && M>0)  // keep interpolating.
    {
      if(d3>0 || f3>f0+x3*RHO*d0) // choose subinterval.
      {
	x4=x3; f4=f3; d4=d3;   // move point 3 to point 4.
      }
      else
      {
	x2=x3; f2=f3; d2=d3;  // move point 3 to point 2.
      }
      if(f4>f0)
      {
	x3 = x2-(0.5*d2*((x4-x2)*(x4-x2)))/(f4-f2-d2*(x4-x2));  // quadratic interpolation.
      }
      else
      {
	A = 6.0*(f2-f4)/(x4-x2)+3*(d4+d2);   // cubic interpolation.
	B = 3.0*(f4-f2)-(2*d2+d4)*(x4-x2);
	x3 = x2+(sqrt(B*B-A*d2*(x4-x2)*(x4-x2))-B)/A;
      }
      if(isnan(x3) || isinf(x3))
	x3 = (x2+x4)/2.0;                // bisect if there was a numerical problem.
      x3 = max(min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2)); // don't accept too close.
      // add s times x3 to wPlus.
      wPlus.deepCopy(X);
      wPlus.axpy(s, x3);
      setOptParams(wPlus);  
      f3 = computeObjectiveGradParams(df3);
      if(f3<F0) // keep best values.
      {
	F0 = f3;
	dF0.deepCopy(df3);
	X0.deepCopy(wPlus);
      }
      funcEval++;
      M--;
      d3 = df3.dotRowRow(0, s, 0);  // new slope.
    } // end of interpolation.
    if (abs(d3)<-SIG*d0 && f3 < f0+x3*RHO*d0)  // if line search succeeded.
    {
      X.deepCopy(wPlus);
      f0 = f3;
      fX.push_back(f0);
      notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::Iter, CG, f0));
      // Polack-Ribiere CG direction.
      s.scale((df3.norm2Row(0) - df0.dotRowRow(0, df3, 0))/df0.norm2Row(0));
      s.axpy(df3, -1.0); 
      df0.deepCopy(df3); // swap derivatives
      d3=d0; d0=df0.dotRowRow(0, s, 0);
      if(d0>0)
      {
        // not negative --- use steepest descent.	
	s.deepCopy(df0);
	s.negate(); 
	d0 = -s.norm2Row(0);
      }
      x3 = x3* min(RATIO, d3/(d0-__DBL_MIN__));
      ls_failed = false;
    }
    else
    {
      // revert the line search failed, restore best point.
      X.deepCopy(X0);
      f0=F0;
      df0.deepCopy(dF0);
      if(ls_failed || (isIterTerminate() && iter>=getMaxIters()) || (isFuncEvalTerminate() && funcEval>=getMaxFuncEvals()))
      { 
	// line search failed twice in a row, or iterations are exceeded.
	//Need a tolerance check here!!.
	break;
      }
      // restart from steepest descent direction.
      s.deepCopy(df0);
      s.negate();
      d0 = -s.norm2Row(0);
      x3 = 1/(1-d0);
      ls_failed = true;
    }
  }
  if(isIterTerminate() && iter >= getMaxIters())
  {
    notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::MaxIters, CG, f0,
                                            std::numeric_limits<double>::quiet_NaN(),
                                            "cgOptimise: Warning: Maximum number of iterations has been exceeded"));
  }
  if(isFuncEvalTerminate() && funcEval >= getMaxFuncEvals())
  {
    notifyOptimiserEvent(makeOptimiserEvent(OptimiserPhase::MaxIters, CG, f0,
                                            std::numeric_limits<double>::quiet_NaN(),
                                            "cgOptimise: Warning: Maximum number of function evalutaions has been exceeded"));
  }
  
}
