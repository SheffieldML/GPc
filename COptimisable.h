/*This file contains a base class for optimisation. In particular it contains an implementation of the Scaled Conjugate Gradient optimisation routine.

05/11/2005 Changed use of abs (which was calling f2c code) to fabs to improve compatability with MSVC.

21/10/2005 More minor changes from Bill. 

20/10/2005 Minor changes by William V. Baxter for MSVC compatibility. This principally removing an unused variable and changing the file in which the constants were declared. */
#ifndef COPTIMISABLE_H
#define COPTIMISABLE_H
#include "CMatrix.h"
#include "ndlutil.h"
#include "ndlfortran.h"
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

/// Outer-iteration / terminal progress (CIP-001E). Not fired from objective/gradient.
enum class OptimiserPhase
{
  Start,
  Iter,
  Converged,
  MaxIters,
  Failed
};

struct OptimiserEvent
{
  OptimiserPhase phase;
  int optimiser;       // COptimisable::CG / SCG / GD / BFGS / LBFGS
  int iter;
  int funcEval;        // -1 if not tracked
  double objective;
  double gradNorm;     // NaN if unused
  double stepOrScale;  // e.g. SCG lambda; NaN if unused
  std::string message;

  OptimiserEvent()
    : phase(OptimiserPhase::Iter),
      optimiser(-1),
      iter(0),
      funcEval(-1),
      objective(0.0),
      gradNorm(std::numeric_limits<double>::quiet_NaN()),
      stepOrScale(std::numeric_limits<double>::quiet_NaN()),
      message()
  {
  }
};

typedef std::function<void(const OptimiserEvent&)> OptimiserObserver;

// abstract base class for making a class optimisable.
class COptimisable {
  
 public:
  enum 
  {
    CG, 
    SCG, 
    GD, 
    BFGS,
    LBFGS
  };
  COptimisable()
  {
    // set default optimisation parameters
    setVerbosity(2);
    setDefaultOptimiser(SCG);
    setFuncEvalTerminate(false);
    setIterTerminate(true);
    setMaxFuncEvals(1000);
    setMaxIters(1000);
    setObjectiveTol(1e-6);
    setParamTol(1e-6);
    setLearnRate(0.01);
    setMomentum(0.9);
    iter = 0;
    funcEval = 0;
    defaultTextProgress = true;
    progressStream = &std::cout;
    logStream = 0;
    progressLinesPrinted = 0;
    inPlaceProgressMode = -1; // -1 auto (TTY), 0 off, 1 on
  }
  virtual ~COptimisable() {}
  virtual inline void setVerbosity(int val) const 
  {
    verbosity = val;
  }  
  virtual inline int getVerbosity() const 
  {
    return verbosity;
  }

  /// Register an outer-iteration observer (CIP-001E). Cleared with clearOptimiserObservers().
  void addOptimiserObserver(const OptimiserObserver& obs)
  {
    optimiserObservers.push_back(obs);
  }
  void clearOptimiserObservers()
  {
    optimiserObservers.clear();
  }
  /// When true (default), verbosity-gated text is written to the progress stream.
  void setDefaultTextProgress(bool on)
  {
    defaultTextProgress = on;
  }
  bool isDefaultTextProgress() const
  {
    return defaultTextProgress;
  }
  void setOptimiserProgressStream(std::ostream* os)
  {
    progressStream = os;
    progressLinesPrinted = 0;
  }
  std::ostream* getOptimiserProgressStream() const
  {
    return progressStream;
  }
  /// Optional append-only record of events (independent of live TTY redraw).
  void setOptimiserLogStream(std::ostream* os)
  {
    logStream = os;
  }
  std::ostream* getOptimiserLogStream() const
  {
    return logStream;
  }
  /// Force in-place ANSI redraw: -1 auto (TTY), 0 off, 1 on.
  void setInPlaceProgress(int mode) const
  {
    inPlaceProgressMode = mode;
    progressLinesPrinted = 0;
  }

  /// Optional multi-line model snapshot for live progress (CGp overrides).
  virtual std::string progressSnapshot(const OptimiserEvent& /*event*/) const
  {
    return std::string();
  }

  /// Fire default text (if enabled) then all registered observers.
  void notifyOptimiserEvent(const OptimiserEvent& event) const;
  virtual unsigned int getOptNumParams() const=0;
  virtual void getOptParams(CMatrix& param) const=0;
  virtual void setOptParams(const CMatrix& param)=0;
  virtual double computeObjectiveGradParams(CMatrix& g) const=0;
  virtual double computeObjectiveVal() const=0;

  inline void setDirection(const CMatrix& vals)
  {
    DIMENSIONMATCH(vals.getCols()==getOptNumParams());
    DIMENSIONMATCH(vals.getRows()==1);
    direction.deepCopy(vals);
  }
  inline void getDirection(CMatrix& vals) const
  {
    DIMENSIONMATCH(vals.getCols()==getOptNumParams());
    DIMENSIONMATCH(vals.getRows()==1);
    DIMENSIONMATCH(direction.dimensionsMatch(vals));
    vals.deepCopy(direction);
  }
  void checkGradients();  
  void gdOptimise();
  void gdPullbackOptimise();
  //void netlabScgOptimise();
  void lbfgsOptimise();
  void scgOptimise();
  void cgOptimise();
  //void lineMinimisation(const CMatrix& direction);
  double oneDObjectiveVal(double val);
  //void lineMinimisation();
  //void bracketMinimum(double& a, double& b, double& c, double& fa, unsigned int maxStep);
  void setLearnRate(double val)
  {
    learnRate = val;
  }
  double getLearnRate() const
  {
    return learnRate;
  }
  void setMomentum(double val)
  {
    momentum = val;
  }
  double getMomentum() const
  { 
    return momentum;
  }
  void setFuncEvalTerminate(bool val)
  {
    funcEvalTerminate = val;
  }
  bool isFuncEvalTerminate() const
  {
    return funcEvalTerminate;
  }
  void setIterTerminate(bool val)
  {
    iterTerminate = val;
  }
  bool isIterTerminate() const
  {
    return iterTerminate;
  }
  void setMaxFuncEvals(unsigned int val)
  {
    maxFuncVal = val;
  }
  unsigned int getMaxFuncEvals() const
  {
    return maxFuncVal;
  }
  void setMaxIters(unsigned int val)
  {
    maxIters = val;
  }
  unsigned int getMaxIters() const
  {
    return maxIters;
  }
  /// Completed algorithm / reverse-communication steps in the last optimiser run.
  unsigned int getIter() const
  {
    return iter;
  }
  /// Objective/gradient evaluations in the last optimiser run.
  unsigned int getFuncEval() const
  {
    return funcEval;
  }
  void setObjectiveTol(double val)
  {
    objectiveTol = val;
  }
  double getObjectiveTol() const
  {
    return objectiveTol;
  }
  void setParamTol(double val)
  {
    parameterTol = val;
  }
  double getParamTol() const
  {
    return parameterTol;
  }
  void setDefaultOptimiser(int val) const
  {
    defaultOptimiser = val;
  }
  int getDefaultOptimiser() const
  {
    return defaultOptimiser;
  }
  void setDefaultOptimiserStr(string val)
  {
    if(val == "conjgrad")
      defaultOptimiser = CG;
    else if(val == "scg")
      defaultOptimiser = SCG;
    else if(val == "quasinew" || val == "lbfgs")
      defaultOptimiser = BFGS;
    else if(val == "graddesc")
      defaultOptimiser = GD;
    else 
      throw ndlexceptions::NotImplementedError("Unknown optimisation");
  }
  string getDefaultOptimiserStr() const
  {
    switch(defaultOptimiser)
    {
    case CG:
      return "conjgrad";
    case SCG:
      return "scg";
    case BFGS:
      return "quasinew";
    case GD:
      return "graddesc";
    default:
      throw ndlexceptions::NotImplementedError("Unknown optimisation.");

    }
  }
  void runDefaultOptimiser()
  {
    switch(defaultOptimiser)
    {
    case CG:
      cgOptimise();
      break;
    case SCG:
      scgOptimise();
      break;
    case BFGS:
      lbfgsOptimise();
      break;
    case GD:
      gdOptimise();
      break;
    default:
      throw ndlexceptions::NotImplementedError("Unknown optimisation.");
      
    }
  }
 private:
  
  double objectiveTol;
  double parameterTol;
  
  #ifndef SWIG
    const static bool evalFunc=true;
    const static double phi;
    const static double cphi;
    const static double smallNum;
  #else
    static bool evalFunc=true;
    static double phi;
    static double cphi;
    static double smallNum;
  #endif
  double learnRate;
  double momentum;

  unsigned int iter;
  unsigned int funcEval;

  unsigned int maxIters;
  unsigned int maxFuncVal;

  mutable int verbosity; 

  mutable int defaultOptimiser;

  bool funcEvalTerminate;
  bool iterTerminate;

  bool defaultTextProgress;
  mutable std::ostream* progressStream;
  mutable std::ostream* logStream;
  mutable unsigned int progressLinesPrinted;
  mutable int inPlaceProgressMode;
  mutable std::vector<OptimiserObserver> optimiserObservers;
  
  CMatrix direction; // direction for 1-D optimisation.
  CMatrix paramStoreOne;
  CMatrix paramStoreTwo;

  void writeDefaultOptimiserText(const OptimiserEvent& event) const;
  void writeOptimiserLogLine(const OptimiserEvent& event) const;
  std::string buildProgressBlock(const OptimiserEvent& event) const;
  bool shouldUseInPlaceProgress() const;
  static const char* optimiserName(int optimiserId);
  OptimiserEvent makeOptimiserEvent(OptimiserPhase phase,
                                    int optimiserId,
                                    double objectiveVal,
                                    double stepOrScale = std::numeric_limits<double>::quiet_NaN(),
                                    const std::string& message = std::string()) const;
}; 

class CProbabilisticOptimisable : public COptimisable
{
 public:
  CProbabilisticOptimisable() : COptimisable() {}
  virtual double logLikelihood() const=0;
  virtual double logLikelihoodGradient(CMatrix& g) const=0;
  virtual double computeObjectiveGradParams(CMatrix& g) const 
  {
    double L = logLikelihoodGradient(g);
    g.negate();
    return -L;
  }
  virtual double computeObjectiveVal() const 
  {
    return -logLikelihood();
  }
};
#endif
