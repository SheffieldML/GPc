#include "CSparsePpa.h"

CSparsePpa::CSparsePpa(const CMatrix& inData, const CMatrix& targetData,
                       CKern& kernel, CNoise& outer,
                       int approx, unsigned int actSize,
                       double beta, int verbos)
  : X(inData), y(targetData), kern(kernel), outerNoise(outer),
    pInnerNoise(nullptr), pGp(nullptr),
    betaVal(beta), approxType(approx), actSetSize(actSize),
    verbosity(verbos),
    numData(static_cast<int>(targetData.getRows())),
    numTarget(static_cast<int>(targetData.getCols()))
{
  DIMENSIONMATCH(X.getRows() == y.getRows());
  if(approxType == CGp::FTC)
    {
      approxType = CGp::DTC;
      if(actSetSize == 0)
        actSetSize = static_cast<unsigned int>(numData);
    }
  if(actSetSize == 0)
    throw ndlexceptions::Error("CSparsePpa: sparse approx needs actSetSize > 0");

  f.resize(numData, numTarget);
  fBar.resize(numData, numTarget);
  g.resize(numData, numTarget);
  nu.resize(numData, numTarget);
  gamma.resize(numData, numTarget);
  ff.resize(numData, numTarget);
  f.zeros();
  fBar.zeros();
  g.zeros();
  nu.zeros();
  gamma.zeros();
  ff.zeros();

  pInnerNoise = new CGaussianNoise(&f);
  for(unsigned int j = 0; j < pInnerNoise->getOutputDim(); j++)
    pInnerNoise->setBiasVal(0.0, j);
  pInnerNoise->setParam(1e-12, pInnerNoise->getNumParams() - 1);

  pGp = new CGp(static_cast<unsigned int>(X.getCols()),
                static_cast<unsigned int>(numTarget),
                &X, &f, &kern, pInnerNoise,
                approxType, actSetSize, verbosity);
  pGp->setBetaVal(betaVal);
  pGp->fixBeta(true);
  pGp->setOutputScaleLearnt(false);
  if(pGp->isSparseApproximation())
    {
      pGp->setInducingFixed(true);
      if(actSetSize == static_cast<unsigned int>(numData))
        setInducingToX();
    }
  outerNoise.setVerbosity(verbosity);
  init();
}

CSparsePpa::~CSparsePpa()
{
  delete pGp;
  delete pInnerNoise;
}

void CSparsePpa::setInducingToX()
{
  if(!pGp->isSparseApproximation())
    throw ndlexceptions::Error("setInducingToX requires a sparse approximation");
  if(actSetSize != static_cast<unsigned int>(numData))
    throw ndlexceptions::Error("setInducingToX requires actSetSize == N");
  pGp->setInducingVals(X);
  pGp->setInducingFixed(true);
  pGp->invalidateCache();
}

void CSparsePpa::fixInducing(bool val)
{
  pGp->setInducingFixed(val);
}

void CSparsePpa::syncGpTargets()
{
  pGp->setTargetVals(f);
  pGp->invalidateCache();
}

void CSparsePpa::updateNuG()
{
  for(int i = 0; i < numData; i++)
    outerNoise.getNuG(g, nu, i);
}

void CSparsePpa::init()
{
  outerNoise.setMus(0.0);
  for(int i = 0; i < numData; i++)
    {
      const double dk = kern.diagComputeElement(X, i);
      for(int j = 0; j < numTarget; j++)
        outerNoise.setVarSigma(dk, i, j);
    }
  f.zeros();
  fBar.zeros();
  eStep();
}

void CSparsePpa::eStep()
{
  syncGpTargets();
  pGp->posteriorMean(fBar, X);

  outerNoise.setMus(fBar);
  for(int i = 0; i < numData; i++)
    for(int j = 0; j < numTarget; j++)
      outerNoise.setVarSigma(1.0 / betaVal, i, j);
  updateNuG();
  for(int i = 0; i < numData; i++)
    for(int j = 0; j < numTarget; j++)
      {
        const double gVal = g.getVal(i, j);
        gamma.setVal((gVal * gVal - nu.getVal(i, j)) / 2.0, i, j);
        f.setVal(fBar.getVal(i, j) + gVal / betaVal, i, j);
      }
  for(int j = 0; j < numTarget; j++)
    {
      const double b2 = betaVal * betaVal;
      for(int i = 0; i < numData; i++)
        {
          const double mVal = fBar.getVal(i, j);
          const double ffVal = 2.0 / b2 * gamma.getVal(i, j) + 1.0 / betaVal
            + 2.0 * mVal * f.getVal(i, j) - mVal * mVal;
          ff.setVal(ffVal, i, j);
        }
    }
}

void CSparsePpa::mStep(int kernIters)
{
  if(kern.getNumParams() == 0 || kernIters <= 0)
    return;
  syncGpTargets();
  const int oldVerb = pGp->getVerbosity();
  pGp->setVerbosity(0);
  pGp->setMaxIters(static_cast<unsigned int>(kernIters));
  pGp->runDefaultOptimiser();
  pGp->setVerbosity(oldVerb);
  pGp->invalidateCache();
}

double CSparsePpa::monitorLogLikelihood() const
{
  // Interim Phase C monitor: inducing approx LML on ⟨f⟩ with noise 1/β.
  // Not the dense PPA approxLogLikelihood (undefined for rank-M priors).
  const_cast<CSparsePpa*>(this)->syncGpTargets();
  return pGp->logLikelihood();
}

void CSparsePpa::posteriorMeanVar(CMatrix& mu, CMatrix& varSigma,
                                  const CMatrix& Xin) const
{
  const_cast<CSparsePpa*>(this)->syncGpTargets();
  pGp->posteriorMeanVar(mu, varSigma, Xin);
}

void CSparsePpa::out(CMatrix& yPred, const CMatrix& Xin) const
{
  CMatrix mu(Xin.getRows(), numTarget);
  CMatrix vs(Xin.getRows(), numTarget);
  posteriorMeanVar(mu, vs, Xin);
  outerNoise.out(yPred, mu, vs);
}

void CSparsePpa::display(ostream& os) const
{
  os << "Sparse PPA: approx=" << pGp->getApproximationStr()
     << " M=" << actSetSize
     << " N=" << numData
     << " beta=" << betaVal << endl;
  os << "Kernel:" << endl;
  kern.display(os);
  os << "Outer noise: " << outerNoise.getName() << endl;
}

void CSparsePpa::optimise(int maxIters, int kernIters)
{
  if(verbosity > 0)
    {
      cout << "Initial sparse PPA:" << endl;
      display(cout);
    }
  for(int counter = 0; counter < maxIters; counter++)
    {
      mStep(kernIters);
      eStep();
      if(verbosity > 0)
        {
          const double ll = monitorLogLikelihood();
          cout << "CSparsePpa iter " << (counter + 1)
               << " monitor(CGp LL) " << ll << endl;
        }
    }
  if(verbosity > 0)
    {
      cout << "Final sparse PPA:" << endl;
      display(cout);
    }
}
