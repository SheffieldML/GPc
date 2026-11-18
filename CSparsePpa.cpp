#include "CSparsePpa.h"
#include "ndlstrutil.h"
#include <fstream>
#include <iomanip>

namespace {
void allocSiteStorage(CSparsePpa& m, int numData, int numTarget)
{
  m.f.resize(numData, numTarget);
  m.fBar.resize(numData, numTarget);
  m.g.resize(numData, numTarget);
  m.nu.resize(numData, numTarget);
  m.gamma.resize(numData, numTarget);
  m.ff.resize(numData, numTarget);
  m.f.zeros();
  m.fBar.zeros();
  m.g.zeros();
  m.nu.zeros();
  m.gamma.zeros();
  m.ff.zeros();
}
}  // namespace

void CSparsePpa::constructGp(bool runInitEstep)
{
  if(approxType == CGp::FTC)
    {
      approxType = CGp::DTC;
      if(actSetSize == 0)
        actSetSize = static_cast<unsigned int>(numData);
    }
  if(actSetSize == 0)
    throw ndlexceptions::Error("CSparsePpa: sparse approx needs actSetSize > 0");

  allocSiteStorage(*this, numData, numTarget);

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
  if(runInitEstep)
    init();
}

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
  constructGp(true);
}

CSparsePpa::CSparsePpa(const CMatrix& inData, const CMatrix& targetData,
                       CKern& kernel, CNoise& outer,
                       int approx, unsigned int actSize,
                       double beta, int verbos, CSparsePpaLoadedTag)
  : X(inData), y(targetData), kern(kernel), outerNoise(outer),
    pInnerNoise(nullptr), pGp(nullptr),
    betaVal(beta), approxType(approx), actSetSize(actSize),
    verbosity(verbos),
    numData(static_cast<int>(targetData.getRows())),
    numTarget(static_cast<int>(targetData.getCols()))
{
  DIMENSIONMATCH(X.getRows() == y.getRows());
  constructGp(false);
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

void CSparsePpa::setSiteState(const CMatrix& fBarIn, const CMatrix& fIn)
{
  DIMENSIONMATCH(fBarIn.dimensionsMatch(fBar));
  DIMENSIONMATCH(fIn.dimensionsMatch(f));
  fBar.deepCopy(fBarIn);
  f.deepCopy(fIn);
  syncGpTargets();
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

double CSparsePpa::ppaBound() const
{
  // Sparse variational monitor (CIP-0026 Phase C):
  //   L ≈ log p_approx(⟨f⟩ | X, u)  [CGp DTC/DTCVAR/FITC/PITC bound]
  //     + E[log p(y | fBar)] with site variance 1/β (mean-field; no N×N Cov)
  // Dense approxLogLikelihood also has KL terms in K^{-1}/fBarfBar; those need
  // a full q(u) derivation. DTCVAR is the preferred GP term (Titsias VFE).
  const_cast<CSparsePpa*>(this)->syncGpTargets();
  double L = pGp->logLikelihood();
  CMatrix vs(numData, numTarget);
  vs.setVals(1.0 / betaVal);
  L += outerNoise.logLikelihood(fBar, vs, y);
  return L;
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
     << " beta=" << betaVal
     << " inducingFixed=" << (pGp->isInducingFixed() ? "true" : "false")
     << endl;
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
  double oldBound = ppaBound();
  for(int counter = 0; counter < maxIters; counter++)
    {
      mStep(kernIters);
      eStep();
      const double bound = ppaBound();
      if(verbosity > 0)
        {
          cout << "CSparsePpa iter " << (counter + 1)
               << " ppaBound " << bound
               << " (delta " << (bound - oldBound) << ")" << endl;
        }
      // Soft signal only — mean-field M-step need not be monotone.
      oldBound = bound;
    }
  if(verbosity > 0)
    {
      cout << "Final sparse PPA:" << endl;
      display(cout);
    }
}

void writeSparsePpaToStream(const CSparsePpa& model, ostream& out)
{
  out << "ppaVersion=" << SPARSE_PPAVERSION << endl;
  out << "approx=" << model.gp().getApproximationStr() << endl;
  out << "numActive=" << model.getNumActive() << endl;
  out << "numTrainData=" << model.X.getRows() << endl;
  out << "numProcesses=" << model.y.getCols() << endl;
  out << "numFeatures=" << model.X.getCols() << endl;
  out << "beta=" << model.getBeta() << endl;
  out << "inducingFixed=" << (model.gp().isInducingFixed() ? 1 : 0) << endl;

  writeKernToStream(*model.gp().getKernel(), out);
  writeNoiseToStream(model.getOuterNoise(), out);

  // Inducing points X_u (numActive x numFeatures).
  const CMatrix& Xu = model.gp().X_u;
  for(unsigned int i = 0; i < Xu.getRows(); i++)
    {
      for(unsigned int j = 0; j < Xu.getCols(); j++)
        {
          out << Xu.getVal(i, j);
          if(j + 1 < Xu.getCols())
            out << " ";
        }
      out << endl;
    }

  // Dense blocks: y, fBar, f, X (row-major text).
  const unsigned int N = model.X.getRows();
  const unsigned int P = model.y.getCols();
  const unsigned int Q = model.X.getCols();
  for(unsigned int i = 0; i < N; i++)
    {
      for(unsigned int j = 0; j < P; j++)
        out << model.y.getVal(i, j) << (j + 1 < P ? " " : "");
      out << endl;
    }
  for(unsigned int i = 0; i < N; i++)
    {
      for(unsigned int j = 0; j < P; j++)
        out << model.fBar.getVal(i, j) << (j + 1 < P ? " " : "");
      out << endl;
    }
  for(unsigned int i = 0; i < N; i++)
    {
      for(unsigned int j = 0; j < P; j++)
        out << model.f.getVal(i, j) << (j + 1 < P ? " " : "");
      out << endl;
    }
  for(unsigned int i = 0; i < N; i++)
    {
      for(unsigned int j = 0; j < Q; j++)
        out << model.X.getVal(i, j) << (j + 1 < Q ? " " : "");
      out << endl;
    }
}

void writeSparsePpaToFile(const CSparsePpa& model, const string& modelFileName,
                          const string& comment)
{
  ofstream out(modelFileName.c_str());
  if(!out)
    throw ndlexceptions::FileWriteError(modelFileName);
  out << setiosflags(ios::scientific);
  out << setprecision(17);
  if(comment.size() > 0)
    out << "# " << comment << endl;
  writeSparsePpaToStream(model, out);
  out.close();
}

static string readKeyedLine(istream& in, const string& key)
{
  string line;
  vector<string> tokens;
  ndlstrutil::getline(in, line);
  while(line.size() > 0 && line[0] == '#')
    {
      tokens.clear();
      ndlstrutil::getline(in, line);
    }
  ndlstrutil::tokenise(tokens, line, "=");
  if(tokens.size() != 2 || tokens[0] != key)
    throw ndlexceptions::FileFormatError();
  return tokens[1];
}

CSparsePpa* readSparsePpaFromStream(istream& in, int verbosity)
{
  if(readKeyedLine(in, "ppaVersion") != SPARSE_PPAVERSION)
    throw ndlexceptions::FileFormatError();
  const string approxStr = readKeyedLine(in, "approx");
  const unsigned int numActive = static_cast<unsigned int>(atoi(readKeyedLine(in, "numActive").c_str()));
  const int numTrainData = atoi(readKeyedLine(in, "numTrainData").c_str());
  const int numProcesses = atoi(readKeyedLine(in, "numProcesses").c_str());
  const int numFeatures = atoi(readKeyedLine(in, "numFeatures").c_str());
  const double beta = atof(readKeyedLine(in, "beta").c_str());
  const bool inducingFixed = (atoi(readKeyedLine(in, "inducingFixed").c_str()) != 0);

  int approxType = CGp::DTC;
  if(approxStr == "dtc")
    approxType = CGp::DTC;
  else if(approxStr == "dtcvar")
    approxType = CGp::DTCVAR;
  else if(approxStr == "fitc")
    approxType = CGp::FITC;
  else if(approxStr == "pitc")
    approxType = CGp::PITC;
  else
    throw ndlexceptions::FileFormatError();

  CKern* pkern = readKernFromStream(in);
  CNoise* pnoise = readNoiseFromStream(in);

  CMatrix Xu(numActive, numFeatures);
  for(unsigned int i = 0; i < numActive; i++)
    {
      string line;
      vector<string> tokens;
      ndlstrutil::getline(in, line);
      ndlstrutil::tokenise(tokens, line, " ");
      if(tokens.size() != numFeatures)
        throw ndlexceptions::FileFormatError();
      for(unsigned int j = 0; j < numFeatures; j++)
        Xu.setVal(atof(tokens[j].c_str()), i, j);
    }

  CMatrix* pX = new CMatrix(numTrainData, numFeatures, 0.0);
  CMatrix* py = new CMatrix(numTrainData, numProcesses);
  CMatrix fBar(numTrainData, numProcesses);
  CMatrix f(numTrainData, numProcesses);
  auto readBlock = [&](CMatrix& A, int rows, int cols) {
    for(int i = 0; i < rows; i++)
      {
        string line;
        vector<string> tokens;
        ndlstrutil::getline(in, line);
        ndlstrutil::tokenise(tokens, line, " ");
        if(static_cast<int>(tokens.size()) != cols)
          throw ndlexceptions::FileFormatError();
        for(int j = 0; j < cols; j++)
          A.setVal(atof(tokens[j].c_str()), i, j);
      }
  };
  readBlock(*py, numTrainData, numProcesses);
  readBlock(fBar, numTrainData, numProcesses);
  readBlock(f, numTrainData, numProcesses);
  readBlock(*pX, numTrainData, numFeatures);

  pnoise->setTarget(py);
  CSparsePpa* model = new CSparsePpa(*pX, *py, *pkern, *pnoise,
                                     approxType, numActive, beta, verbosity,
                                     CSparsePpaLoadedTag());
  model->gp().setInducingVals(Xu);
  model->gp().setInducingFixed(inducingFixed);
  model->setSiteState(fBar, f);
  return model;
}

CSparsePpa* readSparsePpaFromFile(const string& modelFileName, int verbosity)
{
  ifstream in(modelFileName.c_str());
  if(!in)
    throw ndlexceptions::FileReadError(modelFileName);
  return readSparsePpaFromStream(in, verbosity);
}
