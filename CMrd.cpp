/* Manifold Relevance Determination — CIP-0018. */
#include "CMrd.h"
#include "ndlutil.h"
#include <cmath>

using namespace std;

CMrd::CMrd(const vector<CMatrix>& Ys, const vector<CKern*>& kerns,
           unsigned int latDim, unsigned int numInducing, int verbosity)
  : numData(0), latentDim(latDim), numActive(numInducing)
{
  setVerbosity(verbosity);
  if(Ys.size() < 2)
    throw ndlexceptions::Error("CMrd requires at least two views.");
  if(Ys.size() != kerns.size())
    throw ndlexceptions::Error("CMrd: one kernel per view required.");
  numData = Ys[0].getRows();
  for(size_t v = 1; v < Ys.size(); v++)
    if(Ys[v].getRows() != numData)
      throw ndlexceptions::Error("CMrd: all views must share N.");
  if(numActive < 1 || numActive > numData)
    throw ndlexceptions::Error("CMrd: numInducing must be in 1..N.");

  sTransform = CTransform::defaultPositive();
  initStoreage();

  for(size_t v = 0; v < Ys.size(); v++)
  {
    if(kerns[v] == 0)
      throw ndlexceptions::Error("CMrd: null kernel.");
    if(kerns[v]->getInputDim() != latentDim)
      throw ndlexceptions::Error("CMrd: kernel input dim must equal q.");
    CBayesGplvm* view = new CBayesGplvm(kerns[v], Ys[v], latentDim, numActive, verbosity);
    view->setOwnKern(true); // CMrd takes ownership of caller kernels
    views.push_back(view);
    ownKern.push_back(true);
  }
  initVals();
}

CMrd::~CMrd()
{
  for(size_t v = 0; v < views.size(); v++)
    delete views[v]; // deletes kern when view->isOwnKern()
}

void CMrd::initStoreage()
{
  mu.resize(numData, latentDim);
  S.resize(numData, latentDim);
  Z.resize(numActive, latentDim);
}

void CMrd::initVals()
{
  initXpcaConcat();
  S.setVals(0.5);
  initZFromMu();
  syncViews();
}

void CMrd::initXpcaConcat()
{
  // Concatenate observation columns then PCA into mu (GPy PCA_concat spirit).
  unsigned int Dtot = 0;
  for(size_t v = 0; v < views.size(); v++)
    Dtot += views[v]->getOutputDim();
  CMatrix Ycat(numData, Dtot);
  unsigned int col = 0;
  for(size_t v = 0; v < views.size(); v++)
  {
    const CMatrix& Yv = views[v]->getY();
    for(unsigned int j = 0; j < Yv.getCols(); j++, col++)
      for(unsigned int n = 0; n < numData; n++)
        Ycat.setVal(Yv.getVal(n, j), n, col);
  }

  CMatrix ymean = meanCol(Ycat);
  CMatrix covm(Dtot, Dtot);
  covm.setSymmetric(true);
  covm.syrk(Ycat, 1.0 / (double)numData, 0.0, "u", "t");
  ymean.trans();
  covm.syr(ymean, -1.0, "u");
  CMatrix eigVals(1, Dtot);
  covm.syev(eigVals, "v", "u");
  const unsigned int qUse = latentDim < Dtot ? latentDim : Dtot;
  CMatrix Winv(Dtot, latentDim);
  Winv.zeros();
  for(unsigned int i = 0; i < qUse; i++)
  {
    Winv.copyColCol(i, covm, Dtot - 1 - i);
    const double ev = eigVals.getVal(Dtot - 1 - i);
    if(ev > 0.0)
      Winv.scaleCol(i, 1.0 / sqrt(ev));
  }
  mu.gemm(Ycat, Winv, 1.0, 0.0, "n", "n");
  CMatrix meanX = meanCol(mu);
  for(unsigned int i = 0; i < mu.getRows(); i++)
    mu.axpyRowRow(i, meanX, 0, -1.0);
}

void CMrd::initZFromMu()
{
  for(unsigned int m = 0; m < numActive; m++)
  {
    const unsigned int n = (m * numData) / numActive;
    for(unsigned int q = 0; q < latentDim; q++)
      Z.setVal(mu.getVal(n % numData, q), m, q);
  }
}

void CMrd::syncViews() const
{
  for(size_t v = 0; v < views.size(); v++)
    views[v]->setLatents(mu, S, Z);
}

CBayesGplvm* CMrd::getView(unsigned int v)
{
  BOUNDCHECK(v < views.size());
  return views[v];
}

const CBayesGplvm* CMrd::getView(unsigned int v) const
{
  BOUNDCHECK(v < views.size());
  return views[v];
}

double CMrd::logLikelihood() const
{
  syncViews();
  double L = 0.0;
  for(size_t v = 0; v < views.size(); v++)
    L += views[v]->varDtcBound();
  // Single KL (not V times).
  L -= views[0]->klDivergence();
  return L;
}

unsigned int CMrd::getOptNumParams() const
{
  unsigned int n = 0;
  for(size_t v = 0; v < views.size(); v++)
    n += views[v]->getKern()->getNumParams() + 1; // kern + beta
  n += numActive * latentDim; // Z
  n += numData * latentDim;   // mu
  n += numData * latentDim;   // S
  return n;
}

void CMrd::getOptParams(CMatrix& param) const
{
  DIMENSIONMATCH(param.getRows() == 1 && param.getCols() == getOptNumParams());
  syncViews();
  unsigned int c = 0;
  for(size_t v = 0; v < views.size(); v++)
  {
    CBayesGplvm* view = views[v];
    CMatrix vp(1, view->getOptNumParams());
    view->getOptParams(vp);
    // View packing: [kern...][beta][Z...][mu...][S...] — take kern+beta only.
    const unsigned int nk = view->getKern()->getNumParams();
    for(unsigned int i = 0; i < nk + 1; i++)
      param.setVal(vp.getVal(i), c++);
  }
  for(unsigned int m = 0; m < numActive; m++)
    for(unsigned int q = 0; q < latentDim; q++)
      param.setVal(Z.getVal(m, q), c++);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      param.setVal(mu.getVal(n, q), c++);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      param.setVal(sTransform->xtoa(S.getVal(n, q)), c++);
  DIMENSIONMATCH(c == getOptNumParams());
}

void CMrd::setOptParams(const CMatrix& param)
{
  DIMENSIONMATCH(param.getRows() == 1 && param.getCols() == getOptNumParams());
  unsigned int c = 0;
  for(size_t v = 0; v < views.size(); v++)
  {
    CBayesGplvm* view = views[v];
    CMatrix vp(1, view->getOptNumParams());
    view->getOptParams(vp); // seed with current Z/mu/S
    const unsigned int nk = view->getKern()->getNumParams();
    for(unsigned int i = 0; i < nk + 1; i++)
      vp.setVal(param.getVal(c++), i);
    // Keep view's Z/mu/S slots; overwritten by sync after shared write.
    view->setOptParams(vp);
  }
  for(unsigned int m = 0; m < numActive; m++)
    for(unsigned int q = 0; q < latentDim; q++)
      Z.setVal(param.getVal(c++), m, q);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      mu.setVal(param.getVal(c++), n, q);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      S.setVal(sTransform->atox(param.getVal(c++)), n, q);
  DIMENSIONMATCH(c == getOptNumParams());
  syncViews();
}

double CMrd::logLikelihoodGradient(CMatrix& g) const
{
  DIMENSIONMATCH(g.getRows() == 1 && g.getCols() == getOptNumParams());
  g.zeros();
  syncViews();

  CMatrix gZ(numActive, latentDim);
  CMatrix gMu(numData, latentDim);
  CMatrix gS(numData, latentDim);
  gZ.zeros();
  gMu.zeros();
  gS.zeros();

  unsigned int c = 0;
  for(size_t v = 0; v < views.size(); v++)
  {
    CBayesGplvm* view = views[v];
    const unsigned int nk = view->getKern()->getNumParams();
    CMatrix gKern(1, nk);
    CMatrix gZv(numActive, latentDim);
    CMatrix gMuv(numData, latentDim);
    CMatrix gSv(numData, latentDim);
    double gBeta = 0.0;
    view->computeVarDtcGrads(gKern, gZv, gMuv, gSv, gBeta, false);
    for(unsigned int i = 0; i < nk; i++)
      g.setVal(gKern.getVal(i), c++);
    g.setVal(gBeta, c++);
    for(unsigned int m = 0; m < numActive; m++)
      for(unsigned int q = 0; q < latentDim; q++)
        gZ.addVal(gZv.getVal(m, q), m, q);
    for(unsigned int n = 0; n < numData; n++)
      for(unsigned int q = 0; q < latentDim; q++)
      {
        gMu.addVal(gMuv.getVal(n, q), n, q);
        gS.addVal(gSv.getVal(n, q), n, q);
      }
  }

  // Single KL on shared latents.
  CMatrix gMuKl(numData, latentDim);
  CMatrix gSKl(numData, latentDim);
  views[0]->klGradient(gMuKl, gSKl);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
    {
      gMu.addVal(-gMuKl.getVal(n, q), n, q);
      gS.addVal(-gSKl.getVal(n, q), n, q);
    }

  for(unsigned int m = 0; m < numActive; m++)
    for(unsigned int q = 0; q < latentDim; q++)
      g.setVal(gZ.getVal(m, q), c++);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      g.setVal(gMu.getVal(n, q), c++);
  for(unsigned int n = 0; n < numData; n++)
    for(unsigned int q = 0; q < latentDim; q++)
      g.setVal(gS.getVal(n, q) * sTransform->gradfact(S.getVal(n, q)), c++);

  DIMENSIONMATCH(c == getOptNumParams());
  return logLikelihood();
}

void CMrd::optimise(unsigned int iters)
{
  if(getVerbosity() > 2 && getOptNumParams() < 80)
    checkGradients();
  setMaxIters(iters);
  runDefaultOptimiser();
}

void CMrd::display(ostream& os) const
{
  syncViews();
  os << "MRD:" << endl;
  os << "  V=" << views.size() << " N=" << numData << " q=" << latentDim
     << " M=" << numActive << endl;
  os << "  ELBO=" << logLikelihood() << endl;
  for(size_t v = 0; v < views.size(); v++)
  {
    os << "  view " << v << ": D=" << views[v]->getOutputDim()
       << " beta=" << views[v]->getBeta() << endl;
    views[v]->getKern()->display(os);
  }
}

const CRbfardKern* CMrd::findRbfard(const CKern* k) const
{
  if(k == 0)
    return 0;
  // v1: direct RBF-ARD only (compound ARD readout deferred).
  if(k->getType() == "rbfard")
    return dynamic_cast<const CRbfardKern*>(k);
  return 0;
}

void CMrd::getInputScales(unsigned int view, vector<double>& scales) const
{
  BOUNDCHECK(view < views.size());
  const CRbfardKern* ard = findRbfard(views[view]->getKern());
  if(ard == 0)
    throw ndlexceptions::Error("getInputScales: view has no RBF-ARD kernel.");
  scales.resize(latentDim);
  for(unsigned int q = 0; q < latentDim; q++)
    scales[q] = ard->getParam(2 + q);
}

void CMrd::factorizeSpace(vector<unsigned int>& shared, vector<unsigned int>& private0,
                          vector<unsigned int>& private1, double threshold) const
{
  if(views.size() != 2)
    throw ndlexceptions::Error("factorizeSpace currently requires exactly two views.");
  vector<double> s0, s1;
  getInputScales(0, s0);
  getInputScales(1, s1);
  double m0 = 0.0, m1 = 0.0;
  for(unsigned int q = 0; q < latentDim; q++)
  {
    if(s0[q] > m0) m0 = s0[q];
    if(s1[q] > m1) m1 = s1[q];
  }
  if(m0 <= 0.0) m0 = 1.0;
  if(m1 <= 0.0) m1 = 1.0;
  shared.clear();
  private0.clear();
  private1.clear();
  for(unsigned int q = 0; q < latentDim; q++)
  {
    const double a = s0[q] / m0;
    const double b = s1[q] / m1;
    const bool aOn = a > threshold;
    const bool bOn = b > threshold;
    if(aOn && bOn)
      shared.push_back(q);
    else if(aOn)
      private0.push_back(q);
    else if(bOn)
      private1.push_back(q);
  }
}

void CMrd::latentMeanVar(unsigned int view, CMatrix& means, CMatrix& vars,
                         const CMatrix& Xstar) const
{
  BOUNDCHECK(view < views.size());
  syncViews();
  views[view]->latentMeanVar(means, vars, Xstar);
}

void CMrd::predictiveMeanVar(unsigned int view, CMatrix& means, CMatrix& vars,
                             const CMatrix& Xstar) const
{
  BOUNDCHECK(view < views.size());
  syncViews();
  views[view]->predictiveMeanVar(means, vars, Xstar);
}

void CMrd::posteriorMeanVar(unsigned int view, CMatrix& means, CMatrix& vars,
                            const CMatrix& Xstar) const
{
  // CIP-0031: deprecated alias — same as predictiveMeanVar.
  predictiveMeanVar(view, means, vars, Xstar);
}

void CMrd::writeParamsToStream(ostream& os) const
{
  syncViews();
  writeToStream(os, "baseType", "mrd");
  writeToStream(os, "type", "mrd");
  writeToStream(os, "numViews", (int)views.size());
  writeToStream(os, "numData", numData);
  writeToStream(os, "latentDim", latentDim);
  writeToStream(os, "numActive", numActive);
  mu.toStream(os);
  S.toStream(os);
  Z.toStream(os);
  for(size_t v = 0; v < views.size(); v++)
  {
    writeToStream(os, "view", (int)v);
    views[v]->writeParamsToStream(os);
  }
}

void CMrd::readParamsFromStream(istream& is)
{
  string tbase = getBaseTypeStream(is);
  if(tbase != "mrd")
    throw ndlexceptions::StreamFormatError("baseType", "Expected mrd");
  string t = getTypeStream(is);
  if(t != "mrd")
    throw ndlexceptions::StreamFormatError("type", "Expected mrd");
  const unsigned int V = (unsigned int)readIntFromStream(is, "numViews");
  numData = readIntFromStream(is, "numData");
  latentDim = readIntFromStream(is, "latentDim");
  numActive = readIntFromStream(is, "numActive");
  initStoreage();
  mu.fromStream(is);
  S.fromStream(is);
  Z.fromStream(is);

  for(size_t v = 0; v < views.size(); v++)
    delete views[v];
  views.clear();
  ownKern.clear();

  for(unsigned int v = 0; v < V; v++)
  {
    const int vv = readIntFromStream(is, "view");
    if((unsigned int)vv != v)
      throw ndlexceptions::StreamFormatError("view", "Unexpected view index");
    CMatrix Ytmp(1, 1);
    CRbfKern kernTmp(1);
    kernTmp.setInitParam();
    CBayesGplvm* view = new CBayesGplvm(&kernTmp, Ytmp, 1, 1, getVerbosity());
    view->readParamsFromStream(is); // owns streamed kern
    views.push_back(view);
    ownKern.push_back(true);
  }
  syncViews();
}
