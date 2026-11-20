/* CIP-0015 Phase A: multi-output helpers + coregionalized regression factory. */
#include "CMultioutput.h"
#include "ndlutil.h"
#include <cmath>

using namespace std;

namespace {

void appendIndexWhite(CCmpndKern* signal, unsigned int numOutputs)
{
  // Same layout as createIcmKern's coregionalize: inputDim=1, activeDims={q}
  // where q is the index column of the wide (q_spatial+1) design.
  const unsigned int wideDim = signal->getInputDim();
  BOUNDCHECK(wideDim >= 1);
  const unsigned int qIndex = wideDim - 1;
  CIndexWhiteKern* iw = new CIndexWhiteKern(1, numOutputs);
  iw->setInitParam();
  vector<unsigned int> idx(1, qIndex);
  iw->setActiveDims(idx);
  for(unsigned int d = 0; d < numOutputs; d++)
    iw->setParam(0.1, d);
  signal->addKern(iw);
  delete iw;
}

} // namespace

void buildXY(const vector<CMatrix>& Xlist, const vector<CMatrix>& Ylist,
             CMatrix& Xout, CMatrix& Yout)
{
  if(Xlist.empty())
    throw ndlexceptions::Error("buildXY: Xlist is empty");
  if(Xlist.size() != Ylist.size())
    throw ndlexceptions::Error("buildXY: Xlist and Ylist size mismatch");

  const unsigned int q = Xlist[0].getCols();
  unsigned int N = 0;
  for(size_t v = 0; v < Xlist.size(); v++)
  {
    if(Xlist[v].getCols() != q)
      throw ndlexceptions::Error("buildXY: all X blocks must share input dim q");
    if(Ylist[v].getCols() != 1)
      throw ndlexceptions::Error("buildXY: each Y block must be N_v x 1");
    if(Ylist[v].getRows() != Xlist[v].getRows())
      throw ndlexceptions::Error("buildXY: X and Y row counts must match per output");
    N += Xlist[v].getRows();
  }

  Xout.resize(N, q + 1);
  Yout.resize(N, 1);
  unsigned int row = 0;
  for(size_t v = 0; v < Xlist.size(); v++)
  {
    const unsigned int Nv = Xlist[v].getRows();
    for(unsigned int n = 0; n < Nv; n++, row++)
    {
      for(unsigned int j = 0; j < q; j++)
        Xout.setVal(Xlist[v].getVal(n, j), row, j);
      Xout.setVal(static_cast<double>(v), row, q);
      Yout.setVal(Ylist[v].getVal(n, 0), row, 0);
    }
  }
}

vector<CRowSlice> getSlices(const vector<unsigned int>& lengths)
{
  vector<CRowSlice> slices;
  unsigned int start = 0;
  for(size_t i = 0; i < lengths.size(); i++)
  {
    CRowSlice s;
    s.start = start;
    s.stop = start + lengths[i];
    slices.push_back(s);
    start = s.stop;
  }
  return slices;
}

vector<CRowSlice> getSlices(const vector<CMatrix>& Xlist)
{
  vector<unsigned int> lengths(Xlist.size());
  for(size_t i = 0; i < Xlist.size(); i++)
    lengths[i] = Xlist[i].getRows();
  return getSlices(lengths);
}

vector<vector<CRowSlice> > indexToSlices(const CMatrix& indexCol)
{
  if(indexCol.getCols() != 1)
    throw ndlexceptions::Error("indexToSlices: index column must be N x 1");
  const unsigned int N = indexCol.getRows();
  if(N == 0)
    return vector<vector<CRowSlice> >();

  int maxIdx = -1;
  vector<int> ind(N);
  for(unsigned int n = 0; n < N; n++)
  {
    const double x = indexCol.getVal(n, 0);
    const long long rounded = llround(x);
    if(fabs(x - static_cast<double>(rounded)) >= ndlutil::MATCHTOL)
      throw ndlexceptions::Error("indexToSlices: non-integer index");
    if(rounded < 0)
      throw ndlexceptions::Error("indexToSlices: negative index");
    ind[n] = static_cast<int>(rounded);
    if(ind[n] > maxIdx)
      maxIdx = ind[n];
  }

  vector<vector<CRowSlice> > ret(static_cast<size_t>(maxIdx + 1));
  unsigned int runStart = 0;
  for(unsigned int n = 1; n <= N; n++)
  {
    const bool end = (n == N);
    const bool changed = !end && ind[n] != ind[n - 1];
    if(end || changed)
    {
      CRowSlice s;
      s.start = runStart;
      s.stop = n;
      ret[static_cast<size_t>(ind[runStart])].push_back(s);
      runStart = n;
    }
  }
  return ret;
}

void stampOutputIndex(CMatrix& X, unsigned int outputIndex)
{
  if(X.getCols() < 1)
    throw ndlexceptions::Error("stampOutputIndex: X has no columns");
  const unsigned int idxCol = X.getCols() - 1;
  for(unsigned int n = 0; n < X.getRows(); n++)
    X.setVal(static_cast<double>(outputIndex), n, idxCol);
}

CCmpndKern* createCoregionalizedKern(const CKern& spatial, unsigned int numOutputs,
                                     unsigned int rank, bool withIndexWhite)
{
  if(numOutputs < 1)
    throw ndlexceptions::Error("createCoregionalizedKern: numOutputs >= 1 required");
  CTensorKern* icm = createIcmKern(spatial, numOutputs, rank);
  CCmpndKern* cmpnd = new CCmpndKern(icm->getInputDim());
  cmpnd->addKern(icm);
  delete icm;
  if(withIndexWhite)
    appendIndexWhite(cmpnd, numOutputs);
  return cmpnd;
}

CCmpndKern* createCoregionalizedLcmKern(const vector<const CKern*>& spatials,
                                        unsigned int numOutputs, unsigned int rank,
                                        bool withIndexWhite)
{
  if(numOutputs < 1)
    throw ndlexceptions::Error("createCoregionalizedLcmKern: numOutputs >= 1 required");
  CCmpndKern* lcm = createLcmKern(spatials, numOutputs, rank);
  if(withIndexWhite)
    appendIndexWhite(lcm, numOutputs);
  return lcm;
}

void CCoregionalizedRegression::initCommon(const vector<CMatrix>& Xlist,
                                           const vector<CMatrix>& Ylist,
                                           CKern* signalKern, bool withIndexWhite,
                                           int verbosity)
{
  numOutputs = static_cast<unsigned int>(Xlist.size());
  slices = ::getSlices(Xlist);
  pX = new CMatrix();
  py = new CMatrix();
  buildXY(Xlist, Ylist, *pX, *py);

  (void)withIndexWhite; // signalKern already includes indexwhite when requested
  pkern = signalKern;

  pnoise = new CGaussianNoise(py);
  // Per-output noise is in indexwhite; keep spherical likelihood variance tiny/fixed.
  pnoise->setVariance(1e-6);
  model = new CGp(pkern, pnoise, pX, CGp::FTC, 0, verbosity);
  model->setTargetVals(py);
  model->fixLikelihoodVariance(true);
}

CCoregionalizedRegression::CCoregionalizedRegression(const vector<CMatrix>& Xlist,
                                                     const vector<CMatrix>& Ylist,
                                                     const CKern& spatial,
                                                     unsigned int rank,
                                                     bool withIndexWhite,
                                                     int verbosity)
  : numOutputs(0), pX(0), py(0), pkern(0), pnoise(0), model(0)
{
  CKern* signal = createCoregionalizedKern(spatial, static_cast<unsigned int>(Xlist.size()),
                                           rank, withIndexWhite);
  initCommon(Xlist, Ylist, signal, withIndexWhite, verbosity);
}

CCoregionalizedRegression::CCoregionalizedRegression(const vector<CMatrix>& Xlist,
                                                     const vector<CMatrix>& Ylist,
                                                     const vector<const CKern*>& spatials,
                                                     unsigned int rank,
                                                     bool withIndexWhite,
                                                     int verbosity)
  : numOutputs(0), pX(0), py(0), pkern(0), pnoise(0), model(0)
{
  CKern* signal = createCoregionalizedLcmKern(spatials,
                                              static_cast<unsigned int>(Xlist.size()),
                                              rank, withIndexWhite);
  initCommon(Xlist, Ylist, signal, withIndexWhite, verbosity);
}

CCoregionalizedRegression::~CCoregionalizedRegression()
{
  delete model;
  delete pnoise;
  delete pkern;
  delete py;
  delete pX;
}
