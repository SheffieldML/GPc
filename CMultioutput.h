/* CIP-0015 Phase A: GPy util.multioutput helpers + coregionalized CGp factory. */
#ifndef CMULTIOUTPUT_H
#define CMULTIOUTPUT_H

#include <vector>
#include "CKern.h"
#include "CGp.h"
#include "CNoise.h"

using namespace std;

// Half-open row range [start, stop), GPy slice spirit.
struct CRowSlice
{
  unsigned int start;
  unsigned int stop;
};

// Stack X_list / Y_list like GPy build_XY: vertical stack, append output index
// as last column of X (doubles 0..D-1). Y is one column (stacked targets).
// Throws if lists empty/mismatched, X cols differ, or Y is not N_v x 1.
void buildXY(const vector<CMatrix>& Xlist, const vector<CMatrix>& Ylist,
             CMatrix& Xout, CMatrix& Yout);

// Contiguous block slices from per-output row counts (GPy get_slices).
vector<CRowSlice> getSlices(const vector<unsigned int>& lengths);
vector<CRowSlice> getSlices(const vector<CMatrix>& Xlist);

// Contiguous runs of equal integer indices (GPy index_to_slices flattened
// to one slice list per output, in output order). indexCol is N x 1 doubles
// cast with CIP-000D rules (llround + MATCHTOL).
vector<vector<CRowSlice> > indexToSlices(const CMatrix& indexCol);

// Stamp / overwrite the last column of X with outputIndex (for X* prediction).
void stampOutputIndex(CMatrix& X, unsigned int outputIndex);

// Owned CCmpndKern: ICM (clone of spatial) + optional indexwhite.
// Caller deletes. Signal is createIcmKern; indexwhite variances default to 0.1.
CCmpndKern* createCoregionalizedKern(const CKern& spatial, unsigned int numOutputs,
                                     unsigned int rank = 1, bool withIndexWhite = true);

// Owned CCmpndKern: LCM + optional indexwhite. Caller deletes.
CCmpndKern* createCoregionalizedLcmKern(const vector<const CKern*>& spatials,
                                        unsigned int numOutputs, unsigned int rank = 1,
                                        bool withIndexWhite = true);

// Thin GPCoregionalizedRegression-style bundle. Owns stacked X/y, kernel,
// spherical Gaussian noise, and CGp (FTC). Per-output noise lives in indexwhite
// when enabled; likelihood.variance is fixed small so CIP-0027 does not fight it.
class CCoregionalizedRegression
{
 public:
  CCoregionalizedRegression(const vector<CMatrix>& Xlist, const vector<CMatrix>& Ylist,
                            const CKern& spatial, unsigned int rank = 1,
                            bool withIndexWhite = true, int verbosity = 0);
  CCoregionalizedRegression(const vector<CMatrix>& Xlist, const vector<CMatrix>& Ylist,
                            const vector<const CKern*>& spatials, unsigned int rank = 1,
                            bool withIndexWhite = true, int verbosity = 0);
  ~CCoregionalizedRegression();

  CCoregionalizedRegression(const CCoregionalizedRegression&) = delete;
  CCoregionalizedRegression& operator=(const CCoregionalizedRegression&) = delete;

  CGp& getModel() { return *model; }
  const CGp& getModel() const { return *model; }
  const CMatrix& getX() const { return *pX; }
  const CMatrix& getY() const { return *py; }
  CKern* getKern() { return pkern; }
  const CKern* getKern() const { return pkern; }
  unsigned int getNumOutputs() const { return numOutputs; }
  const vector<CRowSlice>& getSlices() const { return slices; }

 private:
  void initCommon(const vector<CMatrix>& Xlist, const vector<CMatrix>& Ylist,
                  CKern* signalKern, bool withIndexWhite, int verbosity);

  unsigned int numOutputs;
  vector<CRowSlice> slices;
  CMatrix* pX;
  CMatrix* py;
  CKern* pkern;
  CGaussianNoise* pnoise;
  CGp* model;
};

#endif
