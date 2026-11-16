#include "CKern.h"
#include <cmath>
#include <sstream>
using namespace std;

static int fails = 0;

static void expect(bool cond, const string& msg)
{
  if(cond)
    cout << "OK: " << msg << endl;
  else
  {
    cout << "FAILURE: " << msg << endl;
    fails++;
  }
}

static void testIndexCast()
{
  expect(CKern::indexFromDouble(0.0, 3)==0, "index 0.0");
  expect(CKern::indexFromDouble(1.0, 3)==1, "index 1.0");
  expect(CKern::indexFromDouble(2.0, 3)==2, "index 2.0");
  expect(CKern::indexFromDouble(1.0 + 0.5*ndlutil::MATCHTOL, 3)==1, "index near-integer");
  bool threw = false;
  try { CKern::indexFromDouble(1.7, 3); }
  catch(...) { threw = true; }
  expect(threw, "non-integer 1.7 throws");
  threw = false;
  try { CKern::indexFromDouble(3.0, 3); }
  catch(...) { threw = true; }
  expect(threw, "out-of-range 3.0 throws");
  threw = false;
  try { CKern::indexFromDouble(-1.0, 3); }
  catch(...) { threw = true; }
  expect(threw, "negative index throws");
}

static void testActiveDimsHelpers()
{
  CRbfKern rbf(2);
  expect(rbf.isDefaultActiveDims(), "RBF default active dims");
  expect(rbf.getActiveDims().size()==2, "RBF activeDims size");
  CMatrix X(2, 2);
  X.setVal(0.0, 0, 0); X.setVal(0.0, 0, 1);
  X.setVal(1.0, 1, 0); X.setVal(2.0, 1, 1);
  // Default path uses dist2Row.
  expect(fabs(rbf.dist2Active(X, 0, X, 1) - 5.0)<ndlutil::MATCHTOL, "dist2Active default");

  vector<unsigned int> dims(2);
  dims[0] = 1; dims[1] = 0;
  rbf.setActiveDims(dims);
  expect(!rbf.isDefaultActiveDims(), "RBF non-default active dims");
  // Same columns as default, non-default order: still (0-2)^2+(0-1)^2 = 5.
  expect(fabs(rbf.dist2Active(X, 0, X, 1) - 5.0)<ndlutil::MATCHTOL, "dist2Active reordered");

  // Wide X with spatial activeDims {0} (ICM layout): ignore index column.
  CRbfKern space(1);
  vector<unsigned int> sd(1, 0);
  space.setActiveDims(sd);
  CMatrix Xw(2, 2);
  Xw.setVal(0.0, 0, 0); Xw.setVal(0.0, 0, 1);
  Xw.setVal(3.0, 1, 0); Xw.setVal(9.0, 1, 1);
  expect(fabs(space.dist2Active(Xw, 0, Xw, 1) - 9.0)<ndlutil::MATCHTOL, "dist2Active wide X");
}

static void testCoregAlone()
{
  CCoregionalizeKern kern(2, 1);
  CMatrix W(2, 1);
  W.setVal(1.0, 0, 0);
  W.setVal(2.0, 1, 0);
  CMatrix kappa(1, 2);
  kappa.setVal(0.5, 0);
  kappa.setVal(0.25, 1);
  kern.setW(W);
  kern.setKappa(kappa);
  // B = [[1.5, 2], [2, 4.25]]
  expect(fabs(kern.getB().getVal(0,0)-1.5)<ndlutil::MATCHTOL, "B00");
  expect(fabs(kern.getB().getVal(0,1)-2.0)<ndlutil::MATCHTOL, "B01");
  expect(fabs(kern.getB().getVal(1,1)-4.25)<ndlutil::MATCHTOL, "B11");
  expect(kern.getOutputDim()==2, "coreg outputDim");
  expect(kern.getRank()==1, "coreg rank");
  expect(fabs(kern.getWhite())<ndlutil::MATCHTOL, "coreg white");

  CMatrix X(3, 1);
  X.setVal(0.0, 0, 0);
  X.setVal(1.0, 1, 0);
  X.setVal(0.0, 2, 0);
  CMatrix K(3, 3);
  kern.compute(K, X);
  expect(fabs(K.getVal(0,0)-1.5)<ndlutil::MATCHTOL, "K00");
  expect(fabs(K.getVal(0,1)-2.0)<ndlutil::MATCHTOL, "K01");
  expect(fabs(K.getVal(1,1)-4.25)<ndlutil::MATCHTOL, "K11");
  expect(fabs(K.getVal(0,2)-1.5)<ndlutil::MATCHTOL, "K02 same index");

  // Column kappa path + variance rescale.
  CMatrix kappaCol(2, 1);
  kappaCol.setVal(0.5, 0, 0);
  kappaCol.setVal(0.25, 1, 0);
  kern.setKappa(kappaCol);
  expect(fabs(kern.getB().getVal(0,0)-1.5)<ndlutil::MATCHTOL, "B00 after col kappa");
  double var0 = kern.getVariance();
  expect(fabs(var0 - (1.5+4.25)/2.0)<ndlutil::MATCHTOL, "getVariance");
  kern.setVariance(2.0*var0);
  expect(fabs(kern.getVariance()-2.0*var0)<1e-12, "setVariance doubles mean diag(B)");
  // Early-return path when current variance is non-positive.
  CMatrix W0(2, 1); W0.zeros();
  CMatrix k0(1, 2); k0.zeros();
  kern.setW(W0);
  kern.setKappa(k0);
  kern.setVariance(1.0); // no-op
  expect(fabs(kern.getVariance())<ndlutil::MATCHTOL, "setVariance no-op at zero");
  // Restore usable W/kappa for gradient checks below.
  kern.setW(W);
  kern.setKappa(kappa);

  bool threw = false;
  try
  {
    CMatrix bad(1, 1);
    bad.setVal(1.0, 0);
    kern.setKappa(bad);
  }
  catch(...) { threw = true; }
  expect(threw, "bad kappa shape throws");

  // Gradients w.r.t. W/kappa.
  CMatrix cov(3, 3);
  cov.zeros();
  cov.setVal(1.0, 0, 0);
  cov.setVal(1.0, 1, 1);
  cov.setVal(1.0, 2, 2);
  cov.setVal(0.5, 0, 1); cov.setVal(0.5, 1, 0);
  cov.setVal(0.25, 0, 2); cov.setVal(0.25, 2, 0);
  CMatrix g(1, kern.getNumParams());
  kern.getGradParams(g, X, cov, false);
  expect(fabs(kern.getGradParam(0, X, cov) - g.getVal(0))<1e-12, "getGradParam matches getGradParams");
  CMatrix gReg(1, kern.getNumParams());
  kern.getGradParams(gReg, X, cov, true); // hits addPriorGrad
  CMatrix X2(2, 1);
  X2.setVal(0.0, 0, 0);
  X2.setVal(1.0, 1, 0);
  CMatrix cov12(3, 2);
  cov12.zeros();
  cov12.setVal(1.0, 0, 0);
  cov12.setVal(0.5, 1, 1);
  CMatrix g12(1, kern.getNumParams());
  kern.getGradParams(g12, X, X2, cov12, false);
  expect(fabs(kern.getGradParam(1, X, X2, cov12) - g12.getVal(1))<1e-12, "cross getGradParam");

  CMatrix gX(3, 1);
  gX.setVals(1.0);
  kern.getGradX(gX, X, 0, X, true); // addG: leave existing
  expect(fabs(gX.getVal(0,0)-1.0)<ndlutil::MATCHTOL, "coreg getGradX addG");
  kern.getGradX(gX, X, 0, X, false);
  expect(fabs(gX.getVal(0,0))<ndlutil::MATCHTOL, "coreg getGradX zeros");
  kern.getDiagGradX(gX, X, false);
  expect(fabs(gX.getVal(0,0))<ndlutil::MATCHTOL, "coreg getDiagGradX zeros");
  kern.getDiagGradX(gX, X, true);

  // Alternate constructors + clone.
  CCoregionalizeKern k2(1, 2, 1);
  expect(k2.getInputDim()==1 && k2.getOutputDim()==2, "inDim,outDim,rank ctor");
  CCoregionalizeKern k3(X, 2, 1);
  expect(k3.getOutputDim()==2, "X ctor");
  CKern* cloned = kern.clone();
  expect(cloned->getType()=="coregionalize", "clone type");
  expect(kern.equals(*cloned), "clone equals");
  delete cloned;
  ostringstream oss;
  kern.display(oss);
  expect(oss.str().find("coregionalize")!=string::npos || oss.str().size()>0, "display writes");
}

static void testIcm()
{
  // 1-D spatial + index column. Two outputs, unequal counts.
  CMatrix X(4, 2);
  // rows: (0.0,0), (1.0,0), (0.0,1), (2.0,1)
  X.setVal(0.0, 0, 0); X.setVal(0.0, 0, 1);
  X.setVal(1.0, 1, 0); X.setVal(0.0, 1, 1);
  X.setVal(0.0, 2, 0); X.setVal(1.0, 2, 1);
  X.setVal(2.0, 3, 0); X.setVal(1.0, 3, 1);

  CRbfKern* space = new CRbfKern(1);
  space->setInverseWidth(1.0);
  space->setVariance(1.0);
  vector<unsigned int> sd(1, 0);
  space->setActiveDims(sd);
  CCoregionalizeKern* B = new CCoregionalizeKern(2, 1);
  CMatrix W(2, 1);
  W.setVal(1.0, 0, 0);
  W.setVal(0.5, 1, 0);
  CMatrix kappa(1, 2);
  kappa.setVal(0.1, 0);
  kappa.setVal(0.2, 1);
  B->setW(W);
  B->setKappa(kappa);
  vector<unsigned int> id(1, 1);
  B->setActiveDims(id);
  CTensorKern tens(2);
  tens.addKern(space);
  tens.addKern(B);

  CMatrix K(4, 4);
  tens.compute(K, X);

  // Hand check K[0,1]: same output 0, rbf(0,1)=exp(-0.5*1)=exp(-0.5), B00=1.1
  double k01 = exp(-0.5)*1.1;
  expect(fabs(K.getVal(0,1)-k01)<1e-12, "ICM K01");
  // K[0,2]: different outputs, rbf(0,0)=1, B01=0.5
  expect(fabs(K.getVal(0,2)-0.5)<1e-12, "ICM K02");
  // GradX on index column from RBF path via tensor: spatial factor ignores col 1.
  CMatrix gX(4, 2);
  tens.getGradX(gX, X, 0, X);
  bool indexGradZero = true;
  for(unsigned int i=0; i<4; i++)
    if(fabs(gX.getVal(i, 1))>1e-14)
      indexGradZero = false;
  expect(indexGradZero, "ICM getGradX index column is zero");

  delete space;
  delete B;
}

static void testIcmLcmHelpers()
{
  CRbfKern spatial(1);
  spatial.setInverseWidth(1.0);
  spatial.setVariance(1.0);
  CTensorKern* icm = createIcmKern(spatial, 2, 1);
  expect(icm->getNumKerns()==2, "ICM helper two factors");
  expect(icm->getInputDim()==2, "ICM helper inputDim q+1");

  CMatrix X(3, 2);
  X.setVal(0.0, 0, 0); X.setVal(0.0, 0, 1);
  X.setVal(1.0, 1, 0); X.setVal(0.0, 1, 1);
  X.setVal(0.5, 2, 0); X.setVal(1.0, 2, 1);
  CMatrix K(3, 3);
  icm->compute(K, X);
  expect(K.getVal(0,0)>0.0, "ICM helper K00 positive");

  // Spatial already has activeDims: createIcmKern keeps them.
  CRbfKern spatial2(1);
  vector<unsigned int> sd(1, 0);
  spatial2.setActiveDims(sd);
  CTensorKern* icm2 = createIcmKern(spatial2, 2, 1);
  expect(icm2->getNumKerns()==2, "ICM helper with preset activeDims");
  delete icm2;
  delete icm;

  vector<const CKern*> parts;
  CRbfKern s1(1);
  CRbfKern s2(1);
  parts.push_back(&s1);
  parts.push_back(&s2);
  CCmpndKern* lcm = createLcmKern(parts, 2, 1);
  expect(lcm->getNumKerns()==2, "LCM two ICM terms");
  expect(lcm->getInputDim()==2, "LCM inputDim q+1");
  delete lcm;

  bool threw = false;
  try
  {
    vector<const CKern*> empty;
    createLcmKern(empty, 2, 1);
  }
  catch(...) { threw = true; }
  expect(threw, "empty LCM throws");
}

static void testRbfActiveDimsGradPaths()
{
  // Default activeDims: updateX cache + getGradParams(X,cov).
  CRbfKern rbf(1);
  rbf.setInverseWidth(1.0);
  rbf.setVariance(1.0);
  CMatrix X(3, 1);
  X.setVal(0.0, 0, 0);
  X.setVal(1.0, 1, 0);
  X.setVal(2.0, 2, 0);
  rbf.updateX(X);
  expect(rbf.isUpdateXused(), "updateX cache enabled for default dims");
  CMatrix cov(3, 3);
  cov.zeros();
  for(unsigned int i=0; i<3; i++)
    cov.setVal(1.0, i, i);
  cov.setVal(0.2, 0, 1); cov.setVal(0.2, 1, 0);
  cov.setSymmetric(true);
  CMatrix g(1, 2);
  rbf.getGradParams(g, X, cov, false);
  expect(g.getVal(1)!=0.0 || g.getVal(0)!=0.0, "RBF getGradParams with cache");

  // Non-default activeDims skips cache; force getGradParams else branch.
  CRbfKern space(1);
  vector<unsigned int> sd(1, 0);
  space.setActiveDims(sd);
  space.setInverseWidth(1.0);
  space.setVariance(1.0);
  CMatrix Xw(3, 2);
  for(unsigned int i=0; i<3; i++)
  {
    Xw.setVal(static_cast<double>(i), i, 0);
    Xw.setVal(0.0, i, 1);
  }
  space.updateX(Xw);
  expect(!space.isUpdateXused(), "updateX skipped for ICM layout");
  CMatrix gw(1, 2);
  space.getGradParams(gw, Xw, cov, false);
  CMatrix X2(2, 2);
  X2.setVal(0.0, 0, 0); X2.setVal(0.0, 0, 1);
  X2.setVal(1.5, 1, 0); X2.setVal(0.0, 1, 1);
  CMatrix cov12(3, 2);
  cov12.setVals(0.1);
  CMatrix g12(1, 2);
  space.getGradParams(g12, Xw, X2, cov12, false);
  CMatrix gX(3, 2);
  space.getGradX(gX, Xw, 0, Xw, false);
  expect(fabs(gX.getVal(0, 1))<1e-14, "RBF getGradX ignores inactive col");
  space.getGradX(gX, Xw, 0, Xw, true); // addG path
  // Copy ctor must keep a non-default activeDims permutation (inputDim 2).
  CRbfKern rbf2(2);
  vector<unsigned int> perm(2); perm[0]=1; perm[1]=0;
  rbf2.setActiveDims(perm);
  CRbfKern rbf2Copy(rbf2);
  expect(!rbf2Copy.isDefaultActiveDims(), "RBF copy preserves activeDims");
  expect(rbf2Copy.getActiveDims()[0]==1 && rbf2Copy.getActiveDims()[1]==0, "RBF copy activeDims order");
  CMatrix d(3, 1);
  space.diagCompute(d, Xw);
  expect(fabs(d.getVal(0)-1.0)<ndlutil::MATCHTOL, "RBF diagCompute");
}

static void testIndexWhite()
{
  CIndexWhiteKern iw(2);
  iw.setParam(0.3, 0);
  iw.setParam(0.7, 1);
  vector<unsigned int> id(1, 0);
  iw.setActiveDims(id);
  CMatrix X(3, 1);
  X.setVal(0.0, 0, 0);
  X.setVal(1.0, 1, 0);
  X.setVal(0.0, 2, 0);
  expect(fabs(iw.diagComputeElement(X, 0)-0.3)<ndlutil::MATCHTOL, "iw diag 0");
  expect(fabs(iw.diagComputeElement(X, 1)-0.7)<ndlutil::MATCHTOL, "iw diag 1");
  expect(fabs(iw.computeElement(X, 0, X, 1))<ndlutil::MATCHTOL, "iw off-diag 0");
  expect(fabs(iw.getWhite())<ndlutil::MATCHTOL, "iw white");
  expect(iw.getOutputDim()==2, "iw outputDim");

  CMatrix cov(3, 3);
  cov.zeros();
  cov.setVal(2.0, 0, 0);
  cov.setVal(5.0, 1, 1);
  cov.setVal(3.0, 2, 2);
  expect(fabs(iw.getGradParam(0, X, cov)-5.0)<ndlutil::MATCHTOL, "iw grad var0"); // rows 0 and 2
  expect(fabs(iw.getGradParam(1, X, cov)-5.0)<ndlutil::MATCHTOL, "iw grad var1");
  // Cross-set overload returns 0 by design.
  CMatrix X2(2, 1);
  X2.setVal(0.0, 0, 0);
  X2.setVal(1.0, 1, 0);
  CMatrix cov12(3, 2);
  cov12.setVals(1.0);
  expect(fabs(iw.getGradParam(0, X, X2, cov12))<ndlutil::MATCHTOL, "iw cross getGradParam is 0");

  expect(fabs(iw.getVariance()-(0.3+0.7)/2.0)<ndlutil::MATCHTOL, "iw getVariance");
  iw.setVariance(0.4);
  expect(fabs(iw.getParam(0)-0.4)<ndlutil::MATCHTOL, "iw setVariance");
  expect(fabs(iw.getParam(1)-0.4)<ndlutil::MATCHTOL, "iw setVariance both");

  CMatrix gX(3, 1);
  gX.setVals(2.0);
  iw.getGradX(gX, X, 0, X, true);
  expect(fabs(gX.getVal(0,0)-2.0)<ndlutil::MATCHTOL, "iw getGradX addG");
  iw.getGradX(gX, X, 0, X, false);
  expect(fabs(gX.getVal(0,0))<ndlutil::MATCHTOL, "iw getGradX zeros");
  iw.getDiagGradX(gX, X, false);
  expect(fabs(gX.getVal(0,0))<ndlutil::MATCHTOL, "iw getDiagGradX zeros");
  iw.getDiagGradX(gX, X, true);

  CIndexWhiteKern iw2(1, 2);
  expect(iw2.getInputDim()==1 && iw2.getOutputDim()==2, "iw inDim,outDim ctor");
  CIndexWhiteKern iw3(X, 2);
  expect(iw3.getOutputDim()==2, "iw X ctor");
  CKern* cloned = iw.clone();
  expect(cloned->getType()=="indexwhite", "iw clone type");
  // After setVariance, clone of original iw still has 0.4 params if cloned after;
  // rebuild a fresh one for equals.
  CIndexWhiteKern iwA(2);
  iwA.setParam(0.2, 0);
  iwA.setParam(0.8, 1);
  CKern* c2 = iwA.clone();
  expect(iwA.equals(*c2), "iw clone equals");
  delete c2;
  delete cloned;
}

static void testStreamRoundTrip()
{
  CCoregionalizeKern a(3, 2);
  CMatrix W(3, 2);
  for(unsigned int i=0; i<3; i++)
    for(unsigned int j=0; j<2; j++)
      W.setVal(0.1*(i+1)*(j+1), i, j);
  CMatrix kappa(1, 3);
  // Avoid 0.5: matrix streams write hex floats, and 0x1p-1 has no '.', so
  // CMatrix's atoi path would round-trip it as 0 (pre-existing stream quirk).
  kappa.setVal(0.4, 0); kappa.setVal(0.55, 1); kappa.setVal(0.6, 2);
  a.setW(W);
  a.setKappa(kappa);
  stringstream ss;
  a.toStream(ss);
  CKern* loaded = readKernFromStream(ss);
  expect(loaded->getType()=="coregionalize", "stream type");
  expect(a.equals(*loaded), "coregionalize stream round-trip");
  delete loaded;

  CIndexWhiteKern iw(2);
  iw.setParam(0.2, 0);
  iw.setParam(0.8, 1);
  stringstream ss2;
  iw.toStream(ss2);
  CKern* loaded2 = readKernFromStream(ss2);
  expect(loaded2->getType()=="indexwhite", "indexwhite stream type");
  expect(iw.equals(*loaded2), "indexwhite stream round-trip");
  delete loaded2;
}

static void testTensorRejectsIndexWhite()
{
  CTensorKern t(2);
  CRbfKern rbf(1);
  t.addKern(&rbf);
  CIndexWhiteKern iw(2);
  bool threw = false;
  try { t.addKern(&iw); }
  catch(...) { threw = true; }
  expect(threw, "tensor rejects indexwhite");
}

int main()
{
  try
  {
    testIndexCast();
    testActiveDimsHelpers();
    testCoregAlone();
    testIcm();
    testIcmLcmHelpers();
    testRbfActiveDimsGradPaths();
    testIndexWhite();
    testStreamRoundTrip();
    testTensorRejectsIndexWhite();
    cout << "Number of failures: " << fails << "." << endl;
    return fails>0 ? 1 : 0;
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
