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
  bool threw = false;
  try { CKern::indexFromDouble(1.7, 3); }
  catch(...) { threw = true; }
  expect(threw, "non-integer 1.7 throws");
  threw = false;
  try { CKern::indexFromDouble(3.0, 3); }
  catch(...) { threw = true; }
  expect(threw, "out-of-range 3.0 throws");
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

  CMatrix cov(3, 3);
  cov.zeros();
  cov.setVal(2.0, 0, 0);
  cov.setVal(5.0, 1, 1);
  cov.setVal(3.0, 2, 2);
  expect(fabs(iw.getGradParam(0, X, cov)-5.0)<ndlutil::MATCHTOL, "iw grad var0"); // rows 0 and 2
  expect(fabs(iw.getGradParam(1, X, cov)-5.0)<ndlutil::MATCHTOL, "iw grad var1");
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
    testCoregAlone();
    testIcm();
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
