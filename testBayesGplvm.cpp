/* CIP-0017: Bayesian GP-LVM ELBO, checkGradients, GPy fixture compare. */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "CBayesGplvm.h"
#include "ndlutil.h"

using namespace std;

namespace {
const unsigned int N = 6;
const unsigned int D = 3;
const unsigned int Q = 2;
const unsigned int M = 3;

// Looser than MATCHTOL: GPy vs C++ Cholesky / BLAS stack differences.
const double FIXTURE_TOL = 1e-6;

CMatrix makeY()
{
  CMatrix Y(N, D);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int d = 0; d < D; d++)
      Y.setVal(sin(0.3 * n + 0.7 * d) + 0.1 * n, n, d);
  return Y;
}

int failNear(double a, double b, double tol, const string& what)
{
  if(fabs(a - b) > tol)
  {
    cout << "FAILURE: " << what << " a=" << a << " b=" << b
         << " |diff|=" << fabs(a - b) << endl;
    return 1;
  }
  return 0;
}

int failNearMat(const CMatrix& A, const CMatrix& B, double tol, const string& what)
{
  if(A.getRows() != B.getRows() || A.getCols() != B.getCols())
  {
    cout << "FAILURE: " << what << " shape mismatch" << endl;
    return 1;
  }
  double maxDiff = 0.0;
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = 0; j < A.getCols(); j++)
    {
      const double d = fabs(A.getVal(i, j) - B.getVal(i, j));
      if(d > maxDiff)
        maxDiff = d;
    }
  if(maxDiff > tol)
  {
    cout << "FAILURE: " << what << " maxAbsDiff=" << maxDiff << endl;
    return 1;
  }
  return 0;
}

string nextToken(istream& in)
{
  string tok;
  while(in >> tok)
  {
    if(!tok.empty() && tok[0] == '#')
    {
      string rest;
      getline(in, rest);
      continue;
    }
    return tok;
  }
  return string();
}

void expectKey(istream& in, const string& key)
{
  const string tok = nextToken(in);
  if(tok != key)
    throw ndlexceptions::Error("fixture: expected key " + key + " got " + tok);
}

double readDouble(istream& in)
{
  const string tok = nextToken(in);
  return atof(tok.c_str());
}

unsigned int readUint(istream& in)
{
  return (unsigned int)readDouble(in);
}

void readMatrix(istream& in, const string& key, CMatrix& A)
{
  expectKey(in, key);
  const unsigned int rows = readUint(in);
  const unsigned int cols = readUint(in);
  A.resize(rows, cols);
  for(unsigned int i = 0; i < rows; i++)
    for(unsigned int j = 0; j < cols; j++)
      A.setVal(readDouble(in), i, j);
}

void readVector(istream& in, const string& key, vector<double>& v)
{
  expectKey(in, key);
  const unsigned int n = readUint(in);
  v.resize(n);
  for(unsigned int i = 0; i < n; i++)
    v[i] = readDouble(in);
}

int testCheckGradients()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  kern.setInverseWidth(1.0);
  kern.setVariance(1.0);
  CBayesGplvm model(&kern, Y, Q, M, 0);
  model.setBeta(2.0);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < Q; q++)
      model.getS().setVal(0.3, n, q);

  const unsigned int nParams = model.getOptNumParams();
  CMatrix analytic(1, nParams);
  CMatrix params(1, nParams);
  model.getOptParams(params);
  model.computeObjectiveGradParams(analytic);

  const double change = ndlutil::GRADCHANGE;
  double maxAbsDiff = 0.0;
  double maxRelDiff = 0.0;
  for(unsigned int j = 0; j < nParams; j++)
  {
    const double orig = params.getVal(j);
    params.setVal(orig + change, j);
    model.setOptParams(params);
    const double plus = model.computeObjectiveVal();
    params.setVal(orig - change, j);
    model.setOptParams(params);
    const double minus = model.computeObjectiveVal();
    params.setVal(orig, j);
    model.setOptParams(params);
    const double numerical = 0.5 * (plus - minus) / change;
    const double a = analytic.getVal(j);
    const double diff = fabs(a - numerical);
    if(diff > maxAbsDiff)
      maxAbsDiff = diff;
    const double scale = 1.0 + fabs(a) + fabs(numerical);
    const double rel = diff / scale;
    if(rel > maxRelDiff)
      maxRelDiff = rel;
  }

  const double absTol = 5e-3;
  const double relTol = 5e-3;
  if(maxAbsDiff > absTol && maxRelDiff > relTol)
  {
    cout << "FAILURE: checkGradients maxAbs=" << maxAbsDiff
         << " maxRel=" << maxRelDiff << endl;
    model.checkGradients();
    return 1;
  }
  cout << "OK: checkGradients (maxAbs=" << maxAbsDiff
       << " maxRel=" << maxRelDiff << ")" << endl;
  return 0;
}

int testElboFinite()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 0);
  const double L = model.logLikelihood();
  if(std::isnan(L) || std::isinf(L))
  {
    cout << "FAILURE: ELBO not finite: " << L << endl;
    return 1;
  }
  cout << "OK: ELBO finite (" << L << ")" << endl;
  return 0;
}

int testStreamRoundTrip()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 0);
  model.setBeta(1.7);
  vector<int> labs(N);
  for(unsigned int i = 0; i < N; i++)
    labs[i] = (int)(i % 3) + 1;
  model.setLabels(labs);

  stringstream ss;
  model.toStream(ss);
  CBayesGplvm model2(&kern, Y, Q, M, 0);
  model2.fromStream(ss);
  if(fabs(model.getBeta() - model2.getBeta()) > 1e-12)
  {
    cout << "FAILURE: beta stream mismatch" << endl;
    return 1;
  }
  if(model.getMu().maxAbsDiff(model2.getMu()) > 1e-12)
  {
    cout << "FAILURE: mu stream mismatch" << endl;
    return 1;
  }
  if(!model2.isLabels() || model2.getMinLabelVal() != 1 || model2.getMaxLabelVal() != 3)
  {
    cout << "FAILURE: label stream min/max" << endl;
    return 1;
  }
  if(model2.getLabel(0) != labs[0] || model2.getLabel(N - 1) != labs[N - 1])
  {
    cout << "FAILURE: label stream values" << endl;
    return 1;
  }
  cout << "OK: stream round-trip beta/mu/labels" << endl;
  return 0;
}

int testFileRoundTrip()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 0);
  model.setBeta(2.3);
  const string path = "/tmp/bayes_gplvm_roundtrip.model";
  model.toFile(path, "testBayesGplvm");
  CRbfKern kern2(Q);
  kern2.setInitParam();
  CBayesGplvm loaded(&kern2, Y, Q, M, 0);
  loaded.fromFile(path);
  if(fabs(loaded.getBeta() - 2.3) > 1e-12
     || loaded.getLatentDim() != Q
     || loaded.getNumActive() != M
     || loaded.getMu().maxAbsDiff(model.getMu()) > 1e-12)
  {
    cout << "FAILURE: file round-trip" << endl;
    return 1;
  }
  cout << "OK: file round-trip" << endl;
  return 0;
}

int testLabelsErrors()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 0);
  int fail = 0;
  try
  {
    vector<int> bad(N - 1, 1);
    model.setLabels(bad);
    cout << "FAILURE: setLabels length mismatch should throw" << endl;
    fail++;
  }
  catch(ndlexceptions::Error&)
  {
    // expected
  }
  try
  {
    (void)model.getMaxLabelVal();
    cout << "FAILURE: getMaxLabelVal without labels should throw" << endl;
    fail++;
  }
  catch(ndlexceptions::Error&)
  {
    // expected
  }
  try
  {
    (void)model.getMinLabelVal();
    cout << "FAILURE: getMinLabelVal without labels should throw" << endl;
    fail++;
  }
  catch(ndlexceptions::Error&)
  {
    // expected
  }
  if(fail == 0)
    cout << "OK: label error paths" << endl;
  return fail;
}

int testPosteriorMeanVar()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  kern.setInverseWidth(1.0);
  kern.setVariance(1.0);
  CBayesGplvm model(&kern, Y, Q, M, 0);
  model.setBeta(2.0);

  const unsigned int nStar = 4;
  CMatrix Xstar(nStar, Q);
  for(unsigned int i = 0; i < nStar; i++)
    for(unsigned int q = 0; q < Q; q++)
      Xstar.setVal(0.2 * (double)i - 0.3 * (double)q, i, q);

  CMatrix means(nStar, D);
  CMatrix vars(nStar, D);
  model.posteriorMeanVar(means, vars, Xstar);
  for(unsigned int i = 0; i < nStar; i++)
  {
    for(unsigned int d = 0; d < D; d++)
    {
      const double m = means.getVal(i, d);
      const double v = vars.getVal(i, d);
      if(std::isnan(m) || std::isinf(m) || std::isnan(v) || std::isinf(v) || v < 0.0)
      {
        cout << "FAILURE: posteriorMeanVar not finite/nonneg at ("
             << i << "," << d << ")" << endl;
        return 1;
      }
    }
    // Homoscedastic noise: variance identical across outputs.
    if(fabs(vars.getVal(i, 0) - vars.getVal(i, D - 1)) > 1e-12)
    {
      cout << "FAILURE: posterior var should match across outputs" << endl;
      return 1;
    }
  }
  cout << "OK: posteriorMeanVar finite" << endl;
  return 0;
}

int testOptimiseAndDisplay()
{
  CMatrix Y = makeY();
  CRbfKern kern(Q);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, Q, M, 0);
  const double L0 = model.logLikelihood();
  model.setVerbosity(0);
  model.optimise(5);
  const double L1 = model.logLikelihood();
  if(std::isnan(L1) || std::isinf(L1))
  {
    cout << "FAILURE: ELBO after optimise not finite" << endl;
    return 1;
  }
  // SCG maximises ELBO; allow tiny numerical wobble.
  if(L1 + 1e-6 < L0)
  {
    cout << "FAILURE: optimise decreased ELBO " << L0 << " -> " << L1 << endl;
    return 1;
  }
  stringstream ss;
  model.display(ss);
  if(ss.str().find("Bayesian GPLVM") == string::npos
     || ss.str().find("ELBO=") == string::npos)
  {
    cout << "FAILURE: display missing expected text" << endl;
    return 1;
  }
  cout << "OK: optimise (ELBO " << L0 << " -> " << L1 << ") + display" << endl;
  return 0;
}

int testArdElboAndGrads()
{
  // RBF-ARD path through CBayesGplvm (psi + ELBO + packed grads).
  // Full FD vs analytic for ARD input scales is covered in testPsi; here we
  // check the model wiring and that ELBO / grad vector stay finite.
  CMatrix Y = makeY();
  const unsigned int qArd = 3;
  CRbfardKern kern(qArd);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, qArd, M, 0);
  model.setBeta(1.5);
  for(unsigned int n = 0; n < N; n++)
    for(unsigned int q = 0; q < qArd; q++)
      model.getS().setVal(0.25, n, q);

  const double L = model.logLikelihood();
  if(std::isnan(L) || std::isinf(L))
  {
    cout << "FAILURE: ARD ELBO not finite: " << L << endl;
    return 1;
  }

  CMatrix g(1, model.getOptNumParams());
  const double L2 = model.logLikelihoodGradient(g);
  if(std::isnan(L2) || std::isinf(L2) || fabs(L2 - L) > 1e-8)
  {
    cout << "FAILURE: ARD logLikelihoodGradient inconsistent" << endl;
    return 1;
  }
  for(unsigned int j = 0; j < g.getCols(); j++)
  {
    const double v = g.getVal(j);
    if(std::isnan(v) || std::isinf(v))
    {
      cout << "FAILURE: ARD grad NaN/Inf at " << j << endl;
      return 1;
    }
  }
  cout << "OK: ARD ELBO+grad finite (ELBO=" << L << " nParams="
       << model.getOptNumParams() << ")" << endl;
  return 0;
}

int testPcaQExceedsD()
{
  // initXpca pads unused latent dims when q > D.
  CMatrix Y(N, 2);
  for(unsigned int n = 0; n < N; n++)
  {
    Y.setVal(0.4 * n, n, 0);
    Y.setVal(-0.2 * n, n, 1);
  }
  CRbfKern kern(4);
  kern.setInitParam();
  CBayesGplvm model(&kern, Y, 4, 2, 0);
  if(model.getLatentDim() != 4 || model.getMu().getCols() != 4)
  {
    cout << "FAILURE: q>D latent shape" << endl;
    return 1;
  }
  const double L = model.logLikelihood();
  if(std::isnan(L) || std::isinf(L))
  {
    cout << "FAILURE: q>D ELBO not finite" << endl;
    return 1;
  }
  cout << "OK: PCA init with q>D" << endl;
  return 0;
}

int testGpyFixture(const string& path)
{
  ifstream in(path.c_str());
  if(!in.is_open())
  {
    cout << "FAILURE: cannot open fixture " << path << endl;
    return 1;
  }

  expectKey(in, "N");
  const unsigned int n = readUint(in);
  expectKey(in, "D");
  const unsigned int d = readUint(in);
  expectKey(in, "Q");
  const unsigned int q = readUint(in);
  expectKey(in, "M");
  const unsigned int m = readUint(in);
  expectKey(in, "beta");
  const double beta = readDouble(in);
  expectKey(in, "inverseWidth");
  const double iw = readDouble(in);
  expectKey(in, "variance");
  const double var = readDouble(in);

  CMatrix Y, mu, S, Z, psi1Ref, psi2Ref;
  vector<double> psi0Ref, gNatRef;
  readMatrix(in, "Y", Y);
  readMatrix(in, "mu", mu);
  readMatrix(in, "S", S);
  readMatrix(in, "Z", Z);
  readVector(in, "psi0", psi0Ref);
  readMatrix(in, "psi1", psi1Ref);
  readMatrix(in, "psi2", psi2Ref);
  expectKey(in, "elbo");
  const double elboRef = readDouble(in);
  readVector(in, "g_natural", gNatRef);

  if(Y.getRows() != n || Y.getCols() != d || mu.getRows() != n || mu.getCols() != q
     || S.getRows() != n || S.getCols() != q || Z.getRows() != m || Z.getCols() != q)
  {
    cout << "FAILURE: fixture dimension mismatch" << endl;
    return 1;
  }

  CRbfKern kern(q);
  kern.setInitParam();
  kern.setInverseWidth(iw);
  kern.setVariance(var);

  CMatrix Psi1(n, m);
  CMatrix Psi2(m, m);
  const double psi0Sum = kern.psi0(mu, S);
  kern.psi1(Psi1, mu, S, Z);
  kern.psi2(Psi2, mu, S, Z);

  double psi0RefSum = 0.0;
  for(size_t i = 0; i < psi0Ref.size(); i++)
    psi0RefSum += psi0Ref[i];

  int fail = 0;
  fail += failNear(psi0Sum, psi0RefSum, FIXTURE_TOL, "psi0_sum vs GPy");
  fail += failNearMat(Psi1, psi1Ref, FIXTURE_TOL, "psi1 vs GPy");
  fail += failNearMat(Psi2, psi2Ref, FIXTURE_TOL, "psi2 vs GPy");

  CBayesGplvm model(&kern, Y, q, m, 0);
  model.setBeta(beta);
  model.getMu().deepCopy(mu);
  model.getS().deepCopy(S);
  model.getZ().deepCopy(Z);
  model.setKupToDate(false);

  const double elbo = model.logLikelihood();
  fail += failNear(elbo, elboRef, FIXTURE_TOL, "ELBO vs GPy");

  CMatrix gUnc(1, model.getOptNumParams());
  model.logLikelihoodGradient(gUnc);
  // Convert unconstrained grads back to natural (exp transform: gradfact = x).
  vector<double> gNat(model.getOptNumParams());
  unsigned int c = 0;
  gNat[c] = gUnc.getVal(c) / kern.getInverseWidth();
  c++;
  gNat[c] = gUnc.getVal(c) / kern.getVariance();
  c++;
  gNat[c] = gUnc.getVal(c) / model.getBeta();
  c++;
  for(; c < 3 + m * q + n * q; c++)
    gNat[c] = gUnc.getVal(c); // Z, mu
  for(unsigned int i = 0; i < n * q; i++, c++)
  {
    const unsigned int nn = i / q;
    const unsigned int qq = i % q;
    gNat[c] = gUnc.getVal(c) / model.getS().getVal(nn, qq);
  }

  if(gNatRef.size() != gNat.size())
  {
    cout << "FAILURE: g_natural length " << gNat.size()
         << " vs fixture " << gNatRef.size() << endl;
    return fail + 1;
  }
  double maxG = 0.0;
  unsigned int worst = 0;
  for(size_t i = 0; i < gNat.size(); i++)
  {
    const double diff = fabs(gNat[i] - gNatRef[i]);
    if(diff > maxG)
    {
      maxG = diff;
      worst = (unsigned int)i;
    }
  }
  if(maxG > FIXTURE_TOL)
  {
    cout << "FAILURE: g_natural vs GPy maxAbs=" << maxG
         << " at index " << worst
         << " (cpp=" << gNat[worst] << " gpy=" << gNatRef[worst] << ")" << endl;
    fail++;
  }

  if(fail == 0)
    cout << "OK: GPy fixture " << path
         << " (ELBO=" << elbo << " maxGradDiff=" << maxG << ")" << endl;
  return fail;
}

} // namespace

int main()
{
  int fail = 0;
  try
  {
    fail += testElboFinite();
    fail += testStreamRoundTrip();
    fail += testFileRoundTrip();
    fail += testLabelsErrors();
    fail += testPosteriorMeanVar();
    fail += testOptimiseAndDisplay();
    fail += testArdElboAndGrads();
    fail += testPcaQExceedsD();
    fail += testCheckGradients();
    fail += testGpyFixture("testdata/cip0017/bayes_gplvm_rbf_tiny.txt");
    cout << "Number of failures: " << fail << "." << endl;
    return fail > 0 ? 1 : 0;
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << err.getMessage() << endl;
    return 1;
  }
}
