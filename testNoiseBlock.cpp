/* CIP-003E Phase 1: block-oracle contract on CNoise.

   At blockSize 1 every default block oracle must equal the element-wise code it
   wraps, for every existing likelihood:
     expectedLogLikelihoodBlock  == expectedLogLikelihood (with the CSvgp bias selection)
     pointDerivsBlock            == ndlnoise::laplacePointDerivs
     cavityMomentsBlock          == getGradInputs on stored (mu, varSigma, target)
   plus the capability table, the capability/scalar-block guards, and a
   blockSize 2 stand-in that the scalar-only defaults must refuse.
   No MATLAB / matio fixtures. */
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlexceptions.h"

using namespace std;

static int fails = 0;

static void expect(bool cond, const string& msg)
{
  if(cond)
    cout << "PASS: " << msg << endl;
  else
  {
    cout << "FAIL: " << msg << endl;
    fails++;
  }
}

static bool close(double a, double b, double tol = 1e-12)
{
  return fabs(a - b) <= tol * (1.0 + fabs(a) + fabs(b));
}

template <class E>
static bool throwsError(void (*f)(void*), void* arg)
{
  try { f(arg); }
  catch(E&) { return true; }
  catch(...) { return false; }
  return false;
}

namespace {

const unsigned int N = 5;
const unsigned int D = 2;

struct Case
{
  string type;
  CMatrix Y;
  unique_ptr<CNoise> noise;
  unsigned int caps;
};

// Deterministic pseudo-data so the test is reproducible without <random> state.
double h(unsigned int i, unsigned int j, double scale)
{
  return scale * sin(1.7 * i + 2.3 * j + 0.4);
}

void fillTargets(const string& type, CMatrix& Y)
{
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < D; j++)
    {
      double v = h(i, j, 1.5);
      if(type == "probit")
        v = (v > 0.0) ? 1.0 : -1.0;
      else if(type == "ncnm")
        v = (i == 2) ? 0.0 : ((v > 0.0) ? 1.0 : -1.0);  // 0 = missing
      else if(type == "ordered")
        v = (double)((i + j) % 3);
      else if(type == "poisson")
        v = (double)((i + 2 * j) % 4);
      else if(type == "binomial")
        v = (double)((i + j) % 5);          // 0..4 successes of 8
      else if(type == "exponential" || type == "gamma" || type == "skew_exponential")
        v = 0.3 + fabs(v);
      Y.setVal(v, i, j);
    }
}

bool makeCase(const string& type, Case& c)
{
  c.type = type;
  c.Y.resize(N, D);
  fillTargets(type, c.Y);
  CMatrix* y = &c.Y;
  if(type == "gaussian")          c.noise.reset(new CGaussianNoise(y));
  else if(type == "probit")       c.noise.reset(new CProbitNoise(y));
  else if(type == "ncnm")         c.noise.reset(new CNcnmNoise(y));
  else if(type == "ordered")      c.noise.reset(new COrderedNoise(y, 3));
  else if(type == "skew_gaussian")     c.noise.reset(new CSkewGaussianNoise(y));
  else if(type == "skew_exponential")  c.noise.reset(new CSkewExponentialNoise(y));
  else if(type == "student_t")    c.noise.reset(new CStudentTNoise(y));
  else if(type == "poisson")      c.noise.reset(new CPoissonNoise(y));
  else if(type == "binomial")     c.noise.reset(new CBinomialNoise(y, 8));
  else if(type == "exponential")  c.noise.reset(new CExponentialNoise(y));
  else if(type == "gamma")        c.noise.reset(new CGammaNoise(y));
  else return false;
  // Distinct per-output biases where a setter exists (Gaussian). Probit and
  // ordered biases stay 0, but the poisoned ellBias below still proves the oracle
  // selects the bias itself.
  if(CGaussianNoise* g = dynamic_cast<CGaussianNoise*>(c.noise.get()))
    for(unsigned int j = 0; j < D; j++)
      g->setBiasVal(0.1 + 0.2 * j, j);
  for(unsigned int i = 0; i < N; i++)
    for(unsigned int j = 0; j < D; j++)
    {
      c.noise->setMu(h(i, j, 0.8) + 0.1, i, j);
      c.noise->setVarSigma(0.2 + 0.1 * fabs(h(j, i, 1.0)), i, j);
    }
  c.caps = c.noise->capabilities();
  return true;
}

// Stand-in coupling two outputs, to prove the scalar-only defaults refuse.
class CTwoBlockGaussian : public CGaussianNoise
{
 public:
  CTwoBlockGaussian(CMatrix* y) : CGaussianNoise(y) {}
  unsigned int blockSize() const { return 2; }
};

void checkOne(const string& type, unsigned int wantCaps)
{
  const string tag = " [" + type + "]";
  Case c;
  if(!makeCase(type, c))
  {
    expect(false, "unknown type" + tag);
    return;
  }
  CNoise& n = *c.noise;
  expect(n.blockSize() == 1, "blockSize is 1" + tag);
  expect(n.capabilities() == wantCaps, "capability set" + tag);

  // --- expected log-likelihood ---
  if(wantCaps & CNoise::CapExpectedLogLik)
  {
    bool ok = true;
    for(unsigned int i = 0; i < N && ok; i++)
      for(unsigned int j = 0; j < D && ok; j++)
      {
        const double y = n.getTarget(i, j), mu = n.getMu(i, j), var = n.getVarSigma(i, j);
        double gM1 = 0, gV1 = 0, gM2 = 0, gV2 = 0;
        // Reference: the exact sequence CSvgp::expectedLogLik performs today.
        if(CProbitNoise* pn = dynamic_cast<CProbitNoise*>(&n))
          pn->setExpectedLogLikBias(pn->getBiasVal(j));
        else if(COrderedNoise* on = dynamic_cast<COrderedNoise*>(&n))
          on->setExpectedLogLikBias(on->getBiasVal(j));
        const double e1 = n.expectedLogLikelihood(y, mu, var, gM1, gV1);
        // Poison the bias state, then let the oracle set it itself.
        if(CProbitNoise* pn = dynamic_cast<CProbitNoise*>(&n))
          pn->setExpectedLogLikBias(99.0);
        else if(COrderedNoise* on = dynamic_cast<COrderedNoise*>(&n))
          on->setExpectedLogLikBias(99.0);
        const double e2 = n.expectedLogLikelihoodBlock(&y, &mu, &var, j, &gM2, &gV2);
        ok = (e1 == e2) && (gM1 == gM2) && (gV1 == gV2);
      }
    expect(ok, "expectedLogLikelihoodBlock == element-wise (bitwise)" + tag);
  }
  else
  {
    double y = 1.0, mu = 0.0, var = 1.0, gm = 0.0, gv = 0.0;
    bool threw = false;
    try { n.expectedLogLikelihoodBlock(&y, &mu, &var, 0, &gm, &gv); }
    catch(ndlexceptions::NotImplementedError&) { threw = true; }
    expect(threw, "expectedLogLikelihoodBlock refuses unsupported type" + tag);
  }

  // --- Laplace point derivatives ---
  if(wantCaps & CNoise::CapPointDerivs)
  {
    bool ok = true;
    for(unsigned int i = 0; i < N && ok; i++)
      for(unsigned int j = 0; j < D && ok; j++)
      {
        const double y = n.getTarget(i, j), f = n.getMu(i, j);
        double lp = 0, d1 = 0, W = 0, dW = 0;
        ndlnoise::laplacePointDerivs(n, y, f, lp, d1, W, dW);
        double lp2 = 0, g2 = 0;
        CMatrix H, dH;
        n.pointDerivsBlock(&y, &f, j, &lp2, &g2, H, &dH);
        ok = (lp == lp2) && (d1 == g2) && H.getRows() == 1 && H.getCols() == 1
             && (W == H.getVal(0, 0)) && (dW == dH.getVal(0, 0));
      }
    expect(ok, "pointDerivsBlock == laplacePointDerivs (bitwise)" + tag);
  }
  else
  {
    double y = 1.0, f = 0.0, lp = 0.0, g = 0.0;
    CMatrix H;
    bool threw = false;
    try { n.pointDerivsBlock(&y, &f, 0, &lp, &g, H); }
    catch(ndlexceptions::Error&) { threw = true; }
    expect(threw, "pointDerivsBlock refuses unsupported type" + tag);
  }

  // --- cavity moments ---
  if(wantCaps & CNoise::CapCavityMoments)
  {
    bool ok = true;
    for(unsigned int i = 0; i < N && ok; i++)
      for(unsigned int j = 0; j < D && ok; j++)
      {
        double gmu = 0, gvs = 0;
        n.getGradInputs(gmu, gvs, i, j);
        const double y = n.getTarget(i, j), mu = n.getMu(i, j), var = n.getVarSigma(i, j);
        double dm = 0, dv = 0;
        n.cavityMomentsBlock(&y, &mu, &var, j, &dm, &dv);
        ok = (gmu == dm) && (gvs == dv);
      }
    expect(ok, "cavityMomentsBlock == getGradInputs (bitwise)" + tag);
  }
  else
  {
    double y = 1.0, mu = 0.0, var = 1.0, dm = 0.0, dv = 0.0;
    bool threw = false;
    try { n.cavityMomentsBlock(&y, &mu, &var, 0, &dm, &dv); }
    catch(ndlexceptions::NotImplementedError&) { threw = true; }
    expect(threw, "cavityMomentsBlock refuses type without exact cavity moments" + tag);
  }

  // --- guards ---
  {
    bool threwCap = false;
    try { n.requireCapability(CNoise::CapCavityMoments, "test"); }
    catch(ndlexceptions::NotImplementedError&) { threwCap = true; }
    expect(threwCap == !(wantCaps & CNoise::CapCavityMoments),
           "requireCapability agrees with capability set" + tag);
    bool threwScalar = false;
    try { n.requireScalarBlocks("test"); }
    catch(ndlexceptions::NotImplementedError&) { threwScalar = true; }
    expect(!threwScalar, "requireScalarBlocks accepts blockSize 1" + tag);
  }
}

}  // namespace

int main()
{
  const unsigned int P = CNoise::CapPredictive;
  const unsigned int E = CNoise::CapExpectedLogLik;
  const unsigned int L = CNoise::CapPointDerivs;
  const unsigned int C = CNoise::CapCavityMoments;
  try
  {
    checkOne("gaussian", P | E | C);
    checkOne("probit", P | E | L | C);
    checkOne("ncnm", P | C);
    checkOne("ordered", P | E | C);
    checkOne("skew_gaussian", P | E | L);
    checkOne("skew_exponential", P | E | L);
    checkOne("student_t", P | E | L);
    checkOne("poisson", P | E | L);
    checkOne("binomial", P | E | L);
    checkOne("exponential", P | E | L);
    checkOne("gamma", P | E | L);

    // Scalar-only defaults must refuse a coupled block (stand-in likelihood).
    CMatrix Y(N, D);
    fillTargets("gaussian", Y);
    CTwoBlockGaussian two(&Y);
    expect(two.blockSize() == 2, "stand-in reports blockSize 2");
    bool threw = false;
    try { two.requireScalarBlocks("EP"); }
    catch(ndlexceptions::NotImplementedError& e)
    {
      threw = string(e.getMessage()).find("couples 2") != string::npos;
    }
    expect(threw, "requireScalarBlocks refuses blockSize 2 with a clear message");
    double y[2] = {0.1, 0.2}, f[2] = {0.0, 0.0}, lp = 0.0, g[2] = {0, 0};
    CMatrix H;
    threw = false;
    try { two.pointDerivsBlock(y, f, 0, &lp, g, H); }
    catch(ndlexceptions::NotImplementedError&) { threw = true; }
    expect(threw, "scalar-only pointDerivsBlock default refuses blockSize 2");
    double m[2] = {0, 0}, v[2] = {1, 1}, pr[2];
    threw = false;
    try { two.predictBlock(m, v, 0, pr); }
    catch(ndlexceptions::NotImplementedError&) { threw = true; }
    expect(threw, "predictBlock default is NotImplemented");

    // ADF path (getNuG + updateSites): a block likelihood reduces to the scalar
    // path when its block factorises, bitwise.
    {
      CGaussianNoise plain(&Y);
      CTwoBlockGaussian blk(&Y);
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < D; j++)
        {
          const double mm = 0.3 * sin(1.0 + i + 2.0 * j), vv = 0.4 + 0.05 * i + 0.02 * j;
          plain.setMu(mm, i, j); blk.setMu(mm, i, j);
          plain.setVarSigma(vv, i, j); blk.setVarSigma(vv, i, j);
        }
      CMatrix gP(N, D), nP(N, D), gB(N, D), nB(N, D);
      CMatrix mP(N, D), bP(N, D), mB(N, D), bB(N, D);
      bool same = true;
      for(unsigned int i = 0; i < N; i++)
      {
        plain.getNuG(gP, nP, i); blk.getNuG(gB, nB, i);
        plain.updateSites(mP, bP, i, gP, nP, i); blk.updateSites(mB, bB, i, gB, nB, i);
        for(unsigned int j = 0; j < D; j++)
          same = same && gP.getVal(i, j) == gB.getVal(i, j) && nP.getVal(i, j) == nB.getVal(i, j)
                      && mP.getVal(i, j) == mB.getVal(i, j) && bP.getVal(i, j) == bB.getVal(i, j);
      }
      expect(same, "ADF getNuG/updateSites: block path == scalar path (bitwise)");
    }
  }
  catch(ndlexceptions::Error& e)
  {
    cout << "FAIL: exception: " << e.getMessage() << endl;
    fails++;
  }
  cout << "Number of failures: " << fails << endl;
  return fails > 0 ? 1 : 0;
}
