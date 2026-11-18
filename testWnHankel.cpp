/* CIP-001C Phase 3: Hankel / spectralDensity gates for white-noise K_fu. */
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "CKern.h"
#include "ndlutil.h"
#include "ndlstrutil.h"

using namespace std;

namespace {

int gPass = 0;
int gFail = 0;

string ftoa(double x)
{
  ostringstream os;
  os.precision(8);
  os << scientific << x;
  return os.str();
}

void check(bool ok, const string& label)
{
  if(ok)
  {
    ++gPass;
    cout << "PASS: " << label << endl;
  }
  else
  {
    ++gFail;
    cout << "FAIL: " << label << endl;
  }
}

double relErr(double got, double want)
{
  return fabs(got - want) / max(1.0, fabs(want));
}

double rbfClosedForm(double r, double ell, double sigma, double s, unsigned int d)
{
  const double v = ell * ell + 2.0 * s * s;
  const double amp = sigma * pow(2.0 * sqrt(2.0) * ell * s / v, 0.5 * (double)d);
  return amp * exp(-r * r / v);
}

double maxRbfHankelRelErr(unsigned int d, double ell, double sigma, double s,
                          const vector<double>& rs, double* outMaxAbs = 0)
{
  CRbfKern kern(d);
  kern.setInitParam();
  kern.setLengthScale(ell);
  kern.setVariance(sigma * sigma);

  // Single centre at 0; probe points along e_1.
  CMatrix Z(1, d);
  Z.zeros();
  CMatrix X(rs.size(), d);
  X.zeros();
  for(unsigned int i = 0; i < rs.size(); i++)
    X.setVal(rs[i], i, 0);

  CMatrix Kfu;
  kern.wnCrossComputeHankel(Kfu, X, Z, s);

  double maxRel = 0.0;
  double maxAbs = 0.0;
  for(unsigned int i = 0; i < rs.size(); i++)
  {
    const double want = rbfClosedForm(rs[i], ell, sigma, s, d);
    const double got = Kfu.getVal(i, 0);
    maxRel = max(maxRel, relErr(got, want));
    maxAbs = max(maxAbs, fabs(got - want));
  }
  if(outMaxAbs)
    *outMaxAbs = maxAbs;
  return maxRel;
}

// Independent 1-D cosine IFT of F=√S ˆψ (same math as Hankel d=1, separate code).
double cosineIftF(const CStationaryKern& kern, double s, double r, unsigned int nRho)
{
  const unsigned int d = kern.getInputDim();
  auto evalF = [&](double rho) -> double
  {
    const double S = kern.spectralDensity(rho);
    if(!(S > 0.0))
      return 0.0;
    return sqrt(S) * pow(4.0 * M_PI * s * s, 0.25 * (double)d) * exp(-0.5 * s * s * rho * rho);
  };
  const double F0 = evalF(0.0);
  const double thr = 1e-20 * max(fabs(F0), 1e-300);
  double rhoHi = 8.0 / max(s, 1e-12);
  for(int k = 0; k < 50; k++)
  {
    if(evalF(rhoHi) < thr)
      break;
    rhoHi *= 1.2;
  }
  unsigned int nUse = nRho | 1; // odd
  const double dRho = rhoHi / (double)(nUse - 1);
  double acc = 0.0;
  for(unsigned int i = 0; i < nUse; i++)
  {
    const double rho = dRho * (double)i;
    const double yi = evalF(rho) * cos(rho * r);
    if(i == 0 || i + 1 == nUse)
      acc += yi;
    else if((i % 2) == 0)
      acc += 2.0 * yi;
    else
      acc += 4.0 * yi;
  }
  return acc * dRho / (3.0 * M_PI);
}

void gateRbf()
{
  cout << "=== RBF Hankel vs closed form ===" << endl;
  const double ells[] = {0.5, 1.0, 2.0};
  const double sigmas[] = {0.7, 1.2};
  const double ss[] = {0.4, 0.8, 1.5};
  const double rs[] = {0.0, 0.25, 0.5, 1.0, 2.0, 3.0};
  vector<double> rVec(rs, rs + sizeof(rs) / sizeof(rs[0]));

  double globalMaxRel = 0.0;
  for(unsigned int d = 1; d <= 2; d++)
  {
    for(unsigned int ie = 0; ie < 3; ie++)
      for(unsigned int isig = 0; isig < 2; isig++)
        for(unsigned int is = 0; is < 3; is++)
        {
          double maxAbs = 0.0;
          const double mrel = maxRbfHankelRelErr(d, ells[ie], sigmas[isig], ss[is], rVec, &maxAbs);
          globalMaxRel = max(globalMaxRel, mrel);
          const bool ok = mrel < 1e-6;
          check(ok, "RBF d=" + ndlstrutil::itoa(d)
                + " ell=" + ftoa(ells[ie])
                + " sig=" + ftoa(sigmas[isig])
                + " s=" + ftoa(ss[is])
                + " maxRel=" + ftoa(mrel)
                + " maxAbs=" + ftoa(maxAbs));
        }
  }
  cout << "RBF Hankel global max relative error: " << globalMaxRel << endl;
  // Gate: <1e-6; typical cells land near 1e-7 with Simpson / 4k ρ-nodes.
  check(globalMaxRel < 1e-6, "RBF global maxRel < 1e-6");
}

void gateMatern32Conv()
{
  cout << "=== Matérn-3/2 d=1 Hankel vs independent cosine IFT ===" << endl;
  CMatern32Kern kern(1);
  kern.setInitParam();
  kern.setParam(1.2, 0); // lengthScale
  kern.setParam(0.9, 1); // variance
  const double s = 0.55;
  const double xs[] = {0.0, 0.5, 1.0, 1.75};

  CMatrix Z(1, 1);
  Z.zeros();
  CMatrix X(4, 1);
  for(unsigned int i = 0; i < 4; i++)
    X.setVal(xs[i], i, 0);
  CMatrix Kfu;
  kern.wnCrossComputeHankel(Kfu, X, Z, s);

  double maxRel = 0.0;
  for(unsigned int i = 0; i < 4; i++)
  {
    // Brute-force spatial check: denser independent cosine path of √S ˆψ.
    const double want = cosineIftF(kern, s, xs[i], 8193);
    const double got = Kfu.getVal(i, 0);
    const double re = relErr(got, want);
    maxRel = max(maxRel, re);
    check(re < 1e-3, "Matern32 cosine x=" + ftoa(xs[i])
          + " got=" + ftoa(got)
          + " want=" + ftoa(want)
          + " rel=" + ftoa(re));
  }
  cout << "Matern32 cosine max relative error: " << maxRel << endl;
}

void gateExpConv()
{
  cout << "=== Exp d=1 Hankel vs cosine IFT + Matérn-½ spectral consistency ===" << endl;
  CExpKern kern(1);
  kern.setInitParam();
  // inverseWidth = 1/ℓ with ℓ=0.8 → inverseWidth = 1.25
  const double ell = 0.8;
  const double sigma = sqrt(1.1);
  kern.setParam(1.0 / ell, 0);
  kern.setParam(sigma * sigma, 1);
  const double s = 0.45;
  const double xs[] = {0.0, 0.4, 1.2};

  CMatrix Z(1, 1);
  Z.zeros();
  CMatrix X(3, 1);
  for(unsigned int i = 0; i < 3; i++)
    X.setVal(xs[i], i, 0);
  CMatrix Kfu;
  kern.wnCrossComputeHankel(Kfu, X, Z, s);

  double maxRelCos = 0.0;
  for(unsigned int i = 0; i < 3; i++)
  {
    const double wantCos = cosineIftF(kern, s, xs[i], 8193);
    const double got = Kfu.getVal(i, 0);
    const double reCos = relErr(got, wantCos);
    maxRelCos = max(maxRelCos, reCos);
    check(reCos < 1e-3, "Exp cosine x=" + ftoa(xs[i])
          + " got=" + ftoa(got)
          + " want=" + ftoa(wantCos)
          + " rel=" + ftoa(reCos));
  }
  cout << "Exp cosine max relative error: " << maxRelCos << endl;

  // Spectral consistency: Exp S(ρ) matches Matérn ν=1/2 GPML formula.
  // S = σ² * 2ℓ / (1 + ℓ² ρ²) in d=1.
  double maxSpecRel = 0.0;
  const double rhos[] = {0.0, 0.5, 1.0, 2.0, 5.0};
  for(unsigned int i = 0; i < 5; i++)
  {
    const double rho = rhos[i];
    const double want = sigma * sigma * 2.0 * ell / (1.0 + ell * ell * rho * rho);
    const double got = kern.spectralDensity(rho);
    const double re = relErr(got, want);
    maxSpecRel = max(maxSpecRel, re);
    check(re < 1e-12, "Exp S(rho) vs Matérn-1/2 formula rho=" + ftoa(rho)
          + " rel=" + ftoa(re));
  }
  cout << "Exp spectral max relative error: " << maxSpecRel << endl;

  // Smoke: Matérn-5/2 Hankel is finite and positive at r=0
  CMatern52Kern m52(1);
  m52.setInitParam();
  m52.setParam(1.0, 0);
  m52.setParam(1.0, 1);
  CMatrix K0;
  m52.wnCrossComputeHankel(K0, X, Z, 0.5);
  check(K0.getVal(0, 0) > 0.0 && std::isfinite(K0.getVal(0, 0)),
        "Matern52 Hankel Kfu(0) finite positive");
}

} // namespace

int main()
{
  cout << "CIP-001C Phase 3: testWnHankel" << endl;
  gateRbf();
  gateMatern32Conv();
  gateExpConv();
  cout << "Summary: PASS=" << gPass << " FAIL=" << gFail << endl;
  return (gFail == 0) ? 0 : 1;
}
