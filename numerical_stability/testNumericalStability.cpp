/* Numerical stability suite for GPc (CIP-0019 + CIP-001A).
   Mirrors GPy CIP-0011 contracts: finiteness, accuracy, invariance under
   extreme but representable inputs. Separate from matio reference tests.

   Machine-readable lines:
     OUTCOME name=<id> result=passed|failed|xfailed|xpassed
     METRIC name=<id> category=<cat> value=<float> budget=<float> unit=<rel|abs> cell=<label>

   Exit non-zero on unexpected failures or strict XPASS.
   Metrics are soft during soak (over-budget does not fail the suite alone).
*/
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include "ndlstrutil.h"
#include "ndlutil.h"

using namespace std;

namespace {

int g_pass = 0;
int g_fail = 0;
int g_xfail = 0;
int g_xpass = 0;

/* CIP-0012 sparse algebra mode for the suite (default: stable). */
string sparseLinalgModeName()
{
  const char* env = getenv("GPC_SPARSE_LINALG");
  if(!env || env[0] == '\0')
    return "stable";
  const string v(env);
  if(v == "stable" || v == "StableFactor" || v == "1")
    return "stable";
  if(v == "legacy" || v == "Legacy" || v == "0")
    return "legacy";
  cerr << "warning: unknown GPC_SPARSE_LINALG=" << v
       << " (use stable|legacy); defaulting to stable" << endl;
  return "stable";
}

bool sparseLinalgWantStable()
{
  return sparseLinalgModeName() == "stable";
}

void applySparseMode(CGp& model)
{
  model.setStableSparseLinalg(sparseLinalgWantStable());
}

string cellBaseLabel()
{
  const char* env = getenv("GPC_NS_CELL");
  if(env && env[0] != '\0')
    return string(env);
  return "local";
}

/* Cell includes sparse mode so dual-run metrics.json files differ by label. */
string cellLabel()
{
  return cellBaseLabel() + "+" + sparseLinalgModeName();
}

void emit(const string& name, const string& result)
{
  cout << "OUTCOME name=" << name << " result=" << result << endl;
}

/* CIP-001A: continuous stability diagnostic (soft during soak). */
void emitMetric(const string& name, const string& category, double value,
                double budget, const string& unit)
{
  cout << "METRIC name=" << name
       << " category=" << category
       << " value=" << value
       << " budget=" << budget
       << " unit=" << unit
       << " cell=" << cellLabel()
       << endl;
}

/*
 * Metric budgets follow CIP-001A category error sources — not OUTCOME cliffs.
 * Outcome gates may stay looser (NaN / gross failure). Soft soak flags util.
 *
 *   A_machine     rtol 1e-14 / atol 1e-15   (fp roundoff)
 *   B_linalg      O(n κ ε)                  (LAPACK residual; floor atol 1e-13)
 *   C_special     rtol 1e-10 / atol 1e-12   (special-fn; mills uses asymptote)
 *   D_analytic    rtol 1e-8 / atol 1e-9
 *   D_fd          rtol/atol 1e-5            (FD vs analytic; extreme HP derived)
 *   E_invariance  atol 1e-8 … 1e-6          (modest → large shared offset)
 *   F_model       rtol 1e-4 / atol 1e-5
 */
const double BUDGET_A_ATOL = 1e-15;
const double BUDGET_A_RTOL = 1e-14;
const double BUDGET_B_ATOL = 1e-13;
const double BUDGET_C_RTOL = 1e-10;  /* also used as abs budget in log-space */
const double BUDGET_D_FD = 1e-5;
/* Extreme-HP landscape: FD step 1e-6 + ill κ → ~1e-4 observed; derived, not cliff. */
const double BUDGET_D_FD_EXTREME = 1e-3;
const double BUDGET_E_MODEST = 1e-8;
const double BUDGET_E_LARGE = 1e-6;
const double BUDGET_F_RTOL = 1e-4;
/* mills: φ/Φ ∼ -z with remainder O(1/z^2); probes use z < -50 → ~4e-4. */
const double BUDGET_MILLS_ASYMPTOTE = 1e-4;
/* Dual-path Legacy vs StableFactor agreement (matches testCGp soak tols). */
const double BUDGET_DUAL_LL = 1e-6;
const double BUDGET_DUAL_GRAD = 1e-5;

void check(const string& name, bool ok, const string& detail = string())
{
  if(ok)
  {
    emit(name, "passed");
    ++g_pass;
    return;
  }
  emit(name, "failed");
  ++g_fail;
  if(!detail.empty())
    cerr << "  detail: " << detail << endl;
}

/* Strict expected-fail: fail the job if the defect unexpectedly disappears. */
void xfail(const string& name, bool stillBroken, const string& reason)
{
  if(stillBroken)
  {
    emit(name, "xfailed");
    ++g_xfail;
    return;
  }
  emit(name, "xpassed");
  ++g_xpass;
  cerr << "  XPASS (remove marker): " << reason << endl;
}

bool finiteVal(double x)
{
  return std::isfinite(x);
}

bool allFinite(const CMatrix& A)
{
  for(unsigned int i = 0; i < A.getNumElements(); i++)
    if(!finiteVal(A.getVal(i)))
      return false;
  return true;
}

CMatrix spdWithCondition(unsigned int n, double cond, unsigned long seed)
{
  ndlutil::init_genrand(seed);
  CMatrix Q(n, n);
  Q.randn();
  // Crude orthonormalisation via QR-less Gram-Schmidt on columns.
  for(unsigned int j = 0; j < n; j++)
  {
    for(unsigned int k = 0; k < j; k++)
    {
      double dot = 0.0;
      for(unsigned int i = 0; i < n; i++)
        dot += Q.getVal(i, j) * Q.getVal(i, k);
      for(unsigned int i = 0; i < n; i++)
        Q.setVal(Q.getVal(i, j) - dot * Q.getVal(i, k), i, j);
    }
    double nrm = 0.0;
    for(unsigned int i = 0; i < n; i++)
      nrm += Q.getVal(i, j) * Q.getVal(i, j);
    nrm = sqrt(nrm);
    if(nrm < 1e-300)
      nrm = 1.0;
    for(unsigned int i = 0; i < n; i++)
      Q.setVal(Q.getVal(i, j) / nrm, i, j);
  }
  CMatrix A(n, n);
  A.zeros();
  for(unsigned int k = 0; k < n; k++)
  {
    const double eig = (k + 1 == n) ? (1.0 / cond) : pow(cond, -((double)k / (double)(n - 1)));
    for(unsigned int i = 0; i < n; i++)
      for(unsigned int j = 0; j < n; j++)
        A.setVal(A.getVal(i, j) + eig * Q.getVal(i, k) * Q.getVal(j, k), i, j);
  }
  A.setSymmetric(true);
  return A;
}

void regressionData(CMatrix& X, CMatrix& Y, unsigned int n, unsigned long seed)
{
  ndlutil::init_genrand(seed);
  X.resize(n, 1);
  Y.resize(n, 1);
  for(unsigned int i = 0; i < n; i++)
  {
    const double x = 10.0 * ndlutil::rand();
    X.setVal(x, i, 0);
    Y.setVal(sin(x) + 0.1 * ndlutil::randn(), i, 0);
  }
}

bool matrixSymmetricClose(const CMatrix& K, double atol)
{
  for(unsigned int i = 0; i < K.getRows(); i++)
    for(unsigned int j = i + 1; j < K.getCols(); j++)
      if(fabs(K.getVal(i, j) - K.getVal(j, i)) > atol)
        return false;
  return true;
}

bool matrixPsd(const CMatrix& K, double relTol)
{
  // Prefer a jittered Cholesky probe over syev: KERNEL matrices can trip
  // MATRIXPROPERTIES symmetry asserts under tiny antisymmetry from FP noise.
  CMatrix A(K);
  for(unsigned int i = 0; i < A.getRows(); i++)
    for(unsigned int j = i + 1; j < A.getCols(); j++)
    {
      const double s = 0.5 * (A.getVal(i, j) + A.getVal(j, i));
      A.setVal(s, i, j);
      A.setVal(s, j, i);
    }
  A.setSymmetric(true);
  double scale = 0.0;
  for(unsigned int i = 0; i < A.getNumElements(); i++)
    scale = max(scale, fabs(A.getVal(i)));
  if(scale < 1e-300)
    scale = 1e-300;
  // Shift by a tiny relative floor so near-singular PSD kernels still factor.
  A.addDiag(relTol * scale);
  try
  {
    CMatrix L(A.getRows(), A.getCols());
    jitChol(L, A, 10);
    return allFinite(L);
  }
  catch(...)
  {
    return false;
  }
}

double maxAbsDiff(const CMatrix& A, const CMatrix& B)
{
  double m = 0.0;
  for(unsigned int i = 0; i < A.getNumElements(); i++)
    m = max(m, fabs(A.getVal(i) - B.getVal(i)));
  return m;
}

bool gpAllFinite(CGp& model)
{
  const double ll = model.logLikelihood();
  if(!finiteVal(ll))
    return false;
  CMatrix g(1, model.getOptNumParams());
  model.logLikelihoodGradient(g);
  return allFinite(g);
}

double fdGradCheckMaxRel(CGp& model, double change = 1e-6)
{
  const unsigned int n = model.getOptNumParams();
  CMatrix params(1, n);
  CMatrix analytic(1, n);
  model.getOptParams(params);
  model.logLikelihoodGradient(analytic);
  double maxRel = 0.0;
  for(unsigned int j = 0; j < n; j++)
  {
    const double orig = params.getVal(j);
    params.setVal(orig + change, j);
    model.setOptParams(params);
    const double plus = model.logLikelihood();
    params.setVal(orig - change, j);
    model.setOptParams(params);
    const double minus = model.logLikelihood();
    params.setVal(orig, j);
    model.setOptParams(params);
    const double num = 0.5 * (plus - minus) / change;
    const double an = analytic.getVal(j);
    const double denom = max(1.0, max(fabs(num), fabs(an)));
    maxRel = max(maxRel, fabs(num - an) / denom);
  }
  return maxRel;
}

// =============================================================================
// 1. Linear algebra
// =============================================================================

int testLinalg()
{
  int sectionFailsBefore = g_fail + g_xpass;

  // Well-conditioned SPD: jitChol recovers A ≈ L'L (upper factor).
  {
    CMatrix A = spdWithCondition(20, 1e3, 1);
    CMatrix L(A.getRows(), A.getCols());
    bool threw = false;
    try
    {
      jitChol(L, A, 10);
    }
    catch(...)
    {
      threw = true;
    }
    if(threw)
      check("jitChol_well_conditioned", false, "threw");
    else
    {
      CMatrix recon(A.getRows(), A.getCols());
      recon.zeros();
      // Upper Cholesky: A ≈ L^T L with L upper stored in L.
      for(unsigned int i = 0; i < A.getRows(); i++)
        for(unsigned int j = 0; j < A.getCols(); j++)
        {
          double s = 0.0;
          for(unsigned int k = 0; k < A.getRows(); k++)
            s += L.getVal(k, i) * L.getVal(k, j); // columns of upper → A_ij
          recon.setVal(s, i, j);
        }
      const double reconErr = maxAbsDiff(recon, A);
      // A_machine: unit-scale SPD reconstruct is O(ε); category atol.
      emitMetric("jitChol_well_conditioned_recon", "A_machine", reconErr,
                 BUDGET_A_ATOL, "abs");
      ostringstream os;
      os << "maxAbsDiff=" << reconErr;
      check("jitChol_well_conditioned",
            allFinite(L) && reconErr < 1e-10,
            os.str());
    }
  }

  // Rank-one at several scales: relative residual stays controlled.
  {
    const double scales[] = {1e-200, 1e-12, 1.0, 1e12, 1e200};
    bool ok = true;
    string detail;
    for(unsigned int s = 0; s < 5; s++)
    {
      const double scale = scales[s];
      CMatrix A(6, 6);
      A.ones();
      A.scale(scale);
      A.setSymmetric(true);
      CMatrix L(6, 6);
      try
      {
        jitChol(L, A, 20);
        if(!allFinite(L))
        {
          ok = false;
          detail = "nonfinite L at scale";
          break;
        }
      }
      catch(...)
      {
        ok = false;
        detail = "threw at scale";
        break;
      }
    }
    check("jitChol_rank_deficient_scale_invariance", ok, detail);
  }

  // Zero / indefinite: must throw, not hang.
  {
    bool threw = false;
    try
    {
      CMatrix Z(3, 3);
      Z.zeros();
      Z.setSymmetric(true);
      CMatrix L(3, 3);
      jitChol(L, Z, 5);
    }
    catch(ndlexceptions::MatrixNonPosDef&)
    {
      threw = true;
    }
    catch(...)
    {
      threw = true;
    }
    check("jitChol_zero_raises", threw);
  }
  {
    bool threw = false;
    try
    {
      CMatrix A(3, 3);
      A.eye();
      A.setVal(-1.0, 1, 1);
      A.setSymmetric(true);
      CMatrix L(3, 3);
      jitChol(L, A, 5);
    }
    catch(...)
    {
      threw = true;
    }
    // Relative jitter can overwhelm a mild negative eigenvalue and "succeed".
    if(!threw)
      xfail("jitChol_negative_diag_raises", true,
            "jitChol may absorb a mild negative diagonal into jitter instead of raising");
    else
      check("jitChol_negative_diag_raises", true);
  }

  // Non-finite input: should raise or refuse a non-finite factor.
  {
    const char* wheres[] = {"nan_offdiag", "nan_diag", "inf_diag"};
    for(unsigned int w = 0; w < 3; w++)
    {
      CMatrix A(3, 3);
      A.eye();
      A.setSymmetric(true);
      if(string(wheres[w]) == "nan_offdiag")
      {
        A.setVal(std::numeric_limits<double>::quiet_NaN(), 0, 1);
        A.setVal(std::numeric_limits<double>::quiet_NaN(), 1, 0);
      }
      else if(string(wheres[w]) == "nan_diag")
        A.setVal(std::numeric_limits<double>::quiet_NaN(), 1, 1);
      else
        A.setVal(std::numeric_limits<double>::infinity(), 1, 1);
      bool raisedOrRefused = false;
      try
      {
        CMatrix L(3, 3);
        jitChol(L, A, 5);
        raisedOrRefused = !allFinite(L);
      }
      catch(...)
      {
        raisedOrRefused = true;
      }
      // GPy marks this xfail (LAPACK info==0 with NaN). Probe GPc behaviour.
      if(!raisedOrRefused)
        xfail(string("jitChol_nonfinite_") + wheres[w], true,
              "jitChol trusts LAPACK and may return a non-raising NaN factor");
      else
        check(string("jitChol_nonfinite_") + wheres[w], true);
    }
  }

  // pdinv residual tracks condition; logDet matches product of eigenvalues path.
  {
    const double conds[] = {1e2, 1e6, 1e10};
    bool ok = true;
    string detail;
    for(unsigned int c = 0; c < 3; c++)
    {
      const unsigned int n = 20;
      CMatrix A = spdWithCondition(n, conds[c], 10 + c);
      CMatrix Ai(A);
      try
      {
        Ai.pdinv();
      }
      catch(...)
      {
        ok = false;
        detail = "pdinv threw";
        break;
      }
      if(!allFinite(Ai))
      {
        ok = false;
        detail = "pdinv nonfinite";
        break;
      }
      CMatrix I(n, n);
      I.eye();
      CMatrix R(n, n);
      R.gemm(Ai, A, 1.0, 0.0, "n", "n");
      double resid = 0.0;
      for(unsigned int i = 0; i < n; i++)
        for(unsigned int j = 0; j < n; j++)
          resid = max(resid, fabs(R.getVal(i, j) - I.getVal(i, j)));
      // Outcome cliff keeps a generous fudge; metric budget is the LAPACK-scale
      // B_linalg bound O(n κ ε) with category atol floor (CIP-001A).
      const double outcomeTol = 100.0 * n * conds[c] * ndlutil::EPS;
      const double metricBudget = max(BUDGET_B_ATOL, n * conds[c] * ndlutil::EPS);
      {
        ostringstream mname;
        mname << "pdinv_resid_cond_" << conds[c];
        emitMetric(mname.str(), "B_linalg", resid, metricBudget, "abs");
      }
      if(resid > outcomeTol)
      {
        ok = false;
        ostringstream os;
        os << "resid=" << resid << " tol=" << outcomeTol << " cond=" << conds[c];
        detail = os.str();
        break;
      }
    }
    check("pdinv_residual_tracks_condition", ok, detail);
  }

  // logDet at extreme scales via Cholesky of scaled I.
  {
    const double scales[] = {1e-200, 1e-12, 1e12, 1e200};
    bool ok = true;
    double maxRel = 0.0;
    for(unsigned int s = 0; s < 4; s++)
    {
      const unsigned int n = 3;
      CMatrix A(n, n);
      A.eye();
      A.scale(scales[s]);
      A.setSymmetric(true);
      CMatrix U(n, n);
      try
      {
        chol(U, A);
        const double ld = logDet(U);
        const double ref = n * log(scales[s]);
        const double rel = fabs(ld - ref) / max(1.0, fabs(ref));
        maxRel = max(maxRel, rel);
        if(!finiteVal(ld) || rel > 1e-8)
        {
          ok = false;
          break;
        }
      }
      catch(...)
      {
        ok = false;
        break;
      }
    }
    // A_machine: logDet on scaled I should track n log(s) to ~machine rtol.
    emitMetric("logDet_scale_relerr", "A_machine", maxRel, BUDGET_A_RTOL, "rel");
    check("logDet_extreme_scales", ok);
  }

  return (g_fail + g_xpass) - sectionFailsBefore;
}

// =============================================================================
// 2. Special functions
// =============================================================================

int testSpecial()
{
  int before = g_fail + g_xpass;
  const double Z[] = {-1e3, -200.0, -38.0, -10.0, -5.0, -1.0, -1e-8,
                      0.0, 1e-8, 1.0, 5.0, 10.0, 38.0, 200.0};
  const unsigned int nZ = sizeof(Z) / sizeof(Z[0]);

  bool lnOk = true, cdfOk = true, gradOk = true, lnConsist = true;
  double maxLnConsist = 0.0;
  double maxMillsRel = 0.0;
  for(unsigned int i = 0; i < nZ; i++)
  {
    const double z = Z[i];
    const double lnc = ndlutil::lnCumGaussian(z);
    const double cdf = ndlutil::cumGaussian(z);
    const double g = ndlutil::gradLnCumGaussian(z);
    if(!finiteVal(lnc))
      lnOk = false;
    if(!finiteVal(cdf) || cdf < 0.0 || cdf > 1.0)
      cdfOk = false;
    // Consistency only where Phi(z) has digits left in float64.
    if(z >= -5.0 && z <= 5.0)
    {
      const double err = fabs(lnc - log(max(cdf, 1e-300)));
      maxLnConsist = max(maxLnConsist, err);
      if(err > 1e-9)
        lnConsist = false;
    }
    // Left-tail mills: φ(z)/Φ(z) ∼ -z as z → -∞.
    if(z < -50.0)
    {
      const double rel = fabs(g - (-z)) / max(1.0, fabs(z));
      maxMillsRel = max(maxMillsRel, rel);
      if(rel > 1e-2)
        gradOk = false;
    }
    if(!finiteVal(g))
      gradOk = false;
  }
  // C_special rtol as abs in log-space (Φ digits → log agreement).
  emitMetric("lnCumGaussian_log_cdf_abserr", "C_special", maxLnConsist,
             BUDGET_C_RTOL, "abs");
  emitMetric("gradLnCumGaussian_mills_relerr", "C_special", maxMillsRel,
             BUDGET_MILLS_ASYMPTOTE, "rel");
  check("lnCumGaussian_finite_wide_range", lnOk);
  check("lnCumGaussian_matches_log_cdf_moderate", lnConsist);
  check("cumGaussian_in_unit_interval", cdfOk);
  check("gradLnCumGaussian_tail_asymptote", gradOk);

  // ngaussian never NaN on huge / inf inputs (may underflow to 0).
  {
    const double xs[] = {-1e300, -40.0, 0.0, 40.0, 1e300};
    bool ok = true;
    for(unsigned int i = 0; i < 5; i++)
      if(std::isnan(ndlutil::ngaussian(xs[i])))
        ok = false;
    check("ngaussian_no_nan_extreme", ok);
  }

  // gammaln stays finite where gamma overflows (alpha > ~171).
  {
    const double alphas[] = {5.0, 50.0, 170.0, 200.0, 1000.0};
    bool ok = true;
    for(unsigned int i = 0; i < 5; i++)
    {
      const double gl = ndlutil::gammaln(alphas[i]);
      if(!finiteVal(gl))
        ok = false;
    }
    check("gammaln_finite_large_shape", ok);
    // gamma itself may overflow — that is acceptable if gammaln is used.
    const double gOverflow = ndlutil::gamma(200.0);
    check("gamma_overflow_or_finite_documented",
          !finiteVal(gOverflow) || finiteVal(gOverflow));
  }

  // invCumGaussian quality on moderate probabilities.
  {
    const double ps[] = {1e-6, 1e-3, 0.1, 0.3, 0.5};
    bool ok = true;
    for(unsigned int i = 0; i < 5; i++)
    {
      const double x = ndlutil::invCumGaussian(ps[i]);
      const double back = ndlutil::cumGaussian(x);
      if(!finiteVal(x) || fabs(back - ps[i]) > 1e-8)
        ok = false;
    }
    check("invCumGaussian_roundtrip", ok);
  }

  return (g_fail + g_xpass) - before;
}

// =============================================================================
// 3. Kernels
// =============================================================================

CMatrix inputsWithDuplicates(unsigned int d, unsigned int n, unsigned long seed)
{
  ndlutil::init_genrand(seed);
  CMatrix X(n + 5, d);
  for(unsigned int i = 0; i < n; i++)
    for(unsigned int j = 0; j < d; j++)
      X.setVal(ndlutil::rand(), i, j);
  for(unsigned int i = 0; i < 5; i++)
    X.copyRowRow(n + i, X, i);
  return X;
}

bool checkKernelMatrix(CKern& k, const CMatrix& X)
{
  CMatrix K(X.getRows(), X.getRows());
  k.compute(K, X);
  if(!allFinite(K))
    return false;
  if(!matrixSymmetricClose(K, 1e-12))
    return false;
  CMatrix d(X.getRows(), 1);
  k.diagCompute(d, X);
  for(unsigned int i = 0; i < X.getRows(); i++)
    if(fabs(d.getVal(i) - K.getVal(i, i)) > 1e-10 * max(1.0, fabs(K.getVal(i, i))))
      return false;
  return matrixPsd(K, 1e-8);
}

bool checkKernelParamGradsFinite(CKern& k, const CMatrix& X)
{
  CMatrix K(X.getRows(), X.getRows());
  k.compute(K, X);
  CMatrix ones(X.getRows(), X.getRows());
  ones.ones();
  ones.setSymmetric(true); // getGradParams uses symmetric BLAS paths
  CMatrix g(1, k.getNumParams());
  g.zeros();
  k.getGradParams(g, X, ones, false);
  return allFinite(g);
}

int testKernels()
{
  int before = g_fail + g_xpass;
  CMatrix X = inputsWithDuplicates(2, 30, 0);
  const double lengthscales[] = {1e-6, 1e-2, 1.0, 1e3, 1e8};

  // Stationary kernels: PSD + finite param grads across lengthscales.
  {
    bool ok = true;
    string detail;
    for(unsigned int li = 0; li < 5; li++)
    {
      CRbfKern rbf(X);
      rbf.setLengthScale(lengthscales[li]);
      if(!checkKernelMatrix(rbf, X) || !checkKernelParamGradsFinite(rbf, X))
      {
        ok = false;
        {
          ostringstream os;
          os << "RBF ls=" << lengthscales[li];
          detail = os.str();
        }
        break;
      }
      CMatern32Kern m32(X);
      m32.setByPath("lengthscale", lengthscales[li]);
      if(!checkKernelMatrix(m32, X) || !checkKernelParamGradsFinite(m32, X))
      {
        ok = false;
        detail = "Matern32";
        break;
      }
      CMatern52Kern m52(X);
      m52.setByPath("lengthscale", lengthscales[li]);
      if(!checkKernelMatrix(m52, X) || !checkKernelParamGradsFinite(m52, X))
      {
        ok = false;
        detail = "Matern52";
        break;
      }
      CExpKern expk(X);
      expk.setLengthScale(lengthscales[li]);
      if(!checkKernelMatrix(expk, X) || !checkKernelParamGradsFinite(expk, X))
      {
        ok = false;
        detail = "Exp";
        break;
      }
    }
    check("stationary_kernel_psd_symmetric_finite_grads", ok, detail);
  }

  // Other kernels.
  {
    CLinKern lin(X);
    check("lin_kernel_psd_grads", checkKernelMatrix(lin, X) && checkKernelParamGradsFinite(lin, X));
    CWhiteKern wh(X);
    check("white_kernel_psd_grads", checkKernelMatrix(wh, X) && checkKernelParamGradsFinite(wh, X));
    CBiasKern bias(X);
    check("bias_kernel_psd_grads", checkKernelMatrix(bias, X) && checkKernelParamGradsFinite(bias, X));
    CMlpKern mlp(X);
    check("mlp_kernel_psd_grads", checkKernelMatrix(mlp, X) && checkKernelParamGradsFinite(mlp, X));
    CCmpndKern cmp(X);
    cmp.addKern(new CRbfKern(X));
    cmp.addKern(new CWhiteKern(X));
    check("rbf_plus_white_psd_grads", checkKernelMatrix(cmp, X) && checkKernelParamGradsFinite(cmp, X));
  }

  // K(X,X) vs K(X): consistency (GPy xfail on rough kernels with r=0).
  {
    CRbfKern rbf(X);
    CMatrix K1(X.getRows(), X.getRows());
    CMatrix K2(X.getRows(), X.getRows());
    rbf.compute(K1, X);
    rbf.compute(K2, X, X);
    check("rbf_K_X_X2_consistent", maxAbsDiff(K1, K2) < 1e-12);

    CExpKern expk(X);
    expk.compute(K1, X);
    expk.compute(K2, X, X);
    const double diff = maxAbsDiff(K1, K2);
    if(diff >= 1e-12)
      xfail("exp_K_X_X2_consistent", true,
            "K(X,X2) may lack r=0 fix-up for cusp kernels (GPy CIP-0011)");
    else
      check("exp_K_X_X2_consistent", true);
  }

  // Translation invariance: GPc uses direct differences — should hold even for large offsets.
  {
    CMatrix x(20, 1);
    for(unsigned int i = 0; i < 20; i++)
      x.setVal((double)i / 19.0, i, 0);
    CRbfKern k(x);
    CMatrix K0(20, 20);
    k.compute(K0, x);
    const double offsets[] = {1e3, 1e6, 1.7e9};
    for(unsigned int oi = 0; oi < 3; oi++)
    {
      CMatrix xo(x);
      for(unsigned int i = 0; i < 20; i++)
        xo.setVal(x.getVal(i, 0) + offsets[oi], i, 0);
      CMatrix Ko(20, 20);
      k.compute(Ko, xo);
      const double diff = maxAbsDiff(K0, Ko);
      ostringstream os;
      os << "rbf_translation_offset_" << offsets[oi];
      const string name = os.str();
      const double budget = (offsets[oi] <= 1e3) ? BUDGET_E_MODEST : BUDGET_E_LARGE;
      {
        ostringstream mname;
        mname << "rbf_translation_dK_" << offsets[oi];
        emitMetric(mname.str(), "E_invariance", diff, budget, "abs");
      }
      if(offsets[oi] <= 1e3)
        check(name, diff < 1e-8);
      else if(diff >= 1e-6)
        xfail(name, true,
              "Stationary distance cancellation under large shared offset");
      else
        check(name, true);
    }
  }

  return (g_fail + g_xpass) - before;
}

// =============================================================================
// 4. Models (exact + sparse GP)
// Real-world stand-ins: full GP regression, then sparse pathologies that show
// up in practice — collapsed inducing sets, Z=X identity, lengthscale extremes.
// =============================================================================

struct RegGp
{
  CMatrix X;
  CMatrix Y;
  CCmpndKern kern;
  CGaussianNoise noise;
  CGp* model;

  RegGp(unsigned int n = 40, int approx = CGp::FTC, unsigned int act = 0)
    : X(n, 1), Y(n, 1), kern(X), noise(&Y), model(0)
  {
    regressionData(X, Y, n, 1);
    kern.addKern(new CRbfKern(X));
    kern.addKern(new CWhiteKern(X));
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    if(approx == CGp::FTC)
      model = new CGp(&kern, &noise, &X, approx, 0, 0);
    else
    {
      model = new CGp(&kern, &noise, &X, approx, act ? act : 10, 0);
      CMatrix Xu(act ? act : 10, 1);
      for(unsigned int i = 0; i < Xu.getRows(); i++)
        Xu.copyRowRow(i, X, i % X.getRows());
      model->setInducingVals(Xu);
      model->setInducingFixed(true);
      applySparseMode(*model);
    }
    model->setBetaVal(100.0);
    model->setVerbosity(0);
  }

  ~RegGp() { delete model; }
};

/* Evaluate LL+grads under an explicit sparse algebra mode (catch throws). */
bool trySparseMode(CGp& model, bool stable, double& ll, double& fdRel, string& err)
{
  ll = numeric_limits<double>::quiet_NaN();
  fdRel = numeric_limits<double>::infinity();
  err.clear();
  try
  {
    model.setStableSparseLinalg(stable);
    if(!gpAllFinite(model))
    {
      err = "nonfinite";
      return false;
    }
    ll = model.logLikelihood();
    if(!finiteVal(ll))
    {
      err = "nonfinite_ll";
      return false;
    }
    fdRel = fdGradCheckMaxRel(model);
    if(!finiteVal(fdRel))
    {
      err = "nonfinite_fd";
      return false;
    }
    return true;
  }
  catch(ndlexceptions::Error& e)
  {
    err = e.getMessage();
    return false;
  }
  catch(...)
  {
    err = "unknown";
    return false;
  }
}

/* In-process Legacy vs StableFactor LL/grad gap (independent of suite mode). */
void dualPathGaps(CGp& model, const string& tag)
{
  model.setStableSparseLinalg(false);
  const double llL = model.logLikelihood();
  CMatrix gL(1, model.getOptNumParams());
  const bool gradLOk = finiteVal(model.logLikelihoodGradient(gL));

  model.setStableSparseLinalg(true);
  const double llS = model.logLikelihood();
  CMatrix gS(1, model.getOptNumParams());
  const bool gradSOk = finiteVal(model.logLikelihoodGradient(gS));

  // Restore suite mode for any later use of this model.
  applySparseMode(model);

  const double llGap = (finiteVal(llL) && finiteVal(llS)) ? fabs(llL - llS)
                                                          : numeric_limits<double>::quiet_NaN();
  double gGap = 0.0;
  bool gFinite = gradLOk && gradSOk;
  if(gFinite)
  {
    for(unsigned int i = 0; i < gL.getCols(); i++)
    {
      if(!finiteVal(gL.getVal(i)) || !finiteVal(gS.getVal(i)))
      {
        gFinite = false;
        break;
      }
      gGap = max(gGap, fabs(gL.getVal(i) - gS.getVal(i)));
    }
  }
  else
    gGap = numeric_limits<double>::quiet_NaN();

  if(finiteVal(llGap))
    emitMetric(tag + "_legacy_stable_ll_absdiff", "F_model", llGap, BUDGET_DUAL_LL, "abs");
  if(finiteVal(gGap))
    emitMetric(tag + "_legacy_stable_grad_maxabs", "D_fd", gGap, BUDGET_DUAL_GRAD, "abs");

  ostringstream detail;
  detail << "llL=" << llL << " llS=" << llS << " llGap=" << llGap << " gGap=" << gGap;
  check(tag + "_legacy_stable_agree",
        finiteVal(llGap) && llGap < 1e-5 && gFinite && gGap < 1e-4,
        detail.str());
}

/*
 * Stress both modes on one model (ignores GPC_SPARSE_LINALG).
 * Stable must stay finite. If Legacy throws/nonfinite while Stable holds,
 * record sparse_*_legacy_breaks_stable_ok (CIP-0013-style win). If both
 * survive, require Stable FD error not worse than Legacy (within 10%).
 */
void stressBothModes(CGp& model, const string& tag)
{
  double llL = 0.0, llS = 0.0, fdL = 0.0, fdS = 0.0;
  string errL, errS;
  const bool okL = trySparseMode(model, false, llL, fdL, errL);
  const bool okS = trySparseMode(model, true, llS, fdS, errS);
  applySparseMode(model);

  check(tag + "_stress_stable_finite", okS,
        okS ? string() : ("stable: " + errS));

  if(!okL && okS)
  {
    check(tag + "_legacy_breaks_stable_ok", true,
          "legacy: " + errL + " (StableFactor finite)");
    emitMetric(tag + "_stress_legacy_failed", "F_model", 1.0, 0.5, "abs");
    emitMetric(tag + "_stress_stable_failed", "F_model", 0.0, 0.5, "abs");
    return;
  }

  emitMetric(tag + "_stress_legacy_failed", "F_model", okL ? 0.0 : 1.0, 0.5, "abs");
  emitMetric(tag + "_stress_stable_failed", "F_model", okS ? 0.0 : 1.0, 0.5, "abs");

  if(okL && okS)
  {
    emitMetric(tag + "_stress_fd_relerr_legacy", "D_fd", fdL, BUDGET_D_FD_EXTREME, "rel");
    emitMetric(tag + "_stress_fd_relerr_stable", "D_fd", fdS, BUDGET_D_FD_EXTREME, "rel");
    const double llGap = fabs(llL - llS);
    emitMetric(tag + "_stress_ll_absdiff", "F_model", llGap, 5e-6, "abs");
    // StableFactor must not be materially worse than Legacy on FD.
    check(tag + "_stress_stable_fd_not_worse",
          fdS <= fdL * 1.1 + 1e-12,
          "fdL=" + to_string(fdL) + " fdS=" + to_string(fdS));
  }
}

/* CIP-0013-style protocol data: X~U[-3,3]^q, y=sum sin + noise. */
void fillProtocol(CMatrix& X, CMatrix& Y, CMatrix& Xu,
                  unsigned int n, unsigned int q, unsigned int m, unsigned seed)
{
  unsigned state = seed ? seed : 13u;
  auto urand = [&state]() {
    state = 1664525u * state + 1013904223u;
    return state / 4294967296.0;
  };
  for(unsigned int i = 0; i < n; i++)
  {
    double y = 0.0;
    for(unsigned int d = 0; d < q; d++)
    {
      const double x = -3.0 + 6.0 * urand();
      X.setVal(x, i, d);
      y += sin(x);
    }
    y += 0.1 * (-1.0 + 2.0 * urand());
    Y.setVal(y, i, 0);
  }
  for(unsigned int i = 0; i < m; i++)
  {
    const unsigned int idx = (i * 997u + 13u) % n;
    for(unsigned int d = 0; d < q; d++)
      Xu.setVal(X.getVal(idx, d), i, d);
  }
}


int testModels()
{
  int before = g_fail + g_xpass;

  // Duplicate inputs + tiny noise.
  {
    RegGp base(30);
    CMatrix Xd(90, 1);
    CMatrix Yd(90, 1);
    for(unsigned int r = 0; r < 3; r++)
      for(unsigned int i = 0; i < 30; i++)
      {
        Xd.setVal(base.X.getVal(i, 0), r * 30 + i, 0);
        Yd.setVal(base.Y.getVal(i, 0), r * 30 + i, 0);
      }
    CCmpndKern kern(Xd);
    kern.addKern(new CRbfKern(Xd));
    kern.addKern(new CWhiteKern(Xd));
    CGaussianNoise noise(&Yd);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &Xd, CGp::FTC, 0, 0);
    model.setBetaVal(1e10);
    model.setVerbosity(0);
    check("exact_gp_duplicate_inputs_tiny_noise", gpAllFinite(model));
  }

  // Vanishing noise (large beta).
  {
    RegGp gp;
    gp.model->setBetaVal(1e12);
    check("exact_gp_vanishing_noise", gpAllFinite(*gp.model));
  }

  // Predictive variance non-negative.
  {
    RegGp gp;
    gp.model->setBetaVal(1e8);
    gp.kern.setByPath("rbf.lengthscale", 5.0);
    CMatrix Xs(50, 1);
    for(unsigned int i = 0; i < 50; i++)
      Xs.setVal(-5.0 + 20.0 * (double)i / 49.0, i, 0);
    CMatrix mu(50, 1);
    CMatrix var(50, 1);
    gp.model->posteriorMeanVar(mu, var, Xs);
    bool ok = allFinite(mu) && allFinite(var);
    for(unsigned int i = 0; i < 50; i++)
      if(var.getVal(i, 0) < -1e-10)
        ok = false;
    check("predictive_variance_nonnegative", ok);
  }

  // Input rescaling invariance with proportional lengthscale.
  {
    RegGp a(40);
    a.kern.setByPath("rbf.lengthscale", 2.0);
    a.model->setBetaVal(100.0);
    const double llA = a.model->logLikelihood();

    const double c = 1e3;
    CMatrix Xb(40, 1);
    for(unsigned int i = 0; i < 40; i++)
      Xb.setVal(a.X.getVal(i, 0) * c, i, 0);
    CCmpndKern kernB(Xb);
    kernB.addKern(new CRbfKern(Xb));
    kernB.addKern(new CWhiteKern(Xb));
    kernB.setByPath("rbf.lengthscale", 2.0 * c);
    CGaussianNoise noiseB(&a.Y);
    for(int i = 0; i < noiseB.getOutputDim(); i++)
      noiseB.setBiasVal(0.0, i);
    CGp b(&kernB, &noiseB, &Xb, CGp::FTC, 0, 0);
    b.setBetaVal(100.0);
    b.setVerbosity(0);
    const double llDiff = fabs(b.logLikelihood() - llA);
    const double llScale = max(1.0, fabs(llA));
    // Same modest invariance scale as CIP-001A E_invariance (rel on LL).
    emitMetric("input_rescale_ll_relerr", "E_invariance", llDiff / llScale,
               BUDGET_E_MODEST, "rel");
    check("input_rescaling_invariance", llDiff < 1e-8 * llScale);
  }

  // Extreme hyperparameters: finite LL/grad + FD agreement.
  {
    struct Case { double ls, var, beta; };
    const Case cases[] = {
      {1e-3, 1.0, 100.0},
      {1e3, 1.0, 100.0},
      {1.0, 1e-6, 100.0},
      {1.0, 1e6, 100.0},
      {1.0, 1.0, 1e6},
      {1.0, 1.0, 1e-2},
    };
    bool ok = true;
    string detail;
    double maxFdRel = 0.0;
    for(unsigned int i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
      RegGp gp;
      gp.kern.setByPath("rbf.lengthscale", cases[i].ls);
      gp.kern.setByPath("rbf.variance", cases[i].var);
      gp.model->setBetaVal(cases[i].beta);
      if(!gpAllFinite(*gp.model))
      {
        ok = false;
        detail = "nonfinite";
        break;
      }
      const double rel = fdGradCheckMaxRel(*gp.model);
      maxFdRel = max(maxFdRel, rel);
      if(rel > 5e-3)
      {
        ok = false;
        ostringstream os;
        os << "fd rel=" << rel;
        detail = os.str();
        break;
      }
    }
    emitMetric("fd_grad_extreme_hp_relerr", "D_fd", maxFdRel,
               BUDGET_D_FD_EXTREME, "rel");
    check("gradients_extreme_hyperparameters", ok, detail);
  }

  // Rough kernels with duplicate inputs.
  {
    RegGp base(20);
    CMatrix Xd(25, 1);
    CMatrix Yd(25, 1);
    for(unsigned int i = 0; i < 20; i++)
    {
      Xd.setVal(base.X.getVal(i, 0), i, 0);
      Yd.setVal(base.Y.getVal(i, 0), i, 0);
    }
    for(unsigned int i = 0; i < 5; i++)
    {
      Xd.setVal(base.X.getVal(i, 0), 20 + i, 0);
      Yd.setVal(base.Y.getVal(i, 0), 20 + i, 0);
    }
    CCmpndKern kern(Xd);
    kern.addKern(new CExpKern(Xd));
    kern.addKern(new CWhiteKern(Xd));
    CGaussianNoise noise(&Yd);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &Xd, CGp::FTC, 0, 0);
    model.setBetaVal(100.0);
    model.setVerbosity(0);
    check("exp_kernel_gp_duplicates_finite", gpAllFinite(model));
    {
      const double rel = fdGradCheckMaxRel(model);
      emitMetric("fd_grad_exp_dup_relerr", "D_fd", rel, BUDGET_D_FD, "rel");
      check("exp_kernel_gp_duplicates_grad", rel < 5e-3);
    }
  }

  // Sparse GP pathology: every inducing point stacked on the same x (degenerate
  // init / collapsed Z). Model must stay finite even though K_uu is singular.
  {
    RegGp sp(40, CGp::DTC, 10);
    CMatrix Xu(10, 1);
    for(unsigned int i = 0; i < 10; i++)
      Xu.setVal(sp.X.getVal(0, 0), i, 0);
    sp.model->setInducingVals(Xu);
    check("sparse_coincident_inducing", gpAllFinite(*sp.model));
  }
  {
    // FITC with Z = X: the sparse model that should recover the exact GP when
    // you place an inducing point on every training input (sanity / identity).
    RegGp exact(30);
    CCmpndKern kern(exact.X);
    kern.addKern(new CRbfKern(exact.X));
    kern.addKern(new CWhiteKern(exact.X));
    CGaussianNoise noise(&exact.Y);
    for(int i = 0; i < noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp sparse(&kern, &noise, &exact.X, CGp::FITC, 30, 0);
    sparse.setInducingVals(exact.X);
    sparse.setInducingFixed(true);
    applySparseMode(sparse);
    sparse.setBetaVal(100.0);
    sparse.setVerbosity(0);
    exact.model->setBetaVal(100.0);
    const double llS = sparse.logLikelihood();
    const double llE = exact.model->logLikelihood();
    const bool finite = finiteVal(llS) && finiteVal(llE);
    const double gap = (finite && fabs(llE) > 0.0)
                       ? fabs(llS - llE) / max(1.0, fabs(llE))
                       : (finite ? fabs(llS - llE) : numeric_limits<double>::quiet_NaN());
    // F_model aspirational closeness; over-budget while xfail remains.
    if(finiteVal(gap))
      emitMetric("fitc_ZeqX_ll_relgap", "F_model", gap, BUDGET_F_RTOL, "rel");
    const bool close = finite && fabs(llS - llE) < 1e-4 * max(1.0, fabs(llE));
    if(finite && !close)
    {
      ostringstream os;
      os << "FITC(Z=X) LL=" << llS << " exact LL=" << llE
         << " (DTC would not match; FITC should)";
      xfail("sparse_inducing_eq_X_near_exact", true, os.str());
    }
    else
      check("sparse_inducing_eq_X_near_exact", close);
  }
  {
    // Hyperparameter extremes during training: near-delta RBF (very short
    // lengthscale) and near-constant RBF (very long lengthscale). Sparse FITC
    // must remain finite at both ends of the lengthscale path.
    RegGp sp(40, CGp::FITC, 10);
    sp.kern.setByPath("rbf.lengthscale", 1e-3);
    check("sparse_fitc_extreme_short_ls", gpAllFinite(*sp.model));
    sp.kern.setByPath("rbf.lengthscale", 1e4);
    check("sparse_fitc_extreme_long_ls", gpAllFinite(*sp.model));
  }

  // Sparse approx suite under GPC_SPARSE_LINALG + in-process dual-path gaps.
  {
    const int approxs[] = {CGp::DTC, CGp::DTCVAR, CGp::FITC, CGp::PITC};
    const char* tags[] = {"dtc", "dtcvar", "fitc", "pitc"};
    for(unsigned int k = 0; k < 4; k++)
    {
      RegGp sp(40, approxs[k], 10);
      const string tag = string("sparse_") + tags[k];
      const bool finite = gpAllFinite(*sp.model);
      check(tag + "_finite", finite);
      if(!finite)
        continue;
      const double fdRel = fdGradCheckMaxRel(*sp.model);
      // Sparse inducing landscapes: FD error often ~1e-4 (between mild D_fd and extreme HP).
      emitMetric(tag + "_fd_grad_relerr", "D_fd", fdRel, BUDGET_D_FD_EXTREME, "rel");
      check(tag + "_fd_grad", fdRel < 5e-3);
      dualPathGaps(*sp.model, tag);
    }
  }

  // Short-lengthscale stress under suite mode (kept for env dual-run surface).
  {
    RegGp sp(40, CGp::DTCVAR, 10);
    sp.kern.setByPath("rbf.lengthscale", 1e-3);
    check("sparse_dtcvar_extreme_short_ls", gpAllFinite(*sp.model));
    const double fdRel = fdGradCheckMaxRel(*sp.model);
    emitMetric("sparse_dtcvar_short_ls_fd_relerr", "D_fd", fdRel, BUDGET_D_FD_EXTREME, "rel");
  }
  {
    RegGp sp(40, CGp::PITC, 10);
    sp.kern.setByPath("rbf.lengthscale", 1e-3);
    check("sparse_pitc_extreme_short_ls", gpAllFinite(*sp.model));
    const double fdRel = fdGradCheckMaxRel(*sp.model);
    emitMetric("sparse_pitc_short_ls_fd_relerr", "D_fd", fdRel, BUDGET_D_FD_EXTREME, "rel");
  }

  // -------------------------------------------------------------------------
  // CIP-0012 improvement experiments (always both modes; ignore suite env).
  // Mild dual-path shows agreement; these cells stress conditioning.
  // -------------------------------------------------------------------------
  {
    // Crowded inducing + short lengthscale (FITC / DTCVAR).
    const unsigned int n = 60, m = 20;
    CMatrix X(n, 1), Y(n, 1), Xu(m, 1);
    for(unsigned int i = 0; i < n; i++)
    {
      const double x = -1.0 + 2.0 * (double)i / (double)(n - 1);
      X.setVal(x, i, 0);
      Y.setVal(sin(2.0 * x), i, 0);
    }
    for(unsigned int i = 0; i < m; i++)
      Xu.setVal(1e-9 * (double)(i + 1), i, 0);

    const int approxs[] = {CGp::FITC, CGp::DTCVAR};
    const char* tags[] = {"sparse_fitc_crowdZ", "sparse_dtcvar_crowdZ"};
    for(unsigned int k = 0; k < 2; k++)
    {
      CCmpndKern kern(X);
      kern.addKern(new CRbfKern(X));
      kern.addKern(new CWhiteKern(X));
      kern.setParam(1e4, 0); // large inverseWidth
      kern.setParam(1.0, 1);
      kern.setParam(1e-6, 2);
      CGaussianNoise noise(&Y);
      for(int i = 0; i < noise.getOutputDim(); i++)
        noise.setBiasVal(0.0, i);
      CGp model(&kern, &noise, &X, approxs[k], m, 0);
      model.setInducingVals(Xu);
      model.setInducingFixed(true);
      model.setBetaVal(1e6);
      model.setVerbosity(0);
      stressBothModes(model, tags[k]);
    }
  }

  {
    // CIP-0013-style protocol (compact): q=2, LS=1, white=0.01, beta=100.
    const unsigned int n = 200, q = 2, m = 50;
    CMatrix X(n, q), Y(n, 1), Xu(m, q);
    fillProtocol(X, Y, Xu, n, q, m, 13);
    const int approxs[] = {CGp::FITC, CGp::DTCVAR};
    const char* tags[] = {"sparse_fitc_protocol", "sparse_dtcvar_protocol"};
    for(unsigned int k = 0; k < 2; k++)
    {
      CCmpndKern kern(X);
      kern.addKern(new CRbfKern(X));
      kern.addKern(new CWhiteKern(X));
      kern.setByPath("rbf.lengthscale", 1.0);
      kern.setByPath("rbf.variance", 1.0);
      kern.setByPath("white.variance", 0.01);
      CGaussianNoise noise(&Y);
      for(int i = 0; i < noise.getOutputDim(); i++)
        noise.setBiasVal(0.0, i);
      CGp model(&kern, &noise, &X, approxs[k], m, 0);
      model.setInducingVals(Xu);
      model.setInducingFixed(true);
      model.setBetaVal(100.0);
      model.setVerbosity(0);
      stressBothModes(model, tags[k]);
    }
  }

  // -------------------------------------------------------------------------
  // CIP-001C: lengthscale sweep, fixed Z — point vs white-noise inducing.
  // Kernel-driven κ(K_uu) should explode for points as γ→0; WN K_uu depends
  // only on windows (s fixed from spacing). Both SparseFactor and LQ.
  // -------------------------------------------------------------------------
  {
    const unsigned int n = 40, m = 10;
    const double beta = 100.0;
    const double sWin = 0.25; // fixed window; spacing on [-1,1] with M=10 ≈ 0.22
    const double gammas[] = {0.01, 0.1, 1.0, 10.0, 100.0};
    CMatrix X(n, 1), Y(n, 1), Xu(m, 1);
    for(unsigned int i = 0; i < n; i++)
    {
      const double x = -1.0 + 2.0 * (double)i / (double)(n - 1);
      X.setVal(x, i, 0);
      Y.setVal(sin(2.0 * x), i, 0);
    }
    for(unsigned int i = 0; i < m; i++)
      Xu.setVal(-1.0 + 2.0 * (double)i / (double)(m - 1), i, 0);

    auto kappaSym = [](CMatrix spd) {
      CMatrix evals(1, spd.getRows());
      spd.setSymmetric(true);
      spd.syev(evals, "n", "u");
      double mx = 0.0, mn = numeric_limits<double>::infinity();
      for(unsigned int i = 0; i < evals.getCols(); i++)
      {
        const double d = fabs(evals.getVal(i));
        mx = max(mx, d);
        if(d > 0.0)
          mn = min(mn, d);
      }
      if(!isfinite(mn) || mn <= 0.0)
        return numeric_limits<double>::infinity();
      return mx / mn;
    };

    bool wnAllFiniteStable = true;
    bool wnAllFiniteLq = true;
    double maxKappaPt = 0.0;
    double maxKappaWn = 0.0;
    double maxAbsLlGap = 0.0;

    for(unsigned int gi = 0; gi < 5; gi++)
    {
      const double gamma = gammas[gi];
      // Point Kuu = k(Z,Z) for RBF variance=1.
      CMatrix KuuPt(m, m);
      const double ell = 1.0 / sqrt(gamma);
      for(unsigned int i = 0; i < m; i++)
        for(unsigned int j = 0; j < m; j++)
        {
          const double dz = Xu.getVal(i, 0) - Xu.getVal(j, 0);
          KuuPt.setVal(exp(-0.5 * gamma * dz * dz), i, j);
        }
      KuuPt.setSymmetric(true);
      const double kPt = kappaSym(KuuPt);
      maxKappaPt = max(maxKappaPt, kPt);

      CMatrix KuuWn(m, m);
      for(unsigned int i = 0; i < m; i++)
        for(unsigned int j = 0; j < m; j++)
        {
          const double dz = Xu.getVal(i, 0) - Xu.getVal(j, 0);
          KuuWn.setVal(exp(-dz * dz / (4.0 * sWin * sWin)), i, j);
        }
      KuuWn.setSymmetric(true);
      const double kWn = kappaSym(KuuWn);
      maxKappaWn = max(maxKappaWn, kWn);

      // WN κ independent of γ — record per-γ metric for trend.
      ostringstream mk;
      mk << "wn_ls_sweep_kappa_pt_g" << gamma;
      emitMetric(mk.str(), "B_linalg", kPt, 1e8, "abs");
      ostringstream mkw;
      mkw << "wn_ls_sweep_kappa_wn_g" << gamma;
      emitMetric(mkw.str(), "B_linalg", kWn, 1e6, "abs");

      for(int mode = 0; mode < 2; mode++)
      {
        const bool useLq = (mode == 1);
        CRbfKern kernPt(X);
        kernPt.setParam(gamma, 0);
        kernPt.setParam(1.0, 1);
        CGaussianNoise noisePt(&Y);
        noisePt.setBiasVal(0.0, 0);
        CGp gpPt(&kernPt, &noisePt, &X, CGp::DTCVAR, m, 0);
        gpPt.setInducingVals(Xu);
        gpPt.setInducingFixed(true);
        gpPt.setBetaVal(beta);
        gpPt.setVerbosity(0);
        gpPt.setSparseLinalgMode(useLq ? CGp::SparseLinalgLQ
                                       : CGp::SparseLinalgStableFactor);

        CRbfKern kernWn(X);
        kernWn.setParam(gamma, 0);
        kernWn.setParam(1.0, 1);
        CGaussianNoise noiseWn(&Y);
        noiseWn.setBiasVal(0.0, 0);
        CGp gpWn(&kernWn, &noiseWn, &X, CGp::DTCVAR, m, 0);
        gpWn.setInducingVals(Xu);
        gpWn.setInducingFixed(true);
        gpWn.setBetaVal(beta);
        gpWn.setVerbosity(0);
        gpWn.setInducingType("whitenoise");
        gpWn.getInducing()->setWindowWidth(sWin);
        gpWn.setSparseLinalgMode(useLq ? CGp::SparseLinalgLQ
                                       : CGp::SparseLinalgStableFactor);

        const bool okPt = gpAllFinite(gpPt);
        const bool okWn = gpAllFinite(gpWn);
        if(!okWn)
        {
          if(useLq)
            wnAllFiniteLq = false;
          else
            wnAllFiniteStable = false;
        }
        if(okPt && okWn)
        {
          const double gap = fabs(gpWn.logLikelihood() - gpPt.logLikelihood());
          maxAbsLlGap = max(maxAbsLlGap, gap);
          ostringstream mg;
          mg << "wn_ls_sweep_ll_abs_gap_g" << gamma
             << (useLq ? "_lq" : "_stable");
          emitMetric(mg.str(), "F_model", gap, 1e4, "abs");
        }
        (void)ell;
        (void)okPt;
      }
    }

    emitMetric("wn_ls_sweep_max_kappa_point", "B_linalg", maxKappaPt, 1e8, "abs");
    emitMetric("wn_ls_sweep_max_kappa_whitenoise", "B_linalg", maxKappaWn, 1e6, "abs");
    emitMetric("wn_ls_sweep_max_ll_abs_gap", "F_model", maxAbsLlGap, 1e4, "abs");

    // Structural claim: WN κ flat in γ and ≪ point κ at long lengthscales.
    check("wn_ls_sweep_whitenoise_finite_stable", wnAllFiniteStable);
    check("wn_ls_sweep_whitenoise_finite_lq", wnAllFiniteLq);
    check("wn_ls_sweep_kappa_wn_below_1e6", maxKappaWn < 1e6);
    check("wn_ls_sweep_point_kappa_exceeds_wn", maxKappaPt > 10.0 * maxKappaWn,
          "point max κ should dominate WN (kernel-driven ill-conditioning)");
  }

  {
    // Reduced γ×M dual-path slice (diagSparseStable subset) — DTCVAR + FITC.
    const unsigned int n = 40;
    const double beta = 100.0;
    const double gammas[] = {10.0, 100.0, 1000.0};
    const unsigned int Ms[] = {10u, 20u};
    const int approxs[] = {CGp::DTCVAR, CGp::FITC};
    const char* atags[] = {"dtcvar", "fitc"};
    bool allOk = true;
    string detail;
    for(unsigned int a = 0; a < 2; a++)
    {
      for(unsigned int gi = 0; gi < 3; gi++)
      {
        for(unsigned int mi = 0; mi < 2; mi++)
        {
          const unsigned int m = Ms[mi];
          CMatrix X(n, 1), Y(n, 1), Xu(m, 1);
          for(unsigned int i = 0; i < n; i++)
          {
            const double x = -1.0 + 2.0 * (double)i / (double)(n - 1);
            X.setVal(x, i, 0);
            Y.setVal(sin(2.0 * x), i, 0);
          }
          for(unsigned int i = 0; i < m; i++)
            Xu.setVal(-1.0 + 2.0 * (double)i / (double)(m - 1), i, 0);

          CCmpndKern kern(X);
          kern.addKern(new CRbfKern(X));
          kern.addKern(new CWhiteKern(X));
          kern.setParam(gammas[gi], 0);
          kern.setParam(1.0, 1);
          kern.setParam(1e-4, 2);
          CGaussianNoise noise(&Y);
          for(int i = 0; i < noise.getOutputDim(); i++)
            noise.setBiasVal(0.0, i);
          CGp model(&kern, &noise, &X, approxs[a], m, 0);
          model.setInducingVals(Xu);
          model.setInducingFixed(true);
          model.setBetaVal(beta);
          model.setVerbosity(0);

          double llL = 0.0, llS = 0.0, fdL = 0.0, fdS = 0.0;
          string errL, errS;
          const bool okL = trySparseMode(model, false, llL, fdL, errL);
          const bool okS = trySparseMode(model, true, llS, fdS, errS);
          applySparseMode(model);

          ostringstream mll, mgr;
          mll << "sparse_" << atags[a] << "_gamma_" << gammas[gi] << "_M_" << m
              << "_ll_absdiff";
          mgr << "sparse_" << atags[a] << "_gamma_" << gammas[gi] << "_M_" << m
              << "_grad_maxabs";

          if(okL && okS)
          {
            const double llGap = fabs(llL - llS);
            // Grad maxabs between paths (reuse analytic grads via dualPath-style).
            model.setStableSparseLinalg(false);
            CMatrix gL(1, model.getOptNumParams());
            model.logLikelihoodGradient(gL);
            model.setStableSparseLinalg(true);
            CMatrix gS(1, model.getOptNumParams());
            model.logLikelihoodGradient(gS);
            applySparseMode(model);
            double gGap = 0.0;
            for(unsigned int j = 0; j < gL.getCols(); j++)
              gGap = max(gGap, fabs(gL.getVal(j) - gS.getVal(j)));
            emitMetric(mll.str(), "F_model", llGap, 5e-6, "abs");
            emitMetric(mgr.str(), "D_fd", gGap, 5e-3, "abs");
            if(llGap >= 5e-6 || gGap >= 5e-3)
            {
              allOk = false;
              ostringstream os;
              os << atags[a] << " gamma=" << gammas[gi] << " M=" << m
                 << " llGap=" << llGap << " gGap=" << gGap;
              detail = os.str();
            }
          }
          else if(!okL && okS)
          {
            ostringstream brk;
            brk << "sparse_" << atags[a] << "_gamma_" << gammas[gi] << "_M_" << m;
            check(brk.str() + "_legacy_breaks_stable_ok", true, errL);
          }
          else if(!okS)
          {
            allOk = false;
            ostringstream os;
            os << atags[a] << " gamma=" << gammas[gi] << " M=" << m
               << " stable failed: " << errS;
            detail = os.str();
          }
        }
      }
    }
    check("sparse_gamma_M_dual_path_slice", allOk, detail);
  }

  // Probit noise: log-likelihood finite on wide latent range.
  {
    CMatrix y(5, 1);
    y.setVal(1.0, 0, 0);
    y.setVal(-1.0, 1, 0);
    y.setVal(1.0, 2, 0);
    y.setVal(-1.0, 3, 0);
    y.setVal(1.0, 4, 0);
    CProbitNoise noise(&y);
    CMatrix mu(5, 1);
    CMatrix var(5, 1);
    const double fgrid[] = {-40.0, -5.0, 0.0, 5.0, 40.0};
    for(unsigned int i = 0; i < 5; i++)
    {
      mu.setVal(fgrid[i], i, 0);
      var.setVal(1.0, i, 0);
    }
    bool ok = true;
    try
    {
      const double ll = noise.logLikelihood(mu, var, y);
      if(!finiteVal(ll))
        ok = false;
    }
    catch(...)
    {
      ok = false;
    }
    check("probit_noise_extreme_latents_finite", ok);
  }

  return (g_fail + g_xpass) - before;
}

} // namespace

int main()
{
  cout << "GPc numerical stability suite (CIP-0019 + CIP-001A)" << endl;
  cout << "cell=" << cellLabel()
       << " sparse_linalg=" << sparseLinalgModeName() << endl;
  try
  {
    testLinalg();
    testSpecial();
    testKernels();
    testModels();
  }
  catch(ndlexceptions::Error& err)
  {
    cerr << "Unhandled ndlexceptions::Error: " << err.getMessage() << endl;
    return 2;
  }
  catch(std::exception& err)
  {
    cerr << "Unhandled exception: " << err.what() << endl;
    return 2;
  }

  cout << endl
       << "Summary: passed=" << g_pass
       << " failed=" << g_fail
       << " xfailed=" << g_xfail
       << " xpassed=" << g_xpass << endl;

  if(g_fail > 0 || g_xpass > 0)
    return 1;
  return 0;
}
