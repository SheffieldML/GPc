
#include "CNoise.h"

using namespace std;

namespace ndlnoise {

void addObservationNoiseDiag(CMatrix& K, double sigma2)
{
  DIMENSIONMATCH(K.getRows() == K.getCols());
  BOUNDCHECK(sigma2 >= 0.0);
  for(unsigned int i = 0; i < K.getRows(); i++)
    K.setVal(K.getVal(i, i) + sigma2, i, i);
}

void gaussianLogDensity(CMatrix& logDens,
                        const CMatrix& y,
                        const CMatrix& f,
                        double sigma2,
                        const double* biasOrNull)
{
  DIMENSIONMATCH(y.dimensionsMatch(f));
  DIMENSIONMATCH(logDens.dimensionsMatch(y));
  BOUNDCHECK(sigma2 > 0.0);
  const double halfLogTwoPiSigma2 = ndlutil::HALFLOGTWOPI + 0.5 * log(sigma2);
  const double invSigma2 = 1.0 / sigma2;
  for(unsigned int i = 0; i < y.getRows(); i++)
  {
    for(unsigned int j = 0; j < y.getCols(); j++)
    {
      const double b = biasOrNull ? biasOrNull[j] : 0.0;
      const double r = y.getVal(i, j) - f.getVal(i, j) - b;
      logDens.setVal(-halfLogTwoPiSigma2 - 0.5 * invSigma2 * r * r, i, j);
    }
  }
}

void gaussianVariationalExpectations(CMatrix& ell,
                                     CMatrix& gMu,
                                     CMatrix& gVar,
                                     const CMatrix& y,
                                     const CMatrix& mu,
                                     const CMatrix& var,
                                     double sigma2,
                                     const double* biasOrNull)
{
  DIMENSIONMATCH(y.dimensionsMatch(mu));
  DIMENSIONMATCH(mu.dimensionsMatch(var));
  DIMENSIONMATCH(ell.dimensionsMatch(y));
  DIMENSIONMATCH(gMu.dimensionsMatch(y));
  DIMENSIONMATCH(gVar.dimensionsMatch(y));
  BOUNDCHECK(sigma2 > 0.0);
  const double betaLik = 1.0 / sigma2;
  const double constTerm = -ndlutil::HALFLOGTWOPI + 0.5 * log(betaLik);
  for(unsigned int i = 0; i < y.getRows(); i++)
  {
    for(unsigned int j = 0; j < y.getCols(); j++)
    {
      const double b = biasOrNull ? biasOrNull[j] : 0.0;
      const double r = y.getVal(i, j) - mu.getVal(i, j) - b;
      const double v = var.getVal(i, j);
      gMu.setVal(betaLik * r, i, j);
      gVar.setVal(-0.5 * betaLik, i, j);
      ell.setVal(constTerm - 0.5 * betaLik * (r * r + v), i, j);
    }
  }
}

double gaussianLogPredictiveDensity(const CMatrix& y,
                                    const CMatrix& mu,
                                    const CMatrix& var,
                                    double sigma2,
                                    const double* biasOrNull)
{
  DIMENSIONMATCH(y.dimensionsMatch(mu));
  DIMENSIONMATCH(y.dimensionsMatch(var));
  BOUNDCHECK(sigma2 > 0.0);
  double L = 0.0;
  for(unsigned int i = 0; i < y.getRows(); i++)
  {
    for(unsigned int j = 0; j < y.getCols(); j++)
    {
      const double b = biasOrNull ? biasOrNull[j] : 0.0;
      double arg = y.getVal(i, j) - mu.getVal(i, j) - b;
      arg *= arg;
      const double v = var.getVal(i, j) + sigma2;
      L += log(v) + arg / v;
    }
  }
  L += static_cast<double>(y.getRows() * y.getCols()) * log(2.0 * M_PI);
  L *= -0.5;
  return L;
}

void bernoulliProbitEpMoments(double& Zhat,
                              double& muHat,
                              double& sigma2Hat,
                              double y,
                              double tauCav,
                              double vCav)
{
  // Mirror GPy.likelihoods.Bernoulli.moments_match_ep (Probit link).
  BOUNDCHECK(tauCav > 0.0);
  double sign = 0.0;
  if(y == 1.0)
    sign = 1.0;
  else if(y == 0.0 || y == -1.0)
    sign = -1.0;
  else
    throw ndlexceptions::Error(
        "bernoulliProbitEpMoments: labels must be in {0,1} or {-1,1}.");
  const double den = sqrt(tauCav * tauCav + tauCav);
  const double z = sign * vCav / den;
  const double phiOverPhi = ndlutil::gradLnCumGaussian(z);
  const double logZ = ndlutil::lnCumGaussian(z);
  Zhat = exp(logZ);
  muHat = vCav / tauCav + sign * phiOverPhi / den;
  sigma2Hat = 1.0 / tauCav
              - (phiOverPhi / (tauCav * tauCav + tauCav)) * (z + phiOverPhi);
  if(!(sigma2Hat > 0.0))
    sigma2Hat = ndlutil::EPS;
}

void bernoulliProbitEpMoments(CMatrix& Zhat,
                              CMatrix& muHat,
                              CMatrix& sigma2Hat,
                              const CMatrix& y,
                              const CMatrix& tauCav,
                              const CMatrix& vCav)
{
  DIMENSIONMATCH(y.dimensionsMatch(tauCav));
  DIMENSIONMATCH(y.dimensionsMatch(vCav));
  DIMENSIONMATCH(Zhat.dimensionsMatch(y));
  DIMENSIONMATCH(muHat.dimensionsMatch(y));
  DIMENSIONMATCH(sigma2Hat.dimensionsMatch(y));
  for(unsigned int i = 0; i < y.getRows(); i++)
  {
    for(unsigned int j = 0; j < y.getCols(); j++)
    {
      double Zh = 0.0, mh = 0.0, s2 = 0.0;
      bernoulliProbitEpMoments(Zh, mh, s2,
                               y.getVal(i, j),
                               tauCav.getVal(i, j),
                               vCav.getVal(i, j));
      Zhat.setVal(Zh, i, j);
      muHat.setVal(mh, i, j);
      sigma2Hat.setVal(s2, i, j);
    }
  }
}

void bernoulliProbitLaplaceDerivs(double& logp,
                                  double& d1,
                                  double& W,
                                  double& dWdf,
                                  double y,
                                  double f)
{
  double sign = 0.0;
  if(y == 1.0)
    sign = 1.0;
  else if(y == 0.0 || y == -1.0)
    sign = -1.0;
  else
    throw ndlexceptions::Error(
        "bernoulliProbitLaplaceDerivs: labels must be in {0,1} or {-1,1}.");
  const double t = sign * f;
  const double r = ndlutil::gradLnCumGaussian(t);
  logp = ndlutil::lnCumGaussian(t);
  d1 = sign * r;
  // W = −d²logp/df² = r(t+r); dW/df = sign * dW/dt
  W = r * (t + r);
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  const double dr = -r * (t + r);
  const double dWdt = dr * (t + r) + r * (1.0 + dr);
  dWdf = sign * dWdt;
}

double invMillsRatio(double z)
{
  // λ(z)=φ/Φ = d/dz log Φ(z) (GPy gaussian_special / ndlutil::gradLnCumGaussian).
  return ndlutil::gradLnCumGaussian(z);
}

double dInvMillsRatio(double z, double lam)
{
  return -lam * (z + lam);
}

double d2InvMillsRatio(double z, double lam)
{
  return lam * ((z + lam) * (z + 2.0 * lam) - 1.0);
}

void skewGaussianLaplaceDerivs(double& logp,
                               double& d1,
                               double& W,
                               double& dWdf,
                               double y,
                               double f,
                               double variance,
                               double alpha)
{
  BOUNDCHECK(variance > 0.0);
  const double sigma = sqrt(variance);
  const double e = y - f;
  const double z = e / sigma;
  const double delta = alpha * z;
  const double lam = invMillsRatio(delta);
  const double aOverS = alpha / sigma;
  logp = M_LN2 - log(sigma) - ndlutil::HALFLOGTWOPI - 0.5 * z * z
         + ndlutil::lnCumGaussian(delta);
  // dlogp/df with e=y-f (GPy dlogpdf_dlink).
  d1 = e / variance - aOverS * lam;
  const double d2 = -1.0 / variance + aOverS * aOverS * dInvMillsRatio(delta, lam);
  W = -d2;
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  // d³logp/df³ = -(α/σ)³ λ''(δ); dW/df = -d³logp/df³
  dWdf = aOverS * aOverS * aOverS * d2InvMillsRatio(delta, lam);
}

void skewExponentialLaplaceDerivs(double& logp,
                                  double& d1,
                                  double& W,
                                  double& dWdf,
                                  double y,
                                  double f,
                                  double rate,
                                  double alpha)
{
  BOUNDCHECK(rate > 0.0);
  const double e = y - f;
  const double delta = alpha * e;
  const double lam = invMillsRatio(delta);
  const double ae = fabs(e);
  logp = log(rate) - rate * ae + ndlutil::lnCumGaussian(delta);
  const double sgn = (e > 0.0) ? 1.0 : ((e < 0.0) ? -1.0 : 0.0);
  d1 = rate * sgn - alpha * lam;
  const double d2 = alpha * alpha * dInvMillsRatio(delta, lam);
  W = -d2;
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  dWdf = alpha * alpha * alpha * d2InvMillsRatio(delta, lam);
}

void studentTLaplaceDerivs(double& logp,
                           double& d1,
                           double& W,
                           double& dWdf,
                           double y,
                           double f,
                           double variance,
                           double degFree)
{
  BOUNDCHECK(variance > 0.0);
  BOUNDCHECK(degFree > 0.0);
  const double e = y - f;
  const double e2 = e * e;
  const double vs = variance * degFree;
  const double denom = vs + e2;
  logp = ndlutil::gammaln(0.5 * (degFree + 1.0)) - ndlutil::gammaln(0.5 * degFree)
         - 0.5 * log(variance * degFree * M_PI)
         - 0.5 * (degFree + 1.0) * log(1.0 + e2 / vs);
  d1 = ((degFree + 1.0) * e) / denom;
  const double d2 = ((degFree + 1.0) * (e2 - vs)) / (denom * denom);
  W = -d2;
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  // GPy d3logpdf_dlink3; dW/df = −d3
  const double d3 = -(2.0 * (degFree + 1.0) * (-e) * (e2 - 3.0 * vs))
                    / (denom * denom * denom);
  dWdf = -d3;
}

void poissonLogLaplaceDerivs(double& logp,
                             double& d1,
                             double& W,
                             double& dWdf,
                             double y,
                             double f)
{
  BOUNDCHECK(y >= 0.0);
  const double rate = exp(f);
  logp = -rate + y * f - ndlutil::gammaln(y + 1.0);
  d1 = y - rate;
  const double d2 = -rate;
  W = -d2;
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  dWdf = rate; // −d3 with d3 = −exp(f)
}

void binomialProbitLaplaceDerivs(double& logp,
                                 double& d1,
                                 double& W,
                                 double& dWdf,
                                 double y,
                                 double f,
                                 double numTrials)
{
  BOUNDCHECK(numTrials >= 1.0);
  BOUNDCHECK(y >= 0.0);
  BOUNDCHECK(y <= numTrials);
  const double N = numTrials;
  const double Ny = N - y;
  const double lam = invMillsRatio(f);
  const double lamN = invMillsRatio(-f);
  logp = ndlutil::gammaln(N + 1.0) - ndlutil::gammaln(y + 1.0) - ndlutil::gammaln(Ny + 1.0)
         + y * ndlutil::lnCumGaussian(f) + Ny * ndlutil::lnCumGaussian(-f);
  d1 = y * lam - Ny * lamN;
  // dλ/df = −λ(f+λ); dλN/df = d/df λ(-f) = −λ'(-f) = λN((-f)+λN) wait:
  // λ'(-f) = −λN((-f)+λN); d/df λ(-f) = λ'(-f)*(-1) = λN((-f)+λN)
  const double dlam = dInvMillsRatio(f, lam);
  const double dlamN_df = lamN * ((-f) + lamN); // = −λ'(-f)
  const double d2 = y * dlam - Ny * dlamN_df;
  W = -d2;
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  // d²λ/df² = λ''; dW/df = −d³logp/df³ = −(y λ'' − Ny d²λN/df²)
  const double d2lam = d2InvMillsRatio(f, lam);
  // d/df [λN((-f)+λN)] = dlamN_df * ((-f)+λN) + λN * (-1 + dlamN_df)
  const double d2lamN_df = dlamN_df * ((-f) + lamN) + lamN * (-1.0 + dlamN_df);
  dWdf = -(y * d2lam - Ny * d2lamN_df);
}

namespace {
/** Trigamma ψ₁(x) and tetragamma ψ₂(x) via recurrence + asymptotic (Abramowitz). */
double trigamma(double x)
{
  BOUNDCHECK(x > 0.0);
  double result = 0.0;
  while(x < 8.0)
  {
    result += 1.0 / (x * x);
    x += 1.0;
  }
  const double ix = 1.0 / x;
  const double ix2 = ix * ix;
  result += ix + 0.5 * ix2 + ix2 * ix / 6.0 - ix2 * ix2 * ix / 30.0;
  return result;
}
double tetragamma(double x)
{
  BOUNDCHECK(x > 0.0);
  double result = 0.0;
  while(x < 8.0)
  {
    result -= 2.0 / (x * x * x);
    x += 1.0;
  }
  const double ix = 1.0 / x;
  const double ix2 = ix * ix;
  const double ix3 = ix2 * ix;
  result += -ix2 - ix3 - 0.5 * ix2 * ix2 + ix2 * ix3 / 6.0;
  return result;
}
} // namespace

void exponentialLogLaplaceDerivs(double& logp,
                                 double& d1,
                                 double& W,
                                 double& dWdf,
                                 double y,
                                 double f)
{
  BOUNDCHECK(y > 0.0);
  const double rate = exp(f);
  logp = f - y * rate;
  d1 = 1.0 - y * rate;
  const double d2 = -y * rate;
  W = -d2;
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  dWdf = y * rate; // −d3, d3=d2
}

void gammaLogLaplaceDerivs(double& logp,
                           double& d1,
                           double& W,
                           double& dWdf,
                           double y,
                           double f,
                           double beta)
{
  BOUNDCHECK(y > 0.0);
  BOUNDCHECK(beta > 0.0);
  const double lam = exp(f);
  const double alpha = beta * lam;
  logp = alpha * log(beta) - ndlutil::gammaln(alpha) + (alpha - 1.0) * log(y) - beta * y;
  const double dig = ndlutil::digamma(alpha);
  const double tri = trigamma(alpha);
  const double tetra = tetragamma(alpha);
  d1 = alpha * (log(beta * y) - dig);
  const double d2 = d1 - alpha * alpha * tri;
  W = -d2;
  if(W < ndlutil::EPS)
    W = ndlutil::EPS;
  const double d3 = d2 - 2.0 * alpha * alpha * tri - alpha * alpha * alpha * tetra;
  dWdf = -d3;
}

void laplacePointDerivs(const CNoise& noise,
                        double y,
                        double f,
                        double& logp,
                        double& d1,
                        double& W,
                        double& dWdf)
{
  const string t = noise.getType();
  if(t == "probit")
  {
    bernoulliProbitLaplaceDerivs(logp, d1, W, dWdf, y, f);
    return;
  }
  if(t == "student_t")
  {
    const CStudentTNoise& n = dynamic_cast<const CStudentTNoise&>(noise);
    studentTLaplaceDerivs(logp, d1, W, dWdf, y, f, n.getVariance(), n.getDegFree());
    return;
  }
  if(t == "poisson")
  {
    poissonLogLaplaceDerivs(logp, d1, W, dWdf, y, f);
    return;
  }
  if(t == "binomial")
  {
    const CBinomialNoise& n = dynamic_cast<const CBinomialNoise&>(noise);
    binomialProbitLaplaceDerivs(logp, d1, W, dWdf, y, f,
                                static_cast<double>(n.getNumTrials()));
    return;
  }
  if(t == "exponential")
  {
    exponentialLogLaplaceDerivs(logp, d1, W, dWdf, y, f);
    return;
  }
  if(t == "gamma")
  {
    const CGammaNoise& n = dynamic_cast<const CGammaNoise&>(noise);
    gammaLogLaplaceDerivs(logp, d1, W, dWdf, y, f, n.getBeta());
    return;
  }
  if(t == "skew_gaussian")
  {
    const CSkewGaussianNoise& n = dynamic_cast<const CSkewGaussianNoise&>(noise);
    skewGaussianLaplaceDerivs(logp, d1, W, dWdf, y, f, n.getVariance(), n.getAlpha());
    return;
  }
  if(t == "skew_exponential")
  {
    const CSkewExponentialNoise& n = dynamic_cast<const CSkewExponentialNoise&>(noise);
    skewExponentialLaplaceDerivs(logp, d1, W, dWdf, y, f, n.getRate(), n.getAlpha());
    return;
  }
  throw ndlexceptions::Error(
      "ndlnoise::laplacePointDerivs: Laplace not supported for noise type '"
      + t + "'");
}

} // namespace ndlnoise

double CNoise::expectedLogLikelihood(double /*y*/, double /*mu*/, double /*var*/,
                                     double& /*gMu*/, double& /*gVar*/) const
{
  throw ndlexceptions::NotImplementedError(
    "expectedLogLikelihood not implemented for noise type " + getType());
}

double CGaussianNoise::expectedLogLikelihood(double y, double muVal, double var,
                                             double& gMu, double& gVar) const
{
  // p(y|f) = N(y|f, sigma2); E_q[log p] with q(f)=N(mu,var).
  const double betaLik = 1.0 / sigma2;
  const double r = y - muVal;
  gMu = betaLik * r;
  gVar = -0.5 * betaLik;
  return -ndlutil::HALFLOGTWOPI + 0.5 * log(betaLik)
    - 0.5 * betaLik * (r * r + var);
}

namespace {
struct ProbitEllCtx {
  double y;
  double bias;
};
double probitLogLik(double f, void* ctx)
{
  const ProbitEllCtx* c = static_cast<const ProbitEllCtx*>(ctx);
  return ndlutil::lnCumGaussian(c->y * (f + c->bias));
}
double probitLogLikPrime(double f, void* ctx)
{
  const ProbitEllCtx* c = static_cast<const ProbitEllCtx*>(ctx);
  return c->y * ndlutil::gradLnCumGaussian(c->y * (f + c->bias));
}
}

double CProbitNoise::expectedLogLikelihood(double y, double muVal, double var,
                                           double& gMu, double& gVar) const
{
  // E_q[log Φ(y (f + bias))]; Gauss–Hermite n=20 (CIP-002A Phase 4).
  // Bias: use process 0 by default; CSvgp sets process via setExpectedLogLikBias.
  ProbitEllCtx ctx;
  ctx.y = y;
  ctx.bias = ellBias;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, probitLogLik, probitLogLikPrime,
                                     &ctx, gMu, gVar);
}

namespace {
struct OrderedEllCtx {
  int targ;
  double bias;
  double sigma2;
  unsigned int numCats;
  const CMatrix* widths;
};
// log p(y|f) for ordered categorical (observation noise sigma2; thresholds via widths).
double orderedLogLik(double f, void* vctx)
{
  const OrderedEllCtx* c = static_cast<const OrderedEllCtx*>(vctx);
  if(isnan((double)c->targ))
    return 0.0;
  const double s = 1.0 / sqrt(c->sigma2);
  double muAdj = f + c->bias;
  if(c->targ == 0)
    return ndlutil::lnCumGaussian(-muAdj * s);
  if(c->targ > 0 && (unsigned int)c->targ < c->numCats - 1)
  {
    for(unsigned int k = 0; k < (unsigned int)c->targ - 1; k++)
      muAdj -= c->widths->getVal(0, k);
    const double u = muAdj * s;
    const double uprime = (muAdj - c->widths->getVal(0, (unsigned int)c->targ - 1)) * s;
    return ndlutil::lnDiffCumGaussian(u, uprime);
  }
  if((unsigned int)c->targ == c->numCats - 1)
  {
    for(unsigned int k = 0; k < (unsigned int)c->targ - 1; k++)
      muAdj -= c->widths->getVal(0, k);
    return ndlutil::lnCumGaussian(muAdj * s);
  }
  throw ndlexceptions::Error("orderedLogLik: invalid category");
}
double orderedLogLikPrime(double f, void* vctx)
{
  // ∂ log p(y|f) / ∂f — same as getGradInputs gmu at varSigma=0.
  const OrderedEllCtx* c = static_cast<const OrderedEllCtx*>(vctx);
  if(isnan((double)c->targ))
    return 0.0;
  const double s = 1.0 / sqrt(c->sigma2);
  double muAdj = f + c->bias;
  if(c->targ == 0)
  {
    const double z = -muAdj * s;
    return -s * ndlutil::gradLnCumGaussian(z);
  }
  if(c->targ > 0 && (unsigned int)c->targ < c->numCats - 1)
  {
    for(unsigned int k = 0; k < (unsigned int)c->targ - 1; k++)
      muAdj -= c->widths->getVal(0, k);
    const double u = muAdj * s;
    const double uprime = (muAdj - c->widths->getVal(0, (unsigned int)c->targ - 1)) * s;
    const double B1 = ndlutil::gaussOverDiffCumGaussian(u, uprime, 1);
    const double B2 = ndlutil::gaussOverDiffCumGaussian(u, uprime, 2);
    return s * (B1 - B2);
  }
  if((unsigned int)c->targ == c->numCats - 1)
  {
    for(unsigned int k = 0; k < (unsigned int)c->targ - 1; k++)
      muAdj -= c->widths->getVal(0, k);
    return s * ndlutil::gradLnCumGaussian(muAdj * s);
  }
  throw ndlexceptions::Error("orderedLogLikPrime: invalid category");
}
}

double COrderedNoise::expectedLogLikelihood(double y, double muVal, double var,
                                            double& gMu, double& gVar) const
{
  OrderedEllCtx ctx;
  ctx.targ = (int)y;
  ctx.bias = ellBias;
  ctx.sigma2 = sigma2;
  ctx.numCats = numCats;
  ctx.widths = &widths;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, orderedLogLik, orderedLogLikPrime,
                                     &ctx, gMu, gVar);
}

void CNoise::getNuG(CMatrix& g, CMatrix& nu, unsigned int index) const
{
  DIMENSIONMATCH(g.dimensionsMatch(nu));
  DIMENSIONMATCH(g.getRows()==nData);
  DIMENSIONMATCH(g.getCols()==nProcesses);
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<nData);
  double nuval=0.0;
  double gval=0.0;
  
  for(unsigned int j=0; j<g.getCols(); j++)
  {
    getGradInputs(gval, nuval, index, j);
    nuval = gval*gval - 2*nuval;
    if(nuval<0)
    {
      if(isLogConcave())
      {
	throw ndlexceptions::Error("Log concave noise model has value of nu < 0.");
      }
      else 
      {
	nuval=ndlutil::SMALLVAL;
      }
    }
    if(isnan(nuval)) 
      throw ndlexceptions::Error("Nu is NaN");
    if(abs(nuval)<ndlutil::SMALLVAL)
      nuval=ndlutil::EPS;
    nu.setVal(nuval, index, j);
    
    g.setVal(gval, index, j);
  }
}

void CNoise::updateSites(CMatrix& m, CMatrix& beta, unsigned int actIndex, 
			 const CMatrix& g, const CMatrix& nu, 
			 unsigned int index) const
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<nData);
  DIMENSIONMATCH(m.dimensionsMatch(beta));
  BOUNDCHECK(actIndex>=0);
  BOUNDCHECK(actIndex<m.getRows());
  DIMENSIONMATCH(g.dimensionsMatch(nu));
  DIMENSIONMATCH(m.getCols()==g.getCols());
  double nuVal=0.0;
  for(unsigned int j=0; j<m.getCols(); j++)
  {
    nuVal = nu.getVal(index, j);
    beta.setVal(nuVal/(1-nuVal*getVarSigma(index, j)), actIndex, j);
    m.setVal(getMu(index, j) + g.getVal(index, j)/nuVal, actIndex, j);
  }
}
#ifdef _NDLMATLAB
mxArray* CNoise::toMxArray() const
{
  size_t dims[1];
  dims[0] = 1;
  const char *fieldNames[] = {"type", "transforms", "numProcess", "spherical", "nParams", "missing", "logconcave"};

  mxArray* matlabArray = mxCreateStructArray(1, dims, 7, fieldNames);
    
  // type field.
  const char *typeName[1];
  string ty=getType();
  typeName[0] = ty.c_str();
  mxSetField(matlabArray, 0, "type", 
	     mxCreateCharMatrixFromStrings(1, typeName));
  
  // transforms field.
  mxSetField(matlabArray, 0, "transforms", transformsToMxArray());

  // inputDimension field.
  mxSetField(matlabArray, 0, "numProcess", convertMxArray((double)nProcesses));

  // nParams field.
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)nParams));
    
  // spherical field
  mxSetField(matlabArray, 0, "spherical", convertMxArray((double)spherical));

  // missing field
  mxSetField(matlabArray, 0, "missing", convertMxArray((double)missing));
  
  // logConcave field
  mxSetField(matlabArray, 0, "logconcave", convertMxArray((double)logConcave));
  // priors field.
  /// if priors exist need to add the field.

  // Noise specific code.
  addParamToMxArray(matlabArray);
  return matlabArray;

}
void CNoise::fromMxArray(const mxArray* matlabArray) 
{
  string mxType = mxArrayExtractStringField(matlabArray, "type");
  if(mxType!=getType())
    throw ndlexceptions::FileReadError("Error mismatch between saved type, " + mxType + ", and Class type, " + type + ".");
  
  mxArray* transformArray = mxArrayExtractMxArrayField(matlabArray, "transforms");
  if(transformArray!=NULL)
    transformsFromMxArray(transformArray);
  nProcesses = mxArrayExtractIntField(matlabArray, "numProcess");
  spherical = mxArrayExtractBoolField(matlabArray, "spherical");
  logConcave = mxArrayExtractBoolField(matlabArray, "logconcave");
  missing = mxArrayExtractBoolField(matlabArray, "missing");
  // TODO priors ... need to deal with priors
  extractParamFromMxArray(matlabArray);
}
  
  
void CNoise::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)nParams));
  string pName;
  for(unsigned int i=0; i<nParams; i++)
  {
    pName = getParamName(i);
    mxAddField(matlabArray, pName.c_str());      
    mxSetField(matlabArray, 0, pName.c_str(), convertMxArray(getParam(i))); 
  } 
}

void CNoise::extractParamFromMxArray(const mxArray* matlabArray) 
{
  nParams = mxArrayExtractIntField(matlabArray, "nParams");
  initStoreage();
  string pName;
  for(unsigned int i=0; i<nParams; i++)
  {
    pName=getParamName(i);
    setParam(mxArrayExtractDoubleField(matlabArray, pName), i);
  }

}
void CGaussianNoise::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)getNumParams()));
  mxAddField(matlabArray, "bias");
  mxSetField(matlabArray, 0, "bias", bias.toMxArray());
  mxAddField(matlabArray, "sigma2");
  mxSetField(matlabArray, 0, "sigma2", convertMxArray(sigma2));
}

void CGaussianNoise::extractParamFromMxArray(const mxArray* matlabArray) 
{
  setNumParams(mxArrayExtractIntField(matlabArray, "nParams"));
  mxArray* biasField = mxArrayExtractMxArrayField(matlabArray, "bias");
  bias.fromMxArray(biasField);
  sigma2 = mxArrayExtractDoubleField(matlabArray, "sigma2");

}
void CScaleNoise::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)getNumParams()));
  mxAddField(matlabArray, "bias");
  mxSetField(matlabArray, 0, "bias", bias.toMxArray());
  mxAddField(matlabArray, "scale");
  mxSetField(matlabArray, 0, "scale", scale.toMxArray());
}

void CScaleNoise::extractParamFromMxArray(const mxArray* matlabArray) 
{
  setNumParams(mxArrayExtractIntField(matlabArray, "nParams"));
  mxArray* biasField = mxArrayExtractMxArrayField(matlabArray, "bias");
  bias.fromMxArray(biasField);
  mxArray* scaleField = mxArrayExtractMxArrayField(matlabArray, "scale");
  scale.fromMxArray(scaleField);

}

void CProbitNoise::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)getNumParams()));
  mxAddField(matlabArray, "bias");
  mxSetField(matlabArray, 0, "bias", bias.toMxArray());
  mxAddField(matlabArray, "sigma2");
  mxSetField(matlabArray, 0, "sigma2", convertMxArray(sigma2));
}

void CProbitNoise::extractParamFromMxArray(const mxArray* matlabArray) 
{
  setNumParams(mxArrayExtractIntField(matlabArray, "nParams"));
  mxArray* biasField = mxArrayExtractMxArrayField(matlabArray, "bias");
  bias.fromMxArray(biasField);
  sigma2 = mxArrayExtractDoubleField(matlabArray, "sigma2");

}
void CNcnmNoise::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)getNumParams()));
  mxAddField(matlabArray, "bias");
  mxSetField(matlabArray, 0, "bias", bias.toMxArray());
  mxAddField(matlabArray, "sigma2");
  mxSetField(matlabArray, 0, "sigma2", convertMxArray(sigma2));
  mxAddField(matlabArray, "width");
  mxSetField(matlabArray, 0, "width", convertMxArray(width));
  mxAddField(matlabArray, "gamman");
  mxSetField(matlabArray, 0, "gamman", convertMxArray(gamman));
  mxAddField(matlabArray, "gammap");
  mxSetField(matlabArray, 0, "gammap", convertMxArray(gammap));
  mxAddField(matlabArray, "gammaSplit");
  mxSetField(matlabArray, 0, "gammaSplit", convertMxArray(isSplitGamma()));
}

void CNcnmNoise::extractParamFromMxArray(const mxArray* matlabArray) 
{
  setNumParams(mxArrayExtractIntField(matlabArray, "nParams"));
  mxArray* biasField = mxArrayExtractMxArrayField(matlabArray, "bias");
  bias.fromMxArray(biasField);
  sigma2 = mxArrayExtractDoubleField(matlabArray, "sigma2");
  width = mxArrayExtractDoubleField(matlabArray, "width");
  gamman = mxArrayExtractDoubleField(matlabArray, "gamman");
  gammap = mxArrayExtractDoubleField(matlabArray, "gammap");
  int temp = mxArrayExtractIntField(matlabArray, "gammaSplit");
  if(temp)
    splitGamma = true;
  else
    splitGamma = false;

}
void COrderedNoise::addParamToMxArray(mxArray* matlabArray) const
{
  mxAddField(matlabArray, "nParams");
  mxSetField(matlabArray, 0, "nParams", convertMxArray((double)getNumParams()));
  mxAddField(matlabArray, "bias");
  mxSetField(matlabArray, 0, "bias", bias.toMxArray());
  mxAddField(matlabArray, "variance");
  mxSetField(matlabArray, 0, "variance", convertMxArray(sigma2));
  mxAddField(matlabArray, "widths");
  mxSetField(matlabArray, 0, "widths", widths.toMxArray());
  mxAddField(matlabArray, "C");
  mxSetField(matlabArray, 0, "C", convertMxArray((double)getNumCategories()));
}

void COrderedNoise::extractParamFromMxArray(const mxArray* matlabArray) 
{
  setNumParams(mxArrayExtractIntField(matlabArray, "nParams"));
  mxArray* biasField = mxArrayExtractMxArrayField(matlabArray, "bias");
  bias.fromMxArray(biasField);
  setNumCategories(mxArrayExtractIntField(matlabArray, "C"));
  initStoreage();
  sigma2 = mxArrayExtractDoubleField(matlabArray, "variance");
  mxArray* widthsField = mxArrayExtractMxArrayField(matlabArray, "widths");
  widths.fromMxArray(widthsField);
}


#endif
bool CNoise::equals(const CNoise& noise, double tol) const
{
  if(getType()!=noise.getType())
    return false;
  if(getNumParams()!=noise.getNumParams())
    return false;
  CMatrix params(1, getNumParams());
  getParams(params);
  CMatrix noiseParams(1, getNumParams());
  noise.getParams(noiseParams);
  if(!params.equals(noiseParams, tol))
    return false;
  return true;
}
void CNoise::initSiteStoreage()
{
  activeSites().resize(getNumData(), getOutputDim());
  activeSites().zeros();
}

void CNoise::setMu(double val, unsigned int i, unsigned int j)
{
  activeSites().mu.setVal(val, i, j);
}

double CNoise::getMu(unsigned int i, unsigned int j) const
{
  return activeSites().mu.getVal(i, j);
}

void CNoise::setVarSigma(double val, unsigned int i, unsigned int j)
{
  SANITYCHECK(!isnan(val));
  SANITYCHECK(val >= 0);
  activeSites().varSigma.setVal(val, i, j);
}

double CNoise::getVarSigma(unsigned int i, unsigned int j) const
{
  return activeSites().varSigma.getVal(i, j);
}

void CNoise::writeParamsToStream(ostream& out) const
{
  //writeToStream(out, "numData", getNumData());
  // CIP-0033/0034: stream carries parameters only — not py / site caches.
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "outputDim", getOutputDim());
  writeToStream(out, "numParams", getNumParams());
  // CIP-003A: zero-param families (poisson, exponential) must not allocate CMatrix(1,0).
  if(getNumParams() == 0)
    return;
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
  // CIP-003B: additive key. Priors are written only when present, so files
  // without noise priors are byte-identical to the old format and old files
  // load unchanged. (Old readers cannot load files that do carry noise priors.)
  if(getNumPriors()>0)
  {
    writeToStream(out, "numPriors", getNumPriors());
    writePriorsToStream(out);
  }
}
void CNoise::readOptionalPriorsFromStream(istream& in)
{
  if(!in.good() || in.eof())
    return;
  streampos pos = in.tellg();
  if(pos==streampos(-1))
    return;
  string line;
  if(ndlstrutil::getline(in, line) && line.size()>0)
  {
    vector<string> tokens;
    ndlstrutil::tokenise(tokens, line, "=");
    if(tokens.size()==2 && tokens[0]=="numPriors")
    {
      readPriorsFromStream(in, (unsigned int)atol(tokens[1].c_str()));
      return;
    }
  }
  in.clear();
  in.seekg(pos);
}
void CNoise::readParamsFromStream(istream& in)
{
//   //setNumData(readIntFromStream(in, "numData"));
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setOutputDim(readIntFromStream(in, "outputDim"));
  unsigned int numPar = readIntFromStream(in, "numParams");
  setNumData(1);
  initStoreage();
  if(numPar != getNumParams())
    throw ndlexceptions::StreamFormatError("numParams", "Number of parameters in file does not match computed number.");
  // CIP-003A: skip empty param block for zero-param noise types.
  if(numPar > 0) {
    CMatrix par(1, numPar);
    par.fromStream(in);
    setParams(par);
  }
  readOptionalPriorsFromStream(in);
}

CGaussianNoise::~CGaussianNoise()
{
}
void CGaussianNoise::initStoreage()
{
  setNumParams(getOutputDim()+1);
  initSiteStoreage();
  bias.resize(1, getOutputDim());
  clearTransforms();
  // transform sigma2 (the last parameter).
  addTransform(CTransform::defaultPositive(), getNumParams()-1);
  setLogConcave(true);
  setSpherical(true);
  setMissing(false);

}

void CGaussianNoise::_init()
{
  setType("gaussian");
  setName("Gaussian");
}
void CGaussianNoise::initNames()
{
  for(unsigned int j=0; j<getOutputDim(); j++)
    setParamName("bias" + ndlstrutil::itoa(j), j);
  setParamName("sigma2", getOutputDim());
}
void CGaussianNoise::initParams()
{
  bias.deepCopy(meanCol(*py));
}
void CGaussianNoise::initVals()
{
  activeSites().zeros();
  bias.zeros();
  sigma2 = 1e-6;
}

ostream& CGaussianNoise::display(ostream& os)
{
  double b = 0.0;
  os << "Gaussian Noise: " << endl; 
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    b = bias.getVal(j);
    os << "Bias on process " << j << ": " << b << endl; 
  }
  os << "Variance: " << sigma2 << endl;
  return os;
}
void CGaussianNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  if(index<getOutputDim())
    bias.setVal(val, index);
  else
    sigma2=val;
}  
void CGaussianNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(getNumParams()==getOutputDim()+1);
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  DIMENSIONMATCH(getOutputDim()==bias.getCols());
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    bias.setVal(params.getVal(j), j);
  }
  sigma2 = params.getVal(getNumParams()-1);
}
double CGaussianNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  if(index<getOutputDim())
    return bias.getVal(index);
  else
    return sigma2;

}
void CGaussianNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(getNumParams()==getOutputDim()+1);
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  DIMENSIONMATCH(getOutputDim()==bias.getCols());
  for(unsigned int j=0; j<getOutputDim(); j++)
    params.setVal(bias.getVal(j), j);
  params.setVal(sigma2, getNumParams()-1);
}
 
void CGaussianNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getCols()==getNumParams());
  DIMENSIONMATCH(g.getRows()==1);
  double nu=0.0;
  double u=0.0;
  double gsigma2=0.0;
  double b=0.0;
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    double gbias = 0.0;
    b = bias.getVal(j);
    for(unsigned int i=0; i<getNumData(); i++)
    {
      nu = 1/(getVarSigma(i, j)+sigma2);
      u=getTarget(i, j) - getMu(i, j)-b;
      u*=nu;
      gbias+=u;
      gsigma2+=nu-u*u;
    }
    g.setVal(gbias, 0, j);
  }
  g.setVal(-0.5*gsigma2, 0, getNumParams()-1);
}

void CGaussianNoise::getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const
{
  gmu = -bias.getVal(j);
  gvs = 1/(sigma2+getVarSigma(i, j));
  gmu += getTarget(i, j)-getMu(i, j);
  gmu *= gvs;
  gvs = 0.5*(gmu*gmu - gvs);
}
  
void CGaussianNoise::getNuG(CMatrix& g, CMatrix& nu, unsigned int index) const
{
  double nuval=0.0;
  double gval=0.0;
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    nuval=1./(sigma2+getVarSigma(index, j));
    if(isnan(nuval))
    {
      cout << "Sigma2 " << sigma2 << endl;
      cout << "varSigma " << getVarSigma(index, j) << endl;
    }
    SANITYCHECK(!isnan(nuval));
    nu.setVal(nuval, index, j);
    gval=getTarget(index, j)-getMu(index, j)-bias.getVal(j);
    g.setVal(gval*nuval, index, j);
  }
}
void CGaussianNoise::updateSites(CMatrix& m, CMatrix& beta, unsigned int actIndex, 
				 const CMatrix& g, const CMatrix& nu, 
				 unsigned int index) const
{
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    beta.setVal(1/sigma2, actIndex, j);
    m.setVal(getTarget(index, j)-bias.getVal(j), actIndex, j);
  }
}
void CGaussianNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muout));
  DIMENSIONMATCH(muout.dimensionsMatch(varSigmaOut));
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int i=0; i<getOutputDim(); i++)
    cout << "Mean Squared Error on output " << i+1 << ": " << yPred.dist2Col(i, yTest, i)/(double)yTest.getRows() << endl;
}

void CGaussianNoise::out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yPred.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  yPred.deepCopy(muTest);
  for(unsigned int j=0; j<getOutputDim(); j++)
    yPred.addCol(j,bias.getVal(j));
}
void CGaussianNoise::out(CMatrix& yPred, CMatrix& stdDeviations, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yPred.dimensionsMatch(stdDeviations));
  out(yPred, muTest, varSigmaTest);
  for(unsigned int i=0; i<stdDeviations.getRows(); i++)
    for(unsigned int j=0; j<stdDeviations.getCols(); j++)
      stdDeviations.setVal(sqrt(varSigmaTest.getVal(i, j) + sigma2), i, j);
}
void CGaussianNoise::likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, 
				 const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(L.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  double arg=0.0;
  double var=0.0;
  for(unsigned int i=0; i<muTest.getRows(); i++)
  {
    for(unsigned int j=0; j<muTest.getCols(); j++)
    {
      arg = yTest.getVal(i, j) - muTest.getVal(i, j) - bias.getVal(j);
      arg *= arg;
      var = varSigmaTest.getVal(i, j) + sigma2;
      arg = 1/sqrt(2*M_PI*var)*exp(-.5*arg*arg/var);
      L.setVal(arg, i, j);
    }
  }
}

double CGaussianNoise::logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, 
				     const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(varSigmaTest));
  double arg=0.0;
  double var=0.0;
  double L=0.0;
  for(unsigned int i=0; i<muTest.getRows(); i++)
  {
    for(unsigned int j=0; j<muTest.getCols(); j++)
    {
      arg = yTest.getVal(i, j) - muTest.getVal(i, j) - bias.getVal(j);
      arg *= arg;
      var = varSigmaTest.getVal(i, j) + sigma2;
      arg = arg/var;
      L += log(var)+arg;
    }
  }  
  L += muTest.getRows()*muTest.getCols()*log(2*M_PI);
  L *= -0.5;
  return L;
}


CScaleNoise::~CScaleNoise()
{
}
void CScaleNoise::initStoreage()
{
  setNumParams(2*getOutputDim());
  initSiteStoreage();
  bias.resize(1, getOutputDim());
  scale.resize(1, getOutputDim());
  clearTransforms();
  sigma2=1e-6;
  setLogConcave(true);
  setSpherical(true);
  setMissing(false);

}
void CScaleNoise::initVals()
{
  activeSites().zeros();
  bias.zeros();
  scale.ones();
}
void CScaleNoise::_init()
{
  setType("scale");
  setName("Scaled Gaussian");
}
void CScaleNoise::initNames()
{
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    setParamName("bias" + ndlstrutil::itoa(j), j);
    setParamName("scale" + ndlstrutil::itoa(j), getOutputDim()+j);
  }
}
void CScaleNoise::initParams()
{
  scale.deepCopy(varCol(*py));
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    scale.setVal(sqrt(scale.getVal(j)), j);
    if(scale.getVal(j)<ndlutil::EPS)
      scale.setVal(ndlutil::EPS, j);
  }
  // TODO: need to check for missing values.
  bias.deepCopy(meanCol(*py));
}
ostream& CScaleNoise::display(ostream& os)
{
  double b = 0.0;
  double s = 0.0;
  os << "Scale Noise: " << endl; 
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    b = bias.getVal(j);
    os << "Bias on process " << j << ": " << b << endl; 
    s = scale.getVal(j);
    os << "Scale on process " << j << ": " << s << endl; 
  }
  return os;
}
void CScaleNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<2*getNumParams()-1);
  if(index<getOutputDim())
    bias.setVal(val, index);
  else if(index<2*getOutputDim())
    scale.setVal(val, index-getOutputDim());
}  
void CScaleNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(getNumParams()==2*getOutputDim());
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  DIMENSIONMATCH(getOutputDim()==bias.getCols());
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    bias.setVal(params.getVal(j), j);
  }
  for(unsigned int j=0; j<scale.getCols(); j++)
  {
    scale.setVal(params.getVal(j+bias.getCols()), j);
  }
}
double CScaleNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<2*getNumParams());
  if(index<getOutputDim())
    return bias.getVal(index);
  else
    return scale.getVal(index-getOutputDim());

}
void CScaleNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(getNumParams()==2*getOutputDim());
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  DIMENSIONMATCH(getOutputDim()==bias.getCols());
  for(unsigned int j=0; j<getOutputDim(); j++)
    params.setVal(bias.getVal(j), j);
  for(unsigned int j=0; j<getOutputDim(); j++)
    params.setVal(scale.getVal(j), j+getOutputDim());
}
 
void CScaleNoise::getGradParams(CMatrix& g) const
{
  throw ndlexceptions::Error("ScaleNoise doesn't have gradients implemented.");
  /*~
    DIMENSIONMATCH(g.getCols()==getNumParams());
    DIMENSIONMATCH(g.getRows()==1);
    double nu=0.0;
    double u=0.0;
    double gsigma2=0.0;
    double b=0.0;
    for(unsigned int j=0; j<getOutputDim(); j++)
    {
    double gbias = 0.0;
    b = bias.getVal(j);
    s = scale.getVal(j);
    for(unsigned int i=0; i<getNumData(); i++)
    {
    nu = 1/(getVarSigma(i, j)+sigma2);
    u=getTarget(i, j) - getMu(i, j)-b;
    u*=nu;
    gbias+=u;
    gsigma2+=nu-u*u;
    }
    g.setVal(gbias, 0, j);
    g.setVal(gscale, 0, j+getOutputDim());
    }
    ~*/
}

void CScaleNoise::getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const
{
  throw ndlexceptions::Error("ScaleNoise doesn't have gradients implemented.");
  /*~
    gmu = -bias.getVal(j);
    gvs = 1/(sigma2+getVarSigma(i, j));
    gmu += getTarget(i, j)-getMu(i, j);
    gmu *= gvs;
    gvs = 0.5*(gmu*gmu - gvs);
    ~*/
}
  
void CScaleNoise::getNuG(CMatrix& g, CMatrix& nu, unsigned int index) const
{
  throw ndlexceptions::Error("ScaleNoise doesn't have gradients implemented.");
  /*~
    double nuval=0.0;
    double gval=0.0;
    for(unsigned int j=0; j<getOutputDim(); j++)
    {
    nuval=1./(sigma2+getVarSigma(index, j));
    if(isnan(nuval))
    {
    cout << "Sigma2 " << sigma2 << endl;
    cout << "varSigma " << getVarSigma(index, j) << endl;
    }
    SANITYCHECK(!isnan(nuval));
    nu.setVal(nuval, index, j);
    gval=getTarget(index, j)-getMu(index, j)-bias.getVal(j);
    g.setVal(gval*nuval, index, j);
    }
    ~*/
}
void CScaleNoise::updateSites(CMatrix& m, CMatrix& beta, unsigned int actIndex, 
			      const CMatrix& g, const CMatrix& nu, 
			      unsigned int index) const
{
  // beta(actIndex,:) := 1/sigma^2
  // m(actIndex,:) :=  ( y(index,:) - bias(:) ) ./ scale(:)
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    beta.setVal(1/sigma2, actIndex, j);
    m.setVal((getTarget(index, j)-bias.getVal(j))/scale.getVal(j), actIndex, j);
  }
}
void CScaleNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const
{
  throw ndlexceptions::Error("ScaleNoise doesn't have site updates implemented.");
  /*~
    DIMENSIONMATCH(yTest.dimensionsMatch(muout));
    DIMENSIONMATCH(muout.dimensionsMatch(varSigmaOut));
    DIMENSIONMATCH(yTest.getCols()==getOutputDim());
    CMatrix yPred(yTest.getRows(), yTest.getCols());
    out(yPred, muout, varSigmaOut);
    for(unsigned int i=0; i<getOutputDim(); i++)
    cout << "Mean Squared Error on output " << i+1 << ": " << yPred.dist2Col(i, yTest, i)/(double)yTest.getRows() << endl;
    ~*/
}

void CScaleNoise::out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yPred.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  yPred.deepCopy(muTest);
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    yPred.scaleCol(j, scale.getVal(j));
    yPred.addCol(j, bias.getVal(j));
  }
}
void CScaleNoise::out(CMatrix& yPred, CMatrix& errorBars, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yPred.dimensionsMatch(errorBars));
  out(yPred, muTest, varSigmaTest);
  for(unsigned int i=0; i<errorBars.getRows(); i++)
    for(unsigned int j=0; j<errorBars.getCols(); j++)
      errorBars.setVal(sqrt(varSigmaTest.getVal(i, j) + sigma2)*scale.getVal(j), i, j);
}
void CScaleNoise::likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, 
			      const CMatrix& yTest) const
{
  throw ndlexceptions::Error("ScaleNoise doesn't have site likelihoods implemented.");
  /*~
    DIMENSIONMATCH(yTest.getCols()==getOutputDim());
    DIMENSIONMATCH(L.dimensionsMatch(muTest));
    DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
    DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
    double arg=0.0;
    double var=0.0;
    for(unsigned int i=0; i<muTest.getRows(); i++)
    {
    for(unsigned int j=0; j<muTest.getCols(); j++)
    {
    arg = yTest.getVal(i, j) - muTest.getVal(i, j) - bias.getVal(j);
    arg *= arg;
    var = varSigmaTest.getVal(i, j) + sigma2;
    arg = 1/sqrt(2*M_PI*var)*exp(-.5*arg*arg/var);
    L.setVal(arg, i, j);
    }
    }
    ~*/
}

double CScaleNoise::logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, 
				  const CMatrix& yTest) const
{
  throw ndlexceptions::Error("ScaleNoise doesn't have site loglikelihoods implemented.");
  /*~
    DIMENSIONMATCH(yTest.getCols()==getOutputDim());
    DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
    DIMENSIONMATCH(yTest.dimensionsMatch(varSigmaTest));
    double arg=0.0;
    double var=0.0;
    double L=0.0;
    for(unsigned int i=0; i<muTest.getRows(); i++)
    {
    for(unsigned int j=0; j<muTest.getCols(); j++)
    {
    arg = yTest.getVal(i, j) - muTest.getVal(i, j) - bias.getVal(j);
    arg *= arg;
    var = varSigmaTest.getVal(i, j) + sigma2;
    arg = arg/var;
    L += log(var)+arg;
    }
    }  
    L += muTest.getRows()*muTest.getCols()*log(2*M_PI);
    L *= -0.5;
    return L;
    ~*/
}





CProbitNoise::~CProbitNoise()
{
}
void CProbitNoise::initStoreage()
{
  setNumParams(getOutputDim());
  initSiteStoreage();
  bias.resize(1, getOutputDim());
  clearTransforms();
  sigma2 = 1e-6;
  setLogConcave(true);
  setSpherical(false);
  setMissing(false);

}
void CProbitNoise::initVals()
{
  activeSites().zeros();
  bias.zeros();

}
void CProbitNoise::_init()
{
  setType("probit");
  setName("Probit");
  ellBias = 0.0;
}
void CProbitNoise::initNames()
{
  for(unsigned int j=0; j<getOutputDim(); j++)
    setParamName("bias" + ndlstrutil::itoa(j), j); 
}
void CProbitNoise::initParams()
{
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    double nClass1=0.0;
    for(unsigned int i=0; i<getNumData(); i++)
    {
      if(py->getVal(i, j)==1)
	nClass1++;
    }
    bias.setVal(ndlutil::invCumGaussian(nClass1/(double)getNumData()), j);
  }
}

ostream& CProbitNoise::display(ostream& os)
{
  double b = 0.0;
  os << "Probit noise: " << endl;
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    b = bias.getVal(j);
    os << "Bias on process " << j << ": " << b << endl; 
  }
  return os;
}
void CProbitNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  bias.setVal(val, index);
}  
void CProbitNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(getNumParams()==getOutputDim());
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  DIMENSIONMATCH(getOutputDim()==bias.getCols());
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    bias.setVal(params.getVal(j), j);
  }
}
double CProbitNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  if(index<getOutputDim())
    return bias.getVal(index);
  return 0;
}
void CProbitNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(getNumParams()==getOutputDim());
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  DIMENSIONMATCH(getOutputDim()==bias.getCols());
  for(unsigned int j=0; j<getOutputDim(); j++)
    params.setVal(bias.getVal(j), j);
}
 
void CProbitNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getCols()==getNumParams());
  DIMENSIONMATCH(g.getRows()==1);
  double c=0.0;
  double u=0.0;
  double b=0.0;
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    double gbias = 0.0;
    b = bias.getVal(j);
    for(unsigned int i=0; i<getNumData(); i++)
    {
      c = getTarget(i, j)/sqrt(getVarSigma(i, j)+sigma2);
      u=c*(getMu(i, j)+b);
      u=ndlutil::gradLnCumGaussian(u);
      gbias+=u*c;
    }
    g.setVal(gbias, 0, j);
  }
}
void CProbitNoise::getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const
{
  double b = bias.getVal(j);
  double c = getTarget(i, j)/sqrt(sigma2+getVarSigma(i, j));
  double u = c*(getMu(i, j) + b);
  gmu = ndlutil::gradLnCumGaussian(u)*c;
  gvs = -0.5*c*u*gmu;
}

void CProbitNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muout));
  DIMENSIONMATCH(muout.dimensionsMatch(varSigmaOut));
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    int error=0;
    for(unsigned int i=0; i<yTest.getRows(); i++)
    {
      if(yPred.getVal(i, j)!=yTest.getVal(i, j))
	error++;
    }
    double err=((double)error)/((double)yTest.getRows());
    err=err*100.0;
    cout << "Classification error on output " << j+1 << ": " << err << "%." << endl;
  }
}
void CProbitNoise::out(CMatrix& yOut, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yOut.dimensionsMatch(probOut));
  out(yOut, muTest, varSigmaTest);
  likelihoods(probOut, muTest, varSigmaTest, yOut);
}

void CProbitNoise::out(CMatrix& yOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yOut.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  for(unsigned int j=0; j<yOut.getCols(); j++)
  {
    double b = bias.getVal(j);
    for(unsigned int i=0; i<yOut.getRows(); i++)	
    {
      if(muTest.getVal(i, j)>-b)
	yOut.setVal(1.0, i, j);
      else
	yOut.setVal(-1.0, i, j);
    }
  }
}
void CProbitNoise::likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, 
			       const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(L.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  double arg=0.0;
  for(unsigned int j=0; j<muTest.getCols(); j++)
  {
    double b = bias.getVal(j);
    for(unsigned int i=0; i<muTest.getRows(); i++)
    {
      arg = yTest.getVal(i, j)*(muTest.getVal(i, j) + b);
      arg =arg/sqrt(varSigmaTest.getVal(i, j) + sigma2);
      L.setVal(ndlutil::cumGaussian(arg), i, j);
    }
  }
}

double CProbitNoise::logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, 
				   const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(varSigmaTest));
  double arg=0.0;
  double var=0.0;
  double L=0.0;
  for(unsigned int i=0; i<muTest.getRows(); i++)
  {
    for(unsigned int j=0; j<muTest.getCols(); j++)
    {
      arg = muTest.getVal(i, j) + bias.getVal(j);
      arg *= getTarget(i, j);
      var = sqrt(varSigmaTest.getVal(i, j) + sigma2);
      L += ndlutil::lnCumGaussian(arg/var);
    }
  }  
  return L;
}

CNcnmNoise::~CNcnmNoise()
{
}
void CNcnmNoise::initStoreage()
{
  if(isSplitGamma())
    setNumParams(getOutputDim()+2);
  else
    setNumParams(getOutputDim()+1);
  initSiteStoreage();
  bias.resize(1, getOutputDim());
  // sigmoid transforms on gamman and gammap.
  clearTransforms();
  addTransform(new CSigmoidTransform(), getNumParams()-1);
  if(isSplitGamma())
    addTransform(new CSigmoidTransform(), getNumParams()-2);
  // Match GPmat ncnmNoiseParamInit: noise.sigma2 = eps (not optimised).
  sigma2 = ndlutil::EPS;
  width = 1.0;
  
  // sigma2 isn't treated as a parameter.
  setLogConcave(false);
  setSpherical(false);
  setMissing(true);

}
void CNcnmNoise::initVals()
{
  activeSites().zeros();
  bias.zeros();

}
void CNcnmNoise::_init()
{
  setType("ncnm");
  setName("Null Category");
}
void CNcnmNoise::initNames()
{
  for(unsigned int j=0; j<getOutputDim(); j++)
    setParamName("bias" + ndlstrutil::itoa(j), j);
}
void CNcnmNoise::initParams()
{
  double nClass1=0.0;
  double nClass2=0.0;
  double nMissing=0.0;
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    for(unsigned int i=0; i<getNumData(); i++)
    {
      if(py->getVal(i, j)==1.0)	    
	nClass1++;
      else if(py->getVal(i, j)==-1.0)
	nClass2++;
      else
	nMissing++;
    }
    bias.setVal(ndlutil::invCumGaussian(nClass1/(nClass1+nClass2)), j);
  }
  gamman=nMissing/(double)getNumData();
  gammap = gamman;
  
}
ostream& CNcnmNoise::display(ostream& os)
{
  double b = 0.0;
  os << "Ncnm noise: " << endl;
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    b = bias.getVal(j);
    os << "Bias on process " << j << ": " << b << endl; 
  }
  os << "Missing label probability for -ve class: " << gamman << endl;
  os << "Missing label probability for +ve class: " << gammap << endl;
  return os;
}
void CNcnmNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  if(index<getOutputDim())
  {
    bias.setVal(val, index);
    return;
  }
  if(index==getOutputDim())
  {
    gamman=val;
    return;
  }
  if(index==getOutputDim()+1)
  {
    gammap=val;
    return;
  }
}  
void CNcnmNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  unsigned int nProc = getOutputDim();
  for(unsigned int j=0; j<nProc; j++)
  {
    bias.setVal(params.getVal(j), j);
  }
  gamman=params.getVal(nProc);
  if(isSplitGamma())
    gammap=params.getVal(nProc+1);
  else
    gammap=gamman;
}
double CNcnmNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  if(index<getOutputDim())
    return bias.getVal(index);
  if(index==getOutputDim())
    return gamman;
  if(index==getOutputDim()+1) // this only happens if the gammas aren't split.
    return gammap;
  return -1;
}
void CNcnmNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  unsigned int nProc=getOutputDim();
  for(unsigned int j=0; j<nProc; j++)
    params.setVal(bias.getVal(j), j);
  params.setVal(gamman, nProc);
  if(isSplitGamma())
    params.setVal(gammap, nProc+1);
}
 
void CNcnmNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getCols()==getNumParams());
  DIMENSIONMATCH(g.getRows()==1);
  double ggamman=0.0;
  double ggammap=0.0;
  double halfWidth = width/2.0;
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    double gbias = 0.0;
    double b = bias.getVal(j);
    for(unsigned int i=0; i<getNumData(); i++)
    {
      double muAdj = getMu(i, j)+b;
      double c = 1/sqrt(sigma2+getVarSigma(i, j));
      double targVal = getTarget(i, j);
      if(targVal==-1.0)
      {
	muAdj+=halfWidth;
	muAdj=muAdj*c;
	gbias-= c*ndlutil::gradLnCumGaussian(-muAdj);
	ggamman-=1.0/(1.0-gamman);
      }
      else if(targVal==1.0)
      {
	muAdj-=halfWidth;
	muAdj*=c;
	gbias+=c*ndlutil::gradLnCumGaussian(muAdj);
	ggammap-=1.0/(1.0-gammap);
      }
      else
      {
	muAdj+=halfWidth;
	double u=muAdj*c;
	double uprime=(muAdj-width)*c;
	double lndenom = ndlutil::lnCumGaussSum(-u, uprime, gamman, gammap);
	double lnNumer1 = log(gamman) - ndlutil::HALFLOGTWOPI -.5*u*u;
	double lnNumer2 = log(gammap) - ndlutil::HALFLOGTWOPI -.5*uprime*uprime;
	double B1 = exp(lnNumer1-lndenom);
	double B2 = exp(lnNumer2-lndenom);
	gbias+=c*(B2-B1);
	ggammap+=exp(ndlutil::lnCumGaussian(uprime)-lndenom);
	ggamman+=exp(ndlutil::lnCumGaussian(-u)-lndenom);
      }
	      
    }
    g.setVal(gbias, 0, j);
  }
  if(isSplitGamma())
  {
    g.setVal(ggamman, 0, getOutputDim());
    g.setVal(ggammap, 0, getOutputDim()+1);
  }
  else
    g.setVal(ggamman+ggammap, 0, getOutputDim());

}
void CNcnmNoise::getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const
{
  double b = bias.getVal(j);
  double c = 1.0/sqrt(sigma2+getVarSigma(i, j));
  double muAdj = getMu(i, j)+b;
  double targ = getTarget(i, j);
  double halfWidth = width/2.0;
  if(targ==-1.0)
  {
    muAdj+=halfWidth;
    muAdj*=c;
    gmu=-ndlutil::gradLnCumGaussian(-muAdj)*c;
    gvs=-.5*c*muAdj*gmu;
  }
  else if(targ==1.0)
  {
    muAdj-=halfWidth;
    muAdj*=c;
    gmu=ndlutil::gradLnCumGaussian(muAdj)*c;
    gvs=-.5*c*muAdj*gmu;
  }
  else // missing data
  {
    muAdj+=halfWidth;
    double u=c*muAdj;
    double uprime=(muAdj-width)*c;
    double lndenom=ndlutil::lnCumGaussSum(-u, uprime, gamman, gammap);
    double lnNumer1 = log(gamman) - ndlutil::HALFLOGTWOPI -.5*(u*u);
    double lnNumer2 = log(gammap) - ndlutil::HALFLOGTWOPI -.5*(uprime*uprime);
    double B1 = exp(lnNumer1 - lndenom);
    double B2 = exp(lnNumer2 - lndenom);
    gmu = c*(B2-B1);
    gvs = -.5*c*c*(uprime*B2-u*B1);
      
  }
  if(isnan(gmu))
    throw ndlexceptions::Error("gmu is NaN");
  if(isnan(gvs))
    throw ndlexceptions::Error("gvs is NaN");
}  
void CNcnmNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muout));
  DIMENSIONMATCH(muout.dimensionsMatch(varSigmaOut));
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    unsigned int missNum=0;
    unsigned int error=0;
    for(unsigned int i=0; i<yTest.getRows(); i++)
    {
      if(yTest.getVal(i, j)==0.0 || isnan(yTest.getVal(i, j)))
	missNum++;
      else if(yPred.getVal(i, j)!=yTest.getVal(i, j))
	error++;
    }
    if(missNum==yTest.getRows())
      cout << "No labels on output" << j+1 << "." << endl;
    double err=(double)error/(double)(yTest.getRows()-missNum);
    err=err*100.0;
    double miss=(double)missNum/(double)yTest.getRows();
    miss=miss*100.0;
    cout << "Classification error on output " << j+1 << ": " << err << "%, with " << miss << " missing values." << endl;
  }
}
  
void CNcnmNoise::out(CMatrix& yOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yOut.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  for(unsigned int j=0; j<yOut.getCols(); j++)
  {
    double b = bias.getVal(j);
    for(unsigned int i=0; i<yOut.getRows(); i++)	
    {
      double muVal = muTest.getVal(i, j);
      if(muVal>-b)
	yOut.setVal(1.0, i, j);
      else
	yOut.setVal(-1.0, i, j);
    }
  }
}
void CNcnmNoise::out(CMatrix& yOut, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yOut.dimensionsMatch(probOut));
  out(yOut, muTest, varSigmaTest);
  likelihoods(probOut, muTest, varSigmaTest, yOut);
}
void CNcnmNoise::likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, 
			     const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(L.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  for(unsigned int j=0; j<muTest.getCols(); j++)
  {
    double b = bias.getVal(j);
    for(unsigned int i=0; i<muTest.getRows(); i++)
    {
      double muAdj=muTest.getVal(i, j) + b;
      double c=1/sqrt(sigma2+varSigmaTest.getVal(i, j));	  
      double targVal=yTest.getVal(i, j);
      if(targVal==1.0)
      {
	//muAdj-=halfWidth;
	L.setVal(ndlutil::cumGaussian(muAdj*c), i, j);
      }
      else if(targVal==-1.0)
      {
	//muAdj+=halfWidth;
	L.setVal(ndlutil::cumGaussian(-muAdj*c), i, j);
      }
    }
  }
}

double CNcnmNoise::logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, 
				 const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(varSigmaTest));
  double L=0.0;
  double halfWidth = width/2.0;
  double logPosGamma = log(1.0-gammap);
  double logNegGamma = log(1.0-gamman);
  for(unsigned int j=0; j<muTest.getCols(); j++)    
  {
    double b=bias.getVal(j);
    for(unsigned int i=0; i<muTest.getRows(); i++)
    {
      double muAdj = muTest.getVal(i, j) + b;
      double c=1/sqrt(sigma2+varSigmaTest.getVal(i, j));
      double targVal=yTest.getVal(i, j);
      if(targVal==1.0)
      {
	muAdj-=halfWidth;
	L+=ndlutil::lnCumGaussian(muAdj*c);
	L+=logPosGamma;
	      
      }
      else if(targVal==-1.0)
      {
	muAdj+=halfWidth;
	L+=ndlutil::lnCumGaussian(-muAdj*c);
	L+=logNegGamma;
      }
      else // missing data.
      {
	muAdj+=halfWidth;
	double u=muAdj*c;
	double uprime=(muAdj-width)*c;
	L+=ndlutil::lnCumGaussSum(-u, uprime, gamman, gammap);	      
      }
    }
  }  
  return L;
}
void CNcnmNoise::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "numData", getNumData());
  writeToStream(out, "outputDim", getOutputDim());
  writeToStream(out, "numParams", getNumParams());
  writeToStream(out, "gammaSplit", isSplitGamma());
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
}

void CNcnmNoise::readParamsFromStream(istream& in)
{
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setNumData(readIntFromStream(in, "numData"));
  setOutputDim(readIntFromStream(in, "outputDim"));
  int numPar = readIntFromStream(in, "numParams");
  setSplitGamma(readBoolFromStream(in, "gammaSplit"));
  initStoreage();

  CMatrix par(1, numPar);
  par.fromStream(in);
  setNumData(1);
  initStoreage();
  if(numPar==getNumParams())
    setParams(par);
  else
    throw ndlexceptions::StreamFormatError("numParams", "Number of parameters in file does not match computed number.");
}
COrderedNoise::~COrderedNoise()
{
}
void COrderedNoise::initStoreage()
{
  setNumParams(getOutputDim()+getNumCategories()-2);
  initSiteStoreage();
  bias.resize(1, getOutputDim());
  widths.resize(1, getNumCategories()-2);
  gwidth.resize(1, getNumCategories()-2);
  // sigmoid transforms on gamman and gammap.
  clearTransforms();
  for(unsigned int i=0; i<getNumCategories()-2; i++)
    addTransform(CTransform::defaultPositive(), i+getOutputDim());
  sigma2 = 0.1;
  
  // sigma2 isn't treated as a parameter.
  setLogConcave(true);
  setSpherical(false);
  setMissing(true);

}
void COrderedNoise::initVals()
{
  activeSites().zeros();
  bias.zeros();
  widths.setVals(1/(getNumCategories()-2));

}
void COrderedNoise::_init()
{
  setType("ordered");
  setName("Ordered Categorical");
  ellBias = 0.0;
}
void COrderedNoise::initNames()
{
  for(unsigned int j=0; j<getOutputDim(); j++)
    setParamName("bias" + ndlstrutil::itoa(j), j);
  for(unsigned int j=0; j<getNumCategories()-2; j++)
    setParamName("width" + ndlstrutil::itoa(j), j+getOutputDim());
}
void COrderedNoise::initParams()
{
  bias.deepCopy(meanCol(*py));
}
ostream& COrderedNoise::display(ostream& os)
{
  double b = 0.0;
  os << "Ordered Categorical noise: " << endl;
  for(unsigned int j=0; j<bias.getCols(); j++)
  {
    b = bias.getVal(j);
    os << "Bias on process " << j << ": " << b << endl; 
  }
  for(unsigned int j=0; j<widths.getCols(); j++)
  {
    b = widths.getVal(j);
    os << "Width for category " << j << ": " << b << endl; 
  }
  return os;
}
void COrderedNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  if(index<getOutputDim())
  {
    bias.setVal(val, index);
    return;
  }
  if(index>=getOutputDim())
  {
    widths.setVal(val, index-getOutputDim());
    return;
  }
}  
void COrderedNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  unsigned int nProc = getOutputDim();
  for(unsigned int j=0; j<nProc; j++)
  {
    bias.setVal(params.getVal(j), j);
  }
  for(unsigned int j=0; j<getNumCategories()-2; j++)
    widths.setVal(params.getVal(j+getOutputDim()), j);
}
double COrderedNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index>=0);
  BOUNDCHECK(index<getNumParams());
  if(index<getOutputDim())
    return bias.getVal(index);
  if(index>=getOutputDim())
    return widths.getVal(index-getOutputDim());
  return -1;
}
void COrderedNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getCols()==getNumParams());
  DIMENSIONMATCH(params.getRows()==1);
  int nProc=getOutputDim();
  for(unsigned int j=0; j<nProc; j++)
    params.setVal(bias.getVal(j), j);
  for(unsigned int j=0; j<getNumCategories()-2; j++)
    params.setVal(widths.getVal(j), j+getOutputDim());
}
 
void COrderedNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getCols()==getNumParams());
  DIMENSIONMATCH(g.getRows()==1);
  g.zeros();
  gwidth.zeros();
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    double gbias = 0.0;
    double b = bias.getVal(j);
    for(unsigned int i=0; i<getNumData(); i++)
    {
      double muAdj = getMu(i, j)+b;
      // Test NaN before casting: on ARM, (int)NaN is 0 and would look like category 0.
      if(isnan(getTarget(i, j)))
	continue;
      int targVal = (int)getTarget(i, j);
      SANITYCHECK(getTarget(i, j)==(double)targVal);
      double c = 1/sqrt(sigma2+getVarSigma(i, j));
      if(targVal==0)
      {
	muAdj*=c;
	gbias -= c*ndlutil::gradLnCumGaussian(-muAdj);
      }
      else if(targVal>0 && targVal<getNumCategories()-1)
      {
	for(unsigned int k=0; k<targVal-1; k++)
	  muAdj -= widths.getVal(k);
	double u = muAdj*c;
	double uprime = (muAdj-widths.getVal(targVal-1))*c;
	double B1 = ndlutil::gaussOverDiffCumGaussian(u, uprime, 1);
	double B2 = ndlutil::gaussOverDiffCumGaussian(u, uprime, 2);
	gbias += c*(B1-B2);
	double addPart = c*B2;
	for(unsigned int k=0; k<targVal; k++)
	  gwidth.setVal(gwidth.getVal(k)+addPart, k);
	if(targVal>1)
	{
	  addPart = c*B1;
	  for(unsigned int k=0; k<targVal-1; k++)
	    gwidth.setVal(gwidth.getVal(k)-addPart, k);	      
	}
      }
      else if(targVal==getNumCategories()-1)
      {
	for(unsigned int k=0; k<targVal-1; k++)
	  muAdj -= widths.getVal(k);
	muAdj*=c;
	double addPart = c*ndlutil::gradLnCumGaussian(muAdj);
	gbias += addPart;
	if(getNumCategories()>2)
	  for(unsigned int k=0; k<gwidth.getCols(); k++)
	    gwidth.setVal(gwidth.getVal(k)-addPart, k);
      }
      else
	throw ndlexceptions::Error("This point should never be reached in COrderedNoise::getGradParams");
    }
    g.setVal(gbias, 0, j);
  }
  for(unsigned int j=0; j<getNumCategories()-2; j++)
    g.setVal(gwidth.getVal(j), j+getOutputDim());
}
void COrderedNoise::getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const
{
  double b = bias.getVal(j);
  double c = 1.0/sqrt(sigma2+getVarSigma(i, j));
  double muAdj = getMu(i, j)+b;
  // Test NaN before casting: on ARM, (int)NaN is 0 and would look like category 0.
  if(isnan(getTarget(i, j)))
  {
    gmu = 0.0;
    gvs = 0.0;
    return;
  }
  int targ = (int)getTarget(i, j);
  SANITYCHECK(getTarget(i, j)==(double)targ);
  if(targ==0)
  {
    muAdj*=c;
    gmu=-c*ndlutil::gradLnCumGaussian(-muAdj);
    gvs=-.5*gmu*c*muAdj;
  }
  
  else if(targ>0 && targ < getNumCategories() - 1)
  {
    for(unsigned int k=0; k<targ-1; k++)
      muAdj-=widths.getVal(k);
    double u = muAdj*c;
    double uprime = (muAdj - widths.getVal(targ-1))*c;
    double B1 = ndlutil::gaussOverDiffCumGaussian(u, uprime, 1);
    double B2 = ndlutil::gaussOverDiffCumGaussian(u, uprime, 2);
    gmu = c*(B1-B2);
    gvs = -.5*c*c*(u*B1 - uprime*B2);
  }
  else if (targ==getNumCategories()-1) // highest category
  {
    for(unsigned int k=0; k<targ-1; k++)
      muAdj-=widths.getVal(k);
    muAdj*=c;
    gmu = c*ndlutil::gradLnCumGaussian(muAdj);
    gvs = -.5*gmu*c*muAdj;
      
  }
  else
    throw ndlexceptions::Error("This point should never be reached in COrderedNoise::getGradInputs");
}  
void COrderedNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muout));
  DIMENSIONMATCH(muout.dimensionsMatch(varSigmaOut));
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j=0; j<getOutputDim(); j++)
  {
    int error=0;
    for(unsigned int i=0; i<yTest.getRows(); i++)
    {
      if(yPred.getVal(i, j)!=yTest.getVal(i, j))
	error++;
    }
    double err=(double)error/(double)(yTest.getRows());
    err=err*100.0;
    cout << "Classification error on output " << j+1 << ": " << err << "% " << endl;
  }
}
  
void COrderedNoise::out(CMatrix& yOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yOut.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  for(unsigned int j=0; j<yOut.getCols(); j++)
  {
    double b = bias.getVal(j);
    for(unsigned int i=0; i<yOut.getRows(); i++)	
    {
      double muVal = muTest.getVal(i, j)+b;
      double tot = 0.0;
      int counter = 1;
      if(muVal<0)
      {
	yOut.setVal(0.0, i, j);
	continue;
      }
      for(unsigned int k=0; k<widths.getCols(); k++)
	if (muVal>tot)
	{
	  tot+=widths.getVal(i);
	  counter++;
	}
	else
	{
	  yOut.setVal((double)counter, i, j);
	  continue;
	}
    }
      
  }
}
void COrderedNoise::out(CMatrix& yOut, CMatrix& probOut, const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yOut.dimensionsMatch(probOut));
  out(yOut, muTest, varSigmaTest);
  likelihoods(probOut, muTest, varSigmaTest, yOut);
}
void COrderedNoise::likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, 
				const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(L.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
  for(unsigned int j=0; j<muTest.getCols(); j++)
  {
    double b = bias.getVal(j);
    for(unsigned int i=0; i<muTest.getRows(); i++)
    {
      double muAdj=muTest.getVal(i, j) + b;
      double c=1/sqrt(sigma2+varSigmaTest.getVal(i, j));
      // Test NaN before casting: on ARM, (int)NaN is 0 and would look like category 0.
      if(isnan(yTest.getVal(i, j)))
      {
	L.setVal(1.0, i, j);
	continue;
      }
      int targVal=(int)yTest.getVal(i, j);
      SANITYCHECK(yTest.getVal(i, j)==(double)targVal);
      if(targVal==0)
      {
	L.setVal(ndlutil::cumGaussian(-muAdj*c), i, j);
      }
      else if(targVal>0 && targVal < getNumCategories()-1)
      {
	for(unsigned int k=0; k<targVal-1; k++)
	  muAdj+=widths.getVal(k);
	L.setVal(ndlutil::cumGaussian(muAdj*c)-ndlutil::cumGaussian((muAdj - widths.getVal(targVal-1))*c), i, j);
      }
      else if(targVal==getNumCategories()-1)
      {
	for(unsigned int k=0; k<targVal-1; k++)
	  muAdj+=widths.getVal(k);
	L.setVal(ndlutil::cumGaussian(muAdj*c), i, j);
      }
      else
	throw ndlexceptions::Error("This point should never be reached in COrderedNoise::getGradInputs");
    }
  }
}

double COrderedNoise::logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, 
				    const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.getCols()==getOutputDim());
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  DIMENSIONMATCH(yTest.dimensionsMatch(varSigmaTest));
  double L=0.0;
  for(unsigned int j=0; j<muTest.getCols(); j++)
  {
    double b = bias.getVal(j);
    for(unsigned int i=0; i<muTest.getRows(); i++)
    {
      double muAdj=muTest.getVal(i, j) + b;
      double c=1/sqrt(sigma2+varSigmaTest.getVal(i, j));
      // Test NaN before casting: on ARM, (int)NaN is 0 and would look like category 0.
      if(isnan(yTest.getVal(i, j)))
	continue;
      int targVal=(int)yTest.getVal(i, j);
      SANITYCHECK(yTest.getVal(i, j)==(double)targVal);
      if(targVal==0)
      {
	L+=ndlutil::lnCumGaussian(-muAdj*c);
      }
      else if(targVal>0 && targVal < getNumCategories()-1)
      {
	for(unsigned int k=0; k<targVal-1; k++)
	  muAdj-=widths.getVal(k);
	double u = muAdj*c;
	double uprime = (muAdj - widths.getVal(targVal-1))*c;
	L+=ndlutil::lnDiffCumGaussian(u, uprime);
      }
      else if(targVal==getNumCategories()-1)
      {
	for(unsigned int k=0; k<targVal-1; k++)
	  muAdj-=widths.getVal(k);
	L+=ndlutil::lnCumGaussian(muAdj*c);
      }
      else
	throw ndlexceptions::Error("This point should never be reached in COrderedNoise::getGradInputs");

    }
  }  
  return L;
}
void COrderedNoise::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "numData", getNumData());
  writeToStream(out, "outputDim", getOutputDim());
  writeToStream(out, "numParams", getNumParams());
  writeToStream(out, "numCategories", getNumCategories());
  CMatrix par(1, getNumParams());
  getParams(par);
  par.toStream(out);
}

void COrderedNoise::readParamsFromStream(istream& in)
{
//   string tbaseType = getBaseTypeStream(in);
//   if(tbaseType != getBaseType())
//     throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, " + getType() + ".");
//   string ttype = getTypeStream(in);
//   if(ttype != getType())
//     throw ndlexceptions::StreamFormatError("type", "Error mismatch between saved type, " + ttype + ", and Class type, " + getType() + ".");
  setNumData(readIntFromStream(in, "numData"));
  setOutputDim(readIntFromStream(in, "outputDim"));
  unsigned int numPar = readIntFromStream(in, "numParams");
  setNumCategories(readIntFromStream(in, "numCategories"));
  initStoreage();

  CMatrix par(1, numPar);
  par.fromStream(in);
  setNumData(1);
  initStoreage();
  if(numPar==getNumParams())
    setParams(par);
  else
    throw ndlexceptions::StreamFormatError("numParams", "Number of parameters in file does not match computed number.");
      
}

void writeNoiseToStream(const CNoise& noise, ostream& out)
{
  noise.toStream(out);
}

CNoise* readNoiseFromStream(istream& in)
{
  double ver = CStreamInterface::readVersionFromStream(in); 
  string tbaseType = CStreamInterface::getBaseTypeStream(in);
  if(tbaseType != "noise")
    throw ndlexceptions::StreamFormatError("baseType", "Error mismatch between saved base type, " + tbaseType + ", and Class base type, noise.");
  CNoise* pnoise;

  string type = CStreamInterface::getTypeStream(in);
  if(type=="probit")
    pnoise = new CProbitNoise();
  else if(type=="ncnm")
    pnoise = new CNcnmNoise();
  else if(type=="gaussian")
    pnoise = new CGaussianNoise();
  else if(type=="ordered")
    pnoise = new COrderedNoise();
  else if(type=="scale")
    pnoise = new CScaleNoise();
  else if(type=="skew_gaussian")
    pnoise = new CSkewGaussianNoise();
  else if(type=="skew_exponential")
    pnoise = new CSkewExponentialNoise();
  else if(type=="student_t")
    pnoise = new CStudentTNoise();
  else if(type=="poisson")
    pnoise = new CPoissonNoise();
  else if(type=="binomial")
    pnoise = new CBinomialNoise();
  else if(type=="exponential")
    pnoise = new CExponentialNoise();
  else if(type=="gamma")
    pnoise = new CGammaNoise();
  else
    throw ndlexceptions::StreamFormatError("type", "Unknown noise type " + type);
  pnoise->readParamsFromStream(in);  
  return pnoise;
}

// ---- CIP-003A SkewGaussian / SkewExponential ---------------------------------

namespace {
struct SkewGaussEllCtx {
  double y;
  double variance;
  double alpha;
};
double skewGaussLogLik(double f, void* vctx)
{
  const SkewGaussEllCtx* c = static_cast<const SkewGaussEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewGaussianLaplaceDerivs(logp, d1, W, dW, c->y, f, c->variance, c->alpha);
  return logp;
}
double skewGaussLogLikPrime(double f, void* vctx)
{
  const SkewGaussEllCtx* c = static_cast<const SkewGaussEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewGaussianLaplaceDerivs(logp, d1, W, dW, c->y, f, c->variance, c->alpha);
  return d1;
}
struct SkewExpEllCtx {
  double y;
  double rate;
  double alpha;
};
double skewExpLogLik(double f, void* vctx)
{
  const SkewExpEllCtx* c = static_cast<const SkewExpEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewExponentialLaplaceDerivs(logp, d1, W, dW, c->y, f, c->rate, c->alpha);
  return logp;
}
double skewExpLogLikPrime(double f, void* vctx)
{
  const SkewExpEllCtx* c = static_cast<const SkewExpEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewExponentialLaplaceDerivs(logp, d1, W, dW, c->y, f, c->rate, c->alpha);
  return d1;
}
} // namespace

CSkewGaussianNoise::~CSkewGaussianNoise() {}

void CSkewGaussianNoise::_init()
{
  setType("skew_gaussian");
  setName("SkewGaussian");
}

void CSkewGaussianNoise::initStoreage()
{
  setNumParams(2);
  initSiteStoreage();
  clearTransforms();
  addTransform(CTransform::defaultPositive(), 0); // variance
  setLogConcave(false);
  setSpherical(true);
  setMissing(false);
}

void CSkewGaussianNoise::initNames()
{
  setParamName("variance", 0);
  setParamName("alpha", 1);
}

void CSkewGaussianNoise::initVals()
{
  activeSites().zeros();
  variance = 1.0;
  alpha = 0.0;
}

void CSkewGaussianNoise::initParams()
{
  // Reasonable default: match residual scale; alpha starts at 0 (Gaussian).
  if(py != 0 && getNumData() > 0)
  {
    double s = 0.0;
    for(unsigned int i = 0; i < getNumData(); i++)
      for(unsigned int j = 0; j < getOutputDim(); j++)
      {
        const double v = getTarget(i, j);
        s += v * v;
      }
    variance = s / static_cast<double>(getNumData() * getOutputDim());
    if(!(variance > 0.0))
      variance = 1.0;
  }
}

ostream& CSkewGaussianNoise::display(ostream& os)
{
  os << "SkewGaussian Noise: variance=" << variance
     << " alpha=" << alpha << endl;
  return os;
}

void CSkewGaussianNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index < getNumParams());
  if(index == 0)
    setVariance(val);
  else
    setAlpha(val);
}

void CSkewGaussianNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getRows() == 1);
  DIMENSIONMATCH(params.getCols() == getNumParams());
  setVariance(params.getVal(0));
  setAlpha(params.getVal(1));
}

double CSkewGaussianNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index < getNumParams());
  return (index == 0) ? variance : alpha;
}

void CSkewGaussianNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getRows() == 1);
  DIMENSIONMATCH(params.getCols() == getNumParams());
  params.setVal(variance, 0);
  params.setVal(alpha, 1);
}

void CSkewGaussianNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getRows() == 1);
  DIMENSIONMATCH(g.getCols() == getNumParams());
  // Pointwise grads at cavity mean (var→0 / Laplace MAP); GPy dlogpdf_link_dθ.
  double gVar = 0.0;
  double gAlpha = 0.0;
  const double sigma = sqrt(variance);
  for(unsigned int i = 0; i < getNumData(); i++)
  {
    for(unsigned int j = 0; j < getOutputDim(); j++)
    {
      const double e = getTarget(i, j) - getMu(i, j);
      const double z = e / sigma;
      const double delta = alpha * z;
      const double lam = ndlnoise::invMillsRatio(delta);
      gVar += -0.5 / variance + 0.5 * (e * e) / (variance * variance)
              - lam * delta / (2.0 * variance);
      gAlpha += lam * z;
    }
  }
  g.setVal(gVar, 0);
  g.setVal(gAlpha, 1);
}

void CSkewGaussianNoise::getGradInputs(double& gmu, double& gvs,
                                       unsigned int i, unsigned int j) const
{
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewGaussianLaplaceDerivs(logp, d1, W, dW,
                                      getTarget(i, j), getMu(i, j),
                                      variance, alpha);
  gmu = d1;
  gvs = -0.5 * W; // 0.5 * d²logp/df²
}

void CSkewGaussianNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut,
                              const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muout));
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j = 0; j < getOutputDim(); j++)
    cout << "Mean Squared Error on output " << j + 1 << ": "
         << yPred.dist2Col(j, yTest, j) / static_cast<double>(yTest.getRows())
         << endl;
}

void CSkewGaussianNoise::out(CMatrix& yPred, const CMatrix& muTest,
                             const CMatrix& varSigmaTest) const
{
  DIMENSIONMATCH(yPred.dimensionsMatch(muTest));
  const double sigma = sqrt(variance);
  const double delta = alpha / sqrt(1.0 + alpha * alpha);
  const double shift = sigma * delta * sqrt(2.0 / M_PI);
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
      yPred.setVal(muTest.getVal(i, j) + shift, i, j);
}

void CSkewGaussianNoise::out(CMatrix& yPred, CMatrix& errorBarOut,
                             const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  out(yPred, muTest, varSigmaTest);
  const double delta = alpha / sqrt(1.0 + alpha * alpha);
  const double noise = variance * (1.0 - 2.0 * delta * delta / M_PI);
  for(unsigned int i = 0; i < errorBarOut.getRows(); i++)
    for(unsigned int j = 0; j < errorBarOut.getCols(); j++)
      errorBarOut.setVal(sqrt(varSigmaTest.getVal(i, j) + noise), i, j);
}

void CSkewGaussianNoise::likelihoods(CMatrix& L, const CMatrix& muTest,
                                     const CMatrix& varSigmaTest,
                                     const CMatrix& yTest) const
{
  DIMENSIONMATCH(L.dimensionsMatch(muTest));
  for(unsigned int i = 0; i < muTest.getRows(); i++)
  {
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
      ndlnoise::skewGaussianLaplaceDerivs(logp, d1, W, dW,
                                          yTest.getVal(i, j), muTest.getVal(i, j),
                                          variance, alpha);
      L.setVal(exp(logp), i, j);
    }
  }
}

double CSkewGaussianNoise::logLikelihood(const CMatrix& muTest,
                                         const CMatrix& varSigmaTest,
                                         const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  double L = 0.0;
  for(unsigned int i = 0; i < muTest.getRows(); i++)
  {
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      const double vs = varSigmaTest.getVal(i, j);
      if(vs <= ndlutil::EPS)
      {
        double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
        ndlnoise::skewGaussianLaplaceDerivs(logp, d1, W, dW,
                                            yTest.getVal(i, j), muTest.getVal(i, j),
                                            variance, alpha);
        L += logp;
      }
      else
      {
        // log E_q[p] ≈ log( mean_GH exp(log p) ) via GH of log p then? Use E[log p]
        // for site objectives when cavity var is non-zero (VE); for predictive Z
        // prefer logsumexp of density — here E[log p] via GH (matches VE).
        double gMu = 0.0, gVar = 0.0;
        L += expectedLogLikelihood(yTest.getVal(i, j), muTest.getVal(i, j), vs,
                                   gMu, gVar);
      }
    }
  }
  return L;
}

double CSkewGaussianNoise::expectedLogLikelihood(double y, double muVal, double var,
                                                 double& gMu, double& gVar) const
{
  SkewGaussEllCtx ctx;
  ctx.y = y;
  ctx.variance = variance;
  ctx.alpha = alpha;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, skewGaussLogLik,
                                     skewGaussLogLikPrime, &ctx, gMu, gVar);
}

CSkewExponentialNoise::~CSkewExponentialNoise() {}

void CSkewExponentialNoise::_init()
{
  setType("skew_exponential");
  setName("SkewExponential");
}

void CSkewExponentialNoise::initStoreage()
{
  setNumParams(2);
  initSiteStoreage();
  clearTransforms();
  addTransform(CTransform::defaultPositive(), 0); // rate
  setLogConcave(false);
  setSpherical(true);
  setMissing(false);
}

void CSkewExponentialNoise::initNames()
{
  setParamName("rate", 0);
  setParamName("alpha", 1);
}

void CSkewExponentialNoise::initVals()
{
  activeSites().zeros();
  rate = 1.0;
  alpha = 0.0;
}

void CSkewExponentialNoise::initParams()
{
  rate = 1.0;
  alpha = 0.0;
}

ostream& CSkewExponentialNoise::display(ostream& os)
{
  os << "SkewExponential Noise: rate=" << rate
     << " alpha=" << alpha << endl;
  return os;
}

void CSkewExponentialNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index < getNumParams());
  if(index == 0)
    setRate(val);
  else
    setAlpha(val);
}

void CSkewExponentialNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getRows() == 1);
  DIMENSIONMATCH(params.getCols() == getNumParams());
  setRate(params.getVal(0));
  setAlpha(params.getVal(1));
}

double CSkewExponentialNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index < getNumParams());
  return (index == 0) ? rate : alpha;
}

void CSkewExponentialNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getRows() == 1);
  DIMENSIONMATCH(params.getCols() == getNumParams());
  params.setVal(rate, 0);
  params.setVal(alpha, 1);
}

void CSkewExponentialNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getRows() == 1);
  DIMENSIONMATCH(g.getCols() == getNumParams());
  double gRate = 0.0;
  double gAlpha = 0.0;
  for(unsigned int i = 0; i < getNumData(); i++)
  {
    for(unsigned int j = 0; j < getOutputDim(); j++)
    {
      const double e = getTarget(i, j) - getMu(i, j);
      const double lam = ndlnoise::invMillsRatio(alpha * e);
      gRate += 1.0 / rate - fabs(e);
      gAlpha += lam * e;
    }
  }
  g.setVal(gRate, 0);
  g.setVal(gAlpha, 1);
}

void CSkewExponentialNoise::getGradInputs(double& gmu, double& gvs,
                                          unsigned int i, unsigned int j) const
{
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::skewExponentialLaplaceDerivs(logp, d1, W, dW,
                                         getTarget(i, j), getMu(i, j),
                                         rate, alpha);
  gmu = d1;
  gvs = -0.5 * W;
}

void CSkewExponentialNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut,
                                 const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muout));
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j = 0; j < getOutputDim(); j++)
    cout << "Mean Squared Error on output " << j + 1 << ": "
         << yPred.dist2Col(j, yTest, j) / static_cast<double>(yTest.getRows())
         << endl;
}

void CSkewExponentialNoise::out(CMatrix& yPred, const CMatrix& muTest,
                                const CMatrix& /*varSigmaTest*/) const
{
  // No closed-form predictive mean in GPy; use latent mean as point prediction.
  yPred.deepCopy(muTest);
}

void CSkewExponentialNoise::out(CMatrix& yPred, CMatrix& errorBarOut,
                                const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  out(yPred, muTest, varSigmaTest);
  // Laplace scale 1/rate → var 2/rate² when alpha=0; use that as noise floor.
  const double noise = 2.0 / (rate * rate);
  for(unsigned int i = 0; i < errorBarOut.getRows(); i++)
    for(unsigned int j = 0; j < errorBarOut.getCols(); j++)
      errorBarOut.setVal(sqrt(varSigmaTest.getVal(i, j) + noise), i, j);
}

void CSkewExponentialNoise::likelihoods(CMatrix& L, const CMatrix& muTest,
                                        const CMatrix& /*varSigmaTest*/,
                                        const CMatrix& yTest) const
{
  DIMENSIONMATCH(L.dimensionsMatch(muTest));
  for(unsigned int i = 0; i < muTest.getRows(); i++)
  {
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
      ndlnoise::skewExponentialLaplaceDerivs(logp, d1, W, dW,
                                             yTest.getVal(i, j), muTest.getVal(i, j),
                                             rate, alpha);
      L.setVal(exp(logp), i, j);
    }
  }
}

double CSkewExponentialNoise::logLikelihood(const CMatrix& muTest,
                                            const CMatrix& varSigmaTest,
                                            const CMatrix& yTest) const
{
  DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
  double L = 0.0;
  for(unsigned int i = 0; i < muTest.getRows(); i++)
  {
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      const double vs = varSigmaTest.getVal(i, j);
      if(vs <= ndlutil::EPS)
      {
        double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
        ndlnoise::skewExponentialLaplaceDerivs(logp, d1, W, dW,
                                               yTest.getVal(i, j), muTest.getVal(i, j),
                                               rate, alpha);
        L += logp;
      }
      else
      {
        double gMu = 0.0, gVar = 0.0;
        L += expectedLogLikelihood(yTest.getVal(i, j), muTest.getVal(i, j), vs,
                                   gMu, gVar);
      }
    }
  }
  return L;
}

double CSkewExponentialNoise::expectedLogLikelihood(double y, double muVal, double var,
                                                    double& gMu, double& gVar) const
{
  SkewExpEllCtx ctx;
  ctx.y = y;
  ctx.rate = rate;
  ctx.alpha = alpha;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, skewExpLogLik,
                                     skewExpLogLikPrime, &ctx, gMu, gVar);
}

// ---- CIP-003A Phase 2: Student-t / Poisson / Binomial -----------------------

namespace {
struct StudentTEllCtx {
  double y;
  double variance;
  double degFree;
};
double studentTLogLik(double f, void* vctx)
{
  const StudentTEllCtx* c = static_cast<const StudentTEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::studentTLaplaceDerivs(logp, d1, W, dW, c->y, f, c->variance, c->degFree);
  return logp;
}
double studentTLogLikPrime(double f, void* vctx)
{
  const StudentTEllCtx* c = static_cast<const StudentTEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::studentTLaplaceDerivs(logp, d1, W, dW, c->y, f, c->variance, c->degFree);
  return d1;
}
struct PoissonEllCtx { double y; };
double poissonLogLik(double f, void* vctx)
{
  const PoissonEllCtx* c = static_cast<const PoissonEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::poissonLogLaplaceDerivs(logp, d1, W, dW, c->y, f);
  return logp;
}
double poissonLogLikPrime(double f, void* vctx)
{
  const PoissonEllCtx* c = static_cast<const PoissonEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::poissonLogLaplaceDerivs(logp, d1, W, dW, c->y, f);
  return d1;
}
struct BinomialEllCtx {
  double y;
  double numTrials;
};
double binomialLogLik(double f, void* vctx)
{
  const BinomialEllCtx* c = static_cast<const BinomialEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::binomialProbitLaplaceDerivs(logp, d1, W, dW, c->y, f, c->numTrials);
  return logp;
}
double binomialLogLikPrime(double f, void* vctx)
{
  const BinomialEllCtx* c = static_cast<const BinomialEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::binomialProbitLaplaceDerivs(logp, d1, W, dW, c->y, f, c->numTrials);
  return d1;
}
} // namespace

CStudentTNoise::~CStudentTNoise() {}
void CStudentTNoise::_init()
{
  setType("student_t");
  setName("StudentT");
}
void CStudentTNoise::initStoreage()
{
  setNumParams(2);
  initSiteStoreage();
  clearTransforms();
  addTransform(CTransform::defaultPositive(), 0);
  addTransform(CTransform::defaultPositive(), 1);
  setLogConcave(false);
  setSpherical(true);
  setMissing(false);
}
void CStudentTNoise::initNames()
{
  setParamName("variance", 0);
  setParamName("deg_free", 1);
}
void CStudentTNoise::initVals()
{
  activeSites().zeros();
  variance = 1.0;
  degFree = 5.0;
}
void CStudentTNoise::initParams() {}
ostream& CStudentTNoise::display(ostream& os)
{
  os << "StudentT Noise: variance=" << variance << " deg_free=" << degFree << endl;
  return os;
}
void CStudentTNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index < 2);
  if(index == 0) setVariance(val); else setDegFree(val);
}
void CStudentTNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getRows() == 1 && params.getCols() == 2);
  setVariance(params.getVal(0));
  setDegFree(params.getVal(1));
}
double CStudentTNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index < 2);
  return (index == 0) ? variance : degFree;
}
void CStudentTNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getRows() == 1 && params.getCols() == 2);
  params.setVal(variance, 0);
  params.setVal(degFree, 1);
}
void CStudentTNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getRows() == 1 && g.getCols() == 2);
  double gVar = 0.0;
  double gNu = 0.0;
  for(unsigned int i = 0; i < getNumData(); i++)
  {
    for(unsigned int j = 0; j < getOutputDim(); j++)
    {
      const double e = getTarget(i, j) - getMu(i, j);
      const double e2 = e * e;
      const double vs = variance * degFree;
      const double denom = vs + e2;
      // GPy dlogpdf_link_dvar / dlogpdf_link_dv
      gVar += degFree * (e2 - variance) / (2.0 * variance * denom);
      gNu += 0.5 * ndlutil::digamma(0.5 * (degFree + 1.0))
             - 0.5 * ndlutil::digamma(0.5 * degFree)
             - 1.0 / (2.0 * degFree)
             + 0.5 * (degFree + 1.0) * e2 / (degFree * denom)
             - 0.5 * log(1.0 + e2 / vs);
    }
  }
  g.setVal(gVar, 0);
  g.setVal(gNu, 1);
}
void CStudentTNoise::getGradInputs(double& gmu, double& gvs,
                                   unsigned int i, unsigned int j) const
{
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::studentTLaplaceDerivs(logp, d1, W, dW,
                                  getTarget(i, j), getMu(i, j), variance, degFree);
  gmu = d1;
  gvs = -0.5 * W;
}
void CStudentTNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut,
                          const CMatrix& yTest) const
{
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j = 0; j < getOutputDim(); j++)
    cout << "Mean Squared Error on output " << j + 1 << ": "
         << yPred.dist2Col(j, yTest, j) / static_cast<double>(yTest.getRows()) << endl;
}
void CStudentTNoise::out(CMatrix& yPred, const CMatrix& muTest,
                         const CMatrix& /*varSigmaTest*/) const
{
  yPred.deepCopy(muTest);
}
void CStudentTNoise::out(CMatrix& yPred, CMatrix& errorBarOut,
                         const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  out(yPred, muTest, varSigmaTest);
  const double noise = (degFree > 2.0)
                         ? variance * degFree / (degFree - 2.0)
                         : variance;
  for(unsigned int i = 0; i < errorBarOut.getRows(); i++)
    for(unsigned int j = 0; j < errorBarOut.getCols(); j++)
      errorBarOut.setVal(sqrt(varSigmaTest.getVal(i, j) + noise), i, j);
}
void CStudentTNoise::likelihoods(CMatrix& L, const CMatrix& muTest,
                                 const CMatrix& /*varSigmaTest*/,
                                 const CMatrix& yTest) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
      ndlnoise::studentTLaplaceDerivs(logp, d1, W, dW,
                                      yTest.getVal(i, j), muTest.getVal(i, j),
                                      variance, degFree);
      L.setVal(exp(logp), i, j);
    }
}
double CStudentTNoise::logLikelihood(const CMatrix& muTest,
                                     const CMatrix& varSigmaTest,
                                     const CMatrix& yTest) const
{
  double L = 0.0;
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      const double vs = varSigmaTest.getVal(i, j);
      if(vs <= ndlutil::EPS)
      {
        double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
        ndlnoise::studentTLaplaceDerivs(logp, d1, W, dW,
                                        yTest.getVal(i, j), muTest.getVal(i, j),
                                        variance, degFree);
        L += logp;
      }
      else
      {
        double gMu = 0.0, gVar = 0.0;
        L += expectedLogLikelihood(yTest.getVal(i, j), muTest.getVal(i, j), vs, gMu, gVar);
      }
    }
  return L;
}
double CStudentTNoise::expectedLogLikelihood(double y, double muVal, double var,
                                             double& gMu, double& gVar) const
{
  StudentTEllCtx ctx;
  ctx.y = y;
  ctx.variance = variance;
  ctx.degFree = degFree;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, studentTLogLik,
                                     studentTLogLikPrime, &ctx, gMu, gVar);
}

CPoissonNoise::~CPoissonNoise() {}
void CPoissonNoise::_init()
{
  setType("poisson");
  setName("Poisson");
}
void CPoissonNoise::initStoreage()
{
  setNumParams(0);
  initSiteStoreage();
  clearTransforms();
  setLogConcave(true);
  setSpherical(true);
  setMissing(false);
}
void CPoissonNoise::initNames() {}
void CPoissonNoise::initVals() { activeSites().zeros(); }
void CPoissonNoise::initParams() {}
ostream& CPoissonNoise::display(ostream& os)
{
  os << "Poisson Noise (log link)" << endl;
  return os;
}
void CPoissonNoise::setParam(double /*val*/, unsigned int /*index*/)
{
  throw ndlexceptions::Error("CPoissonNoise has no parameters");
}
void CPoissonNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getCols() == 0 || getNumParams() == 0);
}
double CPoissonNoise::getParam(unsigned int /*index*/) const
{
  throw ndlexceptions::Error("CPoissonNoise has no parameters");
}
void CPoissonNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getCols() == 0 || getNumParams() == 0);
}
void CPoissonNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getCols() == 0 || getNumParams() == 0);
}
void CPoissonNoise::getGradInputs(double& gmu, double& gvs,
                                  unsigned int i, unsigned int j) const
{
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::poissonLogLaplaceDerivs(logp, d1, W, dW, getTarget(i, j), getMu(i, j));
  gmu = d1;
  gvs = -0.5 * W;
}
void CPoissonNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut,
                         const CMatrix& yTest) const
{
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j = 0; j < getOutputDim(); j++)
    cout << "Mean Squared Error on output " << j + 1 << ": "
         << yPred.dist2Col(j, yTest, j) / static_cast<double>(yTest.getRows()) << endl;
}
void CPoissonNoise::out(CMatrix& yPred, const CMatrix& muTest,
                        const CMatrix& /*varSigmaTest*/) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
      yPred.setVal(exp(muTest.getVal(i, j)), i, j);
}
void CPoissonNoise::out(CMatrix& yPred, CMatrix& errorBarOut,
                        const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  out(yPred, muTest, varSigmaTest);
  // Var(y|f)=λ; predictive ≈ E[λ] + Var(λ) with λ=e^f, lognormal moments.
  for(unsigned int i = 0; i < errorBarOut.getRows(); i++)
    for(unsigned int j = 0; j < errorBarOut.getCols(); j++)
    {
      const double m = muTest.getVal(i, j);
      const double v = varSigmaTest.getVal(i, j);
      const double meanLam = exp(m + 0.5 * v);
      const double varLam = exp(2.0 * m + v) * (exp(v) - 1.0);
      errorBarOut.setVal(sqrt(meanLam + varLam), i, j);
    }
}
void CPoissonNoise::likelihoods(CMatrix& L, const CMatrix& muTest,
                                const CMatrix& /*varSigmaTest*/,
                                const CMatrix& yTest) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
      ndlnoise::poissonLogLaplaceDerivs(logp, d1, W, dW,
                                        yTest.getVal(i, j), muTest.getVal(i, j));
      L.setVal(exp(logp), i, j);
    }
}
double CPoissonNoise::logLikelihood(const CMatrix& muTest,
                                    const CMatrix& varSigmaTest,
                                    const CMatrix& yTest) const
{
  double L = 0.0;
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      const double vs = varSigmaTest.getVal(i, j);
      if(vs <= ndlutil::EPS)
      {
        double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
        ndlnoise::poissonLogLaplaceDerivs(logp, d1, W, dW,
                                          yTest.getVal(i, j), muTest.getVal(i, j));
        L += logp;
      }
      else
      {
        double gMu = 0.0, gVar = 0.0;
        L += expectedLogLikelihood(yTest.getVal(i, j), muTest.getVal(i, j), vs, gMu, gVar);
      }
    }
  return L;
}
double CPoissonNoise::expectedLogLikelihood(double y, double muVal, double var,
                                            double& gMu, double& gVar) const
{
  PoissonEllCtx ctx;
  ctx.y = y;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, poissonLogLik,
                                     poissonLogLikPrime, &ctx, gMu, gVar);
}

CBinomialNoise::~CBinomialNoise() {}
void CBinomialNoise::_init()
{
  setType("binomial");
  setName("Binomial");
}
void CBinomialNoise::initStoreage()
{
  setNumParams(0);
  initSiteStoreage();
  clearTransforms();
  setLogConcave(true);
  setSpherical(true);
  setMissing(false);
}
void CBinomialNoise::initNames() {}
void CBinomialNoise::initVals() { activeSites().zeros(); }
void CBinomialNoise::initParams() {}
ostream& CBinomialNoise::display(ostream& os)
{
  os << "Binomial Noise (probit): numTrials=" << numTrials << endl;
  return os;
}
void CBinomialNoise::setParam(double /*val*/, unsigned int /*index*/)
{
  throw ndlexceptions::Error("CBinomialNoise has no free parameters");
}
void CBinomialNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getCols() == 0 || getNumParams() == 0);
}
double CBinomialNoise::getParam(unsigned int /*index*/) const
{
  throw ndlexceptions::Error("CBinomialNoise has no free parameters");
}
void CBinomialNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getCols() == 0 || getNumParams() == 0);
}
void CBinomialNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getCols() == 0 || getNumParams() == 0);
}
void CBinomialNoise::getGradInputs(double& gmu, double& gvs,
                                   unsigned int i, unsigned int j) const
{
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::binomialProbitLaplaceDerivs(logp, d1, W, dW,
                                        getTarget(i, j), getMu(i, j),
                                        static_cast<double>(numTrials));
  gmu = d1;
  gvs = -0.5 * W;
}
void CBinomialNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut,
                          const CMatrix& yTest) const
{
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j = 0; j < getOutputDim(); j++)
    cout << "Mean Squared Error on output " << j + 1 << ": "
         << yPred.dist2Col(j, yTest, j) / static_cast<double>(yTest.getRows()) << endl;
}
void CBinomialNoise::out(CMatrix& yPred, const CMatrix& muTest,
                         const CMatrix& /*varSigmaTest*/) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
      yPred.setVal(static_cast<double>(numTrials) * ndlutil::cumGaussian(muTest.getVal(i, j)),
                   i, j);
}
void CBinomialNoise::out(CMatrix& yPred, CMatrix& errorBarOut,
                         const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  out(yPred, muTest, varSigmaTest);
  for(unsigned int i = 0; i < errorBarOut.getRows(); i++)
    for(unsigned int j = 0; j < errorBarOut.getCols(); j++)
    {
      const double p = ndlutil::cumGaussian(muTest.getVal(i, j));
      const double N = static_cast<double>(numTrials);
      errorBarOut.setVal(sqrt(N * p * (1.0 - p) + varSigmaTest.getVal(i, j)), i, j);
    }
}
void CBinomialNoise::likelihoods(CMatrix& L, const CMatrix& muTest,
                                 const CMatrix& /*varSigmaTest*/,
                                 const CMatrix& yTest) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
      ndlnoise::binomialProbitLaplaceDerivs(logp, d1, W, dW,
                                            yTest.getVal(i, j), muTest.getVal(i, j),
                                            static_cast<double>(numTrials));
      L.setVal(exp(logp), i, j);
    }
}
double CBinomialNoise::logLikelihood(const CMatrix& muTest,
                                     const CMatrix& varSigmaTest,
                                     const CMatrix& yTest) const
{
  double L = 0.0;
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      const double vs = varSigmaTest.getVal(i, j);
      if(vs <= ndlutil::EPS)
      {
        double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
        ndlnoise::binomialProbitLaplaceDerivs(logp, d1, W, dW,
                                              yTest.getVal(i, j), muTest.getVal(i, j),
                                              static_cast<double>(numTrials));
        L += logp;
      }
      else
      {
        double gMu = 0.0, gVar = 0.0;
        L += expectedLogLikelihood(yTest.getVal(i, j), muTest.getVal(i, j), vs, gMu, gVar);
      }
    }
  return L;
}
double CBinomialNoise::expectedLogLikelihood(double y, double muVal, double var,
                                             double& gMu, double& gVar) const
{
  BinomialEllCtx ctx;
  ctx.y = y;
  ctx.numTrials = static_cast<double>(numTrials);
  return ndlutil::gaussHermiteExpect(muVal, var, 20, binomialLogLik,
                                     binomialLogLikPrime, &ctx, gMu, gVar);
}
void CBinomialNoise::writeParamsToStream(ostream& out) const
{
  writeToStream(out, "baseType", getBaseType());
  writeToStream(out, "type", getType());
  writeToStream(out, "outputDim", getOutputDim());
  writeToStream(out, "numParams", getNumParams());
  writeToStream(out, "numTrials", static_cast<int>(numTrials));
}
void CBinomialNoise::readParamsFromStream(istream& in)
{
  setOutputDim(readIntFromStream(in, "outputDim"));
  unsigned int numPar = readIntFromStream(in, "numParams");
  setNumTrials(static_cast<unsigned int>(readIntFromStream(in, "numTrials")));
  setNumData(1);
  initStoreage();
  if(numPar != getNumParams())
    throw ndlexceptions::StreamFormatError("numParams",
        "Number of parameters in file does not match computed number.");
}

// ---- CIP-003A Phase 3: Exponential / Gamma ----------------------------------

namespace {
struct ExpEllCtx { double y; };
double expLogLik(double f, void* vctx)
{
  const ExpEllCtx* c = static_cast<const ExpEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::exponentialLogLaplaceDerivs(logp, d1, W, dW, c->y, f);
  return logp;
}
double expLogLikPrime(double f, void* vctx)
{
  const ExpEllCtx* c = static_cast<const ExpEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::exponentialLogLaplaceDerivs(logp, d1, W, dW, c->y, f);
  return d1;
}
struct GammaEllCtx { double y; double beta; };
double gammaLogLik(double f, void* vctx)
{
  const GammaEllCtx* c = static_cast<const GammaEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::gammaLogLaplaceDerivs(logp, d1, W, dW, c->y, f, c->beta);
  return logp;
}
double gammaLogLikPrime(double f, void* vctx)
{
  const GammaEllCtx* c = static_cast<const GammaEllCtx*>(vctx);
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::gammaLogLaplaceDerivs(logp, d1, W, dW, c->y, f, c->beta);
  return d1;
}
} // namespace

CExponentialNoise::~CExponentialNoise() {}
void CExponentialNoise::_init()
{
  setType("exponential");
  setName("Exponential");
}
void CExponentialNoise::initStoreage()
{
  setNumParams(0);
  initSiteStoreage();
  clearTransforms();
  setLogConcave(true);
  setSpherical(true);
  setMissing(false);
}
void CExponentialNoise::initNames() {}
void CExponentialNoise::initVals() { activeSites().zeros(); }
void CExponentialNoise::initParams() {}
ostream& CExponentialNoise::display(ostream& os)
{
  os << "Exponential Noise (log link)" << endl;
  return os;
}
void CExponentialNoise::setParam(double /*val*/, unsigned int /*index*/)
{
  throw ndlexceptions::Error("CExponentialNoise has no parameters");
}
void CExponentialNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getCols() == 0 || getNumParams() == 0);
}
double CExponentialNoise::getParam(unsigned int /*index*/) const
{
  throw ndlexceptions::Error("CExponentialNoise has no parameters");
}
void CExponentialNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getCols() == 0 || getNumParams() == 0);
}
void CExponentialNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getCols() == 0 || getNumParams() == 0);
}
void CExponentialNoise::getGradInputs(double& gmu, double& gvs,
                                      unsigned int i, unsigned int j) const
{
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::exponentialLogLaplaceDerivs(logp, d1, W, dW, getTarget(i, j), getMu(i, j));
  gmu = d1;
  gvs = -0.5 * W;
}
void CExponentialNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut,
                             const CMatrix& yTest) const
{
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j = 0; j < getOutputDim(); j++)
    cout << "Mean Squared Error on output " << j + 1 << ": "
         << yPred.dist2Col(j, yTest, j) / static_cast<double>(yTest.getRows()) << endl;
}
void CExponentialNoise::out(CMatrix& yPred, const CMatrix& muTest,
                            const CMatrix& /*varSigmaTest*/) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
      yPred.setVal(exp(-muTest.getVal(i, j)), i, j); // E[y]=1/λ with λ=e^f
}
void CExponentialNoise::out(CMatrix& yPred, CMatrix& errorBarOut,
                            const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  out(yPred, muTest, varSigmaTest);
  for(unsigned int i = 0; i < errorBarOut.getRows(); i++)
    for(unsigned int j = 0; j < errorBarOut.getCols(); j++)
    {
      const double mean = yPred.getVal(i, j);
      errorBarOut.setVal(sqrt(mean * mean + varSigmaTest.getVal(i, j)), i, j);
    }
}
void CExponentialNoise::likelihoods(CMatrix& L, const CMatrix& muTest,
                                    const CMatrix& /*varSigmaTest*/,
                                    const CMatrix& yTest) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
      ndlnoise::exponentialLogLaplaceDerivs(logp, d1, W, dW,
                                            yTest.getVal(i, j), muTest.getVal(i, j));
      L.setVal(exp(logp), i, j);
    }
}
double CExponentialNoise::logLikelihood(const CMatrix& muTest,
                                        const CMatrix& varSigmaTest,
                                        const CMatrix& yTest) const
{
  double L = 0.0;
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      const double vs = varSigmaTest.getVal(i, j);
      if(vs <= ndlutil::EPS)
      {
        double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
        ndlnoise::exponentialLogLaplaceDerivs(logp, d1, W, dW,
                                              yTest.getVal(i, j), muTest.getVal(i, j));
        L += logp;
      }
      else
      {
        double gMu = 0.0, gVar = 0.0;
        L += expectedLogLikelihood(yTest.getVal(i, j), muTest.getVal(i, j), vs, gMu, gVar);
      }
    }
  return L;
}
double CExponentialNoise::expectedLogLikelihood(double y, double muVal, double var,
                                                double& gMu, double& gVar) const
{
  ExpEllCtx ctx;
  ctx.y = y;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, expLogLik, expLogLikPrime, &ctx, gMu, gVar);
}

CGammaNoise::~CGammaNoise() {}
void CGammaNoise::_init()
{
  setType("gamma");
  setName("Gamma");
}
void CGammaNoise::initStoreage()
{
  setNumParams(1);
  initSiteStoreage();
  clearTransforms();
  addTransform(CTransform::defaultPositive(), 0);
  setLogConcave(true);
  setSpherical(true);
  setMissing(false);
}
void CGammaNoise::initNames() { setParamName("beta", 0); }
void CGammaNoise::initVals()
{
  activeSites().zeros();
  beta = 1.0;
}
void CGammaNoise::initParams() {}
ostream& CGammaNoise::display(ostream& os)
{
  os << "Gamma Noise (log link): beta=" << beta << endl;
  return os;
}
void CGammaNoise::setParam(double val, unsigned int index)
{
  BOUNDCHECK(index == 0);
  setBeta(val);
}
void CGammaNoise::setParams(const CMatrix& params)
{
  DIMENSIONMATCH(params.getRows() == 1 && params.getCols() == 1);
  setBeta(params.getVal(0));
}
double CGammaNoise::getParam(unsigned int index) const
{
  BOUNDCHECK(index == 0);
  return beta;
}
void CGammaNoise::getParams(CMatrix& params) const
{
  DIMENSIONMATCH(params.getRows() == 1 && params.getCols() == 1);
  params.setVal(beta, 0);
}
void CGammaNoise::getGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getRows() == 1 && g.getCols() == 1);
  double gBeta = 0.0;
  for(unsigned int i = 0; i < getNumData(); i++)
    for(unsigned int j = 0; j < getOutputDim(); j++)
    {
      const double y = getTarget(i, j);
      const double lam = exp(getMu(i, j));
      const double alpha = beta * lam;
      // GPy dlogpdf_link_dbeta with link_f = λ
      gBeta += lam * (log(beta * y) + 1.0 - ndlutil::digamma(alpha)) - y;
    }
  g.setVal(gBeta, 0);
}
void CGammaNoise::getGradInputs(double& gmu, double& gvs,
                                unsigned int i, unsigned int j) const
{
  double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
  ndlnoise::gammaLogLaplaceDerivs(logp, d1, W, dW, getTarget(i, j), getMu(i, j), beta);
  gmu = d1;
  gvs = -0.5 * W;
}
void CGammaNoise::test(const CMatrix& muout, const CMatrix& varSigmaOut,
                       const CMatrix& yTest) const
{
  CMatrix yPred(yTest.getRows(), yTest.getCols());
  out(yPred, muout, varSigmaOut);
  for(unsigned int j = 0; j < getOutputDim(); j++)
    cout << "Mean Squared Error on output " << j + 1 << ": "
         << yPred.dist2Col(j, yTest, j) / static_cast<double>(yTest.getRows()) << endl;
}
void CGammaNoise::out(CMatrix& yPred, const CMatrix& muTest,
                      const CMatrix& /*varSigmaTest*/) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
      yPred.setVal(exp(muTest.getVal(i, j)), i, j); // E[y]=λ
}
void CGammaNoise::out(CMatrix& yPred, CMatrix& errorBarOut,
                      const CMatrix& muTest, const CMatrix& varSigmaTest) const
{
  out(yPred, muTest, varSigmaTest);
  for(unsigned int i = 0; i < errorBarOut.getRows(); i++)
    for(unsigned int j = 0; j < errorBarOut.getCols(); j++)
    {
      const double mean = yPred.getVal(i, j);
      errorBarOut.setVal(sqrt(mean / beta + varSigmaTest.getVal(i, j)), i, j);
    }
}
void CGammaNoise::likelihoods(CMatrix& L, const CMatrix& muTest,
                              const CMatrix& /*varSigmaTest*/,
                              const CMatrix& yTest) const
{
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
      ndlnoise::gammaLogLaplaceDerivs(logp, d1, W, dW,
                                      yTest.getVal(i, j), muTest.getVal(i, j), beta);
      L.setVal(exp(logp), i, j);
    }
}
double CGammaNoise::logLikelihood(const CMatrix& muTest,
                                  const CMatrix& varSigmaTest,
                                  const CMatrix& yTest) const
{
  double L = 0.0;
  for(unsigned int i = 0; i < muTest.getRows(); i++)
    for(unsigned int j = 0; j < muTest.getCols(); j++)
    {
      const double vs = varSigmaTest.getVal(i, j);
      if(vs <= ndlutil::EPS)
      {
        double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
        ndlnoise::gammaLogLaplaceDerivs(logp, d1, W, dW,
                                        yTest.getVal(i, j), muTest.getVal(i, j), beta);
        L += logp;
      }
      else
      {
        double gMu = 0.0, gVar = 0.0;
        L += expectedLogLikelihood(yTest.getVal(i, j), muTest.getVal(i, j), vs, gMu, gVar);
      }
    }
  return L;
}
double CGammaNoise::expectedLogLikelihood(double y, double muVal, double var,
                                          double& gMu, double& gVar) const
{
  GammaEllCtx ctx;
  ctx.y = y;
  ctx.beta = beta;
  return ndlutil::gaussHermiteExpect(muVal, var, 20, gammaLogLik, gammaLogLikPrime,
                                     &ctx, gMu, gVar);
}
 
