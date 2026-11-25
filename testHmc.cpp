/* CIP-003B: HMC verification. No .mat fixtures.
   1. Analytic Gaussian target: sample moments agree within Monte Carlo error
      (tolerance from an autocorrelation-based effective sample size).
   2. Leapfrog energy error shrinks as epsilon^2 at fixed trajectory length.
   3. Improper targets are refused at construction.
   4. A rejected proposal restores the model's parameters.
   5. A GP length-scale posterior (other parameters fixed) matches a fine grid
      integration of exp(-E) computed from the same objective. */
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <algorithm>
#include <vector>
#include "CHmc.h"
#include "CGp.h"
#include "CGplvm.h"
#include "CKern.h"
#include "CNoise.h"
#include "CDist.h"
#include "CMatrix.h"
#include "ndlutil.h"

using namespace std;

namespace {

// Gaussian target N(mu, Sigma) in two dimensions: E = 0.5 (a-mu)' S^-1 (a-mu).
class GaussTarget : public COptimisable, public CPosteriorTarget
{
 public:
  GaussTarget(bool proper=true) : proper(proper), a(1, 2)
  {
    a.setVal(0.0, 0); a.setVal(0.0, 1);
    mu[0] = 1.0; mu[1] = -2.0;
    // S = [[2, 0.9],[0.9, 1]]; inverse computed in closed form.
    const double det = 2.0*1.0 - 0.9*0.9;
    si[0][0] = 1.0/det; si[0][1] = -0.9/det; si[1][0] = -0.9/det; si[1][1] = 2.0/det;
  }
  unsigned int getOptNumParams() const { return 2; }
  void getOptParams(CMatrix& p) const { p.deepCopy(a); }
  void setOptParams(const CMatrix& p) { a.deepCopy(p); }
  double computeObjectiveVal() const
  {
    const double d0 = a.getVal(0)-mu[0], d1 = a.getVal(1)-mu[1];
    return 0.5*(d0*(si[0][0]*d0+si[0][1]*d1) + d1*(si[1][0]*d0+si[1][1]*d1));
  }
  double computeObjectiveGradParams(CMatrix& g) const
  {
    const double d0 = a.getVal(0)-mu[0], d1 = a.getVal(1)-mu[1];
    g.setVal(si[0][0]*d0+si[0][1]*d1, 0);
    g.setVal(si[1][0]*d0+si[1][1]*d1, 1);
    return computeObjectiveVal();
  }
  void sampledParams(vector<SampledParam>& ps) const
  {
    ps.clear();
    for(int i=0; i<2; i++)
    {
      SampledParam sp;
      sp.name = i==0 ? "x0" : "x1";
      sp.transform = 0;
      sp.hasPrior = proper;
      sp.constrainedValue = a.getVal(i);
      ps.push_back(sp);
    }
  }
  const COptimisable& optimisable() const { return *this; }
  double mu[2];
  double si[2][2];
 private:
  bool proper;
  CMatrix a;
};

// Effective sample size from the initial positive autocorrelation sequence.
double effectiveSampleSize(const vector<double>& x)
{
  const size_t n = x.size();
  double mean = 0.0;
  for(size_t i=0; i<n; i++) mean += x[i];
  mean /= (double)n;
  double c0 = 0.0;
  for(size_t i=0; i<n; i++) c0 += (x[i]-mean)*(x[i]-mean);
  c0 /= (double)n;
  double tau = 1.0;
  for(size_t lag=1; lag<n/2; lag++)
  {
    double c = 0.0;
    for(size_t i=0; i+lag<n; i++) c += (x[i]-mean)*(x[i+lag]-mean);
    c /= (double)n;
    const double rho = c/c0;
    if(rho<=0.0) break;
    tau += 2.0*rho;
  }
  return (double)n/tau;
}

int checkMoment(const string& what, double estimate, double truth, double se)
{
  if(fabs(estimate-truth) > 4.5*se)
  {
    cout << "FAIL " << what << ": estimate " << estimate << " truth " << truth << " (4.5 se = " << 4.5*se << ")" << endl;
    return 1;
  }
  cout << "PASS " << what << ": " << estimate << " vs " << truth << " (se " << se << ")" << endl;
  return 0;
}

} // namespace

int main()
{
  int fail = 0;

  // 1. Analytic Gaussian moments.
  {
    GaussTarget t;
    CHmc hmc(t, t, 7);
    hmc.setStepSize(0.4);
    hmc.setNumLeapfrog(8);
    hmc.adaptStepSize(300);
    CMatrix S, lp;
    hmc.sample(6000, S, lp);
    cout << "Gaussian: step size " << hmc.getStepSize() << " accept rate " << hmc.acceptRate() << endl;
    const double trueMean[2] = {1.0, -2.0};
    const double trueVar[2] = {2.0, 1.0};
    for(unsigned int d=0; d<2; d++)
    {
      vector<double> x(S.getRows()), x2(S.getRows());
      double m = 0.0, v = 0.0;
      for(unsigned int i=0; i<S.getRows(); i++) { x[i] = S.getVal(i, d); m += x[i]; }
      m /= (double)S.getRows();
      for(unsigned int i=0; i<S.getRows(); i++) { x2[i] = (x[i]-trueMean[d])*(x[i]-trueMean[d]); v += (x[i]-m)*(x[i]-m); }
      v /= (double)S.getRows();
      const double ess = effectiveSampleSize(x);
      const double ess2 = effectiveSampleSize(x2);
      fail += checkMoment("Gaussian mean dim " + ndlstrutil::itoa(d), m, trueMean[d], sqrt(trueVar[d]/ess));
      fail += checkMoment("Gaussian variance dim " + ndlstrutil::itoa(d), v, trueVar[d], sqrt(2.0*trueVar[d]*trueVar[d]/ess2));
    }
    double cov = 0.0, m0 = 0.0, m1 = 0.0;
    const double n = (double)S.getRows();
    for(unsigned int i=0; i<S.getRows(); i++) { m0 += S.getVal(i,0); m1 += S.getVal(i,1); }
    m0 /= n; m1 /= n;
    vector<double> prod(S.getRows());
    for(unsigned int i=0; i<S.getRows(); i++) { prod[i] = (S.getVal(i,0)-1.0)*(S.getVal(i,1)+2.0); cov += prod[i]; }
    cov /= n;
    const double essC = effectiveSampleSize(prod);
    fail += checkMoment("Gaussian covariance", cov, 0.9, sqrt((2.0*1.0+0.81)/essC));
  }

  // 2. Energy error scales as epsilon^2 at fixed trajectory length.
  {
    double err[3];
    const double eps[3] = {0.4, 0.2, 0.1};
    for(int k=0; k<3; k++)
    {
      GaussTarget t;
      CHmc hmc(t, t, 11);
      hmc.setStepSize(eps[k]);
      hmc.setNumLeapfrog((unsigned int)(1.6/eps[k] + 0.5));
      double total = 0.0;
      const int reps = 400;
      for(int r=0; r<reps; r++) { hmc.step(); total += hmc.lastEnergyError(); }
      err[k] = total/reps;
    }
    const double r1 = err[0]/err[1], r2 = err[1]/err[2];
    cout << "Energy error: " << err[0] << " " << err[1] << " " << err[2] << " ratios " << r1 << " " << r2 << endl;
    if(r1 < 2.5 || r1 > 6.5 || r2 < 2.5 || r2 > 6.5)
    {
      cout << "FAIL energy error does not scale as epsilon^2 (ratios should be about 4)" << endl;
      fail++;
    }
    else
      cout << "PASS energy error scales as epsilon^2" << endl;
  }

  // 3. Improper target refused.
  {
    GaussTarget t(false);
    bool threw = false;
    try { CHmc hmc(t, t, 1); }
    catch(ndlexceptions::Error&) { threw = true; }
    if(!threw) { cout << "FAIL improper target was accepted by CHmc" << endl; fail++; }
    else cout << "PASS CHmc refuses an improper target" << endl;
  }

  // 4. A rejected proposal restores the parameters. A huge step size makes
  //    rejection essentially certain.
  {
    GaussTarget t;
    CHmc hmc(t, t, 3);
    hmc.setStepSize(50.0);
    hmc.setNumLeapfrog(5);
    CMatrix before(1, 2), after(1, 2);
    t.getOptParams(before);
    int rejected = 0;
    bool restored = true;
    for(int i=0; i<20; i++)
    {
      CMatrix pre(1, 2);
      t.getOptParams(pre);
      if(!hmc.step())
      {
        rejected++;
        t.getOptParams(after);
        if(fabs(after.getVal(0)-pre.getVal(0))>0 || fabs(after.getVal(1)-pre.getVal(1))>0)
          restored = false;
      }
    }
    if(rejected==0 || !restored)
    {
      cout << "FAIL rejected steps did not restore parameters (rejected " << rejected << ")" << endl;
      fail++;
    }
    else
      cout << "PASS rejection restores parameters (" << rejected << " of 20 rejected)" << endl;
  }

  // 5. GP length-scale posterior against grid integration of the same objective.
  {
    const unsigned int N = 8;
    CMatrix X(N, 1), y(N, 1);
    for(unsigned int i=0; i<N; i++)
    {
      const double xv = -1.0 + 2.0*(double)i/(double)(N-1);
      X.setVal(xv, i, 0);
      y.setVal(sin(2.0*xv), i, 0);
    }
    CCmpndKern kern(X);
    std::unique_ptr<CRbfKern> rbf(new CRbfKern(X));
    for(unsigned int j=0; j<rbf->getNumParams(); j++)
    {
      CGammaDist* kp = new CGammaDist();
      kp->setParam(2.0, 0);
      kp->setParam(1.0, 1);
      rbf->addPrior(kp, j);
    }
    kern.addKern(std::unique_ptr<CKern>(std::move(rbf)));
    CGaussianNoise noise(&y);
    for(int i=0; i<noise.getOutputDim(); i++)
      noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
    model.setLikelihoodVariance(0.1);
    model.fixLikelihoodVariance(true);
    model.fixKernParam(kern.paramIndex("rbf.variance"));
    if(model.getOptNumParams()!=1)
    {
      cout << "FAIL expected a single sampled parameter, got " << model.getOptNumParams() << endl;
      fail++;
    }
    else
    {
      // Grid integration of exp(-E) over the transformed length-scale.
      CMatrix a(1, 1);
      model.getOptParams(a);
      const double a0 = a.getVal(0);
      const double lo = a0-6.0, hi = a0+6.0;
      const int G = 4801;
      vector<double> e(G);
      double emin = 1e300;
      for(int k=0; k<G; k++)
      {
        a.setVal(lo + (hi-lo)*k/(G-1), 0);
        model.setOptParams(a);
        e[k] = model.computeSamplingObjective();
        if(e[k]<emin) emin = e[k];
      }
      double Z = 0.0, M1 = 0.0, M2 = 0.0;
      for(int k=0; k<G; k++)
      {
        const double ak = lo + (hi-lo)*k/(G-1), w = exp(-(e[k]-emin));
        Z += w; M1 += w*ak; M2 += w*ak*ak;
      }
      const double gm = M1/Z, gv = M2/Z - gm*gm;
      model.setOptParams(CMatrix(1, 1, a0));
      CHmc hmc(model, model, 21);
      hmc.setStepSize(0.3);
      hmc.setNumLeapfrog(6);
      hmc.adaptStepSize(300);
      CMatrix S, lp;
      hmc.sample(6000, S, lp);
      vector<double> x(S.getRows()), xv(S.getRows());
      double m = 0.0, v = 0.0;
      for(unsigned int i=0; i<S.getRows(); i++) { x[i] = S.getVal(i, 0); m += x[i]; }
      m /= (double)S.getRows();
      for(unsigned int i=0; i<S.getRows(); i++) { xv[i] = (x[i]-gm)*(x[i]-gm); v += (x[i]-m)*(x[i]-m); }
      v /= (double)S.getRows();
      cout << "GP: step size " << hmc.getStepSize() << " accept rate " << hmc.acceptRate() << endl;
      fail += checkMoment("GP length-scale posterior mean vs grid", m, gm, sqrt(gv/effectiveSampleSize(x)));
      fail += checkMoment("GP length-scale posterior variance vs grid", v, gv, sqrt(2.0*gv*gv/effectiveSampleSize(xv)));
    }
  }

  // 5. CSV output round-trips the in-memory samples (constrained space).
  {
    GaussTarget t;
    CHmc hmc(t, t, 3);
    hmc.setStepSize(0.3);
    hmc.setNumLeapfrog(5);
    CMatrix S, lp;
    hmc.sample(50, S, lp);
    stringstream ss;
    writeSamplesCsv(t, S, lp, ss);
    string line;
    getline(ss, line);
    bool ok = (line == "iteration,log_posterior,x0,x1");
    double maxErr = 0.0;
    unsigned int rows = 0;
    while(getline(ss, line))
    {
      for(size_t k=0; k<line.size(); k++) if(line[k]==',') line[k]=' ';
      istringstream ls(line);
      unsigned int it; double l, x0, x1;
      ls >> it >> l >> x0 >> x1;
      ok = ok && it==rows;
      maxErr = max(maxErr, max(fabs(l-lp.getVal(rows,0)), max(fabs(x0-S.getVal(rows,0)), fabs(x1-S.getVal(rows,1)))));
      rows++;
    }
    if(ok && rows==50 && maxErr<=ndlutil::MATCHTOL)
      cout << "PASS CSV round trip (max error " << maxErr << ")" << endl;
    else { cout << "FAIL CSV round trip: rows " << rows << " max error " << maxErr << endl; fail++; }
  }

  // 6. addPriorByPath makes every sampled parameter proper.
  {
    CMatrix X(5, 1), y(5, 1);
    for(unsigned int i=0; i<5; i++) { X.setVal(0.5*i, i, 0); y.setVal(sin(0.5*i), i, 0); }
    CCmpndKern kern(X);
    kern.addKern(std::unique_ptr<CKern>(new CRbfKern(X)));
    CGaussianNoise noise(&y);
    for(int i=0; i<noise.getOutputDim(); i++) noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
    bool improperBefore = !model.improperReason().empty();
    const char* paths[3] = {"rbf.inv_lengthscale", "rbf.variance", "likelihood.variance"};
    for(int k=0; k<3; k++)
    {
      CGammaDist* g = new CGammaDist();
      g->setParam(2.0, 0);
      g->setParam(1.0, 1);
      model.addPriorByPath(paths[k], g);
    }
    bool properAfter = model.improperReason().empty();
    bool threw = false;
    try { model.addPriorByPath("nonsense.path", new CGammaDist()); }
    catch(...) { threw = true; }
    if(improperBefore && properAfter && threw)
      cout << "PASS addPriorByPath removes improperness and rejects unknown paths" << endl;
    else { cout << "FAIL addPriorByPath: before " << improperBefore << " after " << properAfter << " threw " << threw << endl; fail++; }
  }

  // 6b. Priors on the lengthscale (GPy units): the wrapped density integrates
  //     to one over the stored width, its gradient matches finite differences,
  //     it survives a stream round trip, samples are reported as lengthscales,
  //     and a prior on rbf.lengthscale differs from one on rbf.inv_lengthscale.
  {
    CGammaDist* g = new CGammaDist();
    g->setParam(3.0, 0);
    g->setParam(2.0, 1);
    CPowerViewDist view(g, 0.5);
    // integrate exp(logProb) over the stored width gamma with the trapezoid rule
    // on a log grid (substituting gamma = exp(u), dgamma = gamma du).
    double mass = 0.0;
    const int nGrid = 40000;
    const double u0 = -14.0, u1 = 8.0, du = (u1-u0)/nGrid;
    for(int i=0; i<=nGrid; i++)
    {
      const double gam = exp(u0+i*du);
      const double w = (i==0 || i==nGrid) ? 0.5 : 1.0;
      mass += w*exp(view.logProb(gam))*gam*du;
    }
    double worstGrad = 0.0;
    for(double gam=0.05; gam<5.0; gam*=1.7)
    {
      const double h = 1e-6*gam;
      const double fd = (view.logProb(gam+h)-view.logProb(gam-h))/(2.0*h);
      worstGrad = max(worstGrad, fabs(fd-view.getGradInput(gam))/(1.0+fabs(fd)));
    }
    stringstream ss;
    writeDistToStream(view, ss);
    CDist* back = readDistFromStream(ss);
    const bool roundTrip = back->equals(view) && back->getType()=="powerview"
      && fabs(back->logProb(0.7)-view.logProb(0.7))<1e-12;
    delete back;

    CMatrix X(5, 1), y(5, 1);
    for(unsigned int i=0; i<5; i++) { X.setVal(0.5*i, i, 0); y.setVal(sin(0.5*i), i, 0); }
    CCmpndKern kern(X);
    kern.addKern(std::unique_ptr<CKern>(new CRbfKern(X)));
    CGaussianNoise noise(&y);
    for(int i=0; i<noise.getOutputDim(); i++) noise.setBiasVal(0.0, i);
    CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
    CGammaDist* g2 = new CGammaDist();
    g2->setParam(3.0, 0);
    g2->setParam(2.0, 1);
    model.addPriorByPath("rbf.lengthscale", g2);
    CGammaDist* gv = new CGammaDist();
    gv->setParam(2.0, 0);
    gv->setParam(1.0, 1);
    model.addPriorByPath("rbf.variance", gv);
    CGammaDist* gn = new CGammaDist();
    gn->setParam(2.0, 0);
    gn->setParam(1.0, 1);
    model.addPriorByPath("likelihood.variance", gn);
    vector<SampledParam> sp;
    model.sampledParams(sp);
    bool named = false;
    for(size_t i=0; i<sp.size(); i++)
      if(sp[i].name=="rbf.lengthscale" && sp[i].rawName()=="rbf.inv_lengthscale"
         && fabs(sp[i].reportValue(sp[i].constrainedValue)-1.0/sqrt(sp[i].constrainedValue))<1e-12)
        named = true;
    const bool proper = model.improperReason().empty();
    if(fabs(mass-1.0)<2e-3 && worstGrad<1e-5 && roundTrip && named && proper)
      cout << "PASS lengthscale prior: mass " << mass << ", gradient, stream round trip, naming" << endl;
    else { cout << "FAIL lengthscale prior: mass " << mass << " grad " << worstGrad << " roundTrip " << roundTrip << " named " << named << " proper " << proper << endl; fail++; }
  }

  // 7. Sparse approximations (fixed inducing points): proper target, objective
  //    equals the prior evaluated at variance = 1/beta, gradient matches finite
  //    differences (Jacobian on), sampler runs, and free inducing points refuse.
  {
    const int approx[4] = {CGp::DTC, CGp::DTCVAR, CGp::FITC, CGp::PITC};
    const char* approxName[4] = {"DTC", "DTCVAR", "FITC", "PITC"};
    for(int ai=0; ai<4; ai++)
    {
      const unsigned int N = 30, M = 6;
      CMatrix X(N, 1), y(N, 1), Xu(M, 1);
      for(unsigned int i=0; i<N; i++) { const double xv = 6.0*i/(N-1.0); X.setVal(xv, i, 0); y.setVal(sin(xv)+0.1*cos(7.0*xv), i, 0); }
      for(unsigned int i=0; i<M; i++) Xu.setVal(6.0*i/(M-1.0), i, 0);
      CCmpndKern kern(X);
      kern.addKern(std::unique_ptr<CKern>(new CRbfKern(X)));
      CGaussianNoise noise(&y);
      for(int i=0; i<noise.getOutputDim(); i++) noise.setBiasVal(0.0, i);
      CGp model(&kern, &noise, &X, approx[ai], M, 0);
      model.setInducingVals(Xu);
      model.setBetaVal(10.0);
      model.setInducingFixed(false);
      bool refused = false;
      try { vector<SampledParam> sp; model.sampledParams(sp); } catch(...) { refused = true; }
      model.setInducingFixed(true);
      const string before = model.improperReason();
      for(int k=0; k<3; k++)
      {
        CGammaDist* g = new CGammaDist();
        g->setParam(2.0, 0);
        g->setParam(k==2 ? 10.0 : 1.0, 1);
        model.addPriorByPath(k==0 ? "rbf.inv_lengthscale" : (k==1 ? "rbf.variance" : "likelihood.variance"), g);
      }
      CMatrix a(1, model.getOptNumParams());
      model.getOptParams(a);
      CMatrix g(1, a.getCols()), gFd(1, a.getCols());
      model.computeSamplingObjectiveGrad(g);
      double maxDiff = 0.0;
      for(unsigned int j=0; j<a.getCols(); j++)
      {
        const double h = 1e-6, a0 = a.getVal(j);
        a.setVal(a0+h, j); model.setOptParams(a); const double ep = model.computeSamplingObjective();
        a.setVal(a0-h, j); model.setOptParams(a); const double em = model.computeSamplingObjective();
        a.setVal(a0, j); model.setOptParams(a);
        gFd.setVal((ep-em)/(2.0*h), j);
        maxDiff = max(maxDiff, fabs(gFd.getVal(j)-g.getVal(j))/(1.0+fabs(g.getVal(j))));
      }
      const bool proper = model.improperReason().empty();
      bool ok = refused && !before.empty() && before.find("likelihood.variance")!=string::npos
                && proper && maxDiff<1e-5;
      double accept = 0.0;
      bool positive = true;
      if(ok)
      {
        CHmc hmc(model, model, 5);
        hmc.setStepSize(0.1);
        hmc.setNumLeapfrog(8);
        hmc.adaptStepSize(150);
        CMatrix S, lp;
        hmc.sample(300, S, lp);
        accept = hmc.acceptRate();
        vector<SampledParam> sp;
        model.sampledParams(sp);
        for(unsigned int i=0; i<S.getRows(); i++)
          for(unsigned int j=0; j<S.getCols(); j++)
            positive = positive && sp[j].transform->atox(S.getVal(i, j))>0.0;
        ok = accept>0.4 && positive;
      }
      if(ok) cout << "PASS sparse " << approxName[ai] << ": gradient max rel diff " << maxDiff << ", acceptance " << accept << endl;
      else { cout << "FAIL sparse " << approxName[ai] << ": refused " << refused << " before '" << before << "' proper " << proper << " grad diff " << maxDiff << " accept " << accept << " positive " << positive << endl; fail++; }
    }
    // The noise prior value itself: objective shift equals logProb at variance 1/beta.
    {
      CMatrix X(10, 1), y(10, 1), Xu(4, 1);
      for(unsigned int i=0; i<10; i++) { X.setVal(0.5*i, i, 0); y.setVal(sin(0.5*i), i, 0); }
      for(unsigned int i=0; i<4; i++) Xu.setVal(1.5*i, i, 0);
      CCmpndKern kern(X);
      kern.addKern(std::unique_ptr<CKern>(new CRbfKern(X)));
      CGaussianNoise noise(&y);
      for(int i=0; i<noise.getOutputDim(); i++) noise.setBiasVal(0.0, i);
      CGp model(&kern, &noise, &X, CGp::FITC, 4, 0);
      model.setInducingVals(Xu);
      model.setBetaVal(8.0);
      model.setInducingFixed(true);
      const double l0 = model.logLikelihood();
      CGammaDist* g = new CGammaDist();
      g->setParam(2.0, 0);
      g->setParam(10.0, 1);
      CGammaDist probe;
      probe.setParam(2.0, 0);
      probe.setParam(10.0, 1);
      model.addPriorByPath("likelihood.variance", g);
      const double shift = model.logLikelihood() - l0;
      fail += checkMoment("sparse noise prior value at 1/beta", shift, probe.logProb(1.0/8.0), 1e-9);
    }
  }

  // 8. GPLVM: latent positions are sampled under the N(0, I) regulariser. The
  //    target is proper only with the regulariser or priors; gradients match
  //    finite differences; the sampler runs; unsupported configurations refuse.
  {
    const unsigned int N = 8, D = 3, Q = 2;
    CMatrix Y(N, D);
    for(unsigned int i=0; i<N; i++)
    {
      const double th = 2.0*3.14159265358979*i/N;
      Y.setVal(cos(th), i, 0); Y.setVal(sin(th), i, 1); Y.setVal(0.5*cos(2.0*th), i, 2);
    }
    CMatrix Xk(N, Q);
    CCmpndKern kern(Xk);
    kern.addKern(std::unique_ptr<CKern>(new CRbfKern(Xk)));
    kern.addKern(std::unique_ptr<CKern>(new CWhiteKern(Xk)));
    CScaleNoise noise(&Y);
    CMatrix np(1, 2*D);
    for(unsigned int j=0; j<D; j++) { np.setVal(0.0, j); np.setVal(1.0, j+D); }
    noise.setParams(np);
    CGplvm model(&kern, &noise, Q, 0);
    vector<SampledParam> sp;
    model.sampledParams(sp);
    const bool countOk = sp.size()==model.getOptNumParams() && sp.size()==kern.getNumParams()+N*Q;
    const string noPriors = model.improperReason();
    model.setLatentRegularised(false);
    const string noReg = model.improperReason();
    model.setLatentRegularised(true);
    for(unsigned int j=0; j<kern.getNumParams(); j++)
    {
      CGammaDist* g = new CGammaDist();
      g->setParam(2.0, 0);
      g->setParam(2.0, 1);
      kern.addPriorAt(g, j);
    }
    const bool proper = model.improperReason().empty();
    const bool flagged = noPriors.find("rbf")!=string::npos && noPriors.find("X.")==string::npos
                         && noReg.find("X.0.0")!=string::npos;
    CMatrix a(1, model.getOptNumParams());
    model.getOptParams(a);
    CMatrix g(1, a.getCols());
    model.computeSamplingObjectiveGrad(g);
    double maxDiff = 0.0;
    for(unsigned int j=0; j<a.getCols(); j++)
    {
      const double h = 1e-6, a0 = a.getVal(j);
      a.setVal(a0+h, j); model.setOptParams(a); const double ep = model.computeSamplingObjective();
      a.setVal(a0-h, j); model.setOptParams(a); const double em = model.computeSamplingObjective();
      a.setVal(a0, j); model.setOptParams(a);
      maxDiff = max(maxDiff, fabs((ep-em)/(2.0*h)-g.getVal(j))/(1.0+fabs(g.getVal(j))));
    }
    double accept = 0.0;
    bool ok = countOk && flagged && proper && maxDiff<1e-5;
    if(ok)
    {
      CHmc hmc(model, model, 9);
      hmc.setStepSize(0.05);
      hmc.setNumLeapfrog(10);
      hmc.adaptStepSize(100);
      CMatrix S, lp;
      hmc.sample(200, S, lp);
      accept = hmc.acceptRate();
      ok = accept>0.4;
    }
    model.setInputScaleLearnt(true);
    bool refused = false;
    try { vector<SampledParam> s2; model.sampledParams(s2); } catch(...) { refused = true; }
    if(ok && refused) cout << "PASS GPLVM: gradient max rel diff " << maxDiff << ", acceptance " << accept << endl;
    else { cout << "FAIL GPLVM: count " << countOk << " flagged " << flagged << " proper " << proper << " grad diff " << maxDiff << " accept " << accept << " refused " << refused << endl; fail++; }
  }

  cout << "Number of failures: " << fail << endl;
  return fail > 0 ? 1 : 0;
}
