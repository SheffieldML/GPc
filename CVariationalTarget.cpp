#include "CVariationalTarget.h"
#include "CBayesGplvm.h"
#include "CMrd.h"
#include "ndlexceptions.h"
#include <cmath>

CBoundHyperTarget::CBoundHyperTarget(COptimisable& b) : base(b)
{
}

void CBoundHyperTarget::addKernel(CKern& kern, unsigned int offset, const std::string& prefix)
{
  KernRecord rec;
  rec.kern = &kern;
  rec.offset = offset;
  kernels.push_back(rec);
  for(unsigned int i=0; i<kern.getNumParams(); i++)
  {
    Entry e;
    e.baseIndex = offset+i;
    e.kern = &kern;
    e.kernIndex = i;
    e.kernEntry = static_cast<unsigned int>(kernels.size()-1);
    e.prefix = prefix;
    e.free = true;
    entries.push_back(e);
  }
}

void CBoundHyperTarget::addPrecision(unsigned int index, const std::string& prefix, const CDist* prior)
{
  PrecisionRecord rec;
  rec.baseIndex = index;
  rec.prefix = prefix;
  if(prior!=0)
    rec.prior.reset(prior->clone());
  precisions.push_back(std::move(rec));
  Entry e;
  e.baseIndex = index;
  e.kern = 0;
  e.kernIndex = 0;
  e.kernEntry = static_cast<unsigned int>(precisions.size()-1);
  e.prefix = prefix;
  e.free = true;
  entries.push_back(e);
}

void CBoundHyperTarget::fix(const std::string& name)
{
  bool found = false;
  for(size_t i=0; i<entries.size(); i++)
  {
    SampledParam sp;
    if(entries[i].kern!=0)
      describeKernParam(sp, *entries[i].kern, entries[i].kernIndex);
    else
      sp.name = "likelihood.variance";
    if(entries[i].prefix+sp.name==name || entries[i].prefix+sp.rawName()==name)
    {
      entries[i].free = false;
      found = true;
    }
  }
  if(!found)
    throw ndlexceptions::Error("CBoundHyperTarget::fix: no sampled parameter named '"+name+"'.");
}

std::vector<std::string> CBoundHyperTarget::names() const
{
  std::vector<SampledParam> ps;
  sampledParams(ps);
  std::vector<std::string> out;
  for(size_t i=0; i<ps.size(); i++)
    out.push_back(ps[i].name);
  return out;
}

void CBoundHyperTarget::freeIndices(std::vector<unsigned int>& idx) const
{
  idx.clear();
  for(size_t i=0; i<entries.size(); i++)
    if(entries[i].free)
      idx.push_back(static_cast<unsigned int>(i));
}

void CBoundHyperTarget::sampledParams(std::vector<SampledParam>& params) const
{
  static const CReciprocalExpTransform varianceFromLogBeta;
  params.clear();
  CMatrix full(1, base.getOptNumParams());
  base.getOptParams(full);
  for(size_t i=0; i<entries.size(); i++)
  {
    const Entry& e = entries[i];
    if(!e.free)
      continue;
    SampledParam sp;
    if(e.kern!=0)
    {
      describeKernParam(sp, *e.kern, e.kernIndex);
      sp.name = e.prefix + sp.name;
      sp.nativeName = e.prefix + sp.nativeName;
      sp.transform = e.kern->getParamTransform(e.kernIndex);
      sp.hasPrior = e.kern->paramHasPrior(e.kernIndex);
      sp.constrainedValue = e.kern->getParam(e.kernIndex);
    }
    else
    {
      sp.name = e.prefix + "likelihood.variance";
      sp.transform = &varianceFromLogBeta;
      sp.hasPrior = precisions[e.kernEntry].prior.get()!=0;
      sp.constrainedValue = std::exp(-full.getVal(e.baseIndex));
    }
    params.push_back(sp);
  }
}

unsigned int CBoundHyperTarget::getOptNumParams() const
{
  unsigned int n = 0;
  for(size_t i=0; i<entries.size(); i++)
    if(entries[i].free)
      n++;
  return n;
}

void CBoundHyperTarget::getOptParams(CMatrix& param) const
{
  DIMENSIONMATCH(param.getRows()==1 && param.getCols()==getOptNumParams());
  CMatrix full(1, base.getOptNumParams());
  base.getOptParams(full);
  unsigned int c = 0;
  for(size_t i=0; i<entries.size(); i++)
    if(entries[i].free)
      param.setVal(full.getVal(entries[i].baseIndex), c++);
}

void CBoundHyperTarget::setOptParams(const CMatrix& param)
{
  DIMENSIONMATCH(param.getRows()==1 && param.getCols()==getOptNumParams());
  CMatrix full(1, base.getOptNumParams());
  base.getOptParams(full);
  unsigned int c = 0;
  for(size_t i=0; i<entries.size(); i++)
    if(entries[i].free)
      full.setVal(param.getVal(c++), entries[i].baseIndex);
  base.setOptParams(full);
}

// Log prior of the hyperparameters; if gFull is given, adds its gradient with
// respect to the base optimiser parameters (a-space, log-density convention).
double CBoundHyperTarget::priorTerms(const CMatrix& full, CMatrix* gFull) const
{
  double L = 0.0;
  for(size_t k=0; k<kernels.size(); k++)
  {
    const CKern& kern = *kernels[k].kern;
    L += kern.priorLogProb();
    if(gFull!=0)
    {
      CMatrix gNative(1, kern.getNumParams());
      gNative.zeros();
      kern.addPriorGrad(gNative);
      for(unsigned int i=0; i<kern.getNumParams(); i++)
      {
        const CTransform* t = kern.getParamTransform(i);
        const double f = t==0 ? 1.0 : t->gradfact(kern.getParam(i));
        gFull->addVal(gNative.getVal(i)*f, kernels[k].offset+i);
      }
    }
  }
  for(size_t p=0; p<precisions.size(); p++)
  {
    const CDist* prior = precisions[p].prior.get();
    if(prior==0)
      continue;
    const double variance = std::exp(-full.getVal(precisions[p].baseIndex));
    L += prior->logProb(variance);
    // d logp / da with variance = exp(-a): dlogp/dvariance * (-variance)
    if(gFull!=0)
      gFull->addVal(-prior->getGradInput(variance)*variance, precisions[p].baseIndex);
  }
  return L;
}

double CBoundHyperTarget::computeObjectiveGradParams(CMatrix& g) const
{
  DIMENSIONMATCH(g.getRows()==1 && g.getCols()==getOptNumParams());
  const unsigned int P = base.getOptNumParams();
  CMatrix full(1, P);
  base.getOptParams(full);
  CMatrix gFull(1, P);
  gFull.zeros();
  double E = base.computeObjectiveGradParams(gFull);
  CMatrix gPrior(1, P);
  gPrior.zeros();
  E -= priorTerms(full, &gPrior);
  gFull.axpy(gPrior, -1.0);
  unsigned int c = 0;
  for(size_t i=0; i<entries.size(); i++)
    if(entries[i].free)
      g.setVal(gFull.getVal(entries[i].baseIndex), c++);
  return E;
}

double CBoundHyperTarget::computeObjectiveVal() const
{
  CMatrix full(1, base.getOptNumParams());
  base.getOptParams(full);
  return base.computeObjectiveVal() - priorTerms(full, 0);
}

std::unique_ptr<CBoundHyperTarget> makeBoundHyperTarget(CBayesGplvm& model, const CDist* variancePrior)
{
  std::unique_ptr<CBoundHyperTarget> t(new CBoundHyperTarget(model));
  const unsigned int nk = model.getKern()->getNumParams();
  t->addKernel(*model.getKern(), 0, "");
  t->addPrecision(nk, "", variancePrior);
  return t;
}

std::unique_ptr<CBoundHyperTarget> makeBoundHyperTarget(CMrd& model, const CDist* variancePrior)
{
  std::unique_ptr<CBoundHyperTarget> t(new CBoundHyperTarget(model));
  unsigned int offset = 0;
  for(unsigned int v=0; v<model.getNumViews(); v++)
  {
    CKern* kern = model.getView(v)->getKern();
    const std::string prefix = "view" + ndlstrutil::itoa(v) + ".";
    t->addKernel(*kern, offset, prefix);
    t->addPrecision(offset+kern->getNumParams(), prefix, variancePrior);
    offset += kern->getNumParams()+1;
  }
  return t;
}
