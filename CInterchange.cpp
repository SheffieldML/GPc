#include "CInterchange.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

#include "ndlexceptions.h"

using namespace std;

namespace {

[[noreturn]] void fail(const string& where, const string& what)
{
  throw ndlexceptions::Error("interchange: " + where + ": " + what);
}

const CJson& need(const CJson& obj, const string& key, const string& where)
{
  if(!obj.isObject())
    fail(where, "expected an object");
  if(!obj.has(key))
    fail(where, "missing required field '" + key + "'");
  return obj.at(key);
}

double numberAt(const CJson& v, const string& where)
{
  if(!v.isNumber())
    fail(where, "expected a number");
  return v.asNumber();
}

double positiveAt(const CJson& v, const string& where)
{
  const double x = numberAt(v, where);
  if(!(x > 0.0))
    fail(where, "must be positive");
  return x;
}

unsigned int countAt(const CJson& v, const string& where)
{
  const double x = numberAt(v, where);
  if(x < 1.0 || x != std::floor(x) || x > 1e9)
    fail(where, "must be a positive integer");
  return static_cast<unsigned int>(x);
}

// Dense numeric matrix from an array of equal-length number arrays.
CMatrix matrixAt(const CJson& v, const string& where)
{
  if(!v.isArray() || v.size() == 0)
    fail(where, "expected a non-empty array of rows");
  if(!v.at(0).isArray() || v.at(0).size() == 0)
    fail(where, "rows must be non-empty arrays");
  const size_t rows = v.size();
  const size_t cols = v.at(0).size();
  CMatrix M(static_cast<unsigned int>(rows), static_cast<unsigned int>(cols));
  for(size_t i = 0; i < rows; ++i)
  {
    const CJson& row = v.at(i);
    if(!row.isArray() || row.size() != cols)
      fail(where, "ragged matrix: row " + to_string(i) + " has the wrong length");
    for(size_t j = 0; j < cols; ++j)
      M.setVal(numberAt(row.at(j), where + "[" + to_string(i) + "][" + to_string(j) + "]"),
               static_cast<unsigned int>(i), static_cast<unsigned int>(j));
  }
  return M;
}

CJson matrixToJson(const CMatrix& M)
{
  CJson rows = CJson::makeArray();
  for(unsigned int i = 0; i < M.getRows(); ++i)
  {
    CJson row = CJson::makeArray();
    for(unsigned int j = 0; j < M.getCols(); ++j)
      row.push(CJson::makeNumber(M.getVal(i, j)));
    rows.push(row);
  }
  return rows;
}

int approxFrom(const string& s, const string& where)
{
  if(s == "ftc") return CGp::FTC;
  if(s == "dtc") return CGp::DTC;
  if(s == "dtcvar") return CGp::DTCVAR;
  if(s == "fitc") return CGp::FITC;
  if(s == "pitc")
    fail(where, "approximation 'pitc' is not supported by this reader yet");
  fail(where, "unknown approximation '" + s + "'");
}

int linalgFrom(const string& s, const string& where)
{
  if(s == "lq") return CGp::SparseLinalgLQ;
  if(s == "stable") return CGp::SparseLinalgStableFactor;
  if(s == "legacy") return CGp::SparseLinalgLegacy;
  fail(where, "unknown sparse_linalg '" + s + "'");
}

// GPc realises likelihoods.mixed_gaussian as an index-white kernel plus a frozen
// likelihood variance of this size.
const double kMixedFloor = 1e-6;

// ARD split (CIP-0016): GPc rbfard has 1/l_j^2 = inverseWidth * inputScale_j. We
// store inverseWidth = min(l)^-2 and inputScale_j = (min l / l_j)^2, so the
// smallest lengthscale carries scale one.
bool buildLeafCore(const CJson& node, unsigned int inDim, const string& where,
                   unique_ptr<CKern>& out, bool& isWhite)
{
  isWhite = false;
  const string type = need(node, "type", where).asString();
  if(type == "kern.coregionalize")
  {
    if(inDim != 1)
      fail(where, "kern.coregionalize needs active_dims with exactly one column");
    const unsigned int D = countAt(need(node, "output_dim", where), where + ".output_dim");
    const CMatrix W = matrixAt(need(node, "W", where), where + ".W");
    if(W.getRows() != D)
      fail(where + ".W", "has " + to_string(W.getRows()) + " rows but output_dim is " + to_string(D));
    const CJson& kappa = need(node, "kappa", where);
    if(!kappa.isArray() || kappa.size() != D)
      fail(where + ".kappa", "must be an array of output_dim entries");
    unique_ptr<CCoregionalizeKern> k(new CCoregionalizeKern(1, D, W.getCols()));
    for(unsigned int d = 0; d < D; ++d)
      for(unsigned int r = 0; r < W.getCols(); ++r)
        k->setParam(W.getVal(d, r), d * W.getCols() + r);
    for(unsigned int d = 0; d < D; ++d)
      k->setParam(positiveAt(kappa.at(d), where + ".kappa[" + to_string(d) + "]"),
                  D * W.getCols() + d);
    out = std::move(k);
    return true;
  }
  if(type == "kern.rbf" || type == "kern.matern32" || type == "kern.matern52")
  {
    const double variance = positiveAt(need(node, "variance", where), where + ".variance");
    const CJson& ls = need(node, "lengthscale", where);
    if(ls.isArray())
    {
      if(type != "kern.rbf")
        fail(where, "ARD lengthscales are only supported for kern.rbf");
      if(ls.size() != inDim)
        fail(where + ".lengthscale", "has " + to_string(ls.size())
             + " entries but input_dim is " + to_string(inDim));
      vector<double> l(inDim);
      double lmin = 0.0;
      for(unsigned int j = 0; j < inDim; ++j)
      {
        l[j] = positiveAt(ls.at(j), where + ".lengthscale[" + to_string(j) + "]");
        if(j == 0 || l[j] < lmin)
          lmin = l[j];
      }
      unique_ptr<CRbfardKern> k(new CRbfardKern(inDim));
      k->setByPath("lengthscale", lmin);
      k->setByPath("variance", variance);
      for(unsigned int j = 0; j < inDim; ++j)
      {
        const double r = lmin / l[j];
        k->setByPath("inputScale_" + to_string(j), r * r);
      }
      out = std::move(k);
      return true;
    }
    const double lscalar = positiveAt(ls, where + ".lengthscale");
    unique_ptr<CKern> k;
    if(type == "kern.rbf")
      k.reset(new CRbfKern(inDim));
    else if(type == "kern.matern32")
      k.reset(new CMatern32Kern(inDim));
    else
      k.reset(new CMatern52Kern(inDim));
    k->setByPath("lengthscale", lscalar);
    k->setByPath("variance", variance);
    out = std::move(k);
    return true;
  }
  if(type == "kern.white" || type == "kern.bias")
  {
    const double variance = positiveAt(need(node, "variance", where), where + ".variance");
    unique_ptr<CKern> k;
    if(type == "kern.white")
    {
      k.reset(new CWhiteKern(inDim));
      isWhite = true;
    }
    else
      k.reset(new CBiasKern(inDim));
    k->setByPath("variance", variance);
    out = std::move(k);
    return true;
  }
  return false;
}

// A leaf reads either every input column or the listed active_dims (GPy
// semantics); lengthscale lists must match the number of columns it reads.
bool buildLeaf(const CJson& node, unsigned int totalDim, const string& where,
               unique_ptr<CKern>& out, bool& isWhite)
{
  if(!node.has("active_dims"))
    return buildLeafCore(node, totalDim, where, out, isWhite);
  const CJson& ad = node.at("active_dims");
  if(!ad.isArray() || ad.size() == 0)
    fail(where + ".active_dims", "must be a non-empty array of column indices");
  vector<unsigned int> dims;
  for(size_t i = 0; i < ad.size(); ++i)
  {
    const double v = numberAt(ad.at(i), where + ".active_dims");
    if(v < 0.0 || v != std::floor(v) || v >= totalDim)
      fail(where + ".active_dims", "column " + to_string(v) + " is outside 0.." + to_string(totalDim - 1));
    if(std::find(dims.begin(), dims.end(), static_cast<unsigned int>(v)) != dims.end())
      fail(where + ".active_dims", "repeats a column");
    dims.push_back(static_cast<unsigned int>(v));
  }
  if(!buildLeafCore(node, static_cast<unsigned int>(dims.size()), where, out, isWhite))
    return false;
  out->setActiveDims(dims);
  return true;
}

// Build one kernel node. kern.prod becomes a CTensorKern (product over the same
// inputs); a kern.add inside a product becomes a nested CCmpndKern. Leaves are
// recorded in document order so toJson() can read parameters back. The parent
// takes ownership without cloning, so the raw leaf pointers stay valid.
unique_ptr<CKern> buildKernelNode(const CJson& node, unsigned int inDim, const string& where,
                                  vector<CKern*>& leaves, bool inProduct)
{
  const string type = need(node, "type", where).asString();
  if(type == "kern.add" || type == "kern.prod")
  {
    const bool prod = (type == "kern.prod");
    const CJson& parts = need(node, "parts", where);
    if(!parts.isArray() || parts.size() == 0)
      fail(where + ".parts", "must be a non-empty array");
    unique_ptr<CCmpndKern> sum;
    unique_ptr<CTensorKern> product;
    if(prod) product.reset(new CTensorKern(inDim)); else sum.reset(new CCmpndKern(inDim));
    for(size_t i = 0; i < parts.size(); ++i)
    {
      unique_ptr<CKern> child = buildKernelNode(parts.at(i), inDim,
          where + ".parts[" + to_string(i) + "]", leaves, inProduct || prod);
      if(prod) product->addKern(std::move(child)); else sum->addKern(std::move(child));
    }
    if(prod) return unique_ptr<CKern>(std::move(product));
    return unique_ptr<CKern>(std::move(sum));
  }
  unique_ptr<CKern> leaf;
  bool white = false;
  if(!buildLeaf(node, inDim, where, leaf, white))
    fail(where, "unknown kernel type '" + type + "'");
  if(node.at("type").asString() == "kern.coregionalize" && !node.has("active_dims"))
    fail(where, "kern.coregionalize requires active_dims");
  if(white && inProduct)
    fail(where, "kern.white inside kern.prod is not supported (it is an additive noise term)");
  leaves.push_back(leaf.get());
  return leaf;
}

// Top level: sums are flattened into the model's compound kernel.
void buildKernelTree(const CJson& node, unsigned int inDim, const string& where,
                     CCmpndKern& sum, vector<CKern*>& leaves, bool& hasWhite)
{
  const string type = need(node, "type", where).asString();
  if(type == "kern.add")
  {
    const CJson& parts = need(node, "parts", where);
    if(!parts.isArray() || parts.size() == 0)
      fail(where + ".parts", "must be a non-empty array");
    for(size_t i = 0; i < parts.size(); ++i)
      buildKernelTree(parts.at(i), inDim, where + ".parts[" + to_string(i) + "]",
                      sum, leaves, hasWhite);
    return;
  }
  if(type == "kern.white")
    hasWhite = true;
  sum.addKern(buildKernelNode(node, inDim, where, leaves, false));
}

}  // namespace

CInterchange::~CInterchange() {}

unique_ptr<CInterchange> CInterchange::fromFile(const string& path)
{
  return fromJson(CJson::parseFile(path));
}

unique_ptr<CInterchange> CInterchange::fromJson(const CJson& doc)
{
  if(!doc.isObject())
    fail("document", "expected a JSON object");
  const string version = need(doc, "schema_version", "document").asString();
  if(version != schemaVersion())
    fail("schema_version", "'" + version + "' is not supported (reader handles '"
         + schemaVersion() + "')");

  const string type = need(doc, "type", "document").asString();
  const bool sparse = (type == "models.sparse_gp_regression");
  const bool generic = (type == "models.gp");
  if(!sparse && !generic && type != "models.gp_regression")
    fail("type", "unknown model type '" + type + "'");

  const unsigned int inputDim = countAt(need(doc, "input_dim", "document"), "input_dim");
  if(doc.has("output_dim") && countAt(doc.at("output_dim"), "output_dim") != 1)
    fail("output_dim", "only single-output models are supported");

  unique_ptr<CInterchange> m(new CInterchange());
  m->doc = doc;

  // Data.
  const CJson& data = need(doc, "data", "document");
  m->X = matrixAt(need(data, "X", "data"), "data.X");
  m->Y = matrixAt(need(data, "Y", "data"), "data.Y");
  if(m->X.getCols() != inputDim)
    fail("data.X", "has " + to_string(m->X.getCols()) + " columns but input_dim is "
         + to_string(inputDim));
  if(m->Y.getRows() != m->X.getRows())
    fail("data.Y", "has " + to_string(m->Y.getRows()) + " rows but data.X has "
         + to_string(m->X.getRows()));
  if(m->Y.getCols() != 1)
    fail("data.Y", "only a single output column is supported");

  // Likelihood.
  const CJson& lik = need(doc, "likelihood", "document");
  const string likType = need(lik, "type", "likelihood").asString();
  const bool mixed = (likType == "likelihoods.mixed_gaussian");
  const bool bernoulli = (likType == "likelihoods.bernoulli"), poisson = (likType == "likelihoods.poisson");
  if(generic != (bernoulli || poisson))
    fail("type", "models.gp is for non-Gaussian likelihoods (bernoulli, poisson); got likelihood '" + likType + "'");
  if(!mixed && !bernoulli && !poisson && likType != "likelihoods.gaussian")
    fail("likelihood.type", "only gaussian, mixed_gaussian, bernoulli and poisson likelihoods are supported");
  if(generic)
  {
    // Bernoulli (probit) or Poisson (log) with Laplace inference on a full GP.
    if(!doc.has("inference") || doc.at("inference").asString() != "laplace")
      fail("inference", "non-Gaussian likelihoods need inference: laplace");
    if(doc.has("approximation") && doc.at("approximation").asString() != "ftc")
      fail("approximation", "non-Gaussian models are full GPs (ftc)");
    if(doc.has("mean") || doc.has("normalizer") || doc.has("sparse_linalg") || doc.has("inducing_inputs"))
      fail("document", "mean, normalizer and sparse fields are not defined for non-Gaussian models");
    if(lik.has("variance"))
      fail("likelihood.variance", "bernoulli and poisson likelihoods take no variance");
    const string link = need(lik, "link", "likelihood").asString();
    if(link != (bernoulli ? "probit" : "log"))
      fail("likelihood.link", "'" + link + "' is not valid for " + likType);
    for(unsigned int i = 0; i < m->Y.getRows(); ++i)
    {
      const double y = m->Y.getVal(i, 0);
      if(bernoulli && y != 0.0 && y != 1.0)
        fail("data.Y", "row " + to_string(i) + ": bernoulli labels must be 0 or 1");
      if(poisson && (y < 0.0 || y != std::floor(y)))
        fail("data.Y", "row " + to_string(i) + ": poisson counts must be non-negative integers");
    }
    m->kern.reset(new CCmpndKern(inputDim));
    bool white = false;
    buildKernelTree(need(doc, "kernel", "document"), inputDim, "kernel", *m->kern, m->leaves, white);
    if(white)
      fail("kernel", "kern.white is not supported for non-Gaussian models");
    if(bernoulli)
    {
      m->noise.reset(new CProbitNoise(&m->Y));
      m->noise->setParam(0.0, 0);   // GPy's Bernoulli has no bias: p(y=1|f) = Phi(f)
    }
    else
      m->noise.reset(new CPoissonNoise(&m->Y));
    m->pgp.reset(new CGp(m->kern.get(), m->noise.get(), &m->X, CGp::FTC, 0, 0));
    m->pgp->setBiasVal(0.0, 0);
    m->pgp->setLaplaceInference(true);
    // GPy stops at 1e-4 on the objective, which leaves ~1e-6 error in the predictive; the
    // interchange reader solves to convergence so documents have one meaning.
    m->pgp->setLaplaceSolver(200, 1e-13, 0.0);
    m->pgp->setVerbosity(0);
    m->laplace = true;
    m->nonGaussian = bernoulli ? BernoulliNG : PoissonNG;
    return m;
  }
  vector<double> mixedVar;
  unsigned int mixedCol = 0;
  double sigma2 = 0.0;
  if(mixed)
  {
    const CJson& v = need(lik, "variance", "likelihood");
    if(!v.isArray() || v.size() == 0)
      fail("likelihood.variance", "mixed_gaussian needs an array of per-output variances");
    for(size_t i = 0; i < v.size(); ++i)
    {
      mixedVar.push_back(positiveAt(v.at(i), "likelihood.variance[" + to_string(i) + "]"));
      if(mixedVar.back() <= kMixedFloor)
        fail("likelihood.variance[" + to_string(i) + "]", "must exceed " + to_string(kMixedFloor)
             + " (GPc keeps a frozen likelihood variance of that size)");
    }
    const double c = numberAt(need(lik, "output_index_column", "likelihood"),
                              "likelihood.output_index_column");
    if(c < 0.0 || c != std::floor(c) || c >= inputDim)
      fail("likelihood.output_index_column", "is not a column of data.X");
    mixedCol = static_cast<unsigned int>(c);
    for(unsigned int i = 0; i < m->X.getRows(); ++i)
    {
      const double idx = m->X.getVal(i, mixedCol);
      if(idx < 0.0 || idx != std::floor(idx) || idx >= mixedVar.size())
        fail("data.X", "row " + to_string(i) + " has output index " + to_string(idx)
             + " outside 0.." + to_string(mixedVar.size() - 1));
    }
  }
  else
  {
    if(need(lik, "variance", "likelihood").isArray())
      fail("likelihood.variance", "likelihoods.gaussian takes a single variance");
    sigma2 = positiveAt(need(lik, "variance", "likelihood"), "likelihood.variance");
  }

  // Kernel.
  m->kern.reset(new CCmpndKern(inputDim));
  bool hasWhite = false;
  buildKernelTree(need(doc, "kernel", "document"), inputDim, "kernel",
                  *m->kern, m->leaves, hasWhite);
  m->hasWhite = hasWhite;
  if(mixed)
  {
    // Per-output noise is an index-white kernel; the likelihood variance is frozen
    // at kMixedFloor and subtracted from each wire variance so the Gram is exact.
    unique_ptr<CIndexWhiteKern> iw(new CIndexWhiteKern(1, static_cast<unsigned int>(mixedVar.size())));
    for(unsigned int d = 0; d < mixedVar.size(); ++d)
      iw->setParam(mixedVar[d] - kMixedFloor, d);
    iw->setActiveDims(vector<unsigned int>(1, mixedCol));
    m->indexWhite = iw.get();
    m->kern->addKern(std::move(iw));
    sigma2 = kMixedFloor;
  }

  // Noise: zero mean, so the bias is fixed to zero (GPy semantics).
  {
    unique_ptr<CGaussianNoise> g(new CGaussianNoise(&m->Y));
    g->setBiasVal(0.0, 0);
    m->noise = std::move(g);
  }

  // Target transform. mean.constant c and normalizer both map onto GPc's stored
  // bias/scale (m = (y - bias)/scale). They are exclusive in 0.1: the order in
  // which a mean function and a normalizer apply is not yet specified.
  double bias = 0.0, scale = 1.0;
  if(doc.has("mean") && doc.has("normalizer"))
    fail("document", "'mean' and 'normalizer' cannot both be present in schema 0.1");
  if(doc.has("mean"))
  {
    const CJson& mn = doc.at("mean");
    if(need(mn, "type", "mean").asString() != "mean.constant")
      fail("mean.type", "only mean.constant is supported");
    bias = numberAt(need(mn, "value", "mean"), "mean.value");
  }
  if(doc.has("normalizer"))
  {
    const CJson& nz = doc.at("normalizer");
    const string nt = need(nz, "type", "normalizer").asString();
    bias = numberAt(need(nz, "mean", "normalizer"), "normalizer.mean");
    if(nt == "normalizer.standardize")
      scale = positiveAt(need(nz, "std", "normalizer"), "normalizer.std");
    else if(nt != "normalizer.center")
      fail("normalizer.type", "'" + nt + "' is not supported");
    // Density of y from the density of (y - mean)/std.
    m->targetJacobian = -static_cast<double>(m->Y.getRows())
        * static_cast<double>(m->Y.getCols()) * std::log(scale);
  }

  if(!sparse)
  {
    if(doc.has("approximation") && doc.at("approximation").asString() != "ftc")
      fail("approximation", "models.gp_regression requires 'ftc' (or omit the field)");
    if(doc.has("sparse_linalg") || doc.has("inducing_inputs"))
      fail("document", "sparse_linalg and inducing_inputs are not valid for models.gp_regression");
    m->pgp.reset(new CGp(m->kern.get(), m->noise.get(), &m->X, CGp::FTC, 0, 0));
    m->pgp->setLikelihoodVariance(sigma2);
    // The wire format is additive: Gram = K (including any White) + sigma^2 I.
    if(hasWhite)
      m->pgp->setAllowWhiteAndLikelihoodNoise(true);
    if(mixed)
      m->pgp->fixLikelihoodVariance(true);
  }
  else
  {
    const int approx = approxFrom(need(doc, "approximation", "document").asString(), "approximation");
    if(mixed)
      fail("likelihood.type", "likelihoods.mixed_gaussian is only supported with models.gp_regression");
    if(approx == CGp::FTC)
      fail("approximation", "'ftc' is not a sparse approximation");
    const int linalg = doc.has("sparse_linalg")
        ? linalgFrom(doc.at("sparse_linalg").asString(), "sparse_linalg")
        : static_cast<int>(CGp::SparseLinalgLQ);
    m->Z = matrixAt(need(doc, "inducing_inputs", "document"), "inducing_inputs");
    if(m->Z.getCols() != inputDim)
      fail("inducing_inputs", "has " + to_string(m->Z.getCols())
           + " columns but input_dim is " + to_string(inputDim));
    if(m->Z.getRows() > m->X.getRows())
      fail("inducing_inputs", "more inducing points than data points");
    m->pgp.reset(new CGp(m->kern.get(), m->noise.get(), &m->X, approx, m->Z.getRows(), 0));
    m->pgp->setSparseLinalgMode(linalg);
    m->pgp->setActiveSet(m->Z);
    m->pgp->setInducingFixed(false);
    m->pgp->setByPath("likelihood.variance", sigma2);
  }
  m->pgp->setBiasVal(bias, 0);
  m->pgp->setScaleVal(scale, 0);
  return m;
}

double CInterchange::logLikelihood() const
{
  return -pgp->computeObjectiveVal() + targetJacobian;
}

void CInterchange::predict(const CMatrix& Xnew, vector<double>& mean, vector<double>& var) const
{
  if(Xnew.getCols() != X.getCols())
    fail("predict", "Xnew has " + to_string(Xnew.getCols()) + " columns but the model has "
         + to_string(X.getCols()) + " inputs");
  if(laplace)
  {
    // Moments of y from the latent Gaussian N(mu, s2).
    vector<double> mu, s2;
    predictLatent(Xnew, mu, s2);
    mean.resize(mu.size());
    var.resize(mu.size());
    for(size_t i = 0; i < mu.size(); ++i)
    {
      if(nonGaussian == BernoulliNG)
      {
        const double p = ndlutil::cumGaussian(mu[i] / std::sqrt(1.0 + s2[i]));
        mean[i] = p;
        var[i] = p * (1.0 - p);
      }
      else
      {
        mean[i] = std::exp(mu[i] + 0.5 * s2[i]);
        var[i] = mean[i] + mean[i] * mean[i] * std::expm1(s2[i]);
      }
    }
    return;
  }
  CMatrix mu(Xnew.getRows(), Y.getCols());
  CMatrix vs(Xnew.getRows(), Y.getCols());
  pgp->predictiveMeanVar(mu, vs, Xnew);
  mean.resize(Xnew.getRows());
  var.resize(Xnew.getRows());
  for(unsigned int i = 0; i < Xnew.getRows(); ++i)
  {
    mean[i] = mu.getVal(i, 0);
    var[i] = vs.getVal(i, 0);
  }
}

void CInterchange::predictLatent(const CMatrix& Xnew, vector<double>& mu, vector<double>& var) const
{
  if(!laplace)
    fail("predictLatent", "only Laplace models have a latent predictive");
  CMatrix m(Xnew.getRows(), 1), v(Xnew.getRows(), 1);
  pgp->laplaceLatentMeanVar(m, v, Xnew);
  mu.resize(Xnew.getRows());
  var.resize(Xnew.getRows());
  for(unsigned int i = 0; i < Xnew.getRows(); ++i)
  {
    mu[i] = m.getVal(i, 0);
    var[i] = v.getVal(i, 0);
  }
}

void CInterchange::writeKernelParams(CJson& node, size_t& leafIndex) const
{
  const string type = node.at("type").asString();
  if(type == "kern.add" || type == "kern.prod")
  {
    CJson& parts = node.at("parts");
    for(size_t i = 0; i < parts.size(); ++i)
      writeKernelParams(parts.at(i), leafIndex);
    return;
  }
  const CKern* k = leaves.at(leafIndex++);
  if(type == "kern.coregionalize")
  {
    const CCoregionalizeKern* c = static_cast<const CCoregionalizeKern*>(k);
    const unsigned int D = c->getOutputDim(), R = c->getRank();
    CMatrix W(D, R);
    for(unsigned int d = 0; d < D; ++d)
      for(unsigned int r = 0; r < R; ++r)
        W.setVal(c->getParam(d * R + r), d, r);
    node.set("W", matrixToJson(W));
    CJson kappa = CJson::makeArray();
    for(unsigned int d = 0; d < D; ++d)
      kappa.push(CJson::makeNumber(c->getParam(D * R + d)));
    node.set("kappa", kappa);
    return;
  }
  node.set("variance", CJson::makeNumber(k->getByPath("variance")));
  if(type == "kern.rbf" || type == "kern.matern32" || type == "kern.matern52")
  {
    if(node.at("lengthscale").isArray())
    {
      const double base = k->getByPath("lengthscale");
      CJson ls = CJson::makeArray();
      for(size_t j = 0; j < node.at("lengthscale").size(); ++j)
      {
        const double s = k->getByPath("inputScale_" + to_string(j));
        ls.push(CJson::makeNumber(base / std::sqrt(s)));
      }
      node.set("lengthscale", ls);
    }
    else
      node.set("lengthscale", CJson::makeNumber(k->getByPath("lengthscale")));
  }
}

CJson CInterchange::toJson() const
{
  CJson out = doc;
  out.erase("x-reference");
  size_t leafIndex = 0;
  writeKernelParams(out.at("kernel"), leafIndex);
  if(laplace)
  {
    // no likelihood parameters on the wire
  }
  else if(indexWhite)
  {
    CJson v = CJson::makeArray();
    for(unsigned int d = 0; d < indexWhite->getOutputDim(); ++d)
      v.push(CJson::makeNumber(indexWhite->getParam(d) + pgp->getByPath("likelihood.variance")));
    out.at("likelihood").set("variance", v);
  }
  else
    out.at("likelihood").set("variance",
        CJson::makeNumber(pgp->getByPath("likelihood.variance")));
  // Inducing inputs move during optimisation unless fixed.
  if(pgp->isSparseApproximation())
  {
    CMatrix Z(pgp->getNumActive(), X.getCols());
    for(unsigned int i = 0; i < Z.getRows(); ++i)
      for(unsigned int j = 0; j < Z.getCols(); ++j)
        Z.setVal(pgp->X_u.getVal(i, j), i, j);
    out.set("inducing_inputs", matrixToJson(Z));
  }
  return out;
}

void CInterchange::writeFile(const string& path) const
{
  toJson().dumpFile(path);
}

void CInterchange::writeGpModel(const string& path, const string& comment) const
{
  // An exact model with White sets the allow-white flag in fromJson, and the
  // stream records it, so the additive likelihood survives a reload.
  writeGpToFile(*pgp, path, comment);
}

namespace {

CJson exportKernel(const CKern* k, unsigned int inDim, const string& where, bool whiteOk);

CJson exportLeaf(const CKern* k, unsigned int inDim, const string& where, bool whiteOk);

CJson exportKernel(const CKern* k, unsigned int inDim, const string& where, bool whiteOk)
{
  CJson node = exportLeaf(k, k->getInputDim(), where, whiteOk);
  // A leaf that reads fewer columns than the model has, or a permuted subset, names them.
  const bool composite = dynamic_cast<const CCmpndKern*>(k) || dynamic_cast<const CTensorKern*>(k);
  if(!composite && (k->getInputDim() != inDim || !k->isDefaultActiveDims()))
  {
    CJson dims = CJson::makeArray();
    for(unsigned int d : k->getActiveDims())
      dims.push(CJson::makeNumber(d));
    node.set("active_dims", dims);
  }
  return node;
}

CJson exportLeaf(const CKern* k, unsigned int inDim, const string& where, bool whiteOk)
{
  const string type = k->getType();
  CJson node = CJson::makeObject();
  if(const CCmpndKern* c = dynamic_cast<const CCmpndKern*>(k))
  {
    node.set("type", CJson::makeString("kern.add"));
    CJson parts = CJson::makeArray();
    for(unsigned int i = 0; i < c->getNumKerns(); ++i)
      parts.push(exportKernel(c->getComponent(i), inDim, where + ".parts[" + to_string(i) + "]", whiteOk));
    node.set("parts", parts);
    return node;
  }
  if(dynamic_cast<const CTensorKern*>(k))
  {
    const CTensorKern* t = static_cast<const CTensorKern*>(k);
    node.set("type", CJson::makeString("kern.prod"));
    CJson parts = CJson::makeArray();
    for(unsigned int i = 0; i < t->getNumKerns(); ++i)
      parts.push(exportKernel(t->getComponent(i), inDim, where + ".parts[" + to_string(i) + "]", whiteOk));
    node.set("parts", parts);
    return node;
  }
  if(type == "coregionalize")
  {
    const CCoregionalizeKern* c = static_cast<const CCoregionalizeKern*>(k);
    const unsigned int D = c->getOutputDim(), R = c->getRank();
    node.set("type", CJson::makeString("kern.coregionalize"));
    node.set("output_dim", CJson::makeNumber(D));
    CMatrix W(D, R);
    for(unsigned int d = 0; d < D; ++d)
      for(unsigned int r = 0; r < R; ++r)
        W.setVal(c->getParam(d * R + r), d, r);
    node.set("W", matrixToJson(W));
    CJson kappa = CJson::makeArray();
    for(unsigned int d = 0; d < D; ++d)
      kappa.push(CJson::makeNumber(c->getParam(D * R + d)));
    node.set("kappa", kappa);
    return node;
  }
  if(type == "white")
  {
    if(!whiteOk)
      fail(where, "kern.white cannot be exported from this .model: without the allow-white flag "
           "(gp learn --allow-white-and-noise) White owns the noise, and sparse models treat it as "
           "jitter; both differ from the wire format's additive rule");
    node.set("type", CJson::makeString("kern.white"));
    node.set("variance", CJson::makeNumber(k->getByPath("variance")));
    return node;
  }
  if(type == "rbf" || type == "matern32" || type == "matern52")
  {
    node.set("type", CJson::makeString("kern." + type));
    node.set("variance", CJson::makeNumber(k->getByPath("variance")));
    node.set("lengthscale", CJson::makeNumber(k->getByPath("lengthscale")));
    return node;
  }
  if(type == "rbfard")
  {
    node.set("type", CJson::makeString("kern.rbf"));
    node.set("variance", CJson::makeNumber(k->getByPath("variance")));
    const double base = k->getByPath("lengthscale");
    CJson ls = CJson::makeArray();
    for(unsigned int j = 0; j < inDim; ++j)
      ls.push(CJson::makeNumber(base / std::sqrt(k->getByPath("inputScale_" + to_string(j)))));
    node.set("lengthscale", ls);
    node.set("ARD", CJson::makeBool(true));
    return node;
  }
  if(type == "bias")
  {
    node.set("type", CJson::makeString("kern.bias"));
    node.set("variance", CJson::makeNumber(k->getByPath("variance")));
    return node;
  }
  fail(where, "kernel type '" + type + "' has no wire-format mapping yet");
}

}  // namespace

CJson CInterchange::exportLaplaceModel(const CGp& model)
{
  const string nt = model.getNoiseType();
  if(nt != "probit" && nt != "poisson")
    fail("export", "Laplace inference is on the wire only for probit (Bernoulli) and poisson noise, not '" + nt + "'");
  if(model.isSparseApproximation() || model.getApproximationType() != CGp::FTC)
    fail("export", "non-Gaussian models are full GPs (ftc) in schema 0.1");
  if(model.getBiasVal(0) != 0.0 || model.getScaleVal(0) != 1.0 || model.isOutputScaleLearnt())
    fail("export", "mean and normalizer are not defined for non-Gaussian models in schema 0.1");
  CJson doc = CJson::makeObject();
  doc.set("schema_version", CJson::makeString(schemaVersion()));
  doc.set("type", CJson::makeString("models.gp"));
  doc.set("input_dim", CJson::makeNumber(model.getInputDim()));
  doc.set("output_dim", CJson::makeNumber(1));
  doc.set("approximation", CJson::makeString("ftc"));
  doc.set("inference", CJson::makeString("laplace"));
  doc.set("kernel", exportKernel(model.getKernel(), model.getInputDim(), "kernel", false));
  CJson lik = CJson::makeObject();
  lik.set("type", CJson::makeString(nt == "probit" ? "likelihoods.bernoulli" : "likelihoods.poisson"));
  lik.set("link", CJson::makeString(nt == "probit" ? "probit" : "log"));
  doc.set("likelihood", lik);
  CJson data = CJson::makeObject();
  data.set("X", matrixToJson(*model.pX));
  data.set("Y", matrixToJson(*model.py));
  doc.set("data", data);
  // Verify against a rebuilt model (this also catches a probit bias, which the wire format lacks); both solve the same Laplace problem.
  const double want = fromJson(doc)->logLikelihood();
  const double got = -model.computeObjectiveVal();
  if(!(std::fabs(got - want) <= 1e-5 * (1.0 + std::fabs(want))))
    fail("export", "verification failed: source log likelihood " + to_string(got)
         + " but the exported document gives " + to_string(want));
  return doc;
}

CJson CInterchange::exportModel(const CGp& model)
{
  if(model.getOutputDim() != 1)
    fail("export", "only single-output models are supported");
  const bool laplaceModel = model.isLaplaceInference();
  if(laplaceModel)
    return exportLaplaceModel(model);
  if(!model.isGaussianNoise())
    fail("export", "only Gaussian likelihoods are supported (and Bernoulli/Poisson with Laplace inference)");
  if(model.getApproximationType() == CGp::PITC)
    fail("export", "PITC is not supported");
  if(model.isOutputScaleLearnt())
    fail("export", "the output scale is a learnt parameter with GPc's own prior term; "
         "the wire format only carries a fixed normalizer");
  if(!(model.getScaleVal(0) > 0.0))
    fail("export", "a non-positive output scale cannot be written as a normalizer");
  const double outBias = model.getBiasVal(0), outScale = model.getScaleVal(0);
  const bool sparse = model.isSparseApproximation();

  CJson doc = CJson::makeObject();
  doc.set("schema_version", CJson::makeString(schemaVersion()));
  doc.set("type", CJson::makeString(sparse ? "models.sparse_gp_regression" : "models.gp_regression"));
  doc.set("input_dim", CJson::makeNumber(model.getInputDim()));
  doc.set("output_dim", CJson::makeNumber(1));
  if(sparse)
  {
    const char* names[] = {"ftc", "dtc", "fitc", "pitc", "dtcvar"};
    doc.set("approximation", CJson::makeString(names[model.getApproximationType()]));
    const char* linalg = model.getSparseLinalgMode() == CGp::SparseLinalgLQ ? "lq"
        : model.getSparseLinalgMode() == CGp::SparseLinalgStableFactor ? "stable" : "legacy";
    doc.set("sparse_linalg", CJson::makeString(linalg));
    CMatrix Z(model.getNumActive(), model.getInputDim());
    for(unsigned int i = 0; i < Z.getRows(); ++i)
      for(unsigned int j = 0; j < Z.getCols(); ++j)
        Z.setVal(model.X_u.getVal(i, j), i, j);
    doc.set("inducing_inputs", matrixToJson(Z));
  }
  const bool whiteOk = !sparse && model.isAllowWhiteAndLikelihoodNoise();
  // A top-level index-white component is per-output noise: it leaves the kernel and
  // becomes likelihoods.mixed_gaussian (wire variance = index-white + likelihood variance).
  const CKern* kernel = model.getKernel();
  const CIndexWhiteKern* iw = nullptr;
  vector<const CKern*> others;
  if(const CCmpndKern* top = dynamic_cast<const CCmpndKern*>(kernel))
    for(unsigned int i = 0; i < top->getNumKerns(); ++i)
    {
      const CKern* c = top->getComponent(i);
      if(c->getType() == "indexwhite")
      {
        if(iw)
          fail("export", "more than one index-white component");
        iw = static_cast<const CIndexWhiteKern*>(c);
      }
      else
        others.push_back(c);
    }
  if(iw)
  {
    if(sparse)
      fail("export", "index-white noise is only supported for exact models");
    if(others.empty())
      fail("export", "index-white noise without a covariance kernel");
    if(others.size() == 1)
      doc.set("kernel", exportKernel(others[0], model.getInputDim(), "kernel", whiteOk));
    else
    {
      CJson parts = CJson::makeArray();
      for(size_t i = 0; i < others.size(); ++i)
        parts.push(exportKernel(others[i], model.getInputDim(), "kernel.parts[" + to_string(i) + "]", whiteOk));
      CJson add = CJson::makeObject();
      add.set("type", CJson::makeString("kern.add"));
      add.set("parts", parts);
      doc.set("kernel", add);
    }
  }
  else
    doc.set("kernel", exportKernel(kernel, model.getInputDim(), "kernel", whiteOk));
  CJson lik = CJson::makeObject();
  if(iw)
  {
    lik.set("type", CJson::makeString("likelihoods.mixed_gaussian"));
    CJson v = CJson::makeArray();
    for(unsigned int d = 0; d < iw->getOutputDim(); ++d)
      v.push(CJson::makeNumber(iw->getParam(d) + model.getByPath("likelihood.variance")));
    lik.set("variance", v);
    lik.set("output_index_column", CJson::makeNumber(iw->getActiveDims().at(0)));
  }
  else
  {
    lik.set("type", CJson::makeString("likelihoods.gaussian"));
    lik.set("variance", CJson::makeNumber(model.getByPath("likelihood.variance")));
  }
  doc.set("likelihood", lik);
  CJson data = CJson::makeObject();
  data.set("X", matrixToJson(*model.pX));
  data.set("Y", matrixToJson(*model.py));
  doc.set("data", data);
  double jacobian = 0.0;
  if(outBias != 0.0 || outScale != 1.0)
  {
    CJson nz = CJson::makeObject();
    nz.set("type", CJson::makeString(outScale == 1.0 ? "normalizer.center" : "normalizer.standardize"));
    nz.set("mean", CJson::makeNumber(outBias));
    if(outScale != 1.0)
      nz.set("std", CJson::makeNumber(outScale));
    doc.set("normalizer", nz);
    jacobian = -static_cast<double>(model.getNumData()) * std::log(outScale);
  }

  // Verify: the document must reproduce the source model's log likelihood.
  const double want = -model.computeObjectiveVal() + jacobian;
  const double got = fromJson(doc)->logLikelihood();
  if(!(std::fabs(got - want) <= 1e-8 * (1.0 + std::fabs(want))))
    fail("export", "verification failed: source log likelihood " + to_string(want)
         + " but the exported document gives " + to_string(got)
         + " (a semantic difference the mapping cannot express)");
  return doc;
}
