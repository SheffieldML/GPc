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

// ARD split (CIP-0016): GPc rbfard has 1/l_j^2 = inverseWidth * inputScale_j. We
// store inverseWidth = min(l)^-2 and inputScale_j = (min l / l_j)^2, so the
// smallest lengthscale carries scale one.
bool buildLeaf(const CJson& node, unsigned int inDim, const string& where,
               unique_ptr<CKern>& out, bool& isWhite)
{
  isWhite = false;
  const string type = need(node, "type", where).asString();
  if(node.has("active_dims"))
    fail(where, "active_dims is not supported by this reader");
  if(type == "kern.rbf" || type == "kern.matern32")
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
    else
      k.reset(new CMatern32Kern(inDim));
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
  if(type == "kern.prod")
    fail(where, "kern.prod is not supported by this reader yet");
  unique_ptr<CKern> leaf;
  bool white = false;
  if(!buildLeaf(node, inDim, where, leaf, white))
    fail(where, "unknown kernel type '" + type + "'");
  hasWhite = hasWhite || white;
  // The compound takes ownership without cloning, so the raw pointer stays valid
  // for read-back in toJson().
  CKern* raw = leaf.get();
  sum.addKern(std::move(leaf));
  leaves.push_back(raw);
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
  if(!sparse && type != "models.gp_regression")
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
  if(need(lik, "type", "likelihood").asString() != "likelihoods.gaussian")
    fail("likelihood.type", "only likelihoods.gaussian is supported");
  const double sigma2 = positiveAt(need(lik, "variance", "likelihood"), "likelihood.variance");

  // Kernel.
  m->kern.reset(new CCmpndKern(inputDim));
  bool hasWhite = false;
  buildKernelTree(need(doc, "kernel", "document"), inputDim, "kernel",
                  *m->kern, m->leaves, hasWhite);

  // Noise: zero mean, so the bias is fixed to zero (GPy semantics).
  m->noise.reset(new CGaussianNoise(&m->Y));
  m->noise->setBiasVal(0.0, 0);

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
  }
  else
  {
    const int approx = approxFrom(need(doc, "approximation", "document").asString(), "approximation");
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
  return m;
}

double CInterchange::logLikelihood() const
{
  return -pgp->computeObjectiveVal();
}

void CInterchange::predict(const CMatrix& Xnew, vector<double>& mean, vector<double>& var) const
{
  if(Xnew.getCols() != X.getCols())
    fail("predict", "Xnew has " + to_string(Xnew.getCols()) + " columns but the model has "
         + to_string(X.getCols()) + " inputs");
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

void CInterchange::writeKernelParams(CJson& node, size_t& leafIndex) const
{
  const string type = node.at("type").asString();
  if(type == "kern.add")
  {
    CJson& parts = node.at("parts");
    for(size_t i = 0; i < parts.size(); ++i)
      writeKernelParams(parts.at(i), leafIndex);
    return;
  }
  const CKern* k = leaves.at(leafIndex++);
  node.set("variance", CJson::makeNumber(k->getByPath("variance")));
  if(type == "kern.rbf" || type == "kern.matern32")
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
  out.at("likelihood").set("variance",
      CJson::makeNumber(pgp->getByPath("likelihood.variance")));
  return out;
}

void CInterchange::writeFile(const string& path) const
{
  toJson().dumpFile(path);
}
