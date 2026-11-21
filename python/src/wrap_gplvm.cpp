#include "wrap_gplvm.h"

#include <sstream>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "COptimisable.h"
#include "matrix_numpy.h"
#include "ndlexceptions.h"
#include "ndlutil.h"

namespace {

void checkY(const CMatrix& Y)
{
  if(Y.getRows() < 2)
    throw ndlexceptions::Error("Y must have at least two rows");
  if(Y.getCols() < 1)
    throw ndlexceptions::Error("Y must have at least one column");
}

bool kernelIsPartsList(const py::object& kernel)
{
  return py::isinstance<py::list>(kernel) || py::isinstance<py::tuple>(kernel);
}

double dictGet(const py::dict& d, const char* key, double fallback)
{
  if(!d.contains(key))
    return fallback;
  return py::cast<double>(d[key]);
}

std::string partType(const py::dict& d)
{
  if(!d.contains("type"))
    throw ndlexceptions::Error("kernel part dict missing 'type'");
  return py::cast<std::string>(d["type"]);
}

bool isRbf(const std::string& t)
{
  return t == "rbf" || t == "RBF";
}
bool isRbfArd(const std::string& t)
{
  return t == "rbfard" || t == "RBFARD" || t == "rbf_ard";
}
bool isEq(const std::string& t)
{
  return t == "eq" || t == "EQ" || t == "ExpQuad" || t == "expquad";
}
bool isWhite(const std::string& t)
{
  return t == "white" || t == "White";
}
bool isBias(const std::string& t)
{
  return t == "bias" || t == "Bias";
}
bool isMlp(const std::string& t)
{
  return t == "mlp" || t == "MLP";
}
bool isLin(const std::string& t)
{
  return t == "lin" || t == "linear" || t == "Linear";
}

// Build compound over a latent-space meta matrix (cols = q).
std::string addPartsLatent(CCmpndKern& kern, const CMatrix& Xmeta, const py::list& parts)
{
  if(parts.size() < 1)
    throw ndlexceptions::Error("kernel parts list must be non-empty");

  std::ostringstream label;
  bool first = true;
  for(py::handle h : parts)
  {
    py::dict d = py::reinterpret_borrow<py::dict>(h);
    const std::string type = partType(d);
    if(!first)
      label << "+";
    first = false;

    if(isRbfArd(type) || (isRbf(type) && d.contains("ARD") && py::cast<bool>(d["ARD"])))
    {
      CRbfardKern rbf(Xmeta);
      kern.addKern(&rbf);
      // GPy path alias: rbfard component exposes as rbf.* (CIP-0014).
      kern.setByPath("rbf.variance", dictGet(d, "variance", 1.0));
      label << "rbfard";
    }
    else if(isRbf(type))
    {
      CRbfKern rbf(Xmeta);
      kern.addKern(&rbf);
      kern.setByPath("rbf.lengthscale", dictGet(d, "lengthscale", 1.0));
      kern.setByPath("rbf.variance", dictGet(d, "variance", 1.0));
      label << "rbf";
    }
    else if(isEq(type))
    {
      CEqKern eq(Xmeta);
      kern.addKern(&eq);
      kern.setByPath("ExpQuad.lengthscale", dictGet(d, "lengthscale", 1.0));
      kern.setByPath("ExpQuad.variance", dictGet(d, "variance", 1.0));
      label << "eq";
    }
    else if(isWhite(type))
    {
      CWhiteKern white(Xmeta);
      kern.addKern(&white);
      kern.setByPath("white.variance", dictGet(d, "variance", 1.0));
      label << "white";
    }
    else if(isBias(type))
    {
      CBiasKern bias(Xmeta);
      kern.addKern(&bias);
      kern.setByPath("bias.variance", dictGet(d, "variance", 1.0));
      label << "bias";
    }
    else if(isMlp(type))
    {
      CMlpKern mlp(Xmeta);
      kern.addKern(&mlp);
      kern.setByPath("mlp.weightVariance", dictGet(d, "weight_variance", 1.0));
      kern.setByPath("mlp.biasVariance", dictGet(d, "bias_variance", 1.0));
      kern.setByPath("mlp.variance", dictGet(d, "variance", 1.0));
      label << "mlp";
    }
    else if(isLin(type))
    {
      CLinKern lin(Xmeta);
      kern.addKern(&lin);
      label << "lin";
    }
    else
    {
      throw ndlexceptions::Error(
          "Unknown GPLVM kernel part '" + type
          + "'. Use rbf, rbfard, eq/ExpQuad, white, bias, mlp, or lin.");
    }
  }
  return label.str();
}

// MAP GPLVM CLI default: RBF + Bias + White over latent space.
std::string addMapGplvmDefault(CCmpndKern& kern, const CMatrix& Xmeta)
{
  CRbfKern rbf(Xmeta);
  kern.addKern(&rbf);
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
  CBiasKern bias(Xmeta);
  kern.addKern(&bias);
  CWhiteKern white(Xmeta);
  kern.addKern(&white);
  return "rbf+bias+white";
}

std::string addBayesDefault(CCmpndKern& kern, const CMatrix& Xmeta, bool ard)
{
  if(ard)
  {
    CRbfardKern rbf(Xmeta);
    kern.addKern(&rbf);
    kern.setByPath("rbf.variance", 1.0);
    return "rbfard";
  }
  CRbfKern rbf(Xmeta);
  kern.addKern(&rbf);
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
  return "rbf";
}

std::string buildLatentKernel(CCmpndKern& kern,
                              const CMatrix& Xmeta,
                              py::object kernel,
                              bool mapStyleDefault,
                              bool ardDefault)
{
  if(kernelIsPartsList(kernel))
  {
    py::list parts = py::reinterpret_borrow<py::list>(kernel);
    return addPartsLatent(kern, Xmeta, parts);
  }

  const std::string name = py::cast<std::string>(kernel);
  if(isRbfArd(name) || (isRbf(name) && ardDefault && !mapStyleDefault))
    return addBayesDefault(kern, Xmeta, true);
  if(isRbf(name))
  {
    if(mapStyleDefault)
      return addMapGplvmDefault(kern, Xmeta);
    return addBayesDefault(kern, Xmeta, false);
  }
  if(isEq(name))
  {
    CEqKern eq(Xmeta);
    kern.addKern(&eq);
    kern.setByPath("ExpQuad.lengthscale", 1.0);
    kern.setByPath("ExpQuad.variance", 1.0);
    if(mapStyleDefault)
    {
      CBiasKern bias(Xmeta);
      kern.addKern(&bias);
      CWhiteKern white(Xmeta);
      kern.addKern(&white);
      return "eq+bias+white";
    }
    return "eq";
  }
  if(isMlp(name))
  {
    CMlpKern mlp(Xmeta);
    kern.addKern(&mlp);
    kern.setByPath("mlp.weightVariance", 1.0);
    kern.setByPath("mlp.biasVariance", 1.0);
    kern.setByPath("mlp.variance", 1.0);
    if(mapStyleDefault)
    {
      CBiasKern bias(Xmeta);
      kern.addKern(&bias);
      CWhiteKern white(Xmeta);
      kern.addKern(&white);
      return "mlp+bias+white";
    }
    return "mlp";
  }
  if(isLin(name))
  {
    CLinKern lin(Xmeta);
    kern.addKern(&lin);
    return "lin";
  }
  throw ndlexceptions::Error(
      "GPLVM kernel must be 'rbf', 'rbfard', 'eq'/'ExpQuad', 'mlp', 'lin', "
      "or a parts list (got '" + name + "').");
}

std::pair<py::array_t<double>, py::array_t<double>> predictGplvm(
    CGplvm& model,
    unsigned int latentDim,
    unsigned int outDim,
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
    bool /*includeObservationNoise*/)
{
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != latentDim)
    throw ndlexceptions::Error("Xtest latent dimension does not match input_dim");
  CMatrix mu(Xt.getRows(), outDim);
  CMatrix varSigma(Xt.getRows(), outDim);
  // MAP GPLVM: latentMeanVar == predictiveMeanVar (CIP-0031).
  model.predictiveMeanVar(mu, varSigma, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varSigma)};
}

std::pair<py::array_t<double>, py::array_t<double>> predictBayes(
    CBayesGplvm& model,
    unsigned int latentDim,
    unsigned int outDim,
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
    bool includeObservationNoise)
{
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != latentDim)
    throw ndlexceptions::Error("Xtest latent dimension does not match input_dim");
  CMatrix mu(Xt.getRows(), outDim);
  CMatrix varSigma(Xt.getRows(), outDim);
  if(includeObservationNoise)
    model.predictiveMeanVar(mu, varSigma, Xt);
  else
    model.latentMeanVar(mu, varSigma, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varSigma)};
}

ModelDescription describeFromKern(const std::string& title,
                                  CKern* pkern,
                                  double logLik,
                                  int optNumParams,
                                  const std::vector<std::string>& headers)
{
  ModelDescription desc;
  desc.title = title;
  desc.headerLines = headers;
  desc.hasLogLikelihood = true;
  desc.logLikelihood = logLik;
  desc.optNumParams = optNumParams;
  if(pkern != 0)
  {
    ModelDescription kdesc = pkern->describe();
    for(size_t i = 0; i < kdesc.params.size(); ++i)
      desc.params.push_back(kdesc.params[i]);
  }
  return desc;
}

} // namespace

PyGPLVM::PyGPLVM(py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
                 unsigned int input_dim,
                 py::object kernel,
                 int verbosity)
  : Y(matrixFromNumpy(Yin)), latentDim(input_dim)
{
  checkY(Y);
  if(latentDim < 1)
    throw ndlexceptions::Error("input_dim (latent dimension) must be >= 1");
  if(static_cast<unsigned int>(Y.getCols()) < latentDim)
    throw ndlexceptions::Error(
        "output dimension D must be >= input_dim q for PCA initialisation");

  CMatrix Xmeta(1, latentDim);
  Xmeta.zeros();
  kern = std::make_unique<CCmpndKern>(Xmeta);
  kernelType = buildLatentKernel(*kern, Xmeta, kernel, /*mapStyleDefault=*/true,
                                 /*ardDefault=*/false);

  noise = std::make_unique<CScaleNoise>(&Y);
  ndlutil::init_genrand(0);
  model = std::make_unique<CGplvm>(kern.get(), noise.get(),
                                   static_cast<int>(latentDim), verbosity);
  model->setLatentRegularised(true);
  model->setDefaultOptimiser(CGplvm::SCG);
}

PyGPLVM::~PyGPLVM()
{
  // CGplvm allocates pX with new and has no destructor cleanup.
  if(model && model->pX != 0)
  {
    delete model->pX;
    model->pX = 0;
  }
}

void PyGPLVM::optimize(unsigned int iters, bool messages, py::object callback)
{
  (void)messages;
  model->clearOptimiserObservers();
  model->setDefaultTextProgress(false);
  model->setVerbosity(0);

  const bool haveCallback = !(callback.is_none());
  if(haveCallback)
  {
    py::object cb = callback;
    model->addOptimiserObserver([cb](const OptimiserEvent& e) {
      py::gil_scoped_acquire acquire;
      cb(e);
    });
  }

  {
    py::gil_scoped_release release;
    model->optimise(static_cast<int>(iters));
  }

  model->clearOptimiserObservers();
  model->setDefaultTextProgress(true);
}

double PyGPLVM::objective() const
{
  return model->computeObjectiveVal();
}

std::pair<py::array_t<double>, py::array_t<double>> PyGPLVM::predict(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictGplvm(*model, latentDim, Y.getCols(), Xtest, true);
}

std::pair<py::array_t<double>, py::array_t<double>> PyGPLVM::predict_noiseless(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictGplvm(*model, latentDim, Y.getCols(), Xtest, false);
}

py::array_t<double> PyGPLVM::latent_mean() const
{
  return matrixToNumpy(*model->pX);
}

ModelDescription PyGPLVM::describe() const
{
  std::vector<std::string> headers;
  {
    std::ostringstream oss;
    oss << "N=" << Y.getRows() << " D=" << Y.getCols() << " q=" << latentDim;
    headers.push_back(oss.str());
  }
  headers.push_back(std::string("kernel=") + kernelType);
  return describeFromKern("GPLVM", kern.get(), model->logLikelihood(),
                          static_cast<int>(model->getOptNumParams()), headers);
}

std::string PyGPLVM::to_string() const
{
  return modelDescriptionToString(describe());
}

PyBayesianGPLVM::PyBayesianGPLVM(
    py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
    unsigned int input_dim,
    unsigned int num_inducing,
    py::object kernel,
    bool ard,
    int verbosity)
  : Y(matrixFromNumpy(Yin)), latentDim(input_dim), numActive(num_inducing)
{
  checkY(Y);
  if(latentDim < 1)
    throw ndlexceptions::Error("input_dim (latent dimension) must be >= 1");
  if(numActive < 1 || numActive > Y.getRows())
    throw ndlexceptions::Error("num_inducing must be in 1..N");
  if(static_cast<unsigned int>(Y.getCols()) < latentDim)
    throw ndlexceptions::Error(
        "output dimension D must be >= input_dim q for PCA initialisation");

  CMatrix Xmeta(1, latentDim);
  Xmeta.zeros();
  kern = std::make_unique<CCmpndKern>(Xmeta);
  kernelType = buildLatentKernel(*kern, Xmeta, kernel, /*mapStyleDefault=*/false,
                                 /*ardDefault=*/ard);

  ndlutil::init_genrand(0);
  model = std::make_unique<CBayesGplvm>(kern.get(), Y, latentDim, numActive,
                                        verbosity);
  model->setOwnKern(false);
  model->setDefaultOptimiser(CBayesGplvm::SCG);
}

void PyBayesianGPLVM::optimize(unsigned int iters, bool messages,
                               py::object callback)
{
  (void)messages;
  model->clearOptimiserObservers();
  model->setDefaultTextProgress(false);
  model->setVerbosity(0);

  const bool haveCallback = !(callback.is_none());
  if(haveCallback)
  {
    py::object cb = callback;
    model->addOptimiserObserver([cb](const OptimiserEvent& e) {
      py::gil_scoped_acquire acquire;
      cb(e);
    });
  }

  {
    py::gil_scoped_release release;
    model->optimise(iters);
  }

  model->clearOptimiserObservers();
  model->setDefaultTextProgress(true);
}

double PyBayesianGPLVM::objective() const
{
  return model->computeObjectiveVal();
}

double PyBayesianGPLVM::elbo() const
{
  return model->logLikelihood();
}

std::pair<py::array_t<double>, py::array_t<double>> PyBayesianGPLVM::predict(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictBayes(*model, latentDim, Y.getCols(), Xtest, true);
}

std::pair<py::array_t<double>, py::array_t<double>> PyBayesianGPLVM::predict_noiseless(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictBayes(*model, latentDim, Y.getCols(), Xtest, false);
}

py::array_t<double> PyBayesianGPLVM::latent_mean() const
{
  return matrixToNumpy(model->getMu());
}

py::array_t<double> PyBayesianGPLVM::latent_variance() const
{
  return matrixToNumpy(model->getS());
}

py::array_t<double> PyBayesianGPLVM::inducing_inputs() const
{
  return matrixToNumpy(model->getZ());
}

ModelDescription PyBayesianGPLVM::describe() const
{
  std::vector<std::string> headers;
  {
    std::ostringstream oss;
    oss << "N=" << Y.getRows() << " D=" << Y.getCols() << " q=" << latentDim
        << " M=" << numActive << " beta=" << model->getBeta();
    headers.push_back(oss.str());
  }
  headers.push_back(std::string("kernel=") + kernelType);
  return describeFromKern("Bayesian GPLVM", kern.get(), model->logLikelihood(),
                          static_cast<int>(model->getOptNumParams()), headers);
}

std::string PyBayesianGPLVM::to_string() const
{
  return modelDescriptionToString(describe());
}

namespace {

// Heap kernel for CMrd ownership. Prefer bare RBF-ARD so factorizeSpace works
// (CMrd::findRbfard does not walk compounds).
CKern* makeMrdViewKern(unsigned int latentDim, py::object kernel, bool ard,
                       std::string& labelOut)
{
  CMatrix Xmeta(1, latentDim);
  Xmeta.zeros();

  if(kernel.is_none())
  {
    if(ard)
    {
      CRbfardKern* rbf = new CRbfardKern(latentDim);
      rbf->setInitParam();
      labelOut = "rbfard";
      return rbf;
    }
    CRbfKern* rbf = new CRbfKern(latentDim);
    rbf->setInitParam();
    labelOut = "rbf";
    return rbf;
  }

  if(kernelIsPartsList(kernel))
  {
    CCmpndKern* cmpnd = new CCmpndKern(Xmeta);
    py::list parts = py::reinterpret_borrow<py::list>(kernel);
    labelOut = addPartsLatent(*cmpnd, Xmeta, parts);
    return cmpnd;
  }

  const std::string name = py::cast<std::string>(kernel);
  if(isRbfArd(name) || (isRbf(name) && ard))
  {
    CRbfardKern* rbf = new CRbfardKern(latentDim);
    rbf->setInitParam();
    labelOut = "rbfard";
    return rbf;
  }
  if(isRbf(name))
  {
    CRbfKern* rbf = new CRbfKern(latentDim);
    rbf->setInitParam();
    labelOut = "rbf";
    return rbf;
  }

  // Fall back to compound builder for eq/mlp/lin.
  CCmpndKern* cmpnd = new CCmpndKern(Xmeta);
  labelOut = buildLatentKernel(*cmpnd, Xmeta, kernel, /*mapStyleDefault=*/false,
                               /*ardDefault=*/ard);
  return cmpnd;
}

} // namespace

PyMRD::PyMRD(py::sequence Ylist,
             unsigned int input_dim,
             unsigned int num_inducing,
             py::object kernel,
             bool ard,
             int verbosity)
  : latentDim(input_dim), numActive(num_inducing)
{
  if(latentDim < 1)
    throw ndlexceptions::Error("input_dim (latent dimension) must be >= 1");

  const py::ssize_t V = py::len(Ylist);
  if(V < 2)
    throw ndlexceptions::Error("MRD requires at least two views in Ylist");

  std::vector<CMatrix> Ys;
  Ys.reserve(static_cast<size_t>(V));
  for(py::ssize_t v = 0; v < V; ++v)
  {
    py::array_t<double, py::array::c_style | py::array::forcecast> arr =
        py::reinterpret_borrow<
            py::array_t<double, py::array::c_style | py::array::forcecast>>(
            Ylist[v]);
    CMatrix Yv = matrixFromNumpy(arr);
    checkY(Yv);
    Ys.push_back(Yv);
  }
  for(size_t v = 1; v < Ys.size(); ++v)
    if(Ys[v].getRows() != Ys[0].getRows())
      throw ndlexceptions::Error("All MRD views must share the same N");

  if(numActive < 1 || numActive > Ys[0].getRows())
    throw ndlexceptions::Error("num_inducing must be in 1..N");

  // Distinguishing parts-list vs per-view list: parts-list items are dicts;
  // per-view list items are strings or nested parts lists.
  bool treatAsPerView = false;
  if(!kernel.is_none()
     && (py::isinstance<py::list>(kernel) || py::isinstance<py::tuple>(kernel)))
  {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(kernel);
    if(py::len(seq) == V && V >= 2)
    {
      py::handle first = seq[0];
      if(py::isinstance<py::str>(first)
         || py::isinstance<py::list>(first)
         || py::isinstance<py::tuple>(first))
        treatAsPerView = true;
      // dict → parts list shared across views
    }
  }

  std::vector<CKern*> kerns;
  kerns.reserve(static_cast<size_t>(V));
  std::ostringstream label;
  for(py::ssize_t v = 0; v < V; ++v)
  {
    py::object kv = kernel;
    if(treatAsPerView)
    {
      py::sequence seq = py::reinterpret_borrow<py::sequence>(kernel);
      kv = seq[v];
    }
    std::string lab;
    CKern* k = makeMrdViewKern(latentDim, kv, ard, lab);
    kerns.push_back(k);
    if(v > 0)
      label << ",";
    label << lab;
  }
  kernelType = label.str();

  ndlutil::init_genrand(0);
  model = std::make_unique<CMrd>(Ys, kerns, latentDim, numActive, verbosity);
  // CMrd took ownership of kerns.
  model->setDefaultOptimiser(CMrd::SCG);
}

void PyMRD::optimize(unsigned int iters, bool messages, py::object callback)
{
  (void)messages;
  model->clearOptimiserObservers();
  model->setDefaultTextProgress(false);
  model->setVerbosity(0);

  const bool haveCallback = !(callback.is_none());
  if(haveCallback)
  {
    py::object cb = callback;
    model->addOptimiserObserver([cb](const OptimiserEvent& e) {
      py::gil_scoped_acquire acquire;
      cb(e);
    });
  }

  {
    py::gil_scoped_release release;
    model->optimise(iters);
  }

  model->clearOptimiserObservers();
  model->setDefaultTextProgress(true);
}

double PyMRD::objective() const
{
  return model->computeObjectiveVal();
}

double PyMRD::elbo() const
{
  return model->logLikelihood();
}

std::pair<py::array_t<double>, py::array_t<double>> PyMRD::predict(
    unsigned int view,
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  if(view >= model->getNumViews())
    throw ndlexceptions::Error("view index out of range");
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != latentDim)
    throw ndlexceptions::Error("Xtest latent dimension does not match input_dim");
  const unsigned int Dout = model->getView(view)->getOutputDim();
  CMatrix mu(Xt.getRows(), Dout);
  CMatrix varSigma(Xt.getRows(), Dout);
  model->predictiveMeanVar(view, mu, varSigma, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varSigma)};
}

std::pair<py::array_t<double>, py::array_t<double>> PyMRD::predict_noiseless(
    unsigned int view,
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  if(view >= model->getNumViews())
    throw ndlexceptions::Error("view index out of range");
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != latentDim)
    throw ndlexceptions::Error("Xtest latent dimension does not match input_dim");
  const unsigned int Dout = model->getView(view)->getOutputDim();
  CMatrix mu(Xt.getRows(), Dout);
  CMatrix varSigma(Xt.getRows(), Dout);
  model->latentMeanVar(view, mu, varSigma, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varSigma)};
}

py::array_t<double> PyMRD::latent_mean() const
{
  return matrixToNumpy(model->getMu());
}

py::array_t<double> PyMRD::latent_variance() const
{
  return matrixToNumpy(model->getS());
}

py::array_t<double> PyMRD::inducing_inputs() const
{
  return matrixToNumpy(model->getZ());
}

py::array_t<double> PyMRD::input_scales(unsigned int view) const
{
  std::vector<double> scales;
  model->getInputScales(view, scales);
  py::array_t<double> out(static_cast<py::ssize_t>(scales.size()));
  auto r = out.mutable_unchecked<1>();
  for(size_t i = 0; i < scales.size(); ++i)
    r(static_cast<py::ssize_t>(i)) = scales[i];
  return out;
}

py::dict PyMRD::factorize_space(double threshold) const
{
  std::vector<unsigned int> shared, p0, p1;
  model->factorizeSpace(shared, p0, p1, threshold);
  py::dict out;
  out["shared"] = shared;
  out["private0"] = p0;
  out["private1"] = p1;
  return out;
}

ModelDescription PyMRD::describe() const
{
  ModelDescription desc;
  desc.title = "MRD";
  {
    std::ostringstream oss;
    oss << "N=" << model->getNumData() << " V=" << model->getNumViews()
        << " q=" << latentDim << " M=" << numActive;
    desc.headerLines.push_back(oss.str());
  }
  desc.headerLines.push_back(std::string("kernel=") + kernelType);
  desc.hasLogLikelihood = true;
  desc.logLikelihood = model->logLikelihood();
  desc.optNumParams = static_cast<int>(model->getOptNumParams());
  for(unsigned int v = 0; v < model->getNumViews(); ++v)
  {
    CKern* pk = model->getView(v)->getKern();
    if(pk == 0)
      continue;
    ModelDescription kdesc = pk->describe();
    for(size_t i = 0; i < kdesc.params.size(); ++i)
    {
      ParamRow row = kdesc.params[i];
      row.path = "view" + std::to_string(v) + "." + row.path;
      desc.params.push_back(row);
    }
  }
  return desc;
}

std::string PyMRD::to_string() const
{
  return modelDescriptionToString(describe());
}

unsigned int PyMRD::n() const
{
  return model->getNumData();
}

unsigned int PyMRD::num_views() const
{
  return model->getNumViews();
}

unsigned int PyMRD::output_dim(unsigned int view) const
{
  if(view >= model->getNumViews())
    throw ndlexceptions::Error("view index out of range");
  return model->getView(view)->getOutputDim();
}

void bind_gplvm(py::module_& m)
{
  py::class_<PyGPLVM>(
      m,
      "GPLVM",
      "MAP Gaussian process latent variable model (CGplvm).\n"
      "Prefer gpc.models.GPLVM(Y, input_dim, kernel=...) for GPy-shaped construction.")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    unsigned int, py::object, int>(),
           py::arg("Y"),
           py::arg("input_dim") = 2,
           py::arg("kernel") = py::str("rbf"),
           py::arg("verbosity") = 0,
           "Construct MAP GPLVM from observations Y (N×D).\n"
           "Default kernel is RBF+Bias+White over the latent space (CLI parity).")
      .def("optimize", &PyGPLVM::optimize,
           py::arg("iters") = 100,
           py::arg("messages") = false,
           py::arg("callback") = py::none(),
           "Optimise latent positions and kernel hyperparameters (SCG).")
      .def("objective", &PyGPLVM::objective, "Negative log-likelihood.")
      .def("predict", &PyGPLVM::predict, py::arg("Xtest"),
           "Predictive mean/variance at latent Xtest (CIP-0031).")
      .def("predict_noiseless", &PyGPLVM::predict_noiseless, py::arg("Xtest"),
           "Same as predict for MAP GPLVM (latent == predictive).")
      .def("latent_mean", &PyGPLVM::latent_mean, "Current latent positions X (N×q).")
      .def("describe", &PyGPLVM::describe, "Structured ModelDescription (CIP-001E).")
      .def("__str__", &PyGPLVM::to_string)
      .def("__repr__", &PyGPLVM::to_string)
      .def_property_readonly("n", &PyGPLVM::n)
      .def_property_readonly("input_dim", &PyGPLVM::input_dim)
      .def_property_readonly("output_dim", &PyGPLVM::output_dim)
      .def_property_readonly("kernel_type", &PyGPLVM::kernel_type)
      .def_property_readonly("approx", &PyGPLVM::approx);

  py::class_<PyBayesianGPLVM>(
      m,
      "BayesianGPLVM",
      "Collapsed Bayesian GPLVM with psi statistics (CBayesGplvm; CIP-0017).\n"
      "Prefer gpc.models.BayesianGPLVM(Y, input_dim, ...) for GPy-shaped construction.")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    unsigned int, unsigned int, py::object, bool, int>(),
           py::arg("Y"),
           py::arg("input_dim") = 2,
           py::arg("num_inducing") = 10,
           py::arg("kernel") = py::str("rbf"),
           py::arg("ard") = true,
           py::arg("verbosity") = 0,
           "Construct collapsed Bayesian GPLVM from Y (N×D).\n"
           "Default kernel is RBF-ARD (GPy BayesianGPLVM parity) when ard=True.")
      .def("optimize", &PyBayesianGPLVM::optimize,
           py::arg("iters") = 100,
           py::arg("messages") = false,
           py::arg("callback") = py::none(),
           "Optimise variational q(X), Z, kernel, and β (SCG).")
      .def("objective", &PyBayesianGPLVM::objective, "Negative ELBO.")
      .def("elbo", &PyBayesianGPLVM::elbo, "Evidence lower bound (logLikelihood).")
      .def("predict", &PyBayesianGPLVM::predict, py::arg("Xtest"),
           "Predictive y_* mean/variance at latent Xtest (includes 1/β).")
      .def("predict_noiseless", &PyBayesianGPLVM::predict_noiseless, py::arg("Xtest"),
           "Latent f_* mean/variance at Xtest.")
      .def("latent_mean", &PyBayesianGPLVM::latent_mean, "Variational mean μ (N×q).")
      .def("latent_variance", &PyBayesianGPLVM::latent_variance,
           "Variational diagonal variances S (N×q).")
      .def("inducing_inputs", &PyBayesianGPLVM::inducing_inputs, "Inducing inputs Z (M×q).")
      .def("describe", &PyBayesianGPLVM::describe, "Structured ModelDescription (CIP-001E).")
      .def("__str__", &PyBayesianGPLVM::to_string)
      .def("__repr__", &PyBayesianGPLVM::to_string)
      .def_property_readonly("n", &PyBayesianGPLVM::n)
      .def_property_readonly("input_dim", &PyBayesianGPLVM::input_dim)
      .def_property_readonly("output_dim", &PyBayesianGPLVM::output_dim)
      .def_property_readonly("num_inducing", &PyBayesianGPLVM::num_inducing)
      .def_property_readonly("kernel_type", &PyBayesianGPLVM::kernel_type)
      .def_property_readonly("approx", &PyBayesianGPLVM::approx);

  py::class_<PyMRD>(
      m,
      "MRD",
      "Manifold Relevance Determination (CMrd; CIP-0018).\n"
      "Prefer gpc.models.MRD(Ylist, input_dim, ...) for GPy-shaped construction.")
      .def(py::init<py::sequence, unsigned int, unsigned int, py::object, bool, int>(),
           py::arg("Ylist"),
           py::arg("input_dim") = 2,
           py::arg("num_inducing") = 10,
           py::arg("kernel") = py::none(),
           py::arg("ard") = true,
           py::arg("verbosity") = 0,
           "Construct MRD from a sequence of views Y_v (each N×D_v, V≥2).\n"
           "Default per-view kernel is bare RBF-ARD (GPy MRD / factorize_space).")
      .def("optimize", &PyMRD::optimize,
           py::arg("iters") = 100,
           py::arg("messages") = false,
           py::arg("callback") = py::none(),
           "Optimise shared q(X), Z, and per-view kernels / β (SCG).")
      .def("objective", &PyMRD::objective, "Negative ELBO.")
      .def("elbo", &PyMRD::elbo, "Evidence lower bound (single KL).")
      .def("predict", &PyMRD::predict, py::arg("view"), py::arg("Xtest"),
           "Predictive y_* for view at latent Xtest.")
      .def("predict_noiseless", &PyMRD::predict_noiseless, py::arg("view"),
           py::arg("Xtest"), "Latent f_* for view at Xtest.")
      .def("latent_mean", &PyMRD::latent_mean, "Shared variational mean μ (N×q).")
      .def("latent_variance", &PyMRD::latent_variance,
           "Shared variational diagonal variances S (N×q).")
      .def("inducing_inputs", &PyMRD::inducing_inputs, "Shared inducing inputs Z (M×q).")
      .def("input_scales", &PyMRD::input_scales, py::arg("view"),
           "Per-view RBF-ARD input scales (length q).")
      .def("factorize_space", &PyMRD::factorize_space, py::arg("threshold") = 0.005,
           "Soft shared/private latent dims (two views, RBF-ARD).")
      .def("output_dim", &PyMRD::output_dim, py::arg("view"),
           "Output dimension D_v for a view.")
      .def("describe", &PyMRD::describe, "Structured ModelDescription (CIP-001E).")
      .def("__str__", &PyMRD::to_string)
      .def("__repr__", &PyMRD::to_string)
      .def_property_readonly("n", &PyMRD::n)
      .def_property_readonly("input_dim", &PyMRD::input_dim)
      .def_property_readonly("num_views", &PyMRD::num_views)
      .def_property_readonly("num_inducing", &PyMRD::num_inducing)
      .def_property_readonly("kernel_type", &PyMRD::kernel_type)
      .def_property_readonly("approx", &PyMRD::approx);
}
