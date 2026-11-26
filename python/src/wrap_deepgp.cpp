#include "wrap_deepgp.h"

#include <sstream>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "CAdam.h"
#include "COptimisable.h"
#include "matrix_numpy.h"
#include "ndlexceptions.h"
#include "ndlutil.h"

namespace {

void checkXY(const CMatrix& X, const CMatrix& Y)
{
  if(X.getRows() != Y.getRows())
    throw ndlexceptions::Error("X and Y must have the same number of rows");
  if(Y.getCols() < 1)
    throw ndlexceptions::Error("Y must have at least one column");
}

CKern* makeLayerKern(const std::string& name, unsigned int din)
{
  if(name == "rbf" || name == "RBF")
  {
    CRbfKern* rbf = new CRbfKern(din);
    rbf->setInitParam();
    rbf->setInverseWidth(1.0);
    rbf->setVariance(1.0);
    return rbf;
  }
  if(name == "mlp" || name == "MLP")
  {
    CMlpKern* mlp = new CMlpKern(din);
    mlp->setInitParam();
    mlp->setParam(1.0, 0);
    mlp->setParam(1.0, 1);
    mlp->setParam(1.0, 2);
    return mlp;
  }
  if(name == "lin" || name == "linear" || name == "Linear")
  {
    CLinKern* lin = new CLinKern(din);
    lin->setInitParam();
    return lin;
  }
  throw ndlexceptions::Error(
      "DeepGP kernel must be 'rbf', 'mlp', or 'lin' (got '" + name + "').");
}

std::vector<unsigned int> parseNumInducing(py::object num_inducing,
                                           unsigned int nLayers,
                                           unsigned int nData)
{
  std::vector<unsigned int> M;
  if(py::isinstance<py::int_>(num_inducing))
  {
    const unsigned int m = py::cast<unsigned int>(num_inducing);
    M.assign(nLayers, m);
  }
  else if(py::isinstance<py::list>(num_inducing)
          || py::isinstance<py::tuple>(num_inducing))
  {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(num_inducing);
    if(seq.size() != nLayers)
      throw ndlexceptions::Error(
          "num_inducing list length must equal layers");
    for(size_t i = 0; i < seq.size(); ++i)
      M.push_back(py::cast<unsigned int>(seq[i]));
  }
  else
    throw ndlexceptions::Error(
        "num_inducing must be an int or a sequence of length layers");
  for(unsigned int m : M)
    if(m < 1 || m > nData)
      throw ndlexceptions::Error("num_inducing values must be in 1..N");
  return M;
}

OptimiserEvent toOptimiserEvent(const StochasticOptimiserEvent& se)
{
  OptimiserEvent e;
  e.phase = se.terminal ? OptimiserPhase::MaxIters : OptimiserPhase::Iter;
  e.optimiser = -1;
  e.iter = static_cast<int>(se.iteration);
  e.funcEval = -1;
  e.objective = se.objectiveEstimate;
  e.message = "adam";
  return e;
}

} // namespace

PyDeepGP::PyDeepGP()
  : kernelType("rbf"), inducingType("points"), likelihoodType("gaussian")
{
}

PyDeepGP::PyDeepGP(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xin,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
    unsigned int layers,
    int hidden,
    py::object num_inducing,
    py::object kernel,
    const std::string& inducing,
    double chi,
    double beta,
    bool drop_residual,
    const std::string& likelihood,
    unsigned int num_categories,
    int verbosity,
    bool learn_window_width,
    bool learn_window_widths)
  : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin)),
    inducingType(inducing), likelihoodType(likelihood),
    learnWindowWidth(learn_window_width),
    learnWindowWidths(learn_window_widths)
{
  checkXY(X, Y);
  if(learn_window_width && learn_window_widths)
    throw ndlexceptions::Error(
        "Pass only one of learn_window_width=True (tied) or "
        "learn_window_widths=True (untied).");
  if((learn_window_width || learn_window_widths) && inducing == "hermite")
    throw ndlexceptions::Error(
        "Hermite inducing (E3): Hermite windows have no width; learn_window_width(s) "
        "cannot be used with inducing='hermite'.");
  if((learn_window_width || learn_window_widths) && inducing != "weightspace")
    throw ndlexceptions::Error(
        "learn_window_width(s) is only supported with inducing='weightspace'.");
  if(layers < 1)
    throw ndlexceptions::Error("layers must be >= 1");
  if(inducing != "points" && inducing != "weightspace" && inducing != "hermite")
    throw ndlexceptions::Error("inducing must be 'points', 'weightspace' or 'hermite'");
  if(!(chi > 0.0) || !(beta > 0.0))
    throw ndlexceptions::Error("chi and beta must be positive");

  kernelType = py::cast<std::string>(kernel);
  if(inducing == "weightspace"
     && !(kernelType == "mlp" || kernelType == "MLP"))
    throw ndlexceptions::Error(
        "inducing='weightspace' requires kernel='mlp'");

  if(inducing == "hermite"
     && !(kernelType == "mlp" || kernelType == "MLP"))
    throw ndlexceptions::Error(
        "Hermite inducing (E2): inducing='hermite' requires kernel='mlp', got '" + kernelType + "'.");

  const unsigned int Din = X.getCols();
  const unsigned int Dout = Y.getCols();
  const unsigned int hid =
      (hidden < 0) ? Din : static_cast<unsigned int>(hidden);

  std::vector<unsigned int> dims(layers + 1);
  dims[0] = Din;
  dims[layers] = Dout;
  for(unsigned int ell = 1; ell < layers; ++ell)
    dims[ell] = hid;

  const std::vector<unsigned int> M =
      parseNumInducing(num_inducing, layers, X.getRows());

  std::vector<CKern*> kernPtrs;
  ownedKerns.clear();
  for(unsigned int ell = 0; ell < layers; ++ell)
  {
    CKern* k = makeLayerKern(kernelType, dims[ell]);
    ownedKerns.emplace_back(k);
    kernPtrs.push_back(k);
  }
  if(kernelType == "RBF")
    kernelType = "rbf";
  else if(kernelType == "MLP")
    kernelType = "mlp";
  else if(kernelType == "Linear" || kernelType == "linear")
    kernelType = "lin";

  ndlutil::init_genrand(0);
  model = std::make_unique<CDeepGp>(X, Y, kernPtrs, dims, M, beta, chi,
                                    verbosity);
  for(unsigned int ell = 0; ell < layers; ++ell)
  {
    if(inducing != "points")
      model->setLayerInducingType(ell, inducing);
  }
  if(learn_window_width || learn_window_widths)
    for(unsigned int ell = 0; ell < layers; ++ell)
      model->setLayerLearnWindowWidth(ell, true, learn_window_widths);
  if(drop_residual)
    model->setDropResidual(true);

  if(likelihood == "gaussian" || likelihood == "Gaussian")
  {
    likelihoodType = "gaussian";
  }
  else if(likelihood == "probit" || likelihood == "Probit"
          || likelihood == "bernoulli" || likelihood == "Bernoulli")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("probit likelihood needs a single output");
    ownedNoise = std::make_unique<CProbitNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "probit";
  }
  else if(likelihood == "ordered" || likelihood == "Ordered")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("ordered likelihood needs a single output");
    if(num_categories < 2)
      throw ndlexceptions::Error("num_categories must be >= 2");
    ownedNoise = std::make_unique<COrderedNoise>(&Y, num_categories);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "ordered";
  }
  else if(likelihood == "student_t" || likelihood == "StudentT")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("student_t likelihood needs a single output");
    ownedNoise = std::make_unique<CStudentTNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "student_t";
  }
  else if(likelihood == "poisson" || likelihood == "Poisson")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("poisson likelihood needs a single output");
    ownedNoise = std::make_unique<CPoissonNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "poisson";
  }
  else if(likelihood == "binomial" || likelihood == "Binomial")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("binomial likelihood needs a single output");
    ownedNoise = std::make_unique<CBinomialNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "binomial";
  }
  else if(likelihood == "exponential" || likelihood == "Exponential")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("exponential likelihood needs a single output");
    ownedNoise = std::make_unique<CExponentialNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "exponential";
  }
  else if(likelihood == "gamma" || likelihood == "Gamma")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("gamma likelihood needs a single output");
    ownedNoise = std::make_unique<CGammaNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "gamma";
  }
  else if(likelihood == "skew_gaussian" || likelihood == "SkewGaussian")
  {
    if(Dout != 1)
      throw ndlexceptions::Error("skew_gaussian likelihood needs a single output");
    ownedNoise = std::make_unique<CSkewGaussianNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "skew_gaussian";
  }
  else if(likelihood == "skew_exponential" || likelihood == "SkewExponential")
  {
    if(Dout != 1)
      throw ndlexceptions::Error(
          "skew_exponential likelihood needs a single output");
    ownedNoise = std::make_unique<CSkewExponentialNoise>(&Y);
    model->setLikelihood(ownedNoise.get());
    likelihoodType = "skew_exponential";
  }
  else
    throw ndlexceptions::Error(
        "likelihood must be gaussian, probit, ordered, student_t, poisson, "
        "binomial, exponential, gamma, skew_gaussian, or skew_exponential");
}

void PyDeepGP::optimize(unsigned int iters, int batch_size,
                        unsigned int samples, double adam_lr,
                        double natgrad_gamma, uint64_t seed, bool messages,
                        py::object callback)
{
  (void)messages;
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  if(samples < 1)
    throw ndlexceptions::Error("samples must be >= 1");
  if(!(adam_lr > 0.0))
    throw ndlexceptions::Error("adam_lr must be positive");

  unsigned int B = (batch_size < 0) ? model->getNumData()
                                    : static_cast<unsigned int>(batch_size);
  if(B < 1 || B > model->getNumData())
    throw ndlexceptions::Error("batch_size must be in 1..N");

  model->setNumSamples(samples);
  model->setNatGradGamma(natgrad_gamma);
  model->setSeed(seed);
  model->clearProgressObserver();

  const bool haveCallback = !(callback.is_none());
  if(haveCallback)
  {
    py::object cb = callback;
    model->setProgressObserver([cb](const StochasticOptimiserEvent& se) {
      OptimiserEvent e = toOptimiserEvent(se);
      py::gil_scoped_acquire acquire;
      cb(e);
    });
  }

  // CIP-003C: one CAdam per model so moments and step count survive across
  // calls (chunked optimize == one long run for fixed seed schedule).
  if(!adam)
    adam = std::make_unique<CAdam>(adam_lr);
  adam->setLearningRate(adam_lr);
  adam->setBatchSize(B);
  adam->setMaxIters(iters);
  adam->setSeed(seed);
  {
    py::gil_scoped_release release;
    adam->maximise(*model);
  }
  model->clearProgressObserver();
}

double PyDeepGP::elbo() const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  return model->elboEstimate();
}

std::pair<py::array_t<double>, py::array_t<double>> PyDeepGP::predict(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
    unsigned int samples,
    uint64_t seed) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != model->getInputDim())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");

  CMatrix mu, var;
  const bool addNoise = (likelihoodType == "gaussian");
  model->predictMeanVar(Xt, mu, var, samples, seed, addNoise);
  return {matrixToNumpy(mu), matrixToNumpy(var)};
}

std::pair<py::array_t<double>, py::array_t<double>> PyDeepGP::predict_noiseless(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
    unsigned int samples,
    uint64_t seed) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != model->getInputDim())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");
  CMatrix mu, var;
  model->predictMeanVar(Xt, mu, var, samples, seed, false);
  return {matrixToNumpy(mu), matrixToNumpy(var)};
}

void PyDeepGP::reset_optimizer()
{
  if(adam)
    adam->resetState();
}

void PyDeepGP::save(const std::string& path) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  model->toFile(path, "gpc.models.DeepGP");
}

std::unique_ptr<PyDeepGP> PyDeepGP::load(const std::string& path)
{
  auto self = std::unique_ptr<PyDeepGP>(new PyDeepGP());
  CMatrix Xtmp(1, 1), Ytmp(1, 1);
  Xtmp.zeros();
  Ytmp.zeros();
  CRbfKern ktmp(1);
  ktmp.setInitParam();
  std::vector<CKern*> kerns(1, &ktmp);
  self->model = std::make_unique<CDeepGp>(
      Xtmp, Ytmp, kerns, std::vector<unsigned int>{1u, 1u},
      std::vector<unsigned int>{1u}, 1.0, 1.0, 0);
  self->model->fromFile(path);
  self->X.deepCopy(self->model->getX());
  self->Y.deepCopy(self->model->getY());
  self->inducingType = self->model->getLayerInducingType(0);
  self->likelihoodType = "gaussian";
  self->kernelType = "loaded";
  return self;
}

void PyDeepGP::set_drop_residual(bool val)
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  model->setDropResidual(val);
}

bool PyDeepGP::drop_residual() const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  return model->isDropResidual();
}

double PyDeepGP::jacobian_spread(unsigned int n_probe,
                                 uint64_t probe_seed) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  return model->jacobianSpread(n_probe, probe_seed);
}

ModelDescription PyDeepGP::describe() const
{
  ModelDescription d;
  d.title = "Deep GP";
  if(!model)
    return d;
  std::ostringstream h0;
  h0 << "N=" << model->getNumData()
     << " Din=" << model->getInputDim()
     << " Dout=" << model->getOutputDim()
     << " L=" << model->getNumLayers()
     << " chi=" << model->getChi()
     << " inducing=" << inducingType
     << " residual=" << (model->isDropResidual() ? "off" : "on")
     << " likelihood=" << likelihoodType;
  d.headerLines.push_back(h0.str());
  std::ostringstream h1;
  h1 << "dims:";
  for(unsigned int ell = 0; ell <= model->getNumLayers(); ++ell)
    h1 << " " << model->getHiddenDim(ell);
  d.headerLines.push_back(h1.str());
  // CIP-003D: always show the delivered Hermite layout, so a user who never reads
  // warnings can still see what was built.
  for(unsigned int ell = 0; ell < model->getNumLayers(); ++ell)
    if(model->getLayerInducingType(ell) == "hermite")
      d.headerLines.push_back("layer " + std::to_string(ell) + " hermite layout: " +
                              model->layer(ell).hermiteLayoutDescription());
  d.hasLogLikelihood = true;
  d.logLikelihood = model->elboEstimate();
  d.optNumParams = static_cast<int>(model->getStochasticNumParams());
  return d;
}

std::string PyDeepGP::to_string() const
{
  if(!model)
    return "DeepGP(uninitialised)";
  std::ostringstream os;
  model->display(os);
  return os.str();
}

unsigned int PyDeepGP::n() const
{
  return model ? model->getNumData() : 0;
}

unsigned int PyDeepGP::input_dim() const
{
  return model ? model->getInputDim() : 0;
}

unsigned int PyDeepGP::output_dim() const
{
  return model ? model->getOutputDim() : 0;
}

unsigned int PyDeepGP::num_layers() const
{
  return model ? model->getNumLayers() : 0;
}

unsigned int PyDeepGP::hidden_dim() const
{
  if(!model || model->getNumLayers() < 2)
    return 0;
  return model->getHiddenDim(1);
}

double PyDeepGP::chi() const
{
  return model ? model->getChi() : 0.0;
}

void bind_deepgp(py::module_& m)
{
  py::class_<PyDeepGP>(
      m,
      "DeepGP",
      "Deep Gaussian process by doubly stochastic VI (CIP-0035).\n\n"
      "Prefer gpc.models.DeepGP(X, Y, ...) for the public façade.")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    unsigned int,
                    int,
                    py::object,
                    py::object,
                    const std::string&,
                    double,
                    double,
                    bool,
                    const std::string&,
                    unsigned int,
                    int,
                    bool,
                    bool>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("layers") = 2,
           py::arg("hidden") = -1,
           py::arg("num_inducing") = 10,
           py::arg("kernel") = py::str("rbf"),
           py::arg("inducing") = "points",
           py::arg("chi") = 1.0,
           py::arg("beta") = 1.0,
           py::arg("drop_residual") = false,
           py::arg("likelihood") = "gaussian",
           py::arg("num_categories") = 3,
           py::arg("verbosity") = 0,
           py::arg("learn_window_width") = false,
           py::arg("learn_window_widths") = false)
      .def("optimize",
           &PyDeepGP::optimize,
           py::arg("iters") = 100,
           py::arg("batch_size") = -1,
           py::arg("samples") = 1,
           py::arg("adam_lr") = 1e-2,
           py::arg("natgrad_gamma") = 0.0,
           py::arg("seed") = 0,
           py::arg("messages") = false,
           py::arg("callback") = py::none(),
           "Adam on minibatches (maximises ELBO estimate).")
      .def("elbo", &PyDeepGP::elbo, "Unbiased ELBO estimate on the full data.")
      .def("objective",
           &PyDeepGP::objective,
           "Negated ELBO (for display tables that minimise).")
      .def("predict",
           &PyDeepGP::predict,
           py::arg("Xtest"),
           py::arg("samples") = 0,
           py::arg("seed") = 0,
           "Predictive mean/variance at Xtest of a new observation: for a Gaussian "
           "likelihood the variance includes 1/beta; other likelihoods return "
           "the latent variance. samples=0 uses the mean path; samples>=1 mixes "
           "S hidden-layer draws (CIP-0035). See predict_noiseless().")
      .def("predict_noiseless",
           &PyDeepGP::predict_noiseless,
           py::arg("Xtest"),
           py::arg("samples") = 0,
           py::arg("seed") = 0,
           "Latent-function predictive mean/variance (no observation noise); "
           "otherwise as predict().")
      .def("reset_optimizer", &PyDeepGP::reset_optimizer,
           "Discard Adam moments and step count (kept between optimize calls "
           "by default).")
      .def_property_readonly("optimizer_steps", &PyDeepGP::optimizer_steps)
      .def_property_readonly("learn_window_width", &PyDeepGP::learn_window_width)
      .def_property_readonly("learn_window_widths", &PyDeepGP::learn_window_widths)
      .def("save", &PyDeepGP::save, py::arg("path"), "Write deepgp .model stream.")
      .def_static("load", &PyDeepGP::load, py::arg("path"),
                  "Load a deepgp .model stream.")
      .def("set_drop_residual", &PyDeepGP::set_drop_residual, py::arg("value"))
      .def_property_readonly("drop_residual", &PyDeepGP::drop_residual)
      .def("jacobian_spread",
           &PyDeepGP::jacobian_spread,
           py::arg("n_probe") = 32,
           py::arg("probe_seed") = 0,
           "Median prior Jacobian log-singular-value spread (V7).")
      .def("describe", &PyDeepGP::describe)
      .def("hermite_warnings",
           [](const PyDeepGP& self) { return self.hermite_warnings(); },
           "W1-W4 (CIP-003D) as strings for Hermite layers; empty otherwise.")
      .def("__str__", &PyDeepGP::to_string)
      .def("__repr__", &PyDeepGP::to_string)
      .def_property_readonly("n", &PyDeepGP::n)
      .def_property_readonly("input_dim", &PyDeepGP::input_dim)
      .def_property_readonly("output_dim", &PyDeepGP::output_dim)
      .def_property_readonly("num_layers", &PyDeepGP::num_layers)
      .def_property_readonly("hidden_dim", &PyDeepGP::hidden_dim)
      .def_property_readonly("kernel_type", &PyDeepGP::kernel_type)
      .def_property_readonly("inducing_type", &PyDeepGP::inducing_type)
      .def_property_readonly("likelihood_type", &PyDeepGP::likelihood_type)
      .def_property_readonly("chi", &PyDeepGP::chi)
      .def_property_readonly("approx", &PyDeepGP::approx);
}
