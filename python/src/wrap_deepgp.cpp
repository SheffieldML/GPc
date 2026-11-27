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
  // CIP-0041 B3: ARD kernels, one input scale per layer-input dimension.
  if(name == "rbfard")
  {
    CRbfardKern* k = new CRbfardKern(din);
    k->setInitParam();
    return k;
  }
  if(name == "mlpard")
  {
    CMlpardKern* k = new CMlpardKern(din);
    k->setInitParam();
    return k;
  }
  throw ndlexceptions::Error(
      "DeepGP kernel must be 'rbf', 'mlp', 'lin', 'rbfard' or 'mlpard' (got '" + name + "').");
}


// ---- CIP-0040 Phase 1d: convolutional kernel descriptors ------------------------------------

double dictNum(const py::dict& d, const char* key, double dflt)
{
  return d.contains(key) ? py::cast<double>(d[key]) : dflt;
}

// Patch kernel from the descriptor's `patch` part (rbf / rbfard / mlp / mlpard / linear).
std::unique_ptr<CKern> makePatchKern(const py::dict& part, unsigned int q)
{
  const std::string type = py::cast<std::string>(part["type"]);
  const bool ard = part.contains("ARD") && py::cast<bool>(part["ARD"]);
  if(type == "rbf" || type == "rbfard")
  {
    if(ard || type == "rbfard")
    {
      std::unique_ptr<CRbfardKern> k(new CRbfardKern(q));
      k->setInitParam();
      return std::unique_ptr<CKern>(k.release());
    }
    std::unique_ptr<CRbfKern> k(new CRbfKern(q));
    k->setInitParam();
    const double ell = dictNum(part, "lengthscale", 1.0);
    if(!(ell > 0.0))
      throw ndlexceptions::Error("Conv: patch kernel lengthscale must be positive");
    k->setInverseWidth(1.0 / (ell * ell));
    k->setVariance(dictNum(part, "variance", 1.0));
    return std::unique_ptr<CKern>(k.release());
  }
  if(type == "mlp" || type == "mlpard")
  {
    if(ard || type == "mlpard")
    {
      std::unique_ptr<CMlpardKern> k(new CMlpardKern(q));
      k->setInitParam();
      return std::unique_ptr<CKern>(k.release());
    }
    std::unique_ptr<CMlpKern> k(new CMlpKern(q));
    k->setInitParam();
    k->setParam(dictNum(part, "weight_variance", 1.0), 0);
    k->setParam(dictNum(part, "bias_variance", 1.0), 1);
    k->setParam(dictNum(part, "variance", 1.0), 2);
    return std::unique_ptr<CKern>(k.release());
  }
  if(type == "lin" || type == "linear" || type == "Linear")
  {
    std::unique_ptr<CLinKern> k(new CLinKern(q));
    k->setInitParam();
    k->setVariance(dictNum(part, "variance", 1.0));
    return std::unique_ptr<CKern>(k.release());
  }
  throw ndlexceptions::Error("Conv: patch kernel must be rbf, mlp or linear (got '" + type + "')");
}

std::unique_ptr<CConvKern> makeConvFromPart(const py::dict& part)
{
  std::vector<unsigned int> shape = py::cast<std::vector<unsigned int> >(part["image_shape"]);
  std::vector<unsigned int> pshape = py::cast<std::vector<unsigned int> >(part["patch_shape"]);
  if(shape.size() != 3 || pshape.size() != 2)
    throw ndlexceptions::Error("Conv: image_shape must be (H, W, C) and patch_shape (h, w)");
  const unsigned int stride = part.contains("stride") ? py::cast<unsigned int>(part["stride"]) : 1u;
  const std::string weights = part.contains("weights") ? py::cast<std::string>(part["weights"])
                                                       : std::string("learned");
  CConvKern::WeightMode mode;
  if(weights == "learned")
    mode = CConvKern::WEIGHTS_LEARNED;
  else if(weights == "tied")
    mode = CConvKern::WEIGHTS_TIED;
  else if(weights == "fixed")
    mode = CConvKern::WEIGHTS_FIXED;
  else
    throw ndlexceptions::Error("Conv: weights must be 'learned', 'tied' or 'fixed'");
  const unsigned int q = pshape[0] * pshape[1] * shape[2];
  std::unique_ptr<CKern> patch = makePatchKern(py::cast<py::dict>(part["patch"]), q);
  return std::unique_ptr<CConvKern>(
      new CConvKern(*patch, shape[0], shape[1], shape[2], pshape[0], pshape[1], stride, mode));
}

bool isConvPart(py::handle h)
{
  return py::isinstance<py::dict>(h);
}

// CIP-0041 B1: a string (applied to every layer) or a length-L sequence of strings.
std::vector<std::string> parseLayerStrings(py::object obj, unsigned int nLayers,
                                           const char* what)
{
  std::vector<std::string> out;
  if(py::isinstance<py::str>(obj))
  {
    out.assign(nLayers, py::cast<std::string>(obj));
  }
  else if(py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj))
  {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    if(py::len(seq) != nLayers)
      throw ndlexceptions::Error(std::string(what) + " sequence length ("
                                 + std::to_string(py::len(seq))
                                 + ") must equal layers (" + std::to_string(nLayers) + ")");
    for(py::handle h : seq)
    {
      if(isConvPart(h) && std::string(what) == "kernel")
        out.push_back("conv");
      else if(!py::isinstance<py::str>(h))
        throw ndlexceptions::Error(std::string(what) + " entries must be strings");
      else
        out.push_back(py::cast<std::string>(h));
    }
  }
  else
    throw ndlexceptions::Error(std::string(what)
                               + " must be a string or a sequence of strings of length layers");
  return out;
}

std::string canonicalKernelName(const std::string& n)
{
  if(n == "RBF") return "rbf";
  if(n == "MLP") return "mlp";
  if(n == "RBFARD" || n == "rbf_ard") return "rbfard";
  if(n == "MLPARD" || n == "mlp_ard") return "mlpard";
  if(n == "Linear" || n == "linear") return "lin";
  return n;
}

// "a" when every layer agrees, else "a,b,c".
std::string joinLayerNames(const std::vector<std::string>& v)
{
  bool uniform = true;
  for(const std::string& e : v)
    uniform = uniform && (e == v[0]);
  if(uniform)
    return v[0];
  std::string out;
  for(size_t i = 0; i < v.size(); ++i)
    out += (i ? "," : "") + v[i];
  return out;
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
    py::object inducing,
    double chi,
    double beta,
    bool drop_residual,
    const std::string& likelihood,
    unsigned int num_categories,
    int verbosity,
    bool learn_window_width,
    bool learn_window_widths,
    unsigned int num_classes,
    double likelihood_eps,
    unsigned int likelihood_samples,
    unsigned int likelihood_seed,
    const std::string& diag,
    unsigned int diag_samples,
    bool diag_force,
    py::object mlp_init)
  : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin)),
    likelihoodType(likelihood),
    learnWindowWidth(learn_window_width),
    learnWindowWidths(learn_window_widths)
{
  checkXY(X, Y);
  const bool multiclass = (likelihood == "robustmax" || likelihood == "RobustMax"
                           || likelihood == "softmax" || likelihood == "Softmax");
  if(multiclass)
  {
    // Integer labels (N or N x 1) or one-hot (N x K) -> N x K one-hot, here and only here.
    Y = ndlnoise::oneHotLabels(Y, num_classes);
  }
  if(learn_window_width && learn_window_widths)
    throw ndlexceptions::Error(
        "Pass only one of learn_window_width=True (tied) or "
        "learn_window_widths=True (untied).");
  if(layers < 1)
    throw ndlexceptions::Error("layers must be >= 1");
  if(!(chi > 0.0) || !(beta > 0.0))
    throw ndlexceptions::Error("chi and beta must be positive");

  // CIP-0041 B1: per-layer kernel and inducing family, validated per layer.
  const bool perLayer = !(py::isinstance<py::str>(kernel) && py::isinstance<py::str>(inducing));
  std::vector<std::string> kernNames = parseLayerStrings(kernel, layers, "kernel");
  std::vector<std::string> indNames = parseLayerStrings(inducing, layers, "inducing");
  for(std::string& n : kernNames)
    n = canonicalKernelName(n);
  // CIP-0040: a convolutional kernel (a Conv descriptor dict) is first-layer only and
  // takes inducing patches ('points' is promoted since it cannot apply).
  std::vector<py::dict> convParts(layers);
  if(py::isinstance<py::list>(kernel) || py::isinstance<py::tuple>(kernel))
  {
    unsigned int ell = 0;
    for(py::handle h : py::reinterpret_borrow<py::sequence>(kernel))
    {
      if(isConvPart(h))
        convParts[ell] = py::reinterpret_borrow<py::dict>(h);
      ++ell;
    }
  }
  unsigned int nConv = 0;
  for(unsigned int ell = 0; ell < layers; ++ell)
  {
    if(kernNames[ell] != "conv")
      continue;
    ++nConv;
    if(ell != 0)
      throw ndlexceptions::Error("layer " + std::to_string(ell)
                                 + ": a convolutional kernel is supported on the first layer only "
                                   "(deep convolutional GPs are CIP-0040 Phase 4)");
    if(indNames[ell] == "points")
      indNames[ell] = "patches";
  }
  unsigned int nWeightspace = 0, nHermite = 0;
  for(unsigned int ell = 0; ell < layers; ++ell)
  {
    const std::string pre = perLayer ? "layer " + std::to_string(ell) + ": " : "";
    const std::string& ind = indNames[ell];
    if(ind != "points" && ind != "weightspace" && ind != "hermite" && ind != "patches")
      throw ndlexceptions::Error(pre + "inducing must be 'points', 'weightspace', 'hermite' or 'patches'");
    if(ind == "patches" && kernNames[ell] != "conv")
      throw ndlexceptions::Error(pre + "inducing='patches' requires a convolutional kernel (gpc.kern.Conv)");
    const bool isConv = kernNames[ell] == "conv";
    if(isConv && ind == "points")
      throw ndlexceptions::Error(pre + "a convolutional kernel takes inducing='patches', 'weightspace' or 'hermite'");
    if(isConv && (ind == "weightspace" || ind == "hermite"))
    {
      // CIP-0040 Phase 2: windows over the filters need the MLP patch kernel.
      py::dict pk = py::cast<py::dict>(convParts[ell]["patch"]);
      const std::string ptype = py::cast<std::string>(pk["type"]);
      const bool pard = pk.contains("ARD") && py::cast<bool>(pk["ARD"]);
      if(ptype != "mlp" || pard)
        throw ndlexceptions::Error(pre + "inducing='" + ind + "' on a convolutional kernel requires "
                                   "a plain MLP patch kernel (got '" + ptype + "')");
    }
    else if(ind == "weightspace" && kernNames[ell] != "mlp")
      throw ndlexceptions::Error(pre + "inducing='weightspace' requires kernel='mlp' (got '" + kernNames[ell] + "')");
    if(ind == "hermite" && !isConv && kernNames[ell] != "mlp")
      throw ndlexceptions::Error(pre + "Hermite inducing (E2): inducing='hermite' requires kernel='mlp', got '"
                                 + kernNames[ell] + "'.");
    nWeightspace += (ind == "weightspace");
    nHermite += (ind == "hermite");
  }
  if((learn_window_width || learn_window_widths) && nWeightspace == 0)
  {
    if(nHermite > 0)
      throw ndlexceptions::Error(
          "Hermite inducing (E3): Hermite windows have no width; learn_window_width(s) "
          "cannot be used with inducing='hermite'.");
    throw ndlexceptions::Error(
        "learn_window_width(s) is only supported with inducing='weightspace'.");
  }
  inducingType = joinLayerNames(indNames);
  layerInducingTypes = indNames;

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
    CKern* k;
    if(kernNames[ell] == "conv")
    {
      std::unique_ptr<CConvKern> ck = makeConvFromPart(convParts[ell]);
      if(ck->getInputDim() != dims[ell])
        throw ndlexceptions::Error("Conv: image_shape implies " + std::to_string(ck->getInputDim())
                                   + " input columns but X has " + std::to_string(dims[ell]));
      k = ck.release();
    }
    else
      k = makeLayerKern(kernNames[ell], dims[ell]);
    ownedKerns.emplace_back(k);
    kernPtrs.push_back(k);
  }
  // CIP-0040 Phase 1c: diagonal estimator.  Unbiased only for the bound of a Gaussian
  // single-layer model; hidden layers take sqrt of a noisy variance, and non-Gaussian
  // likelihoods are not linear in the diagonal.
  if(diag != "exact" && diag != "estimate")
    throw ndlexceptions::Error("diag must be 'exact' or 'estimate'");
  if(diag == "estimate")
  {
    if(nConv == 0)
      throw ndlexceptions::Error("diag='estimate' requires a convolutional kernel");
    const bool gaussian = (likelihood == "gaussian" || likelihood == "Gaussian");
    if(!diag_force && !(layers == 1 && gaussian))
      throw ndlexceptions::Error(
          "diag='estimate' is unbiased only for a single-layer model with a Gaussian likelihood "
          "(the variance enters the bound linearly there); pass diag_force=True to use it anyway "
          "(biased), or use a larger stride to cut the cost exactly.");
    if(diag_samples < 2)
      throw ndlexceptions::Error("diag_samples must be at least 2");
    static_cast<CConvKern*>(ownedKerns[0].get())->setDiagEstimator(diag_samples, 0);
  }
  kernelType = joinLayerNames(kernNames);
  layerKernelTypes = kernNames;

  ndlutil::init_genrand(0);
  model = std::make_unique<CDeepGp>(X, Y, kernPtrs, dims, M, beta, chi,
                                    verbosity);
  for(unsigned int ell = 0; ell < layers; ++ell)
  {
    if(indNames[ell] != "points")
      model->setLayerInducingType(ell, indNames[ell]);
  }
  if(learn_window_width || learn_window_widths)
    for(unsigned int ell = 0; ell < layers; ++ell)
      if(indNames[ell] == "weightspace")
        model->setLayerLearnWindowWidth(ell, true, learn_window_widths);
  if(drop_residual)
    model->setDropResidual(true);

  // CIP-0041 B2: 'scaled' sets the MLP weight variance to 1 / mean ||h||^2 and the bias
  // variance to 0.1, so that ||xhat||^2 = w ||h||^2 + b starts near 1.1.  'unit' (the
  // default) leaves the kernel as built.
  {
    const std::vector<std::string> inits = parseLayerStrings(mlp_init, layers, "mlp_init");
    for(unsigned int ell = 0; ell < layers; ++ell)
    {
      const std::string pre = perLayer ? "layer " + std::to_string(ell) + ": " : "";
      if(inits[ell] != "unit" && inits[ell] != "scaled")
        throw ndlexceptions::Error(pre + "mlp_init must be 'unit' or 'scaled'");
      if(inits[ell] == "unit")
        continue;
      if(kernNames[ell] != "mlp")
        throw ndlexceptions::Error(pre + "mlp_init='scaled' requires kernel='mlp' (got '"
                                   + kernNames[ell] + "')");
      const double msn = model->layerInputMeanSqNorm(ell);
      if(!(msn > 0.0))
        throw ndlexceptions::Error(pre + "mlp_init='scaled': layer input has zero norm");
      CKern* k = model->layer(ell).getKern();
      k->setParam(1.0 / msn, 0);
      k->setParam(0.1, 1);
      model->layer(ell).updateKernelMatrices();
    }
  }

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
  else if(likelihood == "robustmax" || likelihood == "RobustMax")
  {
    if(!(likelihood_eps > 0.0 && likelihood_eps < 1.0))
      throw ndlexceptions::Error("robustmax eps must be in (0, 1)");
    model->adoptLikelihood(std::unique_ptr<CNoise>(
        new CRobustMaxNoise(&Y, Dout, likelihood_eps)));
    likelihoodType = "robustmax";
  }
  else if(likelihood == "softmax" || likelihood == "Softmax")
  {
    if(likelihood_samples < 2)
      throw ndlexceptions::Error("softmax samples must be at least 2");
    model->adoptLikelihood(std::unique_ptr<CNoise>(
        new CSoftmaxNoise(&Y, Dout, likelihood_samples, likelihood_seed)));
    likelihoodType = "softmax";
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
        "likelihood must be gaussian, probit, robustmax, softmax, ordered, student_t, poisson, "
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

py::array_t<double> PyDeepGP::predict_proba(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
    unsigned int samples,
    uint64_t seed) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  CNoise* lik = model->getLikelihood();
  if(lik == 0 || lik->blockSize() < 2)
    throw ndlexceptions::Error(
        "predict_proba needs a multi-class likelihood (robustmax or softmax)");
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != model->getInputDim())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");
  CMatrix mu, var;
  model->predictMeanVar(Xt, mu, var, samples, seed, false);
  const unsigned int K = lik->blockSize();
  CMatrix P(mu.getRows(), mu.getCols());
  std::vector<double> m(K), v(K), p(K);
  for(unsigned int i = 0; i < mu.getRows(); i++)
    for(unsigned int j0 = 0; j0 + K <= mu.getCols(); j0 += K)
    {
      for(unsigned int k = 0; k < K; k++)
      {
        m[k] = mu.getVal(i, j0 + k);
        v[k] = var.getVal(i, j0 + k);
      }
      lik->predictBlock(&m[0], &v[0], j0, &p[0]);
      for(unsigned int k = 0; k < K; k++)
        P.setVal(p[k], i, j0 + k);
    }
  return matrixToNumpy(P);
}

void PyDeepGP::reset_optimizer()
{
  if(adam)
    adam->resetState();
}

py::array_t<double> conv_kern_K(
    py::dict part,
    py::array_t<double, py::array::c_style | py::array::forcecast> Xin,
    py::object X2in)
{
  std::unique_ptr<CConvKern> k = makeConvFromPart(part);
  CMatrix X = matrixFromNumpy(Xin);
  if(X.getCols() != k->getInputDim())
    throw ndlexceptions::Error("X has " + std::to_string(X.getCols()) + " columns but the Conv kernel expects "
                               + std::to_string(k->getInputDim()));
  if(X2in.is_none())
  {
    CMatrix K(X.getRows(), X.getRows());
    k->compute(K, X);
    return matrixToNumpy(K);
  }
  CMatrix X2 = matrixFromNumpy(py::cast<py::array_t<double, py::array::c_style | py::array::forcecast> >(X2in));
  if(X2.getCols() != k->getInputDim())
    throw ndlexceptions::Error("X2 has " + std::to_string(X2.getCols()) + " columns but the Conv kernel expects "
                               + std::to_string(k->getInputDim()));
  CMatrix K(X.getRows(), X2.getRows());
  k->compute(K, X, X2);
  return matrixToNumpy(K);
}

py::array_t<double> PyDeepGP::conv_filters() const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  const CConvKern* ck = dynamic_cast<const CConvKern*>(model->layer(0).getKern());
  if(ck == 0)
    throw ndlexceptions::Error("the first layer is not convolutional");
  const CMatrix& Z = model->layer(0).getInducingCentres();
  const unsigned int M = Z.getRows(), h = ck->getPatchHeight(), w = ck->getPatchWidth(), C = ck->numChannels();
  py::array_t<double> out({(py::ssize_t)M, (py::ssize_t)h, (py::ssize_t)w, (py::ssize_t)C});
  double* o = out.mutable_data();
  // A patch vector is ordered (di, dj, c): row-major (h, w, C).
  for(unsigned int m = 0; m < M; m++)
    for(unsigned int j = 0; j < h * w * C; j++)
      o[(size_t)m * h * w * C + j] = Z.getVal(m, j);
  return out;
}

py::array_t<double> PyDeepGP::conv_patch_weights() const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  const CConvKern* ck = dynamic_cast<const CConvKern*>(model->layer(0).getKern());
  if(ck == 0)
    throw ndlexceptions::Error("the first layer is not convolutional");
  const std::vector<double>& w = ck->patchWeights();
  py::array_t<double> out((py::ssize_t)w.size());
  for(size_t i = 0; i < w.size(); i++)
    out.mutable_data()[i] = w[i];
  return out;
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
  for(unsigned int ell = 0; ell < self->model->getNumLayers(); ++ell)
    self->layerInducingTypes.push_back(self->model->getLayerInducingType(ell));
  self->inducingType = joinLayerNames(self->layerInducingTypes);
  self->likelihoodType = (self->model->getLikelihood() != 0)
                             ? self->model->getLikelihood()->getType()
                             : std::string("gaussian");
  self->kernelType = "loaded";
  self->layerKernelTypes.assign(self->model->getNumLayers(), "loaded");
  return self;
}

static unsigned int layerParamIndex(const CKern& k, const std::string& name)
{
  for(unsigned int i = 0; i < k.getNumParams(); i++)
    if(k.getParamName(i) == name)
      return i;
  throw ndlexceptions::Error("Unknown kernel parameter '" + name + "'.");
}

std::vector<std::string> PyDeepGP::layer_param_paths(unsigned int ell) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  const CKern* k = model->layer(ell).getKern();
  std::vector<std::string> out;
  for(unsigned int i = 0; i < k->getNumParams(); i++)
    out.push_back(k->getParamName(i));
  return out;
}

double PyDeepGP::layer_kl(unsigned int ell) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  if(ell >= model->getNumLayers())
    throw ndlexceptions::Error("layer index out of range");
  return model->layer(ell).klQ();
}

double PyDeepGP::get_layer_param(unsigned int ell, const std::string& path) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  const CKern* k = model->layer(ell).getKern();
  return k->getParam(layerParamIndex(*k, path));
}

void PyDeepGP::set_layer_param(unsigned int ell, const std::string& path, double val)
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  CKern* k = model->layer(ell).getKern();
  k->setParam(val, layerParamIndex(*k, path));
  model->layer(ell).updateKernelMatrices();
}

double PyDeepGP::layer_input_mean_sqnorm(unsigned int ell) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  if(ell >= model->getNumLayers())
    throw ndlexceptions::Error("layer index out of range");
  return model->layerInputMeanSqNorm(ell);
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
                                 uint64_t probe_seed,
                                 bool prior) const
{
  return std::get<0>(jacobian_stats(n_probe, probe_seed, prior));
}

std::tuple<double, double> PyDeepGP::jacobian_stats(unsigned int n_probe,
                                                    uint64_t probe_seed,
                                                    bool prior) const
{
  if(!model)
    throw ndlexceptions::Error("DeepGP is uninitialised");
  if(n_probe < 1)
    throw ndlexceptions::Error("n_probe must be >= 1");
  double spread, meanLog;
  model->jacobianStats(n_probe, probe_seed, prior, spread, meanLog);
  return std::make_tuple(spread, meanLog);
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
  // CIP-0041 B3: ARD scales are part of what was built; show every parameter of ARD layers.
  for(unsigned int ell = 0; ell < model->getNumLayers(); ++ell)
  {
    const CKern* k = model->layer(ell).getKern();
    const std::string t = k->getType();
    if(t != "rbfard" && t != "mlpard")
      continue;
    std::ostringstream os;
    os << "layer " << ell << " " << t << ":";
    for(unsigned int i = 0; i < k->getNumParams(); ++i)
      os << " " << k->getParamName(i) << "=" << k->getParam(i);
    d.headerLines.push_back(os.str());
  }
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
  m.def("conv_kern_K", &conv_kern_K, py::arg("part"), py::arg("X"), py::arg("X2") = py::none(),
        "Evaluate a convolutional kernel descriptor (gpc.kern.Conv.to_parts()[0]) on X (and X2).");
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
                    py::object,
                    double,
                    double,
                    bool,
                    const std::string&,
                    unsigned int,
                    int,
                    bool,
                    bool,
                    unsigned int,
                    double,
                    unsigned int,
                    unsigned int,
                    const std::string&,
                    unsigned int,
                    bool,
                    py::object>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("layers") = 2,
           py::arg("hidden") = -1,
           py::arg("num_inducing") = 10,
           py::arg("kernel") = py::str("rbf"),
           py::arg("inducing") = py::str("points"),
           py::arg("chi") = 1.0,
           py::arg("beta") = 1.0,
           py::arg("drop_residual") = false,
           py::arg("likelihood") = "gaussian",
           py::arg("num_categories") = 3,
           py::arg("verbosity") = 0,
           py::arg("learn_window_width") = false,
           py::arg("learn_window_widths") = false,
           py::arg("num_classes") = 0,
           py::arg("likelihood_eps") = 1e-3,
           py::arg("likelihood_samples") = 256,
           py::arg("likelihood_seed") = 12345,
           py::arg("diag") = "exact",
           py::arg("diag_samples") = 64,
           py::arg("diag_force") = false,
           py::arg("mlp_init") = py::str("unit"))
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
      .def("predict_proba",
           &PyDeepGP::predict_proba,
           py::arg("Xtest"),
           py::arg("samples") = 0,
           py::arg("seed") = 0,
           "Class probabilities (N x K) at Xtest for a robustmax or softmax likelihood.")
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
      .def("layer_param_paths", &PyDeepGP::layer_param_paths, py::arg("layer"),
           "Kernel parameter paths of one layer (as accepted by set_layer_param).")
      .def("get_layer_param", &PyDeepGP::get_layer_param, py::arg("layer"), py::arg("path"))
      .def("layer_kl", &PyDeepGP::layer_kl, py::arg("layer"),
           "KL(q(v) || N(0, I)) summed over the layer outputs (nats).")
      .def("set_layer_param", &PyDeepGP::set_layer_param, py::arg("layer"), py::arg("path"),
           py::arg("value"), "Set one kernel parameter of one layer (before optimisation).")
      .def("layer_input_mean_sqnorm", &PyDeepGP::layer_input_mean_sqnorm, py::arg("layer"),
           "Mean over the data of the squared norm of layer's input (mean-path).")
      .def_property_readonly("drop_residual", &PyDeepGP::drop_residual)
      .def("jacobian_spread",
           &PyDeepGP::jacobian_spread,
           py::arg("n_probe") = 32,
           py::arg("probe_seed") = 0,
           py::arg("prior") = true,
           "Median Jacobian log-singular-value spread (V7). prior=True draws the whitened "
           "variables from N(0, I) per layer; prior=False evaluates at the variational mean.")
      .def("jacobian_stats",
           &PyDeepGP::jacobian_stats,
           py::arg("n_probe") = 32,
           py::arg("probe_seed") = 0,
           py::arg("prior") = true,
           "(median spread, median mean log singular value) of the input-output Jacobian.")
      .def("conv_filters", &PyDeepGP::conv_filters,
           "Inducing patches of the convolutional first layer as an (M, h, w, C) array.")
      .def("conv_patch_weights", &PyDeepGP::conv_patch_weights,
           "Patch weights w_p of the convolutional first layer (length P, row-major positions).")
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
      .def_property_readonly("layer_kernel_types", &PyDeepGP::layer_kernel_types,
                             "Kernel type of each layer.")
      .def_property_readonly("layer_inducing_types", &PyDeepGP::layer_inducing_types,
                             "Inducing family of each layer.")
      .def_property_readonly("likelihood_type", &PyDeepGP::likelihood_type)
      .def_property_readonly("chi", &PyDeepGP::chi)
      .def_property_readonly("approx", &PyDeepGP::approx);
}
