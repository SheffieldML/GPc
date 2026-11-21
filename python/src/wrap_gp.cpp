#include "wrap_gp.h"

#include <sstream>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "matrix_numpy.h"
#include "ndlexceptions.h"

namespace {

bool isEqKernelName(const std::string& kernel)
{
  return kernel == "eq" || kernel == "ExpQuad" || kernel == "expquad";
}

bool isRbfKernelName(const std::string& kernel)
{
  return kernel == "rbf" || kernel == "RBF";
}

bool isMlpKernelName(const std::string& kernel)
{
  return kernel == "mlp" || kernel == "MLP";
}

bool isArccosKernelName(const std::string& kernel)
{
  return kernel == "arccos" || kernel == "arccosine" || kernel == "ArcCosine"
      || kernel == "ARCCOS";
}

bool isWhiteKernelName(const std::string& kernel)
{
  return kernel == "white" || kernel == "White" || kernel == "WHITE";
}

bool isBiasKernelName(const std::string& kernel)
{
  return kernel == "bias" || kernel == "Bias" || kernel == "BIAS";
}

enum class SignalKind
{
  Rbf,
  Eq,
  Mlp,
  ArcCos
};

SignalKind parseSignalKind(const std::string& kernel)
{
  if(isEqKernelName(kernel))
    return SignalKind::Eq;
  if(isRbfKernelName(kernel))
    return SignalKind::Rbf;
  if(isMlpKernelName(kernel))
    return SignalKind::Mlp;
  if(isArccosKernelName(kernel))
    return SignalKind::ArcCos;
  throw ndlexceptions::Error(
      "Unknown kernel '" + kernel
      + "'. Use 'rbf', 'eq' / 'ExpQuad', 'mlp', or 'arccos' / 'arccosine'.");
}

std::string signalKindLabel(SignalKind kind)
{
  switch(kind)
  {
  case SignalKind::Eq:
    return "eq";
  case SignalKind::Mlp:
    return "mlp";
  case SignalKind::ArcCos:
    return "arccos";
  case SignalKind::Rbf:
  default:
    return "rbf";
  }
}

void addOptionalWhite(CCmpndKern& kern, const CMatrix& X)
{
  CWhiteKern white(X);
  kern.addKern(&white);
}

// CIP-0027 Phase B: string-kernel default is signal only (no auto Bias/White).
// Observation noise is likelihood.variance (FTC) or β (sparse).
// MLP defaults match GPy.kern.MLP: variance=1, weight_variance=1, bias_variance=1
// (no data-dependent scaling — callers / notebooks set scale for large |X|).
void addFtcStyleCompound(CCmpndKern& kern,
                         const CMatrix& X,
                         SignalKind kind,
                         unsigned int arccosOrder)
{
  switch(kind)
  {
  case SignalKind::Eq:
  {
    CEqKern eq(X);
    kern.addKern(&eq);
    kern.setByPath("ExpQuad.lengthscale", 1.0);
    kern.setByPath("ExpQuad.variance", 1.0);
    break;
  }
  case SignalKind::Mlp:
  {
    CMlpKern mlp(X);
    kern.addKern(&mlp);
    kern.setByPath("mlp.weightVariance", 1.0);
    kern.setByPath("mlp.biasVariance", 1.0);
    kern.setByPath("mlp.variance", 1.0);
    break;
  }
  case SignalKind::ArcCos:
  {
    CArcCosineKern ac(X);
    ac.setOrder(arccosOrder);
    kern.addKern(&ac);
    kern.setByPath("arccos.variance", 1.0);
    break;
  }
  case SignalKind::Rbf:
  default:
  {
    CRbfKern rbf(X);
    kern.addKern(&rbf);
    kern.setByPath("rbf.lengthscale", 1.0);
    kern.setByPath("rbf.variance", 1.0);
    break;
  }
  }
}

// CIP-0028 weight-space: one mlp|arccos signal + optional white (no Bias).
void addWeightSpaceCompound(CCmpndKern& kern,
                            const CMatrix& X,
                            SignalKind kind,
                            unsigned int arccosOrder)
{
  switch(kind)
  {
  case SignalKind::Mlp:
  {
    CMlpKern mlp(X);
    kern.addKern(&mlp);
    addOptionalWhite(kern, X);
    kern.setByPath("mlp.weightVariance", 1.0);
    kern.setByPath("mlp.biasVariance", 1.0);
    kern.setByPath("mlp.variance", 1.0);
    break;
  }
  case SignalKind::ArcCos:
  {
    CArcCosineKern ac(X);
    ac.setOrder(arccosOrder);
    kern.addKern(&ac);
    addOptionalWhite(kern, X);
    kern.setByPath("arccos.variance", 1.0);
    break;
  }
  default:
    throw ndlexceptions::Error(
        "inducing='weightspace' requires kernel='mlp' or 'arccos' "
        "(CIP-0028; RBF/ExpQuad use point or whitenoise inducing).");
  }
}

double dictGet(const py::dict& d, const char* key, double fallback)
{
  if(!d.contains(key))
    return fallback;
  return py::cast<double>(d[key]);
}

unsigned int dictGetUInt(const py::dict& d, const char* key, unsigned int fallback)
{
  if(!d.contains(key))
    return fallback;
  return static_cast<unsigned int>(py::cast<double>(d[key]));
}

std::string partType(const py::dict& d)
{
  if(!d.contains("type"))
    throw ndlexceptions::Error("kernel part dict missing 'type'");
  return py::cast<std::string>(d["type"]);
}

// Build exactly the listed components (GPy-style; no auto Bias/White).
std::string addPartsCompound(CCmpndKern& kern, const CMatrix& X, const py::list& parts)
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

    if(isRbfKernelName(type))
    {
      CRbfKern rbf(X);
      kern.addKern(&rbf);
      kern.setByPath("rbf.lengthscale", dictGet(d, "lengthscale", 1.0));
      kern.setByPath("rbf.variance", dictGet(d, "variance", 1.0));
      label << "rbf";
    }
    else if(isEqKernelName(type))
    {
      CEqKern eq(X);
      kern.addKern(&eq);
      kern.setByPath("ExpQuad.lengthscale", dictGet(d, "lengthscale", 1.0));
      kern.setByPath("ExpQuad.variance", dictGet(d, "variance", 1.0));
      label << "eq";
    }
    else if(isWhiteKernelName(type))
    {
      CWhiteKern white(X);
      kern.addKern(&white);
      kern.setByPath("white.variance", dictGet(d, "variance", 1.0));
      label << "white";
    }
    else if(isBiasKernelName(type))
    {
      CBiasKern bias(X);
      kern.addKern(&bias);
      kern.setByPath("bias.variance", dictGet(d, "variance", 1.0));
      label << "bias";
    }
    else if(isMlpKernelName(type))
    {
      CMlpKern mlp(X);
      kern.addKern(&mlp);
      kern.setByPath("mlp.weightVariance", dictGet(d, "weight_variance", 1.0));
      kern.setByPath("mlp.biasVariance", dictGet(d, "bias_variance", 1.0));
      kern.setByPath("mlp.variance", dictGet(d, "variance", 1.0));
      label << "mlp";
    }
    else if(isArccosKernelName(type))
    {
      CArcCosineKern ac(X);
      const unsigned int order = dictGetUInt(d, "order", 0);
      if(order > 1)
        throw ndlexceptions::Error("arccos order must be 0 or 1");
      ac.setOrder(order);
      kern.addKern(&ac);
      kern.setByPath("arccos.variance", dictGet(d, "variance", 1.0));
      label << "arccos";
    }
    else
    {
      throw ndlexceptions::Error(
          "Unknown kernel part type '" + type
          + "'. Use rbf, eq/ExpQuad, white, bias, mlp, or arccos.");
    }
  }
  return label.str();
}

bool kernelIsPartsList(const py::object& kernel)
{
  return py::isinstance<py::list>(kernel) || py::isinstance<py::tuple>(kernel);
}

void checkXY(const CMatrix& X, const CMatrix& Y)
{
  if(X.getRows() != Y.getRows())
    throw ndlexceptions::Error("X and Y must have the same number of rows");
  if(Y.getCols() < 1)
    throw ndlexceptions::Error("Y must have at least one column");
}

std::pair<py::array_t<double>, py::array_t<double>> predictImpl(
    CGp& model,
    const CMatrix& X,
    const CMatrix& Y,
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
    bool includeObservationNoise)
{
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != X.getCols())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");

  CMatrix mu(Xt.getRows(), Y.getCols());
  CMatrix varSigma(Xt.getRows(), Y.getCols());
  // CIP-0031: predictive (y_*) vs latent (f_*).
  if(includeObservationNoise)
    model.predictiveMeanVar(mu, varSigma, Xt);
  else
    model.latentMeanVar(mu, varSigma, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varSigma)};
}

} // namespace

PyGPRegression::PyGPRegression(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xin,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
    py::object kernel,
    int verbosity,
    unsigned int arccos_order)
    : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin))
{
  checkXY(X, Y);
  if(arccos_order > 1)
    throw ndlexceptions::Error("arccos_order must be 0 or 1");

  kern = std::make_unique<CCmpndKern>(X);
  if(kernelIsPartsList(kernel))
  {
    py::list parts = py::reinterpret_borrow<py::list>(kernel);
    kernelType = addPartsCompound(*kern, X, parts);
  }
  else
  {
    const std::string kernelName = py::cast<std::string>(kernel);
    const SignalKind kind = parseSignalKind(kernelName);
    kernelType = signalKindLabel(kind);
    addFtcStyleCompound(*kern, X, kind, arccos_order);
  }

  noise = std::make_unique<CGaussianNoise>(&Y);
  for(unsigned int j = 0; j < Y.getCols(); ++j)
    noise->setBiasVal(0.0, j);

  model = std::make_unique<CGp>(kern.get(), noise.get(), &X, CGp::FTC, 0, verbosity);
  model->setLikelihoodVariance(1.0);
  // CIP-0027 Phase C: explicit White in a kern object → legacy White-owned
  // observation noise (matches ./gp learn -k white). Likelihood-owned noise
  // is the string-kernel default (no White).
  try
  {
    (void)kern->getByPath("white.variance");
    model->fixLikelihoodVariance(true);
  }
  catch(ndlexceptions::Error&)
  {
  }
}

void PyGPRegression::set_by_path(const std::string& path, double val)
{
  model->setByPath(path, val);
}

double PyGPRegression::get_by_path(const std::string& path) const
{
  return model->getByPath(path);
}

std::vector<std::string> PyGPRegression::param_paths() const
{
  return model->paramPaths();
}

void PyGPRegression::optimize(unsigned int iters, bool messages, py::object callback)
{
  // Python façade owns messaging (paramz-style in-place table). Never enable
  // C++ CLI Iteration lines, Initial model dumps, or checkGradients here.
  (void)messages;
  model->clearOptimiserObservers();
  model->setDefaultTextProgress(false);
  model->setVerbosity(0);

  const bool haveCallback = !(callback.is_none());
  if(haveCallback)
  {
    // Hold a copy for the duration of optimise; clear afterwards.
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

double PyGPRegression::objective() const
{
  return model->computeObjectiveVal();
}

std::pair<py::array_t<double>, py::array_t<double>> PyGPRegression::predict(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictImpl(*model, X, Y, Xtest, true);
}

std::pair<py::array_t<double>, py::array_t<double>> PyGPRegression::predict_noiseless(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictImpl(*model, X, Y, Xtest, false);
}

PyGPClassification::PyGPClassification(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xin,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
    py::object kernel,
    const std::string& inference,
    int verbosity)
    : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin)), inferenceType("ep")
{
  checkXY(X, Y);
  if(Y.getCols() != 1)
    throw ndlexceptions::Error("GPClassification requires a single output column.");
  for(unsigned int i = 0; i < Y.getRows(); ++i)
  {
    const double yi = Y.getVal(i, 0);
    if(!(yi == 0.0 || yi == 1.0 || yi == -1.0))
      throw ndlexceptions::Error(
          "GPClassification labels must be in {0,1} or {-1,1}.");
  }

  kern = std::make_unique<CCmpndKern>(X);
  if(kernelIsPartsList(kernel))
  {
    py::list parts = py::reinterpret_borrow<py::list>(kernel);
    kernelType = addPartsCompound(*kern, X, parts);
  }
  else
  {
    const std::string kernelName = py::cast<std::string>(kernel);
    const SignalKind kind = parseSignalKind(kernelName);
    kernelType = signalKindLabel(kind);
    addFtcStyleCompound(*kern, X, kind, 0);
  }

  noise = std::make_unique<CProbitNoise>(&Y);
  model = std::make_unique<CGp>(kern.get(), noise.get(), &X, CGp::FTC, 0, verbosity);
  if(inference == "ep" || inference == "EP")
  {
    inferenceType = "ep";
    model->setEpInference(true);
  }
  else if(inference == "laplace" || inference == "Laplace")
  {
    inferenceType = "laplace";
    model->setLaplaceInference(true);
  }
  else
    throw ndlexceptions::Error(
        "GPClassification inference must be 'ep' or 'laplace' (FTC only).");
}

void PyGPClassification::set_by_path(const std::string& path, double val)
{
  model->setByPath(path, val);
}

double PyGPClassification::get_by_path(const std::string& path) const
{
  return model->getByPath(path);
}

std::vector<std::string> PyGPClassification::param_paths() const
{
  return model->paramPaths();
}

void PyGPClassification::optimize(unsigned int iters, bool messages, py::object callback)
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

double PyGPClassification::objective() const
{
  return model->computeObjectiveVal();
}

std::pair<py::array_t<double>, py::array_t<double>> PyGPClassification::predict_noiseless(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  if(inferenceType != "ep")
    throw ndlexceptions::Error(
        "predict_noiseless for Laplace classification is not wired yet; use inference='ep'.");
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != X.getCols())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");
  CMatrix mu, varDiag;
  model->epLatentMeanVar(mu, varDiag, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varDiag)};
}

py::array_t<double> PyGPClassification::predict_proba(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  auto mv = predict_noiseless(Xtest);
  CMatrix mu = matrixFromNumpy(mv.first);
  CMatrix var = matrixFromNumpy(mv.second);
  CMatrix yPos(mu.getRows(), 1);
  yPos.setVals(1.0);
  CMatrix prob(mu.getRows(), 1);
  noise->likelihoods(prob, mu, var, yPos); // P(y=+1)
  return matrixToNumpy(prob);
}

std::string PyGPClassification::to_string() const
{
  std::ostringstream os;
  os << "GPClassification(n=" << n() << ", inference=" << inferenceType
     << ", kernel=" << kernelType << ", approx=ftc)";
  return os.str();
}

ModelDescription PyGPClassification::describe() const
{
  return model->describe();
}

PySparseGPRegression::PySparseGPRegression(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xin,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
    py::array_t<double, py::array::c_style | py::array::forcecast> Zin,
    const std::string& approx,
    bool inducing_fixed,
    int verbosity,
    py::object kernel,
    const std::string& inducing,
    py::object window_width,
    unsigned int arccos_order,
    bool learn_window_width,
    bool learn_window_widths)
    : X(matrixFromNumpy(Xin)),
      Y(matrixFromNumpy(Yin)),
      Z(matrixFromNumpy(Zin)),
      approxType("dtcvar"),
      inducingFixed(inducing_fixed),
      learnWindowWidth(learn_window_width),
      learnWindowWidths(learn_window_widths)
{
  checkXY(X, Y);
  if(arccos_order > 1)
    throw ndlexceptions::Error("arccos_order must be 0 or 1");

  int approxEnum = CGp::DTCVAR;
  if(approx == "dtcvar" || approx == "DTCVAR")
  {
    approxEnum = CGp::DTCVAR;
    approxType = "dtcvar";
  }
  else if(approx == "fitc" || approx == "FITC")
  {
    approxEnum = CGp::FITC;
    approxType = "fitc";
  }
  else if(approx == "dtc" || approx == "DTC")
  {
    approxEnum = CGp::DTC;
    approxType = "dtc";
  }
  else
  {
    throw ndlexceptions::Error(
        "SparseGPRegression approx='" + approx
        + "' is not supported. Use 'dtcvar' (default), 'fitc', or 'dtc'.");
  }

  if(Z.getRows() < 1)
    throw ndlexceptions::Error("Z must have at least one inducing row");
  if(Z.getCols() != X.getCols())
    throw ndlexceptions::Error("Z input dimension must match X");
  if(Z.getRows() > X.getRows())
    throw ndlexceptions::Error("Number of inducing points must be <= number of data");

  numActive = Z.getRows();

  bool weightSpace = false;
  if(inducing == "points" || inducing == "point")
  {
    inducingType = "points";
    weightSpace = false;
  }
  else if(inducing == "weightspace" || inducing == "weight_space" || inducing == "ridge")
  {
    inducingType = "weightspace";
    weightSpace = true;
  }
  else
  {
    throw ndlexceptions::Error(
        "SparseGPRegression inducing='" + inducing
        + "' is not supported. Use 'points' (default) or 'weightspace'.");
  }

  kern = std::make_unique<CCmpndKern>(X);
  if(kernelIsPartsList(kernel))
  {
    if(weightSpace)
      throw ndlexceptions::Error(
          "Custom gpc.kern compounds are not supported with inducing='weightspace'; "
          "use kernel='mlp' or 'arccos' string names.");
    py::list parts = py::reinterpret_borrow<py::list>(kernel);
    kernelType = addPartsCompound(*kern, X, parts);
  }
  else
  {
    const std::string kernelName = py::cast<std::string>(kernel);
    const SignalKind kind = parseSignalKind(kernelName);
    kernelType = signalKindLabel(kind);
    if(weightSpace)
      addWeightSpaceCompound(*kern, X, kind, arccos_order);
    else
      addFtcStyleCompound(*kern, X, kind, arccos_order);
  }

  noise = std::make_unique<CGaussianNoise>(&Y);
  for(unsigned int j = 0; j < Y.getCols(); ++j)
    noise->setBiasVal(0.0, j);

  model = std::make_unique<CGp>(
      kern.get(), noise.get(), &X, approxEnum, numActive, verbosity);
  model->setActiveSet(Z);
  model->setInducingFixed(inducingFixed);
  model->setBetaVal(1.0);

  if(weightSpace)
  {
    model->setInducingType("weightspace");
    CInducing* pInd = model->getInducing();
    if(pInd == 0)
      throw ndlexceptions::Error("weight-space inducing object missing after setInducingType");

    // CIP-0032: window_width may be a scalar or length-M vector.
    bool haveVectorS = false;
    if(!window_width.is_none())
    {
      if(py::isinstance<py::float_>(window_width) || py::isinstance<py::int_>(window_width))
      {
        const double s0 = py::cast<double>(window_width);
        if(s0 > 0.0)
          pInd->setWindowWidth(s0);
      }
      else
      {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(window_width);
        if(!arr)
          throw ndlexceptions::Error("window_width must be a float or a 1-D array of length M.");
        if(arr.ndim() != 1 || static_cast<unsigned int>(arr.shape(0)) != numActive)
          throw ndlexceptions::Error("window_width array must have shape (M,) matching Z rows.");
        CMatrix sVec(numActive, 1);
        auto r = arr.unchecked<1>();
        for(unsigned int m = 0; m < numActive; m++)
        {
          if(!(r(m) > 0.0))
            throw ndlexceptions::Error("window_width entries must be positive.");
          sVec.setVal(r(m), m, 0);
        }
        CInducingWeightSpace* pWs = dynamic_cast<CInducingWeightSpace*>(pInd);
        if(pWs == 0)
          throw ndlexceptions::Error("per-unit window_width requires weight-space inducing.");
        pWs->setWindowWidths(sVec);
        haveVectorS = true;
      }
    }

    if(learnWindowWidth && learnWindowWidths)
      throw ndlexceptions::Error(
          "Pass only one of learn_window_width=True (tied) or learn_window_widths=True (untied).");

    if(learnWindowWidths || haveVectorS)
    {
      CInducingWeightSpace* pWs = dynamic_cast<CInducingWeightSpace*>(pInd);
      if(pWs == 0)
        throw ndlexceptions::Error("learn_window_widths requires weight-space inducing.");
      if(!pWs->isWindowWidthUntied())
      {
        // Untie at current scalar (or keep vector already set).
        CMatrix sVec(numActive, 1);
        const double s0 = pWs->getWindowWidth();
        for(unsigned int m = 0; m < numActive; m++)
          sVec.setVal(s0, m, 0);
        pWs->setWindowWidths(sVec);
      }
    }

    if(learnWindowWidth || learnWindowWidths)
    {
      if(inducingFixed)
        throw ndlexceptions::Error(
            "learn_window_width(s)=True requires inducing_fixed=False "
            "(s is packed with free inducing parameters).");
      pInd->setLearnWindowWidth(true);
    }
  }
  else if(learnWindowWidth || learnWindowWidths)
  {
    throw ndlexceptions::Error(
        "learn_window_width(s) is only supported with inducing='weightspace'.");
  }
}

void PySparseGPRegression::set_by_path(const std::string& path, double val)
{
  model->setByPath(path, val);
}

double PySparseGPRegression::get_by_path(const std::string& path) const
{
  return model->getByPath(path);
}

std::vector<std::string> PySparseGPRegression::param_paths() const
{
  return model->paramPaths();
}

void PySparseGPRegression::optimize(unsigned int iters, bool messages, py::object callback)
{
  // Python façade owns messaging; never enable C++ CLI progress here.
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

double PySparseGPRegression::objective() const
{
  return model->computeObjectiveVal();
}

std::pair<py::array_t<double>, py::array_t<double>> PySparseGPRegression::predict(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictImpl(*model, X, Y, Xtest, true);
}

std::pair<py::array_t<double>, py::array_t<double>> PySparseGPRegression::predict_noiseless(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictImpl(*model, X, Y, Xtest, false);
}

py::array_t<double> PySparseGPRegression::inducing_inputs() const
{
  return matrixToNumpy(Z);
}

double PySparseGPRegression::window_width() const
{
  const CInducing* pInd = model->getInducing();
  if(pInd == 0)
    return 0.0;
  return pInd->getWindowWidth();
}

py::array_t<double> PySparseGPRegression::window_widths() const
{
  const CInducing* pInd = model->getInducing();
  if(pInd == 0 || pInd->numWindowWidths() == 0)
    return py::array_t<double>(0);
  const unsigned int M = pInd->numWindowWidths();
  py::array_t<double> out(static_cast<py::ssize_t>(M));
  auto r = out.mutable_unchecked<1>();
  for(unsigned int m = 0; m < M; m++)
    r(m) = pInd->getWindowWidthAt(m);
  return out;
}

ModelDescription PyGPRegression::describe() const
{
  return model->describe();
}

std::string PyGPRegression::to_string() const
{
  return modelDescriptionToString(model->describe());
}

ModelDescription PySparseGPRegression::describe() const
{
  return model->describe();
}

std::string PySparseGPRegression::to_string() const
{
  return modelDescriptionToString(model->describe());
}

void bind_gp(py::module_& m)
{
  py::enum_<OptimiserPhase>(m, "OptimiserPhase")
      .value("Start", OptimiserPhase::Start)
      .value("Iter", OptimiserPhase::Iter)
      .value("Converged", OptimiserPhase::Converged)
      .value("MaxIters", OptimiserPhase::MaxIters)
      .value("Failed", OptimiserPhase::Failed);

  py::class_<OptimiserEvent>(m, "OptimiserEvent")
      .def_readonly("phase", &OptimiserEvent::phase)
      .def_readonly("optimiser", &OptimiserEvent::optimiser)
      .def_readonly("iter", &OptimiserEvent::iter)
      .def_readonly("func_eval", &OptimiserEvent::funcEval)
      .def_readonly("objective", &OptimiserEvent::objective)
      .def_readonly("grad_norm", &OptimiserEvent::gradNorm)
      .def_readonly("step_or_scale", &OptimiserEvent::stepOrScale)
      .def_readonly("message", &OptimiserEvent::message);

  py::class_<ParamRow>(m, "ParamRow")
      .def_readonly("path", &ParamRow::path)
      .def_readonly("value", &ParamRow::value)
      .def_readonly("transform", &ParamRow::transform)
      .def_readonly("fixed", &ParamRow::fixed)
      .def_readonly("role", &ParamRow::role);

  py::class_<ModelDescription>(m, "ModelDescription")
      .def_readonly("title", &ModelDescription::title)
      .def_readonly("header_lines", &ModelDescription::headerLines)
      .def_readonly("params", &ModelDescription::params)
      .def_readonly("has_log_likelihood", &ModelDescription::hasLogLikelihood)
      .def_readonly("log_likelihood", &ModelDescription::logLikelihood)
      .def_readonly("opt_num_params", &ModelDescription::optNumParams);

  py::class_<PyGPRegression>(
      m,
      "GPRegression",
      "Full / FTC Gaussian process regression (CIP-001D / CIP-001E).\n\n"
      "Prefer gpc.models.GPRegression(X, Y, kernel=...) for GPy-shaped construction.\n"
      "Parameters use CIP-0014 GPy-natural units via set_by_path / get_by_path.\n"
      "Does not claim API identity with GPy.")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::object,
                    int,
                    unsigned int>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("kernel") = py::str("rbf"),
           py::arg("verbosity") = 0,
           py::arg("arccos_order") = 0,
           "Construct FTC Gaussian GP.\n\n"
           "kernel: string name ('rbf', 'eq'/'ExpQuad', 'mlp', 'arccos') builds the\n"
           "CIP-0027 signal-only compound (noise is likelihood.variance), or a parts\n"
           "list from gpc.kern compounds.")
      .def("set_by_path",
           &PyGPRegression::set_by_path,
           py::arg("path"),
           py::arg("value"),
           "Set a parameter by GPy-like path (e.g. 'rbf.lengthscale') in natural units.")
      .def("get_by_path",
           &PyGPRegression::get_by_path,
           py::arg("path"),
           "Get a parameter by GPy-like path in natural units.")
      .def("param_paths",
           &PyGPRegression::param_paths,
           "List trainable parameter paths for this model.")
      .def("optimize",
           &PyGPRegression::optimize,
           py::arg("iters") = 100,
           py::arg("messages") = false,
           py::arg("callback") = py::none(),
           "Run SCG for iters iterations (minimises -log-likelihood).\n"
           "messages= is handled by gpc.models.GPRegression (in-place table);\n"
           "this binding only installs callback=.")
      .def("objective",
           &PyGPRegression::objective,
           "Return the SCG objective (-log-likelihood).")
      .def("predict",
           &PyGPRegression::predict,
           py::arg("Xtest"),
           "Predictive mean/variance of y_* at Xtest (includes observation noise).\n"
           "Returns (mu, var) NumPy arrays. CIP-0031 / GPy predict.")
      .def("predict_noiseless",
           &PyGPRegression::predict_noiseless,
           py::arg("Xtest"),
           "Latent mean/variance of f_* at Xtest (no observation noise).\n"
           "Returns (mu, var) NumPy arrays. CIP-0031 / GPy predict_noiseless.")
      .def("describe", &PyGPRegression::describe, "Structured ModelDescription (CIP-001E).")
      .def("__str__", &PyGPRegression::to_string)
      .def("__repr__", &PyGPRegression::to_string)
      .def_property_readonly("n", &PyGPRegression::n, "Number of training points.")
      .def_property_readonly("input_dim", &PyGPRegression::input_dim, "Input dimension q.")
      .def_property_readonly("output_dim", &PyGPRegression::output_dim, "Output dimension d.")
      .def_property_readonly("kernel_type",
                             &PyGPRegression::kernel_type,
                             "Kernel label (string name or compound part summary).")
      .def_property_readonly("approx", &PyGPRegression::approx, "Approximation type ('ftc').");

  py::class_<PyGPClassification>(
      m,
      "GPClassification",
      "FTC Bernoulli–probit classification with EP or Laplace (CIP-0033 Phase D).\n\n"
      "Prefer gpc.models.GPClassification(X, Y, kernel=..., inference='ep').\n"
      "Sparse EP/EPDTC is out of scope (FTC-only).")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::object,
                    const std::string&,
                    int>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("kernel") = py::str("rbf"),
           py::arg("inference") = "ep",
           py::arg("verbosity") = 0,
           "Construct FTC classification GP.\n\n"
           "Y labels in {0,1} or {-1,1}. inference: 'ep' (default) or 'laplace'.")
      .def("set_by_path", &PyGPClassification::set_by_path, py::arg("path"), py::arg("value"))
      .def("get_by_path", &PyGPClassification::get_by_path, py::arg("path"))
      .def("param_paths", &PyGPClassification::param_paths)
      .def("optimize",
           &PyGPClassification::optimize,
           py::arg("iters") = 100,
           py::arg("messages") = false,
           py::arg("callback") = py::none())
      .def("objective", &PyGPClassification::objective)
      .def("predict_noiseless",
           &PyGPClassification::predict_noiseless,
           py::arg("Xtest"),
           "Latent mean/variance of f_* (EP).")
      .def("predict_proba",
           &PyGPClassification::predict_proba,
           py::arg("Xtest"),
           "P(y=1 | Xtest) via probit on EP latent predictive.")
      .def("describe", &PyGPClassification::describe)
      .def("__str__", &PyGPClassification::to_string)
      .def("__repr__", &PyGPClassification::to_string)
      .def_property_readonly("n", &PyGPClassification::n)
      .def_property_readonly("input_dim", &PyGPClassification::input_dim)
      .def_property_readonly("output_dim", &PyGPClassification::output_dim)
      .def_property_readonly("kernel_type", &PyGPClassification::kernel_type)
      .def_property_readonly("inference", &PyGPClassification::inference)
      .def_property_readonly("approx", &PyGPClassification::approx);

  py::class_<PySparseGPRegression>(
      m,
      "SparseGPRegression",
      "Sparse Gaussian process regression (DTCVAR / FITC / DTC; CIP-001D / CIP-001E).\n\n"
      "Prefer gpc.models.SparseGPRegression(X, Y, Z, kernel=...) for GPy-shaped construction.\n"
      "Does not claim API identity with GPy.")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    const std::string&,
                    bool,
                    int,
                    py::object,
                    const std::string&,
                    py::object,
                    unsigned int,
                    bool,
                    bool>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("Z"),
           py::arg("approx") = "dtcvar",
           py::arg("inducing_fixed") = false,
           py::arg("verbosity") = 0,
           py::arg("kernel") = py::str("rbf"),
           py::arg("inducing") = "points",
           py::arg("window_width") = -1.0,
           py::arg("arccos_order") = 0,
           py::arg("learn_window_width") = false,
           py::arg("learn_window_widths") = false,
           "Construct sparse Gaussian GP with inducing inputs Z.\n\n"
           "approx: 'dtcvar' (default), 'fitc', or 'dtc'.\n"
           "kernel: string name or gpc.kern parts list (parts list needs inducing='points').\n"
           "inducing: 'points' (default) or 'weightspace' (mlp/arccos string kernels).\n"
           "inducing_fixed defaults False (matches CGp); pass True to freeze Z/W.\n"
           "window_width: scalar or length-M array (CIP-0032 per-unit s_m).\n"
           "learn_window_width: optimise global tied s; learn_window_widths: untied s_m.")
      .def("set_by_path",
           &PySparseGPRegression::set_by_path,
           py::arg("path"),
           py::arg("value"),
           "Set a parameter by GPy-like path in natural units.")
      .def("get_by_path",
           &PySparseGPRegression::get_by_path,
           py::arg("path"),
           "Get a parameter by GPy-like path in natural units.")
      .def("param_paths",
           &PySparseGPRegression::param_paths,
           "List trainable parameter paths for this model.")
      .def("optimize",
           &PySparseGPRegression::optimize,
           py::arg("iters") = 100,
           py::arg("messages") = false,
           py::arg("callback") = py::none(),
           "Run SCG for iters iterations (minimises -log-likelihood).\n"
           "messages= is handled by gpc.models.SparseGPRegression (in-place table);\n"
           "this binding only installs callback=.")
      .def("objective",
           &PySparseGPRegression::objective,
           "Return the SCG objective (-log-likelihood).")
      .def("predict",
           &PySparseGPRegression::predict,
           py::arg("Xtest"),
           "Predictive mean/variance of y_* at Xtest (includes observation noise).\n"
           "Returns (mu, var) NumPy arrays. CIP-0031 / GPy predict.")
      .def("predict_noiseless",
           &PySparseGPRegression::predict_noiseless,
           py::arg("Xtest"),
           "Latent mean/variance of f_* at Xtest (no observation noise).\n"
           "Returns (mu, var) NumPy arrays. CIP-0031 / GPy predict_noiseless.")
      .def("describe",
           &PySparseGPRegression::describe,
           "Structured ModelDescription (CIP-001E).")
      .def("__str__", &PySparseGPRegression::to_string)
      .def("__repr__", &PySparseGPRegression::to_string)
      .def("inducing_inputs",
           &PySparseGPRegression::inducing_inputs,
           "Return inducing input locations Z as a NumPy array.")
      .def_property_readonly("n", &PySparseGPRegression::n, "Number of training points.")
      .def_property_readonly(
          "input_dim", &PySparseGPRegression::input_dim, "Input dimension q.")
      .def_property_readonly(
          "output_dim", &PySparseGPRegression::output_dim, "Output dimension d.")
      .def_property_readonly(
          "num_inducing", &PySparseGPRegression::num_inducing, "Number of inducing points M.")
      .def_property_readonly("approx", &PySparseGPRegression::approx, "Sparse approximation name.")
      .def_property_readonly(
          "kernel_type", &PySparseGPRegression::kernel_type, "Kernel label / compound summary.")
      .def_property_readonly(
          "inducing_type", &PySparseGPRegression::inducing_type, "'points' or 'weightspace'.")
      .def_property_readonly(
          "inducing_fixed",
          &PySparseGPRegression::inducing_fixed,
          "True if inducing inputs are held fixed during optimise.")
      .def_property_readonly(
          "learn_window_width",
          &PySparseGPRegression::learn_window_width,
          "True if global tied weight-space window width s is free.")
      .def_property_readonly(
          "learn_window_widths",
          &PySparseGPRegression::learn_window_widths,
          "True if per-unit weight-space window widths s_m are free (CIP-0032).")
      .def_property_readonly(
          "window_width",
          &PySparseGPRegression::window_width,
          "Weight-space ridge window width (tied scalar / first unit; 0 if unused).")
      .def_property_readonly(
          "window_widths",
          &PySparseGPRegression::window_widths,
          "Per-unit weight-space window widths as a length-M array (CIP-0032).");
}
