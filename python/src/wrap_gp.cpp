#include "wrap_gp.h"

#include <sstream>

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

void addBiasWhite(CCmpndKern& kern, const CMatrix& X)
{
  CBiasKern bias(X);
  CWhiteKern white(X);
  kern.addKern(&bias);
  kern.addKern(&white);
}

void addOptionalWhite(CCmpndKern& kern, const CMatrix& X)
{
  CWhiteKern white(X);
  kern.addKern(&white);
}

// FTC / point-inducing compound: signal + bias + white (GPy-familiar default).
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
    addBiasWhite(kern, X);
    kern.setByPath("ExpQuad.lengthscale", 1.0);
    kern.setByPath("ExpQuad.variance", 1.0);
    break;
  }
  case SignalKind::Mlp:
  {
    CMlpKern mlp(X);
    kern.addKern(&mlp);
    addBiasWhite(kern, X);
    kern.setByPath("mlp.weightVariance", 10.0);
    kern.setByPath("mlp.biasVariance", 10.0);
    kern.setByPath("mlp.variance", 1.0);
    break;
  }
  case SignalKind::ArcCos:
  {
    CArcCosineKern ac(X);
    ac.setOrder(arccosOrder);
    kern.addKern(&ac);
    addBiasWhite(kern, X);
    kern.setByPath("arccos.variance", 1.0);
    break;
  }
  case SignalKind::Rbf:
  default:
  {
    CRbfKern rbf(X);
    kern.addKern(&rbf);
    addBiasWhite(kern, X);
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
    kern.setByPath("mlp.weightVariance", 10.0);
    kern.setByPath("mlp.biasVariance", 10.0);
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
      kern.setByPath("mlp.weightVariance", dictGet(d, "weight_variance", 10.0));
      kern.setByPath("mlp.biasVariance", dictGet(d, "bias_variance", 10.0));
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
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest)
{
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != X.getCols())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");

  CMatrix mu(Xt.getRows(), Y.getCols());
  CMatrix varSigma(Xt.getRows(), Y.getCols());
  model.posteriorMeanVar(mu, varSigma, Xt);
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
  model->setBetaVal(1.0);
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

void PyGPRegression::optimize(unsigned int iters)
{
  model->optimise(iters);
}

double PyGPRegression::objective() const
{
  return model->computeObjectiveVal();
}

std::pair<py::array_t<double>, py::array_t<double>> PyGPRegression::predict(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictImpl(*model, X, Y, Xtest);
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
    double window_width,
    unsigned int arccos_order)
    : X(matrixFromNumpy(Xin)),
      Y(matrixFromNumpy(Yin)),
      Z(matrixFromNumpy(Zin)),
      approxType("dtcvar"),
      inducingFixed(inducing_fixed)
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
    if(window_width > 0.0)
    {
      CInducing* pInd = model->getInducing();
      if(pInd == 0)
        throw ndlexceptions::Error("weight-space inducing object missing after setInducingType");
      pInd->setWindowWidth(window_width);
    }
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

void PySparseGPRegression::optimize(unsigned int iters)
{
  model->optimise(iters);
}

double PySparseGPRegression::objective() const
{
  return model->computeObjectiveVal();
}

std::pair<py::array_t<double>, py::array_t<double>> PySparseGPRegression::predict(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const
{
  return predictImpl(*model, X, Y, Xtest);
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

void bind_gp(py::module_& m)
{
  py::class_<PyGPRegression>(m, "GPRegression")
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
           "FTC Gaussian GP. kernel is a string name (default signal+bias+white) "
           "or a parts list from gpc.kern compounds.")
      .def("set_by_path", &PyGPRegression::set_by_path, py::arg("path"), py::arg("value"))
      .def("get_by_path", &PyGPRegression::get_by_path, py::arg("path"))
      .def("param_paths", &PyGPRegression::param_paths)
      .def("optimize", &PyGPRegression::optimize, py::arg("iters") = 100)
      .def("objective", &PyGPRegression::objective)
      .def("predict", &PyGPRegression::predict, py::arg("Xtest"))
      .def_property_readonly("n", &PyGPRegression::n)
      .def_property_readonly("input_dim", &PyGPRegression::input_dim)
      .def_property_readonly("output_dim", &PyGPRegression::output_dim)
      .def_property_readonly("kernel_type", &PyGPRegression::kernel_type)
      .def_property_readonly("approx", &PyGPRegression::approx);

  py::class_<PySparseGPRegression>(m, "SparseGPRegression")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    const std::string&,
                    bool,
                    int,
                    py::object,
                    const std::string&,
                    double,
                    unsigned int>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("Z"),
           py::arg("approx") = "dtcvar",
           py::arg("inducing_fixed") = true,
           py::arg("verbosity") = 0,
           py::arg("kernel") = py::str("rbf"),
           py::arg("inducing") = "points",
           py::arg("window_width") = -1.0,
           py::arg("arccos_order") = 0,
           "Sparse Gaussian GP. kernel string or gpc.kern parts list "
           "(parts list requires inducing='points').")
      .def("set_by_path", &PySparseGPRegression::set_by_path, py::arg("path"), py::arg("value"))
      .def("get_by_path", &PySparseGPRegression::get_by_path, py::arg("path"))
      .def("param_paths", &PySparseGPRegression::param_paths)
      .def("optimize", &PySparseGPRegression::optimize, py::arg("iters") = 100)
      .def("objective", &PySparseGPRegression::objective)
      .def("predict", &PySparseGPRegression::predict, py::arg("Xtest"))
      .def("inducing_inputs", &PySparseGPRegression::inducing_inputs)
      .def_property_readonly("n", &PySparseGPRegression::n)
      .def_property_readonly("input_dim", &PySparseGPRegression::input_dim)
      .def_property_readonly("output_dim", &PySparseGPRegression::output_dim)
      .def_property_readonly("num_inducing", &PySparseGPRegression::num_inducing)
      .def_property_readonly("approx", &PySparseGPRegression::approx)
      .def_property_readonly("kernel_type", &PySparseGPRegression::kernel_type)
      .def_property_readonly("inducing_type", &PySparseGPRegression::inducing_type)
      .def_property_readonly("inducing_fixed", &PySparseGPRegression::inducing_fixed)
      .def_property_readonly("window_width", &PySparseGPRegression::window_width);
}
