#include "wrap_gp.h"

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

// FTC / point-inducing compound: signal + bias + white (GPy-familiar).
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
    const std::string& kernel,
    int verbosity,
    unsigned int arccos_order)
    : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin)), kernelType(kernel)
{
  checkXY(X, Y);
  if(arccos_order > 1)
    throw ndlexceptions::Error("arccos_order must be 0 or 1");

  const SignalKind kind = parseSignalKind(kernel);
  kernelType = signalKindLabel(kind);

  kern = std::make_unique<CCmpndKern>(X);
  addFtcStyleCompound(*kern, X, kind, arccos_order);

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
    const std::string& kernel,
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

  const SignalKind kind = parseSignalKind(kernel);
  kernelType = signalKindLabel(kind);

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
  if(weightSpace)
    addWeightSpaceCompound(*kern, X, kind, arccos_order);
  else
    addFtcStyleCompound(*kern, X, kind, arccos_order);

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
                    const std::string&,
                    int,
                    unsigned int>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("kernel") = "rbf",
           py::arg("verbosity") = 0,
           py::arg("arccos_order") = 0,
           "FTC Gaussian GP with compound signal + bias + white. "
           "kernel='rbf' (default), 'eq' / 'ExpQuad', 'mlp' (Williams arcsin), "
           "or 'arccos' / 'arccosine' (order via arccos_order=0|1).")
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
                    const std::string&,
                    const std::string&,
                    double,
                    unsigned int>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("Z"),
           py::arg("approx") = "dtcvar",
           py::arg("inducing_fixed") = true,
           py::arg("verbosity") = 0,
           py::arg("kernel") = "rbf",
           py::arg("inducing") = "points",
           py::arg("window_width") = -1.0,
           py::arg("arccos_order") = 0,
           "Sparse Gaussian GP. approx='dtcvar'|'fitc'|'dtc'. "
           "kernel='rbf'|'eq'|'mlp'|'arccos'. "
           "inducing='points' (default) or 'weightspace' (CIP-0028; needs mlp/arccos). "
           "For weight-space, Z rows set M; optional window_width overrides ridge s.")
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
