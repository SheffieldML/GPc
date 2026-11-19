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

void addDefaultCompoundKernel(CCmpndKern& kern, const CMatrix& X)
{
  CRbfKern rbf(X);
  CBiasKern bias(X);
  CWhiteKern white(X);
  kern.addKern(&rbf);
  kern.addKern(&bias);
  kern.addKern(&white);
  kern.setByPath("rbf.lengthscale", 1.0);
  kern.setByPath("rbf.variance", 1.0);
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
    int verbosity)
    : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin)), kernelType(kernel)
{
  checkXY(X, Y);

  kern = std::make_unique<CCmpndKern>(X);
  if(isEqKernelName(kernel))
  {
    CEqKern eq(X);
    CBiasKern bias(X);
    CWhiteKern white(X);
    kern->addKern(&eq);
    kern->addKern(&bias);
    kern->addKern(&white);
    kern->setByPath("ExpQuad.lengthscale", 1.0);
    kern->setByPath("ExpQuad.variance", 1.0);
    kernelType = "eq";
  }
  else if(isRbfKernelName(kernel))
  {
    CRbfKern rbf(X);
    CBiasKern bias(X);
    CWhiteKern white(X);
    kern->addKern(&rbf);
    kern->addKern(&bias);
    kern->addKern(&white);
    kern->setByPath("rbf.lengthscale", 1.0);
    kern->setByPath("rbf.variance", 1.0);
    kernelType = "rbf";
  }
  else
  {
    throw ndlexceptions::Error(
        "Unknown kernel '" + kernel + "'. Use 'rbf' or 'eq' (GPy ExpQuad).");
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
    int verbosity)
    : X(matrixFromNumpy(Xin)),
      Y(matrixFromNumpy(Yin)),
      Z(matrixFromNumpy(Zin)),
      approxType("dtcvar"),
      inducingFixed(inducing_fixed)
{
  checkXY(X, Y);

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

  kern = std::make_unique<CCmpndKern>(X);
  addDefaultCompoundKernel(*kern, X);

  noise = std::make_unique<CGaussianNoise>(&Y);
  for(unsigned int j = 0; j < Y.getCols(); ++j)
    noise->setBiasVal(0.0, j);

  model = std::make_unique<CGp>(
      kern.get(), noise.get(), &X, approxEnum, Z.getRows(), verbosity);
  model->setActiveSet(Z);
  model->setInducingFixed(inducingFixed);
  model->setBetaVal(1.0);
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

void bind_gp(py::module_& m)
{
  py::class_<PyGPRegression>(m, "GPRegression")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    const std::string&,
                    int>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("kernel") = "rbf",
           py::arg("verbosity") = 0,
           "FTC Gaussian GP with compound SE + bias + white. "
           "kernel='rbf' (default) or 'eq' / 'ExpQuad' for GPy lengthscale-first paths.")
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
                    int>(),
           py::arg("X"),
           py::arg("Y"),
           py::arg("Z"),
           py::arg("approx") = "dtcvar",
           py::arg("inducing_fixed") = true,
           py::arg("verbosity") = 0,
           "Sparse Gaussian GP with inducing inputs Z. "
           "approx='dtcvar' (default), 'fitc', or 'dtc'.")
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
      .def_property_readonly("inducing_fixed", &PySparseGPRegression::inducing_fixed);
}
