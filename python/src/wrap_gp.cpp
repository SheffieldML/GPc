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

} // namespace

PyGPRegression::PyGPRegression(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xin,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
    const std::string& kernel,
    int verbosity)
    : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin)), kernelType(kernel)
{
  if(X.getRows() != Y.getRows())
    throw ndlexceptions::Error("X and Y must have the same number of rows");
  if(Y.getCols() < 1)
    throw ndlexceptions::Error("Y must have at least one column");

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
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != X.getCols())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");

  CMatrix mu(Xt.getRows(), Y.getCols());
  CMatrix varSigma(Xt.getRows(), Y.getCols());
  model->posteriorMeanVar(mu, varSigma, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varSigma)};
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
      .def_property_readonly("kernel_type", &PyGPRegression::kernel_type);
}
