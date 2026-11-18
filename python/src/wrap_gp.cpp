#include "wrap_gp.h"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "matrix_numpy.h"
#include "ndlexceptions.h"

PyGPRegression::PyGPRegression(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xin,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yin,
    int verbosity)
    : X(matrixFromNumpy(Xin)), Y(matrixFromNumpy(Yin))
{
  if(X.getRows() != Y.getRows())
    throw ndlexceptions::Error("X and Y must have the same number of rows");
  if(Y.getCols() < 1)
    throw ndlexceptions::Error("Y must have at least one column");

  kern = std::make_unique<CCmpndKern>(X);
  {
    CRbfKern rbf(X);
    CBiasKern bias(X);
    CWhiteKern white(X);
    kern->addKern(&rbf);
    kern->addKern(&bias);
    kern->addKern(&white);
  }
  kern->setByPath("rbf.lengthscale", 1.0);
  kern->setByPath("rbf.variance", 1.0);

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

void PyGPRegression::optimize(unsigned int iters, bool messages, py::object callback)
{
  model->clearOptimiserObservers();
  model->setDefaultTextProgress(messages);
  // CGp::optimise prints the final model when verbosity>0; keep quiet unless messages.
  model->setVerbosity(messages ? 3 : 0);

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
  CMatrix Xt = matrixFromNumpy(Xtest);
  if(Xt.getCols() != X.getCols())
    throw ndlexceptions::Error("Xtest input dimension does not match training X");

  CMatrix mu(Xt.getRows(), Y.getCols());
  CMatrix varSigma(Xt.getRows(), Y.getCols());
  model->posteriorMeanVar(mu, varSigma, Xt);
  return {matrixToNumpy(mu), matrixToNumpy(varSigma)};
}

ModelDescription PyGPRegression::describe() const
{
  return model->describe();
}

std::string PyGPRegression::to_string() const
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
      .def_readonly("log_likelihood", &ModelDescription::logLikelihood);

  py::class_<PyGPRegression>(m, "GPRegression")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    int>(),
           py::arg("X"), py::arg("Y"), py::arg("verbosity") = 0,
           "FTC Gaussian GP with compound RBF + bias + white kernel.")
      .def("set_by_path", &PyGPRegression::set_by_path, py::arg("path"), py::arg("value"))
      .def("get_by_path", &PyGPRegression::get_by_path, py::arg("path"))
      .def("param_paths", &PyGPRegression::param_paths)
      .def("optimize", &PyGPRegression::optimize,
           py::arg("iters") = 100,
           py::arg("messages") = false,
           py::arg("callback") = py::none(),
           "Run the C++ optimiser. Optional outer-iteration callback (CIP-001E).")
      .def("objective", &PyGPRegression::objective)
      .def("predict", &PyGPRegression::predict, py::arg("Xtest"))
      .def("describe", &PyGPRegression::describe)
      .def("__str__", &PyGPRegression::to_string)
      .def("__repr__", &PyGPRegression::to_string)
      .def_property_readonly("n", &PyGPRegression::n)
      .def_property_readonly("input_dim", &PyGPRegression::input_dim)
      .def_property_readonly("output_dim", &PyGPRegression::output_dim);
}
