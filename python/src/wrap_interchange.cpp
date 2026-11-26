#include "wrap_interchange.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "CInterchange.h"
#include "CJson.h"
#include "matrix_numpy.h"

namespace {

// A model built from a gp-scuderia JSON document by CInterchange. All format
// mappings (ARD split, noise, refusals) live in C++; Python only passes JSON.
class PyInterchangeModel
{
public:
  explicit PyInterchangeModel(const std::string& json)
      : impl(CInterchange::fromJson(CJson::parse(json)))
  {
  }

  double objective() const { return -impl->logLikelihood(); }
  std::string to_json() const { return impl->toJson().dump(); }
  void set_allow_white_and_noise(bool v) { impl->gp().setAllowWhiteAndLikelihoodNoise(v); }
  bool allow_white_and_noise() const { return impl->gp().isAllowWhiteAndLikelihoodNoise(); }
  void set_by_path(const std::string& p, double v) { impl->gp().setByPath(p, v); }
  double get_by_path(const std::string& p) const { return impl->gp().getByPath(p); }
  std::vector<std::string> param_paths() const { return impl->gp().paramPaths(); }

  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xnew) const
  {
    CMatrix Xn = matrixFromNumpy(Xnew);
    std::vector<double> mean, var;
    impl->predict(Xn, mean, var);
    py::array_t<double> m(mean.size()), v(var.size());
    std::copy(mean.begin(), mean.end(), m.mutable_data());
    std::copy(var.begin(), var.end(), v.mutable_data());
    return {m, v};
  }

private:
  std::unique_ptr<CInterchange> impl;
};

}  // namespace

void bind_interchange(py::module_& m)
{
  py::class_<PyInterchangeModel>(m, "InterchangeModel",
      "GP model built from an interchange JSON document by the C++ CInterchange.")
      .def(py::init<const std::string&>(), py::arg("json"))
      .def("objective", &PyInterchangeModel::objective)
      .def("to_json", &PyInterchangeModel::to_json)
      .def("predict", &PyInterchangeModel::predict, py::arg("Xnew"))
      .def("set_allow_white_and_noise", &PyInterchangeModel::set_allow_white_and_noise)
      .def("allow_white_and_noise", &PyInterchangeModel::allow_white_and_noise)
      .def("set_by_path", &PyInterchangeModel::set_by_path)
      .def("get_by_path", &PyInterchangeModel::get_by_path)
      .def("param_paths", &PyInterchangeModel::param_paths);
}
