#ifndef GPC_PYTHON_WRAP_GP_H
#define GPC_PYTHON_WRAP_GP_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"

namespace py = pybind11;

// Owns data + kernel + noise + model for FTC Gaussian regression (CIP-001D Phase B).
class PyGPRegression
{
public:
  PyGPRegression(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                 py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                 const std::string& kernel = "rbf",
                 int verbosity = 0);

  void set_by_path(const std::string& path, double val);
  double get_by_path(const std::string& path) const;
  std::vector<std::string> param_paths() const;

  void optimize(unsigned int iters = 100);
  double objective() const;

  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;

  unsigned int n() const { return X.getRows(); }
  unsigned int input_dim() const { return X.getCols(); }
  unsigned int output_dim() const { return Y.getCols(); }
  const std::string& kernel_type() const { return kernelType; }
  std::string approx() const { return "ftc"; }

private:
  CMatrix X;
  CMatrix Y;
  std::string kernelType;
  std::unique_ptr<CCmpndKern> kern;
  std::unique_ptr<CGaussianNoise> noise;
  std::unique_ptr<CGp> model;
};

// Sparse DTCVAR / FITC / DTC regression with inducing inputs Z (CIP-001D Phase C).
class PySparseGPRegression
{
public:
  PySparseGPRegression(
      py::array_t<double, py::array::c_style | py::array::forcecast> X,
      py::array_t<double, py::array::c_style | py::array::forcecast> Y,
      py::array_t<double, py::array::c_style | py::array::forcecast> Z,
      const std::string& approx = "dtcvar",
      bool inducing_fixed = true,
      int verbosity = 0);

  void set_by_path(const std::string& path, double val);
  double get_by_path(const std::string& path) const;
  std::vector<std::string> param_paths() const;

  void optimize(unsigned int iters = 100);
  double objective() const;

  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;

  unsigned int n() const { return X.getRows(); }
  unsigned int input_dim() const { return X.getCols(); }
  unsigned int output_dim() const { return Y.getCols(); }
  unsigned int num_inducing() const { return Z.getRows(); }
  std::string approx() const { return approxType; }
  bool inducing_fixed() const { return inducingFixed; }
  py::array_t<double> inducing_inputs() const;

private:
  CMatrix X;
  CMatrix Y;
  CMatrix Z;
  std::string approxType;
  bool inducingFixed;
  std::unique_ptr<CCmpndKern> kern;
  std::unique_ptr<CGaussianNoise> noise;
  std::unique_ptr<CGp> model;
};

void bind_gp(py::module_& m);

#endif /* GPC_PYTHON_WRAP_GP_H */
