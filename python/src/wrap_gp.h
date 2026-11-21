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
#include "CModelDescription.h"
#include "CNoise.h"
#include "COptimisable.h"

namespace py = pybind11;

// Owns data + kernel + noise + model for FTC Gaussian regression (CIP-001D Phase B / F).
class PyGPRegression
{
public:
  PyGPRegression(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                 py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                 py::object kernel = py::str("rbf"),
                 int verbosity = 0,
                 unsigned int arccos_order = 0);

  void set_by_path(const std::string& path, double val);
  double get_by_path(const std::string& path) const;
  std::vector<std::string> param_paths() const;

  void optimize(unsigned int iters = 100,
                bool messages = false,
                py::object callback = py::none());
  double objective() const;

  // CIP-0031: predict → predictiveMeanVar (y_*); predict_noiseless → latent (f_*).
  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;
  std::pair<py::array_t<double>, py::array_t<double>> predict_noiseless(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;

  std::string to_string() const;
  ModelDescription describe() const;

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

// Sparse DTCVAR / FITC / DTC regression (CIP-001D Phase C / F; CIP-0028 weight-space).
class PySparseGPRegression
{
public:
  PySparseGPRegression(
      py::array_t<double, py::array::c_style | py::array::forcecast> X,
      py::array_t<double, py::array::c_style | py::array::forcecast> Y,
      py::array_t<double, py::array::c_style | py::array::forcecast> Z,
      const std::string& approx = "dtcvar",
      bool inducing_fixed = false,
      int verbosity = 0,
      py::object kernel = py::str("rbf"),
      const std::string& inducing = "points",
      double window_width = -1.0,
      unsigned int arccos_order = 0,
      bool learn_window_width = false);

  void set_by_path(const std::string& path, double val);
  double get_by_path(const std::string& path) const;
  std::vector<std::string> param_paths() const;

  void optimize(unsigned int iters = 100,
                bool messages = false,
                py::object callback = py::none());
  double objective() const;

  // CIP-0031: predict → predictiveMeanVar (y_*); predict_noiseless → latent (f_*).
  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;
  std::pair<py::array_t<double>, py::array_t<double>> predict_noiseless(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;

  std::string to_string() const;
  ModelDescription describe() const;

  unsigned int n() const { return X.getRows(); }
  unsigned int input_dim() const { return X.getCols(); }
  unsigned int output_dim() const { return Y.getCols(); }
  unsigned int num_inducing() const { return numActive; }
  std::string approx() const { return approxType; }
  const std::string& kernel_type() const { return kernelType; }
  const std::string& inducing_type() const { return inducingType; }
  bool inducing_fixed() const { return inducingFixed; }
  bool learn_window_width() const { return learnWindowWidth; }
  double window_width() const;
  py::array_t<double> inducing_inputs() const;

private:
  CMatrix X;
  CMatrix Y;
  CMatrix Z;
  unsigned int numActive = 0;
  std::string approxType;
  std::string kernelType;
  std::string inducingType;
  bool inducingFixed;
  bool learnWindowWidth;
  std::unique_ptr<CCmpndKern> kern;
  std::unique_ptr<CGaussianNoise> noise;
  std::unique_ptr<CGp> model;
};

void bind_gp(py::module_& m);

#endif /* GPC_PYTHON_WRAP_GP_H */
