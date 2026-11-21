/* CIP-001D follow-on: pybind façade for MAP GPLVM and collapsed Bayesian GPLVM. */
#ifndef GPC_PYTHON_WRAP_GPLVM_H
#define GPC_PYTHON_WRAP_GPLVM_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "CBayesGplvm.h"
#include "CGplvm.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CModelDescription.h"
#include "CMrd.h"
#include "CNoise.h"

namespace py = pybind11;

// Owns Y + compound kernel + CScaleNoise + CGplvm (MAP; CIP-0015 / GPLVM CLI).
class PyGPLVM
{
public:
  PyGPLVM(py::array_t<double, py::array::c_style | py::array::forcecast> Y,
          unsigned int input_dim = 2,
          py::object kernel = py::str("rbf"),
          int verbosity = 0);

  ~PyGPLVM();

  void optimize(unsigned int iters = 100,
                bool messages = false,
                py::object callback = py::none());
  double objective() const;

  // CIP-0031: MAP GPLVM latent == predictive.
  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;
  std::pair<py::array_t<double>, py::array_t<double>> predict_noiseless(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;

  py::array_t<double> latent_mean() const;

  ModelDescription describe() const;
  std::string to_string() const;

  unsigned int n() const { return Y.getRows(); }
  unsigned int input_dim() const { return latentDim; }
  unsigned int output_dim() const { return Y.getCols(); }
  const std::string& kernel_type() const { return kernelType; }
  std::string approx() const { return "ftc"; }

private:
  CMatrix Y;
  unsigned int latentDim = 2;
  std::string kernelType;
  std::unique_ptr<CCmpndKern> kern;
  std::unique_ptr<CScaleNoise> noise;
  std::unique_ptr<CGplvm> model;
};

// Owns Y + compound kernel + CBayesGplvm (collapsed / psi; CIP-0017).
class PyBayesianGPLVM
{
public:
  PyBayesianGPLVM(py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                  unsigned int input_dim = 2,
                  unsigned int num_inducing = 10,
                  py::object kernel = py::str("rbf"),
                  bool ard = true,
                  int verbosity = 0);

  void optimize(unsigned int iters = 100,
                bool messages = false,
                py::object callback = py::none());
  double objective() const;
  double elbo() const;

  // CIP-0031: predict → predictive (+1/β); predict_noiseless → latent f_*.
  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;
  std::pair<py::array_t<double>, py::array_t<double>> predict_noiseless(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;

  py::array_t<double> latent_mean() const;
  py::array_t<double> latent_variance() const;
  py::array_t<double> inducing_inputs() const;

  ModelDescription describe() const;
  std::string to_string() const;

  unsigned int n() const { return Y.getRows(); }
  unsigned int input_dim() const { return latentDim; }
  unsigned int output_dim() const { return Y.getCols(); }
  unsigned int num_inducing() const { return numActive; }
  const std::string& kernel_type() const { return kernelType; }
  std::string approx() const { return "vardtc"; }

private:
  CMatrix Y;
  unsigned int latentDim = 2;
  unsigned int numActive = 10;
  std::string kernelType;
  std::unique_ptr<CCmpndKern> kern;
  std::unique_ptr<CBayesGplvm> model;
};

// Owns CMrd (CIP-0018). Kernels are handed to CMrd (per-view ownership).
class PyMRD
{
public:
  // Ylist: sequence of N×D_v arrays (V >= 2). kernel: shared string/parts, or
  // a sequence of per-view specs. Default per view is bare RBF-ARD (GPy MRD /
  // factorizeSpace).
  PyMRD(py::sequence Ylist,
        unsigned int input_dim = 2,
        unsigned int num_inducing = 10,
        py::object kernel = py::none(),
        bool ard = true,
        int verbosity = 0);

  void optimize(unsigned int iters = 100,
                bool messages = false,
                py::object callback = py::none());
  double objective() const;
  double elbo() const;

  std::pair<py::array_t<double>, py::array_t<double>> predict(
      unsigned int view,
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;
  std::pair<py::array_t<double>, py::array_t<double>> predict_noiseless(
      unsigned int view,
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest) const;

  py::array_t<double> latent_mean() const;
  py::array_t<double> latent_variance() const;
  py::array_t<double> inducing_inputs() const;
  py::array_t<double> input_scales(unsigned int view) const;

  // GPy factorize_space spirit; requires exactly two views and RBF-ARD kernels.
  py::dict factorize_space(double threshold = 0.005) const;

  ModelDescription describe() const;
  std::string to_string() const;

  unsigned int n() const;
  unsigned int input_dim() const { return latentDim; }
  unsigned int num_views() const;
  unsigned int num_inducing() const { return numActive; }
  unsigned int output_dim(unsigned int view) const;
  const std::string& kernel_type() const { return kernelType; }
  std::string approx() const { return "vardtc"; }

private:
  unsigned int latentDim = 2;
  unsigned int numActive = 10;
  std::string kernelType;
  std::unique_ptr<CMrd> model;
};

void bind_gplvm(py::module_& m);

#endif
