/* CIP-0035: pybind façade for CDeepGp (gpc.models.DeepGP). */
#ifndef GPC_PYTHON_WRAP_DEEPGP_H
#define GPC_PYTHON_WRAP_DEEPGP_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "CAdam.h"
#include "CDeepGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CModelDescription.h"
#include "CNoise.h"

namespace py = pybind11;

class PyDeepGP
{
public:
  PyDeepGP(py::array_t<double, py::array::c_style | py::array::forcecast> X,
           py::array_t<double, py::array::c_style | py::array::forcecast> Y,
           unsigned int layers = 2,
           int hidden = -1,
           py::object num_inducing = py::int_(10),
           py::object kernel = py::str("rbf"),
           const std::string& inducing = "points",
           double chi = 1.0,
           double beta = 1.0,
           bool drop_residual = false,
           const std::string& likelihood = "gaussian",
           unsigned int num_categories = 3,
           int verbosity = 0,
           bool learn_window_width = false,
           bool learn_window_widths = false,
           unsigned int num_classes = 0,
           double likelihood_eps = 1e-3,
           unsigned int likelihood_samples = 256,
           unsigned int likelihood_seed = 12345);

  void optimize(unsigned int iters = 100,
                int batch_size = -1,
                unsigned int samples = 1,
                double adam_lr = 1e-2,
                double natgrad_gamma = 0.0,
                uint64_t seed = 0,
                bool messages = false,
                py::object callback = py::none());

  double elbo() const;
  double objective() const { return -elbo(); }

  std::pair<py::array_t<double>, py::array_t<double>> predict(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
      unsigned int samples = 0,
      uint64_t seed = 0) const;

  // Noise-free predictive (latent f) mean/variance; same arguments as predict.
  std::pair<py::array_t<double>, py::array_t<double>> predict_noiseless(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
      unsigned int samples = 0,
      uint64_t seed = 0) const;

  // Discard Adam moments / step count (otherwise kept between optimize calls).
  // Class probabilities (N x K) at Xtest for a multi-class likelihood; samples/seed as predict().
  py::array_t<double> predict_proba(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xtest,
      unsigned int samples = 0,
      uint64_t seed = 0) const;

  void reset_optimizer();
  unsigned long optimizer_steps() const { return adam ? adam->getStepCount() : 0; }

  bool learn_window_width() const { return learnWindowWidth && !learnWindowWidths; }
  bool learn_window_widths() const { return learnWindowWidths; }

  void save(const std::string& path) const;
  static std::unique_ptr<PyDeepGP> load(const std::string& path);

  void set_drop_residual(bool val);
  // Per-layer kernel parameters, named as CGp::paramPaths (e.g. 'kern.weight_variance').
  std::vector<std::string> layer_param_paths(unsigned int ell) const;
  double get_layer_param(unsigned int ell, const std::string& path) const;
  // KL(q(v) || N(0, I)) summed over the layer's outputs (diagnostic).
  double layer_kl(unsigned int ell) const;
  void set_layer_param(unsigned int ell, const std::string& path, double val);
  bool drop_residual() const;
  double jacobian_spread(unsigned int n_probe = 32, uint64_t probe_seed = 0) const;

  ModelDescription describe() const;
  std::string to_string() const;

  unsigned int n() const;
  unsigned int input_dim() const;
  unsigned int output_dim() const;
  unsigned int num_layers() const;
  unsigned int hidden_dim() const;
  const std::string& kernel_type() const { return kernelType; }
  const std::string& inducing_type() const { return inducingType; }
  std::vector<std::string> hermite_warnings() const
  {
    return model ? model->hermiteWarnings() : std::vector<std::string>();
  }
  const std::string& likelihood_type() const { return likelihoodType; }
  double chi() const;
  std::string approx() const { return "deepgp"; }

private:
  PyDeepGP();  // for load()

  CMatrix X;
  CMatrix Y;
  std::string kernelType;
  std::string inducingType;
  std::string likelihoodType;
  std::vector<std::unique_ptr<CKern> > ownedKerns;
  std::unique_ptr<CNoise> ownedNoise;
  std::unique_ptr<CDeepGp> model;
  std::unique_ptr<CAdam> adam;  // persists moments across optimize() calls
  bool learnWindowWidth = false;
  bool learnWindowWidths = false;
};

void bind_deepgp(py::module_& m);

#endif
