#include "wrap_likelihood.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <vector>

#include "CNoise.h"
#include "ndlexceptions.h"

namespace py = pybind11;

namespace {

py::array_t<double> noise_log_density(
    const std::string& type,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> f,
    py::dict params)
{
  auto ybuf = y.unchecked<1>();
  auto fbuf = f.unchecked<1>();
  if(ybuf.shape(0) != fbuf.shape(0))
    throw ndlexceptions::Error("noise_log_density: y and f must have the same length");
  const py::ssize_t n = ybuf.shape(0);
  py::array_t<double> out(n);
  auto obuf = out.mutable_unchecked<1>();

  auto getd = [&](const char* key, double def) -> double {
    if(!params.contains(key))
      return def;
    return py::cast<double>(params[key]);
  };

  for(py::ssize_t i = 0; i < n; ++i)
  {
    double logp = 0.0, d1 = 0.0, W = 0.0, dW = 0.0;
    const double yi = ybuf(i);
    const double fi = fbuf(i);
    if(type == "gaussian" || type == "Gaussian")
    {
      const double var = getd("variance", 1.0);
      const double e = yi - fi;
      logp = -ndlutil::HALFLOGTWOPI - 0.5 * std::log(var) - 0.5 * e * e / var;
    }
    else if(type == "skew_gaussian" || type == "SkewGaussian")
    {
      ndlnoise::skewGaussianLaplaceDerivs(logp, d1, W, dW, yi, fi,
                                          getd("variance", 1.0), getd("alpha", 0.0));
    }
    else if(type == "skew_exponential" || type == "SkewExponential")
    {
      ndlnoise::skewExponentialLaplaceDerivs(logp, d1, W, dW, yi, fi,
                                             getd("rate", 1.0), getd("alpha", 0.0));
    }
    else if(type == "student_t" || type == "StudentT")
    {
      ndlnoise::studentTLaplaceDerivs(logp, d1, W, dW, yi, fi,
                                      getd("variance", 1.0), getd("deg_free", 5.0));
    }
    else if(type == "poisson" || type == "Poisson")
    {
      ndlnoise::poissonLogLaplaceDerivs(logp, d1, W, dW, yi, fi);
    }
    else if(type == "binomial" || type == "Binomial")
    {
      ndlnoise::binomialProbitLaplaceDerivs(logp, d1, W, dW, yi, fi,
                                            getd("num_trials", 1.0));
    }
    else if(type == "exponential" || type == "Exponential")
    {
      ndlnoise::exponentialLogLaplaceDerivs(logp, d1, W, dW, yi, fi);
    }
    else if(type == "gamma" || type == "Gamma")
    {
      ndlnoise::gammaLogLaplaceDerivs(logp, d1, W, dW, yi, fi, getd("beta", 1.0));
    }
    else if(type == "probit" || type == "Bernoulli" || type == "bernoulli")
    {
      ndlnoise::bernoulliProbitLaplaceDerivs(logp, d1, W, dW, yi, fi);
    }
    else
    {
      throw ndlexceptions::Error("noise_log_density: unknown type '" + type + "'");
    }
    obuf(i) = logp;
  }
  return out;
}

std::vector<std::string> likelihood_types()
{
  return {
      "gaussian", "probit", "skew_gaussian", "skew_exponential", "student_t",
      "poisson", "binomial", "exponential", "gamma", "ordered", "ncnm", "scale",
  };
}

}  // namespace

void bind_likelihood(py::module_& m)
{
  m.def(
      "noise_log_density",
      &noise_log_density,
      py::arg("type"),
      py::arg("y"),
      py::arg("f"),
      py::arg("params") = py::dict(),
      "Elementwise log p(y|f) for a CIP-003A likelihood type.\n\n"
      "params keys depend on type (variance, alpha, deg_free, rate, beta, num_trials).");
  m.def(
      "likelihood_types",
      &likelihood_types,
      "Stream / façade likelihood type names known to GPc.");
}
