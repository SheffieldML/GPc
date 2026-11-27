#include "wrap_likelihood.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
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

// CIP-003E Phase 6: multi-class block oracles for the Python façade.
std::unique_ptr<CNoise> makeMulticlass(const std::string& type, unsigned int K, double eps,
                                       unsigned int samples, unsigned int seed, CMatrix* target)
{
  if(type == "robustmax")
    return std::unique_ptr<CNoise>(new CRobustMaxNoise(target, K, eps));
  if(type == "softmax")
    return std::unique_ptr<CNoise>(new CSoftmaxNoise(target, K, samples, seed));
  throw ndlexceptions::Error("multiclass: type must be 'robustmax' or 'softmax', got '" + type + "'");
}

CMatrix matrix2d(const py::array_t<double, py::array::c_style | py::array::forcecast>& a)
{
  auto b = a.unchecked<2>();
  CMatrix M(static_cast<unsigned int>(b.shape(0)), static_cast<unsigned int>(b.shape(1)));
  for(py::ssize_t i = 0; i < b.shape(0); i++)
    for(py::ssize_t j = 0; j < b.shape(1); j++)
      M.setVal(b(i, j), static_cast<unsigned int>(i), static_cast<unsigned int>(j));
  return M;
}

py::array_t<double> multiclass_expected_loglik(
    const std::string& type,
    py::array_t<double, py::array::c_style | py::array::forcecast> labels,
    py::array_t<double, py::array::c_style | py::array::forcecast> mu,
    py::array_t<double, py::array::c_style | py::array::forcecast> var,
    unsigned int K, double eps, unsigned int samples, unsigned int seed)
{
  if(mu.ndim() != 2 || var.ndim() != 2 || mu.shape(1) != K || var.shape(1) != K
     || mu.shape(0) != var.shape(0))
    throw ndlexceptions::Error("multiclass_expected_loglik: mu and var must both be N x K");
  const unsigned int N = static_cast<unsigned int>(mu.shape(0));
  CMatrix lab = (labels.ndim() == 2) ? matrix2d(labels) : CMatrix(N, 1);
  if(labels.ndim() == 1)
  {
    auto l = labels.unchecked<1>();
    if(static_cast<unsigned int>(l.shape(0)) != N)
      throw ndlexceptions::Error("multiclass_expected_loglik: labels need N entries");
    for(unsigned int i = 0; i < N; i++)
      lab.setVal(l(i), i, 0);
  }
  CMatrix Y = ndlnoise::oneHotLabels(lab, K);
  if(Y.getRows() != N)
    throw ndlexceptions::Error("multiclass_expected_loglik: labels need N rows");
  std::unique_ptr<CNoise> noise = makeMulticlass(type, K, eps, samples, seed, &Y);
  py::array_t<double> out(N);
  auto o = out.mutable_unchecked<1>();
  auto m = mu.unchecked<2>();
  auto v = var.unchecked<2>();
  std::vector<double> yr(K), mr(K), vr(K), gm(K), gv(K);
  for(unsigned int i = 0; i < N; i++)
  {
    for(unsigned int k = 0; k < K; k++)
    {
      yr[k] = Y.getVal(i, k);
      mr[k] = m(i, k);
      vr[k] = v(i, k);
    }
    o(i) = noise->expectedLogLikelihoodBlock(&yr[0], &mr[0], &vr[0], 0, &gm[0], &gv[0]);
  }
  return out;
}

py::array_t<double> multiclass_predict_proba(
    const std::string& type,
    py::array_t<double, py::array::c_style | py::array::forcecast> mu,
    py::array_t<double, py::array::c_style | py::array::forcecast> var,
    unsigned int K, double eps, unsigned int samples, unsigned int seed)
{
  if(mu.ndim() != 2 || var.ndim() != 2 || mu.shape(1) != K || var.shape(1) != K
     || mu.shape(0) != var.shape(0))
    throw ndlexceptions::Error("multiclass_predict_proba: mu and var must both be N x K");
  const unsigned int N = static_cast<unsigned int>(mu.shape(0));
  CMatrix Y(N, K);
  Y.zeros();
  std::unique_ptr<CNoise> noise = makeMulticlass(type, K, eps, samples, seed, &Y);
  py::array_t<double> out({static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(K)});
  auto o = out.mutable_unchecked<2>();
  auto m = mu.unchecked<2>();
  auto v = var.unchecked<2>();
  std::vector<double> mr(K), vr(K), p(K);
  for(unsigned int i = 0; i < N; i++)
  {
    for(unsigned int k = 0; k < K; k++)
    {
      mr[k] = m(i, k);
      vr[k] = v(i, k);
    }
    noise->predictBlock(&mr[0], &vr[0], 0, &p[0]);
    for(unsigned int k = 0; k < K; k++)
      o(i, k) = p[k];
  }
  return out;
}

std::vector<std::string> likelihood_types()
{
  return {
      "gaussian", "probit", "skew_gaussian", "skew_exponential", "student_t",
      "poisson", "binomial", "exponential", "gamma", "ordered", "ncnm", "scale",
      "robustmax", "softmax",
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
  m.def("multiclass_expected_loglik", &multiclass_expected_loglik, py::arg("type"),
        py::arg("labels"), py::arg("mu"), py::arg("var"), py::arg("num_classes"),
        py::arg("eps") = 1e-3, py::arg("samples") = 256, py::arg("seed") = 12345,
        "Per-row E_q[log p(y|f)] for 'robustmax' or 'softmax' with q = N(mu, diag(var)) (N x K). "
        "labels: integer labels (N) or one-hot (N x K).");
  m.def("multiclass_predict_proba", &multiclass_predict_proba, py::arg("type"),
        py::arg("mu"), py::arg("var"), py::arg("num_classes"), py::arg("eps") = 1e-3,
        py::arg("samples") = 256, py::arg("seed") = 12345,
        "Class probabilities (N x K) for marginals N(mu, var) under robustmax or softmax.");
  m.def(
      "likelihood_types",
      &likelihood_types,
      "Stream / façade likelihood type names known to GPc.");
}
