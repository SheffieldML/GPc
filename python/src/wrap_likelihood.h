/* CIP-003A: pybind helpers for gpc.likelihoods density oracles. */
#ifndef GPC_PYTHON_WRAP_LIKELIHOOD_H
#define GPC_PYTHON_WRAP_LIKELIHOOD_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_likelihood(py::module_& m);

#endif
