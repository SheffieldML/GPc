/* CIP-0016 A3: pybind wrapper over the canonical C++ CInterchange. */
#ifndef GPC_PYTHON_WRAP_INTERCHANGE_H
#define GPC_PYTHON_WRAP_INTERCHANGE_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_interchange(py::module_& m);

#endif
