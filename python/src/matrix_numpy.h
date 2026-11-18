#ifndef GPC_PYTHON_MATRIX_NUMPY_H
#define GPC_PYTHON_MATRIX_NUMPY_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "CMatrix.h"

namespace py = pybind11;

// NumPy C-order (row-major) ↔ CMatrix column-major copy helpers (CIP-001D Phase A).
CMatrix matrixFromNumpy(py::array_t<double, py::array::c_style | py::array::forcecast> arr);
py::array_t<double> matrixToNumpy(const CMatrix& m);

#endif /* GPC_PYTHON_MATRIX_NUMPY_H */
