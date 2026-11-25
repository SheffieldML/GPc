#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>

#include "CMatrix.h"
#include "matrix_numpy.h"
#include "ndlexceptions.h"
#include "wrap_deepgp.h"
#include "wrap_gp.h"
#include "wrap_gplvm.h"
#include "wrap_likelihood.h"

namespace py = pybind11;

namespace {

void registerExceptions(py::module_& m)
{
  // Deliberately leaked: a function-local static would be destroyed after
  // interpreter finalisation and call dec_ref without the GIL (abort on exit).
  static py::exception<ndlexceptions::Error>& exc =
      *new py::exception<ndlexceptions::Error>(m, "GpcError");
  py::register_exception_translator([](std::exception_ptr p) {
    try
    {
      if(p)
        std::rethrow_exception(p);
    }
    catch(const ndlexceptions::Error& e)
    {
      const std::string msg = e.getMessage().empty() ? e.what() : e.getMessage();
      PyErr_SetString(exc.ptr(), msg.c_str());
    }
  });
}

}  // namespace

PYBIND11_MODULE(_gpc, m)
{
  m.doc() =
      "GPc Python extension (_gpc): NumPy ↔ CMatrix and CGp model bindings "
      "(CIP-001D / CIP-002F). Prefer import gpc for the public façade.";

  registerExceptions(m);

  py::class_<CMatrix>(m, "Matrix", "Column-major dense matrix wrapping GPc CMatrix.")
      .def(py::init<>(), "Empty matrix.")
      .def(py::init<unsigned int, unsigned int>(),
           py::arg("rows"),
           py::arg("cols"),
           "Allocate rows×cols (uninitialized storage per CMatrix).")
      .def_static("from_numpy", &matrixFromNumpy, py::arg("array"),
                  "Copy a NumPy array (C-order) into a column-major CMatrix.")
      .def("to_numpy", &matrixToNumpy,
           "Copy to a new NumPy array in C-order (row-major).")
      .def_property_readonly("rows", &CMatrix::getRows, "Number of rows.")
      .def_property_readonly("cols", &CMatrix::getCols, "Number of columns.")
      .def("get",
           [](const CMatrix& self, unsigned int i, unsigned int j) {
             return self.getVal(i, j);
           },
           py::arg("i"),
           py::arg("j"),
           "Return entry (i, j).")
      .def("set",
           [](CMatrix& self, unsigned int i, unsigned int j, double val) {
             self.setVal(val, i, j);
           },
           py::arg("i"),
           py::arg("j"),
           py::arg("val"),
           "Set entry (i, j) to val.");

  m.def("matrix_from_numpy",
        &matrixFromNumpy,
        py::arg("array"),
        "Copy NumPy array into a CMatrix.");
  m.def("matrix_to_numpy",
        &matrixToNumpy,
        py::arg("matrix"),
        "Copy CMatrix to a NumPy array.");

  bind_gp(m);
  bind_gplvm(m);
  bind_deepgp(m);
  bind_likelihood(m);
}
