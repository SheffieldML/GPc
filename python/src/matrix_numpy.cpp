#include "matrix_numpy.h"

#include <stdexcept>
#include <string>

#include "ndlexceptions.h"

CMatrix matrixFromNumpy(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
  if(arr.ndim() == 1)
  {
    const py::ssize_t n = arr.shape(0);
    if(n <= 0)
      throw ndlexceptions::Error("NumPy array must have positive length");
    auto buf = arr.unchecked<1>();
    CMatrix m(static_cast<unsigned int>(n), 1);
    for(py::ssize_t i = 0; i < n; ++i)
      m.setVal(buf(i), static_cast<unsigned int>(i), 0);
    return m;
  }
  if(arr.ndim() != 2)
    throw ndlexceptions::Error("NumPy array must be 1-D or 2-D");

  const py::ssize_t rows = arr.shape(0);
  const py::ssize_t cols = arr.shape(1);
  if(rows <= 0 || cols <= 0)
    throw ndlexceptions::Error("NumPy array must have positive shape");

  // Row-major NumPy → column-major CMatrix (same layout as SWIG-era ctor).
  auto buf = arr.unchecked<2>();
  CMatrix m(static_cast<unsigned int>(rows), static_cast<unsigned int>(cols));
  for(py::ssize_t i = 0; i < rows; ++i)
    for(py::ssize_t j = 0; j < cols; ++j)
      m.setVal(buf(i, j), static_cast<unsigned int>(i), static_cast<unsigned int>(j));
  return m;
}

py::array_t<double> matrixToNumpy(const CMatrix& m)
{
  const unsigned int rows = m.getRows();
  const unsigned int cols = m.getCols();
  py::array_t<double> out({static_cast<py::ssize_t>(rows),
                           static_cast<py::ssize_t>(cols)});
  auto buf = out.mutable_unchecked<2>();
  for(unsigned int i = 0; i < rows; ++i)
    for(unsigned int j = 0; j < cols; ++j)
      buf(i, j) = m.getVal(i, j);
  return out;
}
