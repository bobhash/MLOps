#include "LinearAlgebra.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(linalg_core, m) {
  m.doc() = R"doc(
    Python bindings for LinearAlgebra library
  )doc";

  py::class_<LinearAlgebra>(m, "LinearAlgebra")
      .def_static("dotPure", &LinearAlgebra::dotPure)
      .def_static("dotBlas", &LinearAlgebra::dotBlas)
      .def_static("matmulPure", &LinearAlgebra::matmulPure)
      .def_static("matmulBlas", &LinearAlgebra::matmulBlas)
      .def_static("dotPureNDArray", &LinearAlgebra::dotPureNDArray,
                  "Dot product, pure C++, accepting numpy arrays")
      .def_static("dotBlasNDArray", &LinearAlgebra::dotBlasNDArray,
                  "Dot product, BLAS, accepting numpy arrays")
      .def_static("matmulPureNDArray", &LinearAlgebra::matmulPureNDArray,
                  "Matrix multiplication, pure C++, numpy arrays")
      .def_static("matmulBlasNDArray", &LinearAlgebra::matmulBlasNDArray,
                  "Matrix multiplication, BLAS, numpy arrays");
}
