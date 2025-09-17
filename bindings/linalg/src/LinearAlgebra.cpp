#include "LinearAlgebra.h"
#include <cblas.h>
#include <iostream>
#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

double LinearAlgebra::dotPureNDArray(py::array_t<double> a,
                                     py::array_t<double> b) {
  py::buffer_info buf_a = a.request(), buf_b = b.request();
  if (buf_a.ndim != 1 || buf_b.ndim != 1)
    throw std::runtime_error("dotPureNDArray expects 1D arrays");
  if (buf_a.shape[0] != buf_b.shape[0])
    throw std::runtime_error("Vectors must be the same size");

  auto *ptr_a = static_cast<double *>(buf_a.ptr);
  auto *ptr_b = static_cast<double *>(buf_b.ptr);

  double result = 0.0;
  for (ssize_t i = 0; i < buf_a.shape[0]; ++i) {
    result += ptr_a[i] * ptr_b[i];
  }
  return result;
}

double LinearAlgebra::dotBlasNDArray(py::array_t<double> a,
                                     py::array_t<double> b) {
  py::buffer_info buf_a = a.request(), buf_b = b.request();
  if (buf_a.ndim != 1 || buf_b.ndim != 1)
    throw std::runtime_error("dotBlasNDArray expects 1D arrays");
  if (buf_a.shape[0] != buf_b.shape[0])
    throw std::runtime_error("Vectors must be the same size");

  auto *ptr_a = static_cast<double *>(buf_a.ptr);
  auto *ptr_b = static_cast<double *>(buf_b.ptr);

  return cblas_ddot(buf_a.shape[0], ptr_a, 1, ptr_b, 1);
}

py::array_t<double> LinearAlgebra::matmulPureNDArray(py::array_t<double> a,
                                                     py::array_t<double> b) {
  py::buffer_info buf_a = a.request(), buf_b = b.request();
  if (buf_a.ndim != 2 || buf_b.ndim != 2)
    throw std::runtime_error("matmulPureNDArray expects 2D arrays");

  ssize_t aRows = buf_a.shape[0];
  ssize_t aCols = buf_a.shape[1];
  ssize_t bRows = buf_b.shape[0];
  ssize_t bCols = buf_b.shape[1];

  if (aCols != bRows)
    throw std::runtime_error("Matrix dimensions do not align");

  auto result = py::array_t<double>({aRows, bCols});
  py::buffer_info buf_res = result.request();

  auto *A = static_cast<double *>(buf_a.ptr);
  auto *B = static_cast<double *>(buf_b.ptr);
  auto *C = static_cast<double *>(buf_res.ptr);

  for (ssize_t i = 0; i < aRows; ++i) {
    for (ssize_t j = 0; j < bCols; ++j) {
      double sum = 0.0;
      for (ssize_t k = 0; k < aCols; ++k) {
        sum += A[i * aCols + k] * B[k * bCols + j];
      }
      C[i * bCols + j] = sum;
    }
  }

  return result;
}

py::array_t<double> LinearAlgebra::matmulBlasNDArray(py::array_t<double> a,
                                                     py::array_t<double> b) {
  py::buffer_info buf_a = a.request(), buf_b = b.request();
  if (buf_a.ndim != 2 || buf_b.ndim != 2)
    throw std::runtime_error("matmulBlasNDArray expects 2D arrays");

  ssize_t aRows = buf_a.shape[0];
  ssize_t aCols = buf_a.shape[1];
  ssize_t bRows = buf_b.shape[0];
  ssize_t bCols = buf_b.shape[1];

  if (aCols != bRows)
    throw std::runtime_error("Matrix dimensions do not align");

  auto result = py::array_t<double>({aRows, bCols});
  py::buffer_info buf_res = result.request();

  auto *A = static_cast<double *>(buf_a.ptr);
  auto *B = static_cast<double *>(buf_b.ptr);
  auto *C = static_cast<double *>(buf_res.ptr);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              aRows, bCols, aCols,
              1.0, A, aCols, B, bCols,
              0.0, C, bCols);

  return result;
}

double LinearAlgebra::dotPure(const std::vector<double> &a,
                              const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("Vectors must be of the same size");
  }

  double result = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

double LinearAlgebra::dotBlas(const std::vector<double> &a,
                              const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("Vectors must be of the same size");
  }
  return cblas_ddot(a.size(), &a[0], 1, &b[0], 1);
}

std::vector<std::vector<double>>
LinearAlgebra::matmulPure(const std::vector<std::vector<double>> &a,
                          const std::vector<std::vector<double>> &b) {
  size_t aRows = a.size();
  size_t aCols = a[0].size();
  size_t bRows = b.size();
  size_t bCols = b[0].size();

  if (aCols != bRows) {
    throw std::runtime_error("The number of columns of the 1st matrix must "
                             "equal the number of rows of the 2nd matrix");
  }

  std::vector<std::vector<double>> result(aRows, std::vector<double>(bCols, 0));

  for (size_t i = 0; i < aRows; ++i) {
    for (size_t j = 0; j < bCols; ++j) {
      for (size_t k = 0; k < aCols; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

std::vector<std::vector<double>>
LinearAlgebra::matmulBlas(const std::vector<std::vector<double>> &a,
                          const std::vector<std::vector<double>> &b) {
  size_t aRows = a.size();
  size_t aCols = a[0].size();
  size_t bRows = b.size();
  size_t bCols = b[0].size();

  if (aCols != bRows) {
    throw std::runtime_error("The number of columns of the 1st matrix must "
                             "equal the number of rows of the 2nd matrix");
  }

  // Convert 2D vectors to 1D vector for BLAS compatibility
  std::vector<double> a_flat(aRows * aCols);
  std::vector<double> b_flat(bRows * bCols);
  std::vector<double> result_flat(aRows * bCols);

  for (size_t i = 0; i < aRows; i++)
    for (size_t j = 0; j < aCols; j++)
      a_flat[i * aCols + j] = a[i][j];

  for (size_t i = 0; i < bRows; i++)
    for (size_t j = 0; j < bCols; j++)
      b_flat[i * bCols + j] = b[i][j];

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, aRows, bCols, aCols,
              1.0, &a_flat[0], aCols, &b_flat[0], bCols, 0.0, &result_flat[0],
              bCols);

  // Convert the result back to a 2D vector
  std::vector<std::vector<double>> result(aRows, std::vector<double>(bCols));
  for (size_t i = 0; i < aRows; i++)
    for (size_t j = 0; j < bCols; j++)
      result[i][j] = result_flat[i * bCols + j];

  return result;
}
