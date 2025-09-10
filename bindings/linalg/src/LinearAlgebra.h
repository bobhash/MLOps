#pragma once

#include <vector>
#include <pybind11/numpy.h>

class LinearAlgebra {
public:
  static double dotPure(const std::vector<double> &a,
                        const std::vector<double> &b);
  static double dotBlas(const std::vector<double> &a,
                        const std::vector<double> &b);
  static std::vector<std::vector<double>>
  matmulPure(const std::vector<std::vector<double>> &a,
             const std::vector<std::vector<double>> &b);
  static std::vector<std::vector<double>>
  matmulBlas(const std::vector<std::vector<double>> &a,
             const std::vector<std::vector<double>> &b);
  static double dotPureNDArray(pybind11::array_t<double> a,
                               pybind11::array_t<double> b);
  static double dotBlasNDArray(pybind11::array_t<double> a,
                               pybind11::array_t<double> b);
  static pybind11::array_t<double>
  matmulPureNDArray(pybind11::array_t<double> a,
                    pybind11::array_t<double> b);
  static pybind11::array_t<double>
  matmulBlasNDArray(pybind11::array_t<double> a,
                    pybind11::array_t<double> b);
};
