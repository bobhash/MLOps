from typing import List, Callable
import time

import numpy as np

import linalg


def py_matrix_multiply(mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
    if len(mat1[0]) != len(mat2):
        raise ValueError("Uncompatible matrix shapes")
    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            sum_val = 0
            for k in range(len(mat2)):
                sum_val += mat1[i][k] * mat2[k][j]
            row_result.append(sum_val)
        result.append(row_result)
    return result


def test_timings(func: Callable, *args):
    _ = func(*args)
    start_time = time.time()
    _ = func(*args)
    end_time = time.time()
    return round(end_time - start_time, 5)


def compare(matrix_size: int) -> None:
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    list_a = matrix_a.tolist()
    list_b = matrix_b.tolist()

    print(
        "Mat mul (Pure Python), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(py_matrix_multiply, list_a, list_b)
        )
    )
    print(
        "Mat mul (Pure C++), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulPure, list_a, list_b)
        )
    )
    print(
        "Mat mul (C++ BLAS), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulBlas, list_a, list_b)
        )
    )
    print(
        "Mat mul (Pure C++ NDArray), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulPureNDArray, matrix_a, matrix_b)
        )
    )
    print(
        "Mat mul (C++ BLAS NDArray), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulBlasNDArray, matrix_a, matrix_b)
        )
    )
    print(
        "Mat mul (Python numpy), size={0}x{0}: {1} seconds\n".format(
            matrix_size, test_timings(np.dot, np.array(list_a), np.array(list_b))
        )
    )


if __name__ == "__main__":
    for size in [10, 50, 100, 300, 500, 700]:
        compare(size)
