#pragma once

#include <vector>
#include <functional>
#include <array>
#include <cstddef>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "linalg_common.hpp"
#include "vec.hpp"


namespace linalg {

class Matrix;
struct LUResult;

class Matrix {
public:
    static constexpr d64 kSingularPivotTol = 100.0; // multiplier on machine epsilon 

    Matrix(u32 rows, u32 cols, d64 init = 0.0);

    Matrix(u32 rows, u32 cols, std::vector<d64> data);

    Matrix(std::initializer_list<std::initializer_list<d64>> init);

    static Matrix zeros(u32 rows, u32 cols);
    static Matrix ones(u32 rows, u32 cols);
    static Matrix identity(u32 n);
    static Matrix diagonal(const std::vector<d64>& diag);

    double& operator()(u32 r, u32 c);
    double operator()(u32 r, u32 c) const;
    Vec operator()(u32 i) const; // Return ith row as a vector

    u32 rows() const;
    u32 cols() const;

    std::vector<Vec> getRows() const;
    std::vector<Vec> getCols() const;

    std::array<u32, 2> shape() const;

    void swapRows(u32 row1, u32 row2);
    void swapCols(u32 col1, u32 col2);

    Matrix operator*(const Matrix& other) const;
    Matrix operator*(d64 s) const;
    Matrix& operator*=(d64 s);

    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);
    Matrix operator/(d64 s) const;
    Matrix& operator/=(d64 s);

    bool operator==(const Matrix& other) const;
    bool isApprox(const Matrix& other, d64 absTol = linalg::kDefaultAbsTol, d64 relTol = linalg::kDefaultRelTol) const;
    bool isZero(d64 absTol = linalg::kDefaultAbsTol) const;
    bool isSymmetric(d64 absTol = linalg::kDefaultAbsTol, d64 relTol = linalg::kDefaultRelTol) const;

    Matrix absDiff(const Matrix& other) const;

    Matrix hadamard(const Matrix& other) const;
    Matrix& hadamardInPlace(const Matrix& other);

    Matrix transpose() const;

    d64 determinant() const;

    Matrix getMinor(u32 i, u32 j) const;

    Matrix getCofactorMatrix() const;

    // perform LU decomposition of the matrix with doolittle choice 
    LUResult LUDecomp() const;

    d64 trace() const;
    // product of diagonal elements
    d64 diagProduct() const;

    Matrix inverse() const;
    Matrix cofactorInversion() const;

    d64 sumElements() const;

    Matrix sum(u32 axis = 0) const;

    // Return a matrix with just rows from start (inclusive) to finish (exclusive)
    Matrix sliceByRows(u32 start, u32 finish) const;
    // Same but for columns
    Matrix sliceByCols(u32 start, u32 finish) const;

    Matrix& matmulInto(const Matrix& other, Matrix& out) const;

    d64 max() const;
    d64 min() const;

private:
    u32 rows_, cols_;
    std::vector<d64> data_;

    u32 linearIndex(u32 r, u32 c) const;
};

std::ostream& operator<<(std::ostream&, const Matrix&);

struct LUResult {
    Matrix P, L, U;
    u32 numSwaps;
};

} // namespace linalg