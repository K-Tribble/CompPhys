#pragma once

#include <vector>
#include <functional>
#include <array>
#include <cstddef>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "linalg/linalg_common.hpp"
#include "types.hpp"
#include "constants.hpp"


namespace linalg {

template <Scalar T> class Vec;

template <Scalar T> class Matrix;

template <Scalar T> struct LUResult;
template <Scalar T> struct QRResult;
template <Scalar T> struct EigenResult;

template <Scalar T>
class Matrix {
public:
    static constexpr RealType<T> kSingularPivotTol = 100.0; // multiplier on machine epsilon 

    Matrix(u32 rows, u32 cols, T init = T{});

    Matrix(u32 rows, u32 cols, std::vector<T> data);

    Matrix(std::initializer_list<std::initializer_list<T>> init);

    Matrix() = default;

    static Matrix<T> zeros(u32 rows, u32 cols);
    static Matrix<T> ones(u32 rows, u32 cols);
    static Matrix<T> identity(u32 n);
    static Matrix<T> diagonal(const std::vector<T>& diag);

    T& operator()(u32 r, u32 c);
    T operator()(u32 r, u32 c) const;
    // Return ith row as a vector
    Vec<T> operator()(u32 i) const; 
    // Return jth col as a vector
    Vec<T> getCol(u32 j) const;

    u32 rows() const;
    u32 cols() const;

    std::vector<Vec<T>> getRows() const;
    std::vector<Vec<T>> getCols() const;

    std::array<u32, 2> shape() const;

    void swapRows(u32 row1, u32 row2);
    void swapCols(u32 col1, u32 col2);

    Matrix<T> operator*(const Matrix<T>& other) const;
    Matrix<T> operator*(T s) const;
    Matrix<T>& operator*=(T s);

    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T> operator/(T s) const;
    Matrix<T>& operator/=(T s);

    bool operator==(const Matrix<T>& other) const;
    bool isApprox(const Matrix<T>& other, RealType<T> absTol = kDefaultAbsTol, RealType<T> relTol = kDefaultRelTol) const;
    bool isZero(RealType<T> absTol = kDefaultAbsTol) const;
    bool isSymmetric(RealType<T> absTol = kDefaultAbsTol, RealType<T> relTol = kDefaultRelTol) const;
    bool isHermitian(RealType<T> absTol = kDefaultAbsTol, RealType<T> relTol = kDefaultRelTol) const;

    Matrix<T> absDiff(const Matrix<T>& other) const;

    Matrix<T> hadamard(const Matrix<T>& other) const;
    Matrix<T>& hadamardInPlace(const Matrix<T>& other);

    Matrix<T> conj() const;
    Matrix<T> transpose() const;
    Matrix<T> adjoint() const;

    T determinant() const;

    Matrix<T> getMinor(u32 i, u32 j) const;

    Matrix<T> getCofactorMatrix() const;

    // perform LU decomposition of the matrix with doolittle choice 
    LUResult<T> LUDecomp() const;
    // perform QR decomposition of the matrix with hosueholder reflections
    QRResult<T> QRDecomp() const;
    // get eigenvalues and eigenvectors of a symmetric matrix
    EigenResult<T> hermitianEigenQR(u32 maxIter = 1000, RealType<T> tol = kDefaultAbsTol) const;

    T trace() const;
    // product of diagonal elements
    T diagProduct() const;
    // Get diagonal elements
    std::vector<T> getDiag() const;
    // Get a Matrix with all elements below main diagonal
    Matrix<T> getLower() const;
    // Get a Matrix with all elements above main diagonal
    Matrix<T> getUpper() const;

    Matrix<T> inverse() const;
    Matrix<T> cofactorInversion() const;

    // The following methdos only find the largest modulus real eigenvalue and its associated real eigenvector
    // returns the eigenvalue with the largest modulus and its associated normalized eigenvector
    std::pair<RealType<T>, Vec<T>> largestEigenPair(u32 power = 50) const;
    // returns the eigenvalue with the smalles modulus and its associated normalized eigenvector
    std::pair<RealType<T>, Vec<T>> smallestEigenPair(u32 power = 50) const;
    // Returns the eigenvalue closest to a value alpha and its associated normalized eigenvector
    std::pair<RealType<T>, Vec<T>> eigenPairClosestTo(RealType<T> alpha, u32 power = 50) const;

    T sumElements() const;

    Matrix<T> sum(u32 axis = 0) const;

    // Return a matrix with just rows from start (inclusive) to finish (exclusive)
    Matrix<T> sliceByRows(u32 start, u32 finish) const;
    // Same but for columns
    Matrix<T> sliceByCols(u32 start, u32 finish) const;

    Matrix<T>& matmulInto(const Matrix<T>& other, Matrix<T>& out) const;

    RealType<T> max() const;
    RealType<T> min() const;

private:
    u32 rows_, cols_;
    std::vector<T> data_;

    u32 linearIndex(u32 r, u32 c) const;
};

template <Scalar T>
std::ostream& operator<<(std::ostream&, const Matrix<T>&);

template <Scalar T>
struct LUResult {
    Matrix<T> P, L, U;
    u32 numSwaps;
};

template <Scalar T>
struct QRResult {
    Matrix<T> Q, R;
};

template <Scalar T>
struct EigenResult {
    std::vector<RealType<T>> eigenvalues;
    Matrix<T> eigenvectors; // collumn i is the eigenvector for eigenvalues[i]
};

} // namespace linalg

#include "linalg/matrix.tpp"