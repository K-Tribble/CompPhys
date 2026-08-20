#include "linalg/matrix.hpp"
#include "linalg/linalg_common.hpp"
#include "linalg/linalg_interop.hpp"
#include "linalg/linalg_solve.hpp"
#include "linalg/vec.hpp"
#include <stdexcept>
#include <cmath>
#include <utility>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <iostream>

namespace linalg {

template <Scalar T>
Matrix<T>::Matrix(u32 rows, u32 cols, T init)
    : rows_(rows), cols_(cols), data_(rows * cols, init) {}

template <Scalar T>
Matrix<T>::Matrix(u32 rows, u32 cols, std::vector<T> data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {}

template <Scalar T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init) 
    : rows_(init.size()), cols_(init.size() ? init.begin()->size() : 0) {
    data_.reserve(rows_ * cols_);
    for (const auto& row : init) {
        if (row.size() != cols_) {
            throw std::invalid_argument("ragged initializer list");
        }
        data_.insert(data_.end(), row.begin(), row.end());
     }
}

template <Scalar T>
Matrix<T> Matrix<T>::zeros(u32 rows, u32 cols) {return Matrix<T>(rows, cols, T{0.0});}

template <Scalar T>
Matrix<T> Matrix<T>::ones(u32 rows, u32 cols) {return Matrix<T>(rows, cols, T{1.0});}

template <Scalar T>
Matrix<T> Matrix<T>::identity(u32 n) {
    Matrix<T> m(n, n, T{0});
    
    for (u32 i = 0; i < n; ++i) {
        m(i, i) = T{1.0};
    }

    return m;
}

template <Scalar T>
Matrix<T> Matrix<T>::diagonal(const std::vector<T>& diag) {
    u32 n = diag.size();
    Matrix m(n, n);

    for (u32 i = 0; i < n; ++i) {
        m(i, i) = diag[i];
    }

    return m;
}

template <Scalar T>
template <Scalar U>
Matrix<U> Matrix<T>::cast() const {
    std::vector<U> out;
    out.reserve(data_.size());
    for (T& v : data_) {
        out.emplace_back(scalar_cast<U>(v));
    }

    return Matrix<U>(rows_, cols_, std::move(out));
}

template <Scalar T>
u32 Matrix<T>::linearIndex(u32 r, u32 c) const {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("matrix index out of range");
    }
    return r * cols_ + c;
}

template <Scalar T>
T& Matrix<T>::operator()(u32 r, u32 c) {
    return data_[linearIndex(r, c)];
}

template <Scalar T>
T Matrix<T>::operator()(u32 r, u32 c) const {
    return data_[linearIndex(r, c)];
}

template <Scalar T>
Vec<T> Matrix<T>::operator()(u32 i) const {
    std::vector<T> buff; 
    buff.reserve(cols_);

    for (u32 j = 0; j < cols_; ++j) {
        buff.emplace_back(data_[linearIndex(i, j)]);
    }

    return Vec(buff);
}

template <Scalar T>
Vec<T> Matrix<T>::getCol(u32 j) const {
    std::vector<T> buff;
    buff.reserve(rows_);

    for (u32 i = 0; i < rows_; ++ i) {
        buff.emplace_back(data_[linearIndex(i, j)]);
    }

    return Vec(buff);
}

template <Scalar T>
u32 Matrix<T>::rows() const {
    return rows_;
}

template <Scalar T>
u32 Matrix<T>::cols() const {
    return cols_;
}

template <Scalar T>
std::vector<Vec<T>> Matrix<T>::getRows() const {
    std::vector<Vec<T>> rows;
    rows.reserve(rows_);

    for (u32 i = 0; i < rows_; ++i) {
        std::vector<T> row;
        row.reserve(cols_);

        for (u32 j = 0; j < cols_; ++j) {
            row.push_back(data_[linearIndex(i, j)]);
        }

        rows.emplace_back(Vec(row));
    }

    return rows;
}

template <Scalar T>
std::vector<Vec<T>> Matrix<T>::getCols() const {
    std::vector<Vec<T>> cols;
    cols.reserve(cols_);

    for (u32 j = 0; j < cols_; ++j) {
        std::vector<T> col;
        col.reserve(rows_);

        for (u32 i = 0; i < rows_; ++i) {
            col.push_back(data_[linearIndex(i, j)]);
        }

        cols.emplace_back(Vec(col));
    }

    return cols;
}

template <Scalar T>
std::array<u32, 2> Matrix<T>::shape() const {
    return {rows_, cols_};
}

template <Scalar T>
void Matrix<T>::swapRows(u32 row1, u32 row2) {
    if (row1 == row2) {
        return;
    }

    if (row1 >= rows_ || row2 >= rows_) {
        throw std::out_of_range("row index out of range");
    }

    for (u32 c = 0; c < cols_; ++c) {
        std::swap(data_[linearIndex(row1, c)], data_[linearIndex(row2, c)]);
    }
}

template <Scalar T>
void Matrix<T>::swapCols(u32 col1, u32 col2) {
    if (col1 == col2) {
        return;
    }

    if (col1 >= cols_ || col2 >= cols_) {
        throw std::out_of_range("column index out of range");
    }

    for (u32 r = 0; r < rows_; ++r) {
        std::swap(data_[linearIndex(r, col1)], data_[linearIndex(r, col2)]);
    }
}

template <Scalar T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("matmul dimension mismatch");
    Matrix<T> result(rows_, other.cols_);
    matmulInto(other, result);
    return result;
}

template <Scalar T>
Matrix<T> Matrix<T>::operator*(T s) const {
    Matrix result(*this);

    detail::scaleInPlace(result.data_, s);

    return result;
}

template <Scalar T>
Matrix<T>& Matrix<T>::operator*=(T s) {
    detail::scaleInPlace(data_, s);

    return *this;
}

template <Scalar T>
Matrix<T> Matrix<T>::operator/(T s) const {
    Matrix<T> result(*this);

    detail::scaleInPlace(result.data_, 1.0 / s);

    return result;
}

template <Scalar T>
Matrix<T>& Matrix<T>::operator/=(T s) {
    detail::scaleInPlace(data_, 1.0 / s);

    return *this;
}

template <Scalar T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
    return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
}

template <Scalar T>
bool Matrix<T>::isApprox(const Matrix<T>& other, RealType<T> absTol, RealType<T> relTol) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) return false;

    return detail::approxEqual(data_, other.data_, absTol, relTol);
}

template <Scalar T>
bool Matrix<T>::isZero(RealType<T> absTol) const {
    return detail::isZero(data_, absTol);
}

template <Scalar T>
bool Matrix<T>::isSymmetric(RealType<T> absTol, RealType<T> relTol) const {
    if (rows_ != cols_) return false;

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = i + 1; j < cols_; ++j) {
            T a = (*this)(i, j);
            T b = (*this)(j, i);
            RealType<T> diff = std::abs(a - b);
            RealType<T> largest = std::max(std::abs(a), std::abs(b));
            if (diff > std::max(absTol, relTol * largest)) return false;
        }
    }

    return true;
}

template <Scalar T>
bool Matrix<T>::isHermitian(RealType<T> absTol, RealType<T> relTol) const {
    if (rows_ != cols_) return false;

    for (u32 i = 0; i < rows_; ++i) {
        // check for real diagonal entries
        if (std::abs(std::imag((*this)(i, i))) > kDefaultAbsTol) {
            return false;
        }
        for (u32 j = i + 1; j < cols_; ++j) {
            T a = (*this)(i, j);
            T b = conjugate((*this)(j, i));
            RealType<T> diff = std::abs(a - b);
            RealType<T> largest = std::max(std::abs(a), std::abs(b));
            if (diff > std::max(absTol, relTol * largest)) return false;
        }
    }

    return true;
}

template <Scalar T>
Matrix<T> Matrix<T>::absDiff(const Matrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Shape mismatch");
    }

    return Matrix(rows_, cols_, detail::elementWise(data_, other.data_, [](T a, T b) {return std::abs(a - b);}));
}

template <Scalar T>
Matrix<T> Matrix<T>::conj() const {
    Matrix<T> result(rows_, cols_);
    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = 0; j < cols_; ++j) {
            result(i, j) = conjugate((*this)(i,j));
        }
    }

    return result;
}

template <Scalar T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols_, rows_);

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = 0; j < cols_; ++j) {
            result(j,i) = (*this)(i,j);
        }
    }

    return result;
}

template <Scalar T>
Matrix<T> Matrix<T>::adjoint() const {
    return conj().transpose();
}

template <Scalar T>
T Matrix<T>::determinant() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Must be a square matrix to calculate determinant");
    }

    if (rows_ == 1) {
        return data_[0];
    }

    if (rows_ == 2) {
        return data_[0] * data_[3] - data_[1] * data_[2];
    }

    const u32 n = rows_;
    Matrix<T> A = *this;
    T det = 1.0;
    const RealType<T> tolerance = std::numeric_limits<RealType<T>>::epsilon() * kSingularPivotTol;

    for (u32 i = 0; i < n; ++i) {
        u32 pivot_row = i;
        for (u32 r = i + 1; r < n; ++r) {
            if (std::abs(A(r, i)) > std::abs(A(pivot_row, i))) {
                pivot_row = r;
            }
        }

        if (pivot_row != i) {
            A.swapRows(i, pivot_row);
            det *= -1.0;
        }

        const T pivot = A(i, i);
        if (std::abs(pivot) <= tolerance) {
            return 0.0;
        }

        det *= pivot;

        for (u32 r = i + 1; r < n; ++r) {
            const T factor = A(r, i) / pivot;
            if (factor == 0.0) {
                continue;
            }

            for (u32 c = i + 1; c < n; ++c) {
                A(r, c) -= factor * A(i, c);
            }
            A(r, i) = 0.0;
        }
    }

    return det;
}

template <Scalar T>
Matrix<T> Matrix<T>::getMinor(u32 i, u32 j) const {
    if (rows_ < 2 || cols_ < 2) {
        throw std::invalid_argument("minor requires at least a 2x2 matrix");
    }
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("minor index out of range");
    }

    Matrix<T> minor(rows_ - 1, cols_ - 1);

    for (u32 r = 0; r < rows_; ++r) {
        if (r == i) {
            continue;
        }
        for (u32 c = 0; c < cols_; ++c) {
            if (c == j) {
                continue;
            }

            const u32 dstRow = (r > i) ? r - 1 : r;
            const u32 dstCol = (c > j) ? c - 1 : c;
            minor(dstRow, dstCol) = (*this)(r, c);
        }
    }

    return minor;
}

template <Scalar T>
Matrix<T> Matrix<T>::getCofactorMatrix() const {
    Matrix<T> cofactor(rows_, cols_);

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = 0; j < cols_; ++j) {
            Matrix<T> minor = getMinor(i, j);
            T minorDet = minor.determinant();
            const RealType<T> sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            cofactor(i, j) = minorDet * sign;
        }
    }

    return cofactor;
}

template <Scalar T>
LUResult<T> Matrix<T>::LUDecomp() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("matrix must be square to LU decompose");
    }

    u32 n = rows_;
    Matrix<T> A = *this;
    Matrix<T> L = identity(n);
    Matrix<T> U(n, n, 0.0);
    std::vector<u32> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    u32 numSwaps = 0;

    for (u32 i = 0; i < n; ++i) {
        u32 pivotRow = i;
        RealType<T> pivotVal = std::abs(A(i, i));
        for (u32 k = i + 1; k < n; ++k) {
            RealType<T> v = std::abs(A(k, i));
            if (v > pivotVal) {
                pivotVal = v;
                pivotRow = k;
            }
        }

        if (pivotVal < kDefaultAbsTol) {
            throw std::runtime_error("cannot decompose singular matrix");
        }

        if (pivotRow != i) {
            A.swapRows(i, pivotRow);
            for (u32 col = 0; col < i; ++col) {
                std::swap(L(i, col), L(pivotRow, col));
            }
            std::swap(perm[i], perm[pivotRow]);
            ++numSwaps;
        }

        for (u32 j = i; j < n; ++j) {
            T sum = A(i, j);
            for (u32 k = 0; k < i; ++k) {
                sum -= L(i, k) * U(k, j);
            }
            U(i, j) = sum;
        }

        for (u32 j = i + 1; j < n; ++j) {
            T sum = A(j, i);
            for (u32 k = 0; k < i; ++k) {
                sum -= L(j, k) * U(k, i);
            }
            L(j, i) = sum / U(i, i);
        }
    }

    // build P such that P * A_original = L * U
    Matrix<T> P(n, n);
    for (u32 i = 0; i < n; ++i) {
        P(i, perm[i]) = 1.0;
    }

    return {std::move(P), std::move(L), std::move(U), numSwaps};
}

template <Scalar T>
QRResult<T> Matrix<T>::QRDecomp() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("matrix must be square for this QR decomposition to work");
    }

    Matrix<T> R(*this);
    Matrix<T> Q = identity(rows_);

    auto padMatrix = [&](Matrix<T>& Q, u32 n) {
        Matrix<T> res = identity(n);
        u32 m = Q.rows_;

        for (u32 i = n - m; i < n; ++i) {
            for (u32 j = n - m; j < n; ++j) {
                res(i, j) = Q(i - n + m, j - n + m);
            }
        }

        return res;
    };

    for (u32 i = 0; i < rows_ - 1; ++i) {
        Vec<T> Ri_col = R.sliceByRows(i, rows_).getCol(i);
        RealType<T> col_norm = Ri_col.norm();
        if (col_norm < kDefaultAbsTol) continue;

        T x0 = Ri_col(0);
        RealType<T> abs_x0 = std::abs(x0);
        T phase = (abs_x0 == RealType<T>(0)) ? T(1) : (x0 / abs_x0);
        T alpha = phase * col_norm;

        Vec<T> v = Ri_col;
        v(0) += alpha;

        RealType<T> v_norm = v.norm();
        if (v_norm < kDefaultAbsTol) continue;

        v.normalize();
        Matrix<T> Qn_prepad = identity(rows_ - i) - (v.outer(v.conj()) * T(2.0));
        Matrix<T> Qn = padMatrix(Qn_prepad, rows_);
        R = Qn * R;
        Q = Q * Qn.adjoint();
    }

    return {std::move(Q), std::move(R)};
}

template <Scalar T>
EigenResult<T> Matrix<T>::hermitianEigenQR(u32 maxIter, RealType<T> tol) const {
    if (rows_ != cols_) {
        throw std::invalid_argument("matrix must be square to have eigenvalues");
    }
    if (!isHermitian()) {
        throw std::invalid_argument("matrix must be symmetric for this eigenvalue algortihm");
    }

    u32 n = rows_;
    Matrix<T> A(*this);
    Matrix<T> eigenvectors = identity(n);
    std::vector<RealType<T>> eigenvalues(n, 0.0);

    u32 m = n;

    while (m > 1) {
        for (u32 iter = 0; iter < maxIter; ++iter) {
            // Wilkonsin shift from the trailing 2x2 block of active mxm submatrix
            RealType<T> a = std::real(A(m - 2, m - 2));
            T b = A(m - 2, m - 1);
            RealType<T> b_abs_sq = std::norm(b);
            RealType<T> d = std::real(A(m - 1, m - 1));

            RealType<T> delta = (a - d) / 2.0;
            RealType<T> mu;
            if (delta == 0.0 && b == 0.0) {
                mu = d;
            } else {
                RealType<T> sign = (delta >= 0.0) ? 1.0 : -1.0;
                mu = d - (sign * b_abs_sq) / (std::abs(delta) + std::sqrt(delta * delta + b_abs_sq)); 
            }

            Matrix<T> activeBlock = A.sliceByRows(0, m).sliceByCols(0, m);
            Matrix<T> shifted = activeBlock - (identity(m) * T(mu));
            QRResult<T> qr = shifted.QRDecomp();
            Matrix<T> newBlock = (qr.R * qr.Q) + (identity(m) * T(mu));

            for (u32 i = 0; i < m; ++i) {
                for (u32 j = 0; j < m; ++j) {
                    A(i, j) = newBlock(i, j);
                }
            }

            Matrix<T> Qpad = identity(n);
            for (u32 i = 0; i < m; ++i) {
                for (u32 j = 0; j < m; ++j) {
                    Qpad(i, j) = qr.Q(i, j);
                }
            }

            eigenvectors = eigenvectors * Qpad;

            if (std::abs(A(m - 1, m - 2)) < tol) {
                break;
            }
        }

        eigenvalues[m - 1] = std::real(A(m - 1, m - 1));
        --m;
    }

    eigenvalues[0] = std::real(A(0, 0));

    return {std::move(eigenvalues), std::move(eigenvectors)};
}

template <Scalar T>
T Matrix<T>::trace() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("matrix must be square to have a trace");
    }

    T tr = 0.0;
    for (u32 i = 0; i < rows_; ++i) {
        tr += data_[linearIndex(i, i)];
    }

    return tr;
}

template <Scalar T>
T Matrix<T>::diagProduct() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("matrix must be square to have a main diagonal");
    }

    T prod = 1.0;
    for (u32 i = 0; i < rows_; ++i) {
        prod *= data_[linearIndex(i, i)];
    }

    return prod;
}

template <Scalar T>
std::vector<T> Matrix<T>::getDiag() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("no diagonal for non-square matrix");
    }

    std::vector<T> diag;
    diag.reserve(rows_);

    for (u32 i = 0; i < rows_; ++i) {
        diag.emplace_back(data_[linearIndex(i, i)]);
    }

    return diag;
}

template <Scalar T>
Matrix<T> Matrix<T>::getLower() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("needs to be square matrix to have defined main diagonal");
    }
    Matrix<T> lower(rows_, cols_, 0.0);

    for (u32 i = 0; i < rows_; ++i) { 
        for (u32 j = 0; j < i; ++j) {
            lower(i, j) = data_[linearIndex(i, j)];
        }
    }

    return lower;
}

template <Scalar T>
Matrix<T> Matrix<T>::getUpper() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("needs to be square matrix to have defined main diagonal");
    }
    Matrix<T> upper(rows_, cols_, 0.0);
    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = i + 1; j < cols_; ++j) {
            upper(i, j) = data_[linearIndex(i, j)];
        }
    }

    return upper;
}

template <Scalar T>
Matrix<T> Matrix<T>::cofactorInversion() const {
    T det = (*this).determinant();
    if (rows_ != cols_) {
        throw std::invalid_argument("Cannot invert rectangular matrix");
    }
    if (std::abs(det) < kDefaultAbsTol) {
        throw std::invalid_argument("Cannot invert singular matrix");
    }
    return (*this).getCofactorMatrix().transpose() / det;
}

template <Scalar T>
Matrix<T> Matrix<T>::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Cannot invert rectangular matrix");
    }

    const u32 n = rows_;

    Matrix<T> A = *this;
    Matrix<T> I = identity(n);

    const RealType<T> tolerance = std::numeric_limits<RealType<T>>::epsilon() * kSingularPivotTol;

    for (u32 j = 0; j < n; ++j) {
        u32 pivot_row = j;
        RealType<T> maxVal = std::abs(A(j, j));
        for(u32 i = j + 1; i < n; ++i) {
            if (std::abs(A(i, j)) > maxVal) {
                maxVal = std::abs(A(i, j));
                pivot_row = i;
            }
        }

        if (maxVal < tolerance) {
                throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }

        A.swapRows(j, pivot_row);
        I.swapRows(j, pivot_row);

        T pivot = A(j, j);

        for (u32 i = 0; i < n; ++i) {
            A(j, i) /= pivot;
            I(j, i) /= pivot;
        }

        for (u32 i = 0; i < n; ++i) {
            if (i == j) continue;
            T factor = A(i, j);
            for (u32 col = 0; col < n; ++col) {
                A(i, col) -= factor * A(j, col);
                I(i, col) -= factor * I(j, col);
            }
        }
    }

    return I;
}

template <Scalar T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Shape mismatch");
    }
    return Matrix<T>(rows_, cols_, detail::elementWise(data_, other.data_, std::plus<T>()));
}

template <Scalar T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::plus<T>());

    return *this;
}

template <Scalar T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Shape mismatch");
    }
        return Matrix(rows_, cols_, detail::elementWise(data_, other.data_, std::minus<T>()));
}

template <Scalar T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::minus<T>());

    return *this;
}

template <Scalar T>
Matrix<T> Matrix<T>::hadamard(const Matrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }

    return Matrix(rows_, cols_, detail::elementWise(data_, other.data_, std::multiplies<T>()));
}

template <Scalar T>
Matrix<T>& Matrix<T>::hadamardInPlace(const Matrix<T>& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::multiplies<T>());

    return *this;
}

template <Scalar T>
Matrix<T> Matrix<T>::sliceByRows(u32 start, u32 finish) const {
    if (start >= finish) {
        throw std::invalid_argument("finish must be greater than start");
    }
    if (finish > rows_) {
        throw std::invalid_argument("finish must not be greater than number of rows");
    }

    u32 new_rows = finish - start;

    Matrix<T> result(new_rows, cols_);

    for (u32 i = start; i < finish; ++i) {
        for (u32 j = 0; j < cols_; ++j) {
            result(i - start, j) = (*this)(i, j);
        }
    }

    return result;
}

template <Scalar T>
Matrix<T> Matrix<T>::sliceByCols(u32 start, u32 finish) const {
    if (start >= finish) {
        throw std::invalid_argument("finish must be greater than start");
    }
    if (finish > cols_) {
        throw std::invalid_argument("finish must not be greater than number of cols");
    }

    u32 new_cols = finish - start;

    Matrix<T> result(rows_, new_cols);

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = start; j < finish; ++ j) {
            result(i, j - start) = (*this)(i, j);
        }
    }

    return result;
}

template <Scalar T>
std::pair<RealType<T>, Vec<T>> Matrix<T>::largestEigenPair(u32 power) const {
    if (rows_ != cols_) {
        throw std::invalid_argument("matrix must be square to have eigenvalues");
    }

    Vec<T> v = Vec<T>::random(rows_);

    for (u32 i = 0; i < power; ++i) {
        v = (*this) * v;
        v.normalize();
    }

    v.normalize(); // normalized eigenvector
    RealType<T> eigenVal = std::real(v.inner(((*this) * v)));

    return {eigenVal, v};
}

template <Scalar T>
std::pair<RealType<T>, Vec<T>> Matrix<T>::smallestEigenPair(u32 power) const {
    if (rows_ != cols_){
        throw std::invalid_argument("matrix must be square to have eigenvalues");
    }

    Vec<T> v = Vec<T>::random(rows_);

    LUResult<T> A_res = LUDecomp();

    for (u32 i = 0; i < power; ++i) {
        v = solve::lu(A_res, v);
        v.normalize();
    }

    v.normalize();
    RealType<T> eigenVal = 1.0 / std::real(v.inner(solve::lu(A_res, v)));

    return {eigenVal, v};
}

template <Scalar T>
std::pair<RealType<T>, Vec<T>> Matrix<T>::eigenPairClosestTo(RealType<T> alpha, u32 power) const {
    if (rows_ != cols_) {
        throw std::invalid_argument("matrix must be square to have eigenvalues");
    }

    Matrix<T> A_prime = (*this) - (identity(rows_) * alpha);
    LUResult<T> A_prime_res = A_prime.LUDecomp();

    Vec<T> v = Vec<T>::random(rows_);
    
    for (u32 i = 0; i < power; ++i) {
        v = solve::lu(A_prime_res, v);
        v.normalize();
    }

    v.normalize();
    RealType<T> eigenVal = alpha + 1.0 / std::real(v.inner(solve::lu(A_prime_res, v)));
    
    return {eigenVal, v};
}

template <Scalar T>
T Matrix<T>::sumElements() const {
    return detail::sumElements(data_);
}

// axis = 0 means sum up each column, leaving a row vector
// axis = 1 means sum up each row, leaving a column vector
template <Scalar T>
Matrix<T> Matrix<T>::sum(u32 axis) const {
    Matrix<T> result = axis == 0 ? Matrix<T>(1, (*this).cols_) : Matrix<T>(((*this).rows_), 1);

    if (!axis) {
        for (u32 j = 0; j < cols_; ++j) {
            for (u32 i = 0; i < rows_; ++i) {
                result(0, j) += (*this)(i, j); 
            }
        }
    } else {
        for (u32 i = 0; i < rows_; ++i) {
            for (u32 j = 0; j < cols_; ++j) {
                result(i, 0) += (*this)(i, j);
            }
        }
    }

    return result;
}

template <Scalar T>
Matrix<T>& Matrix<T>::matmulInto(const Matrix<T>& other, Matrix<T>& out) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("matmul dimension mismatch");
    if (rows_ != out.rows_ || other.cols_ != out.cols_)
        throw std::invalid_argument("output shape mismatch");

    const u32 m = rows_;
    const u32 n = other.cols_;
    const u32 k = cols_;
    
    for (u32 i = 0; i < m; ++i) {
        for (u32 j = 0; j < n; ++j) {
            T sum = 0.0;
            for (u32 p = 0; p < k; ++p) {
                sum += data_[i * cols_ + p] * other.data_[p * other.cols_ + j];
            }
            out.data_[i * out.cols_ + j] = sum;
        }
    }

    return out;
}

template <Scalar T>
T Matrix<T>::max() const {
    return detail::maxElement(data_);
}


template <Scalar T>
T Matrix<T>::min() const {
    return detail::minElement(data_);
}

template <Scalar T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
    if constexpr (is_complex_v<T>) {
        for (u32 i = 0; i < m.rows(); ++i) {
            u32 count = 0;
            for (u32 j = 0; j < m.cols(); ++j) {
                if (count == m.cols() - 1) {
                    os << m(i, j).real() << "+" << m(i, j).imag() << "i" << std::endl;
                } else {
                    os << m(i, j).real() << "+" << m(i, j).imag() << "i, ";
                }

                ++count;
            }
        } 
    }
    else {
        for (u32 i = 0; i < m.rows(); ++i) {
            u32 count = 0;
            for (u32 j = 0; j < m.cols(); ++j) {
                if (count == m.cols() - 1) {
                    os << m(i, j) << std::endl;
                } else {
                    os << m(i, j) << ", ";
                }

                ++count;
            }
        }
    }

    return os;

} 

}// namespace linalg