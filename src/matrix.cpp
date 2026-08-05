#include "matrix.hpp"
#include "linalg_common.hpp"
#include <stdexcept>
#include <cmath>
#include <utility>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <iostream>

namespace linalg {

Matrix::Matrix(u32 rows, u32 cols, d64 init)
    : rows_(rows), cols_(cols), data_(rows * cols, init) {}

Matrix::Matrix(u32 rows, u32 cols, std::vector<d64> data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<d64>> init) 
    : rows_(init.size()), cols_(init.size() ? init.begin()->size() : 0) {
    data_.reserve(rows_ * cols_);
    for (const auto& row : init) {
        if (row.size() != cols_) {
            throw std::invalid_argument("ragged initializer list");
        }
        data_.insert(data_.end(), row.begin(), row.end());
     }
}

Matrix Matrix::zeros(u32 rows, u32 cols) {return Matrix(rows, cols, 0.0);}

Matrix Matrix::ones(u32 rows, u32 cols) {return Matrix(rows, cols, 1.0);}

Matrix Matrix::identity(u32 n) {
    Matrix m(n, n, 0.0);
    
    for (u32 i = 0; i < n; ++i) {
        m(i, i) = 1.0;
    }

    return m;
}

Matrix Matrix::diagonal(const std::vector<d64>& diag) {
    u32 n = diag.size();
    Matrix m(n, n, 1.0);

    for (u32 i = 0; i < n; ++i) {
        m(i, i) = diag[i];
    }

    return m;
}

u32 Matrix::linearIndex(u32 r, u32 c) const {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("matrix index out of range");
    }
    return r * cols_ + c;
}

double& Matrix::operator()(u32 r, u32 c) {
    return data_[linearIndex(r, c)];
}

double Matrix::operator()(u32 r, u32 c) const {
    return data_[linearIndex(r, c)];
}

u32 Matrix::rows() const {
    return rows_;
}

u32 Matrix::cols() const {
    return cols_;
}

std::vector<Vec> Matrix::getRows() const {
    std::vector<Vec> rows;

    for (u32 i = 0; i < rows_; ++i) {
        std::vector<d64> row_i(cols_);
        for (u32 j = 0; j < cols_; ++j) {
            row_i.push_back(data_[linearIndex(i, j)]);
        }
        rows.push_back(Vec(row_i));
    }
    return rows;
}

std::vector<Vec> Matrix::getCols() const {
    std::vector<Vec> cols;

    for (u32 j = 0; j < cols_; ++j) {
        std::vector<d64> col_j(cols_);
        for (u32 i = 0; i < rows_; ++i) {
            col_j.push_back(data_[linearIndex(i, j)]);
        }
        cols.push_back(Vec(col_j));
    }
    return cols;
}

std::array<u32, 2> Matrix::shape() const {
    return {rows_, cols_};
}

void Matrix::swapRows(u32 row1, u32 row2) {
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

void Matrix::swapCols(u32 col1, u32 col2) {
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

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("matmul dimension mismatch");
    Matrix result(rows_, other.cols_);
    matmulInto(other, result);
    return result;
}

Matrix Matrix::operator*(d64 s) const {
    Matrix result(*this);

    detail::scaleInPlace(result.data_, s);

    return result;
}

Matrix& Matrix::operator*=(d64 s) {
    detail::scaleInPlace(data_, s);

    return *this;
}

Matrix Matrix::operator/(d64 s) const {
    Matrix result(*this);

    detail::scaleInPlace(result.data_, 1.0 / s);

    return result;
}

Matrix& Matrix::operator/=(d64 s) {
    detail::scaleInPlace(data_, 1.0 / s);

    return *this;
}

bool Matrix::operator==(const Matrix& other) const {
    return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
}

bool Matrix::isApprox(const Matrix& other, d64 absTol, d64 relTol) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) return false;

    return detail::approxEqual(data_, other.data_, absTol, relTol);
}

bool Matrix::isZero(d64 absTol) const {
    return detail::isZero(data_, absTol);
}

bool Matrix::isSymmetric(d64 absTol, d64 relTol) const {
    if (rows_ != cols_) return false;

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = i + 1; j < cols_; ++j) {
            d64 a = (*this)(i, j);
            d64 b = (*this)(j, i);
            d64 diff = std::fabs(a - b);
            d64 largest = std::max(std::fabs(a), std::fabs(b));
            if (diff > std::max(absTol, relTol * largest)) return false;
        }
    }

    return true;
}

Matrix Matrix::absDiff(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Shape mismatch");
    }

    return Matrix(rows_, cols_, detail::elementWise(data_, other.data_, [](d64 a, d64 b) {return std::fabs(a - b);}));
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = 0; j < cols_; ++j) {
            result(j,i) = (*this)(i,j);
        }
    }

    return result;
}

d64 Matrix::determinant() const {
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
    Matrix A = *this;
    d64 det = 1.0;
    const d64 tolerance = std::numeric_limits<d64>::epsilon() * kSingularPivotTol;

    for (u32 i = 0; i < n; ++i) {
        u32 pivot_row = i;
        for (u32 r = i + 1; r < n; ++r) {
            if (std::fabs(A(r, i)) > std::fabs(A(pivot_row, i))) {
                pivot_row = r;
            }
        }

        if (pivot_row != i) {
            A.swapRows(i, pivot_row);
            det *= -1.0;
        }

        const d64 pivot = A(i, i);
        if (std::fabs(pivot) <= tolerance) {
            return 0.0;
        }

        det *= pivot;

        for (u32 r = i + 1; r < n; ++r) {
            const d64 factor = A(r, i) / pivot;
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

Matrix Matrix::getMinor(u32 i, u32 j) const {
    if (rows_ < 2 || cols_ < 2) {
        throw std::invalid_argument("minor requires at least a 2x2 matrix");
    }
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("minor index out of range");
    }

    Matrix minor(rows_ - 1, cols_ - 1);

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

Matrix Matrix::getCofactorMatrix() const {
    Matrix cofactor(rows_, cols_);

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = 0; j < cols_; ++j) {
            Matrix minor = getMinor(i, j);
            d64 minorDet = minor.determinant();
            const d64 sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            cofactor(i, j) = minorDet * sign;
        }
    }

    return cofactor;
}

Matrix Matrix::inverse() const {
    d64 det = (*this).determinant();
    if (std::fabs(det) < kDefaultAbsTol) {
        throw std::invalid_argument("Cannot invert singular matrix");
    }
    return (*this).getCofactorMatrix().transpose() / det;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Shape mismatch");
    }
    return Matrix(rows_, cols_, detail::elementWise(data_, other.data_, std::plus<d64>()));
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::plus<d64>());

    return *this;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Shape mismatch");
    }
        return Matrix(rows_, cols_, detail::elementWise(data_, other.data_, std::minus<d64>()));
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::minus<d64>());

    return *this;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }
    Matrix result(*this);

    return Matrix(rows_, cols_, detail::elementWise(data_, other.data_, std::multiplies<d64>()));
}

Matrix& Matrix::hadamardInPlace(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::multiplies<d64>());

    return *this;
}

Matrix Matrix::sliceByRows(u32 start, u32 finish) const {
    if (start >= finish) {
        throw std::invalid_argument("finish must be greater than start");
    }
    if (finish > rows_) {
        throw std::invalid_argument("finish must not be greater than number of rows");
    }

    u32 new_rows = finish - start;

    Matrix result(new_rows, cols_);

    for (u32 i = start; i < finish; ++i) {
        for (u32 j = 0; j < cols_; ++j) {
            result(i - start, j) = (*this)(i, j);
        }
    }

    return result;
}

Matrix Matrix::sliceByCols(u32 start, u32 finish) const {
    if (start >= finish) {
        throw std::invalid_argument("finish must be greater than start");
    }
    if (finish > cols_) {
        throw std::invalid_argument("finish must not be greater than number of cols");
    }

    u32 new_cols = finish - start;

    Matrix result(rows_, new_cols);

    for (u32 i = 0; i < rows_; ++i) {
        for (u32 j = start; j < finish; ++ j) {
            result(i, j - start) = (*this)(i, j);
        }
    }

    return result;
}

d64 Matrix::sumElements() const {
    return detail::sumElements(data_);
}

// axis = 0 means sum up each column, leaving a row vector
// axis = 1 means sum up each row, leaving a column vector
Matrix Matrix::sum(u32 axis) const {
    Matrix result = axis ==0 ? Matrix(1, (*this).cols_) : Matrix(((*this).rows_), 1);

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

Matrix& Matrix::matmulInto(const Matrix& other, Matrix& out) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("matmul dimension mismatch");
    if (rows_ != out.rows_ || other.cols_ != out.cols_)
        throw std::invalid_argument("output shape mismatch");

    const u32 m = rows_;
    const u32 n = other.cols_;
    const u32 k = cols_;
    
    for (u32 i = 0; i < m; ++i) {
        for (u32 j = 0; j < n; ++j) {
            d64 sum = 0.0;
            for (u32 p = 0; p < k; ++p) {
                sum += data_[i * cols_ + p] * other.data_[p * other.cols_ + j];
            }
            out.data_[i * out.cols_ + j] = sum;
        }
    }

    return out;
}

d64 Matrix::max() const {
    return detail::maxElement(data_);
}


d64 Matrix::min() const {
    return detail::minElement(data_);
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (u32 i = 0; i < m.rows(); ++i) {
        u32 count = 0;
        for (u32 j = 0; j < m.cols(); ++j) {
            if (count == m.cols() - 1) {
                std::cout << m(i, j) << std::endl;
            } else {
                std::cout << m(i, j) << ", ";
            }

            ++count;
        }
    }

    return os;
}

} // namespace linalg