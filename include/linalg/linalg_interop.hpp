#pragma once

#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "types.hpp"
#include "constants.hpp"

namespace linalg{

template <Scalar T>
Vec<T> operator*(const Matrix<T>& m, const Vec<T>& v); // treats v as a column vector

template <Scalar T>
Vec<T> operator*(const Vec<T>& v, const Matrix<T>& m); // treats v as a row vector

} // namespace linalg

#include "linalg/linalg_interop.tpp"


