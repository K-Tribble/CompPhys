#include "matrix.hpp"
#include "vec.hpp"

namespace linalg{

Vec operator*(const Matrix& m, const Vec& v); // treats v as a column vector
Vec operator*(const Vec& v, const Matrix& m); // treats v as a row vector

} // namespace linalg


