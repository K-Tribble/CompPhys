#pragma once

#include "matrix.hpp"
#include "vec.hpp"
#include "linalg_interop.hpp"

namespace linalg {

namespace solve {

    // Solves Ax=b for when A is lower triangular by forward substitution
    Vec forwardSub(const Matrix& lt, const Vec& rhs);
    // Solves Ax=b for when A is uperr triangular by backward substitution
    Vec backSub(const Matrix& up, const Vec& rhs);

    Vec lu(const LUResult& f, const Vec& b);
    std::vector<Vec> lu(const LUResult& f, const std::vector<Vec>& bs);

    Vec lu(const Matrix& A, const Vec& b);

    std::vector<Vec> lu(const Matrix& A, const std::vector<Vec>& bs);

} // namespace solve
    
} // namespace linalg