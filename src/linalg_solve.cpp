#include "matrix.hpp"
#include "vec.hpp"
#include "linalg_interop.hpp"
#include <stdexcept>

namespace linalg {

namespace solve {

    Vec forwrardSub(const Matrix& lt, const Vec& rhs) {
        // undefined behavior for a non lower triangular matrix

        u32 n = rhs.size();

        if (lt.rows() != n) {
            throw std::invalid_argument("shape mismatch");
        }
        if (lt.rows() != lt.cols()) {
            throw std::invalid_argument("matrix must be square");
        }

        Vec x(n); // solution vector

        for(u32 i = 0; i < n; ++i) {
            d64 sum = rhs(i);
            for (u32 j = 0; j < i; ++j) {
                sum -= lt(i, j) * x(j);
            }
            x(i) = sum / lt(i, i);
        }

        return x;
    }

    Vec backSub(const Matrix& ut, const Vec& rhs) {
        // undefined behavior for a non upper triangular matrix

        u32 n = rhs.size();

        if (ut.rows() != n) {
            throw std::invalid_argument("shape mismatch");
        }
        if (ut.rows() != ut.cols()) {
            throw std::invalid_argument("matrix must be square");
        }

        Vec x(n); // solution vector

        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            d64 sum = rhs(i);
            for (int j = i + 1; j < n; ++j) {
                sum -= ut(i, j) * x(j);
            }
            x(i) = sum / ut(i, i);
        }

        return x;
    }

    Vec lu(const LUResult& f, const Vec& b) {
        u32 n = b.size();

        const Matrix& L = f.L;
        const Matrix& U = f.U;
        const Matrix& P = f.P;

        if (L.cols() != n) {
            throw std::invalid_argument("shape mismatch");
        }

        Vec b_prime = P * b;

        Vec y = forwrardSub(L, b_prime);

        Vec x = backSub(U, y);

        return x;
    }

    std::vector<Vec> lu(const LUResult& f, const std::vector<Vec>& bs) {
        std::vector<Vec> solutions;
        solutions.reserve(bs.size());

        for (auto& b : bs) {
            solutions.emplace_back(lu(f, b));
        }

        return solutions;
    }

    Vec lu(const Matrix& A, const Vec& b) {
        LUResult res = A.LUDecomp(); 
        return lu(res, b);
    }

    std::vector<Vec> lu(const Matrix& A, const std::vector<Vec>& bs) {
        LUResult res = A.LUDecomp();
        return lu(res, bs);
    }

} // namespace solve

} // namespace linalg