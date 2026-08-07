#include "matrix.hpp"
#include "vec.hpp"
#include "linalg_interop.hpp"
#include <stdexcept>

namespace linalg {

    namespace solve {

        Vec linSolve(Matrix A, Vec b) {
            u32 n = b.size();

            if (A.cols() != n) {
                throw std::invalid_argument("shape mismatch");
            }
            if (A.cols() != A.rows()) {
                throw std::invalid_argument("A must be a square matrix");
            }

            LUResult res = A.LUDecomp();
            Matrix L = res.L;
            Matrix U = res.U;
            Matrix P = res.P;

            Vec b_prime = P* b;

            Vec y(n);
            for (u32 i = 0; i < n; ++i) {
                d64 sum = b_prime(i);
                for (u32 j = 0; j < i; ++j) {
                    sum -= L(i, j) * y(j);
                }
                y(i) = sum / L(i, i);
            }

            Vec x(n); // final solution vector
            for (int i = n - 1; i >= 0; --i) {
                d64 sum = y(i);
                for (int j = i + 1; j < n; ++j) {
                    sum -= U(i, j) * x(j);
                }
                x(i) = sum / U(i, i);
            }

            return x;
        }

    } // namespace solve

} // namespace linalg