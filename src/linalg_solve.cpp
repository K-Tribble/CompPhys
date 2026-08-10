#include "matrix.hpp"
#include "vec.hpp"
#include "linalg_interop.hpp"
#include "linalg_solve.hpp"
#include <stdexcept>

namespace linalg {

namespace solve {

    Vec forwardSub(const Matrix& lt, const Vec& rhs) {
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
            for (int j = i + 1; j < static_cast<int>(n); ++j) {
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

        Vec y = forwardSub(L, b_prime);

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

    IterResult runSplitIteration(const Matrix& A, const Vec& b, const Matrix& B, const Matrix& S,
            const solveFn& solve, IterStoppingCondition sc, const u32 maxIter) {
        Vec xi = solve(B, b);

        d64 l = sc.lnorm_ord;
        IterResult result;
        result.lnorm_ord = l;
        result.errType = sc.errType;

        d64 bNorm = b.lnorm(l);

        bool iterate = true;
        bool converged = false;
        u32 numIter = 0;
        Vec lastResidual;

        while (iterate) {
            Vec rhs = b - S * xi;
            Vec x_ip1 = solve(B, rhs);

            d64 err;
            if (sc.errType == errorType::Fractional) {
                Vec diff = x_ip1 - xi;
                d64 denom = xi.lnorm(l);
                err = (denom > kDefaultAbsTol) ? diff.lnorm(l) / denom : diff.lnorm(l);
            } else {
                lastResidual = A * x_ip1 - b;
                err = (bNorm > kDefaultAbsTol) ? lastResidual.lnorm(l) / bNorm : lastResidual.lnorm(l);
            }
            xi = std::move(x_ip1);
            ++numIter;

            if (err < sc.stopCondition) { converged = true; iterate = false; }
            else if (numIter >= maxIter) { iterate = false; }
        }

        result.success = converged;
        result.numIter = numIter;
        result.x_final = xi;
        result.finalResidualVector = (sc.errType == errorType::Residual) ? lastResidual : A * xi - b;
        result.finalFractionalResErr = (bNorm > kDefaultAbsTol)
            ? result.finalResidualVector.lnorm(l) / bNorm
            : result.finalResidualVector.lnorm(l);

        return result;
    }

    IterResult jacobi(const Matrix& A, const Vec& b, IterStoppingCondition sc, const u32 maxIter) {
        std::vector<d64> ADiag = A.getDiag();
        for (d64 x : ADiag) {
            if (std::fabs(x) < kDefaultAbsTol) {
                throw std::invalid_argument("cannot do jacob iteration with matrix that has a zero diagonal element");
            }
        }
        Matrix B = Matrix::diagonal(ADiag);
        Matrix S = A - B;

        solveFn diagSolve = [](const Matrix& B, const Vec& rhs) {
            Vec y(rhs.size());
            for (u32 i = 0; i < rhs.size(); ++i) {
                y(i) = rhs(i) / B(i, i);
            }
            return y;
        };

        return runSplitIteration(A, b, B, S, diagSolve, sc, maxIter);
    }

    IterResult sor(const Matrix& A, const Vec& b, const d64 w, IterStoppingCondition sc,
        const u32 maxIter, SplitType split) {
        if (w <= 0 || w > 2) {
            throw std::invalid_argument("sor requires a relaxation parameter in (0, 2]");
        }
        d64 alpha = 1.0 / w;
        Matrix D = Matrix::diagonal(A.getDiag());
        Matrix B = (split == SplitType::Lower) ? A.getLower() + D * alpha : A.getUpper() + D * alpha;

        Matrix S = A - B;

        solveFn triSolve = (split == SplitType::Lower) 
            ? solveFn([](const Matrix& B, const Vec& rhs) {return forwardSub(B, rhs);})
            : solveFn([](const Matrix& B, const Vec& rhs) {return backSub(B, rhs);});

        return runSplitIteration(A, b, B, S, triSolve, sc, maxIter);
    }

    IterResult gaussSeidel(const Matrix& A, const Vec& b, IterStoppingCondition sc, 
        const u32 maxIter, SplitType split) {
        return sor(A, b, 1.0, sc, maxIter, split);
    }

} // namespace solve

} // namespace linalg