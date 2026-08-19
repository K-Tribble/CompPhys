#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "linalg/linalg_interop.hpp"
#include <stdexcept>

namespace linalg {

namespace solve {

    template <Scalar T>
    Vec<T> forwardSub(const Matrix<T>& lt, const Vec<T>& rhs) {
        // undefined behavior for a non lower triangular matrix

        u32 n = rhs.size();

        if (lt.rows() != n) {
            throw std::invalid_argument("shape mismatch");
        }
        if (lt.rows() != lt.cols()) {
            throw std::invalid_argument("matrix must be square");
        }

        Vec<T> x(n); // solution vector

        for(u32 i = 0; i < n; ++i) {
            T sum = rhs(i);
            for (u32 j = 0; j < i; ++j) {
                sum -= lt(i, j) * x(j);
            }
            x(i) = sum / lt(i, i);
        }

        return x;
    }
    
    template <Scalar T>
    Vec<T> backSub(const Matrix<T>& ut, const Vec<T>& rhs) {
        // undefined behavior for a non upper triangular matrix

        u32 n = rhs.size();

        if (ut.rows() != n) {
            throw std::invalid_argument("shape mismatch");
        }
        if (ut.rows() != ut.cols()) {
            throw std::invalid_argument("matrix must be square");
        }

        Vec<T> x(n); // solution vector

        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            T sum = rhs(i);
            for (int j = i + 1; j < static_cast<int>(n); ++j) {
                sum -= ut(i, j) * x(j);
            }
            x(i) = sum / ut(i, i);
        }

        return x;
    }

    template <Scalar T>
    Vec<T> lu(const LUResult<T>& f, const Vec<T>& b) {
        u32 n = b.size();

        const Matrix<T>& L = f.L;
        const Matrix<T>& U = f.U;
        const Matrix<T>& P = f.P;

        if (L.cols() != n) {
            throw std::invalid_argument("shape mismatch");
        }

        Vec<T> b_prime = P * b;

        Vec<T> y = forwardSub(L, b_prime);

        Vec<T> x = backSub(U, y);

        return x;
    }

    template <Scalar T>
    std::vector<Vec<T>> lu(const LUResult<T>& f, const std::vector<Vec<T>>& bs) {
        std::vector<Vec<T>> solutions;
        solutions.reserve(bs.size());

        for (auto& b : bs) {
            solutions.emplace_back(lu(f, b));
        }

        return solutions;
    }

    template <Scalar T>
    Vec<T> lu(const Matrix<T>& A, const Vec<T>& b) {
        LUResult<T> res = A.LUDecomp(); 
        return lu(res, b);
    }

    template <Scalar T>
    std::vector<Vec<T>> lu(const Matrix<T>& A, const std::vector<Vec<T>>& bs) {
        LUResult<T> res = A.LUDecomp();
        return lu(res, bs);
    }

    template <Scalar T>
    IterResult<T> runSplitIteration(const Matrix<T>& A, const Vec<T>& b, const Matrix<T>& B, const Matrix<T>& S,
            const solveFn<T>& solve, IterStoppingCondition<T> sc, const u32 maxIter) {
        Vec<T> xi = solve(B, b);

        RealType<T> l = sc.lnorm_ord;
        IterResult<T> result;
        result.lnorm_ord = l;
        result.errType = sc.errType;

        RealType<T> bNorm = b.lnorm(l);

        bool iterate = true;
        bool converged = false;
        u32 numIter = 0;
        Vec<T> lastResidual;

        while (iterate) {
            Vec<T> rhs = b - S * xi;
            Vec<T> x_ip1 = solve(B, rhs);

            RealType<T> err;
            if (sc.errType == errorType::Fractional) {
                Vec<T> diff = x_ip1 - xi;
                RealType<T> denom = xi.lnorm(l);
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

    template <Scalar T>
    IterResult<T> jacobi(const Matrix<T>& A, const Vec<T>& b, IterStoppingCondition<T> sc, const u32 maxIter) {
        std::vector<T> ADiag = A.getDiag();
        for (T x : ADiag) {
            if (std::abs(x) < kDefaultAbsTol) {
                throw std::invalid_argument("cannot do jacobi iteration with matrix that has a zero diagonal element");
            }
        }
        Matrix<T> B = Matrix<T>::diagonal(ADiag);
        Matrix<T> S = A - B;

        solveFn<T> diagSolve = [](const Matrix<T>& B, const Vec<T>& rhs) {
            Vec<T> y(rhs.size());
            for (u32 i = 0; i < rhs.size(); ++i) {
                y(i) = rhs(i) / B(i, i);
            }
            return y;
        };

        return runSplitIteration(A, b, B, S, diagSolve, sc, maxIter);
    }

    template <Scalar T>
    IterResult<T> sor(const Matrix<T>& A, const Vec<T>& b, const RealType<T> w, IterStoppingCondition<T> sc,
        const u32 maxIter, SplitType split) {
        if (w <= 0 || w > 2) {
            throw std::invalid_argument("sor requires a relaxation parameter in (0, 2]");
        }
        RealType<T> alpha = 1.0 / w;
        Matrix<T> D = Matrix<T>::diagonal(A.getDiag());
        Matrix<T> B = (split == SplitType::Lower) ? A.getLower() + D * alpha : A.getUpper() + D * alpha;

        Matrix<T> S = A - B;

        solveFn<T> triSolve = (split == SplitType::Lower) 
            ? solveFn<T>([](const Matrix<T>& B, const Vec<T>& rhs) {return forwardSub(B, rhs);})
            : solveFn<T>([](const Matrix<T>& B, const Vec<T>& rhs) {return backSub(B, rhs);});

        return runSplitIteration(A, b, B, S, triSolve, sc, maxIter);
    }

    template <Scalar T>
    IterResult<T> gaussSeidel(const Matrix<T>& A, const Vec<T>& b, IterStoppingCondition<T> sc, 
        const u32 maxIter, SplitType split) {
        return sor(A, b, 1.0, sc, maxIter, split);
    }

} // namespace solve

} // namespace linalg