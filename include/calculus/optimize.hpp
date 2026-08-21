#pragma once

#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "calculus/differentiation.hpp"
#include "types.hpp"
#include "constants.hpp"

namespace calculus {

namespace optimize {

struct OpResult {
    linalg::Vec<d64> x_min;
    d64 f_min;
    d64 finalErr;
    u32 numIter;
    bool converged;
};

    // This calcualtes alpha for the gradient descent method bu backtrack line searching.
    // This chooses alpha such that the function is decreased at each step by a fraction (c1)
    // of what is expected from a linear approximation to prevent stalling or overshooting.
    template <typename F>
    d64 backtrackingLineSearch(F&& func, const linalg::Vec<d64>& xi, d64 fi, const linalg::Vec<d64>& gradient,
                                d64 alpha0 = 1.0, d64 rho = 0.5, d64 c1 = 1e-4,
                                u32 maxBacktracks = 50) {
        d64 alpha = alpha0;
        d64 gradNormSq = gradient.normSquared(); 

        for (u32 i = 0; i < maxBacktracks; ++i) {
            linalg::Vec<d64> xNext = xi - gradient * alpha;
            d64 fNext = func(xNext);

            if (fNext <= fi - c1 * alpha * gradNormSq) {
                break; // Armijo condition satisfied
            }
            alpha *= rho;
        }

        return alpha;
    }

    // This finds the minimum of a function using the gradient descent method. The parameter alpha is calculated at 
    // each step instead of passed to the function, using backtracking line search. Alpha is chosen at each step to 
    // ensure the function decreases by an ammount satisfying the armijo condition. This prevents overshooting in 
    // steep areas, or stalling in shallow areas
    template <typename F>
    OpResult gradDescent(F&& func, linalg::Vec<d64> x0, d64 h = 1e-5,
                        d64 stopCondition = kIterStopCondition, d64 gradTol = kIterStopCondition, u32 maxIter = 1000) {
        d64 fi = func(x0);
        linalg::Vec<d64> xi = x0;

        OpResult res;
        u32 numIter = 0;
        bool converged = false;
        d64 err = 0.0;

        while (numIter < maxIter) {
            linalg::Vec<d64> gradient = differentiate::grad(func, xi, h);

            d64 alpha = backtrackingLineSearch(func, xi, fi, gradient);
            xi = xi - gradient * alpha;

            d64 f_ip1 = func(xi);
            err = std::fabs(f_ip1 - fi);
            fi = f_ip1;

            ++numIter;

            if (err < stopCondition || gradient.norm() < gradTol) { 
                converged = true; 
                break; 
            }
        }

        res.converged = converged;
        res.numIter = numIter;
        res.f_min = fi;
        res.x_min = xi;
        res.finalErr = err;

        return res;
    }

    // This finds the minimum of a function using Newtons method with Levenberg-Marquardt regularization.
    // This allows the minimum to be found even when the process has to pass throguh areas where the hessian is not 
    // positive definite. If its not positive definite and has a negative eigenvalue the hessian can point the gradient
    // in a useless direction. By shifting H by lambda * I, it guarantees positive definiteness which gives a descent direction
    template <typename F>
    OpResult newtonsLM(F&& func, linalg::Vec<d64> x0, d64 h_grad = 1e-5, d64 h_hess = 1e-4,
                        d64 stopCondition = kIterStopCondition, d64 gradTol = kIterStopCondition,
                        u32 maxIter = 1000, d64 lambda0 = 1e-3) {
        linalg::Vec<d64> xi = x0;
        d64 fi = func(x0);
        d64 lambda = lambda0;

        OpResult res;
        u32 numIter = 0;
        bool converged = false;
        d64 err = 0.0;

        while (numIter < maxIter) {
            linalg::Vec<d64> gradient = differentiate::grad(func, xi, h_grad);
            linalg::Matrix<d64> hessian = differentiate::hess(func, xi, h_hess);

            for (size_t k = 0; k < hessian.rows(); ++k) hessian(k, k) += lambda;

            linalg::Vec<d64> x_ip1 = xi - linalg::solve::lu(hessian, gradient);
            d64 f_ip1 = func(x_ip1);

            if (f_ip1 < fi) {
                err = std::fabs(f_ip1 - fi);
                fi = f_ip1;
                xi = x_ip1;
                lambda *= 0.5;
                ++numIter;

                if (err < stopCondition || gradient.norm() < gradTol) {
                    converged = true;
                    break;
                }
            } else {
                lambda *= 2.0;
                ++numIter;
            }
        }

        res.converged = converged;
        res.numIter = numIter;
        res.f_min = fi;
        res.x_min = xi;
        res.finalErr = err;

        return res;
    }

} // namespace optimize

} // namespace calculus