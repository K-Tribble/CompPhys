#pragma once

#include <functional>
#include <vector>
#include "types.hpp"
#include "constants.hpp"

namespace nonlin {
    
    struct RootIterResult {
        bool converged; // if the function has converged, could be if the successive xi have almost no difference between them
        bool foundRoot; // if the iterator actually converged to the root
        d64 root;
        d64 function_val;
        u32 numIter;
        d64 finalErr; // final error is the absolute difference between the final two xi
    };

    // 1D root finders
    template <typename F>
    inline RootIterResult bisection(F&& func, d64 xl, d64 xr, d64 functionTol = kIterStopCondition, d64 xTol = kIterStopCondition, u32 maxIter = 1000) {
        RootIterResult res;

        d64 fl = func(xl);
        d64 fr = func(xr);

        if (fl == 0) {
            res.converged = true;
            res.numIter = 0;
            res.function_val = fl;
            res.finalErr = 0;
            res.root = xl;
            return res;
        }

        if (fr == 0) {
            res.converged = true;
            res.numIter = 0;
            res.function_val = fr;
            res.finalErr = 0;
            res.root = xr;
            return res;
        }

        if (fl * fr > 0) {
            throw std::invalid_argument("initial right and left values must have opposite signs");
        }

        d64 xi = (xl + xr) / 2;
        d64 error = std::fabs(xr - xl) / 2;

        bool converged = false;
        bool foundRoot = false;
        u32 numIter = 0;

        while (numIter < maxIter) {

            xi = (xl + xr) / 2;

            d64 fi = func(xi);

            error = std::fabs(xr - xl) / 2;

            if (std::fabs(fi) < functionTol) {
                converged = true;
                foundRoot = true;
                break;
            } else if (error < xTol) {
                converged = true;
                foundRoot = false;
                break;
            }

            if (fl * fi < 0) {
                xr = xi;
                fr = fi;
            } else {
                xl = xi;
                fl = fi;
            }

            ++numIter;
        }

        res.converged = converged;
        res.foundRoot = foundRoot;
        res.numIter = numIter;
        res.function_val = func(xi);
        res.finalErr = error;
        res.root = xi;

        return res;
    }

    template <typename F, typename F_prime>
    inline RootIterResult newton(F&& func, F_prime&& deriv, d64 x0, d64 functionTol = kIterStopCondition, d64 xTol = kIterStopCondition, u32 maxIter = 1000) {
        RootIterResult res;

        d64 xi = x0;
        d64 error = 0.0;
        u32 numIter = 0;
        bool converged = false;
        bool foundRoot = false;

        while (numIter < maxIter) {
            d64 fi = func(xi);
            d64 f_primei = deriv(xi);

            if (std::fabs(f_primei) < kDefaultAbsTol) {
                throw std::invalid_argument("derivative too close to zero");
            }

            d64 x_ip1 = xi - fi / f_primei;

            error = std::fabs(x_ip1 - xi);
            xi = x_ip1;
            ++numIter;

            if (std::fabs(fi) < functionTol) {
                converged = true;
                foundRoot = true;
                break;
            } else if (error < xTol) {
                converged = true;
                foundRoot = false;
                break;
            }
        }

        res.converged = converged;
        res.foundRoot = foundRoot;
        res.numIter = numIter;
        res.function_val = func(xi);
        res.finalErr = error;
        res.root = xi;

        return res;
    }

    template <typename F>
    inline RootIterResult secant(F&& func, d64 x0, d64 x1, d64 functionTol = kIterStopCondition, d64 xTol = kIterStopCondition, u32 maxIter = 1000) {
        RootIterResult res;

        d64 xim1 = x0;
        d64 xi = x1;

        d64 error = std::fabs(xi - xim1);
        u32 numIter = 0;
        bool converged = false;
        bool foundRoot = false;

        while (numIter < maxIter) {
            d64 fim1 = func(xim1);
            d64 fi = func(xi);

            d64 denominator = fi - fim1;

            if (std::fabs(denominator) < kDefaultAbsTol) {
                throw std::invalid_argument("secant slope too close to zero");
            }

            d64 xip1 = xi - fi * (xi - xim1) / denominator;

            error = std::fabs(xip1 - xi);

            xim1 = xi;
            xi = xip1;

            ++numIter;

            if (std::fabs(fi) < functionTol) {
                converged = true;
                foundRoot = true;
                break;
            } else if (error < xTol) {
                converged = true;
                foundRoot = false;
                break;
            }
        }

        res.converged = converged;
        res.foundRoot = foundRoot;
        res.numIter = numIter;
        res.function_val = func(xi);
        res.finalErr = error;
        res.root = xi;

        return res;
    }


} // namespace nonlin