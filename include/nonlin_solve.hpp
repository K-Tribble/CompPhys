#pragma once

#include <functional>
#include <vector>
#include "types.hpp"
#include "interpolation.hpp"
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
            d64 fi = f_ip1;
            d64 f_primei = deriv(xi);

            if (std::fabs(f_primei) < kDefaultAbsTol) {
                throw std::invalid_argument("derivative too close to zero");
            }

            d64 x_ip1 = xi - fi / f_primei;

            error = std::fabs(x_ip1 - xi);
            xi = x_ip1;
            ++numIter;

            d64 f_ip1 = func(xi);

            if (std::fabs(f_ip1) < functionTol) {
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

    // Brent's method: combines bisection, secant, and inverse quadratic interpolation (IQI).
    // IQI is done by reusing lagrangePolynomial() from interpolation.hpp: instead of
    // interpolating f as a function of x, the roles are swapped to interpolate x as a
    // function of f through (fa,a), (fb,b), (fc,c), and evaluated at f = 0.
    // xl and xr must bracket the root, i.e. func(xl) and func(xr) must have opposite signs.
    template <typename F>
    inline RootIterResult brent(F&& func, d64 xl, d64 xr, d64 functionTol = kIterStopCondition, d64 xTol = kIterStopCondition, u32 maxIter = 1000) {
        RootIterResult res;
 
        d64 a = xl;
        d64 b = xr;
        d64 fa = func(a);
        d64 fb = func(b);
 
        if (fa == 0) {
            res.converged = true;
            res.foundRoot = true;
            res.numIter = 0;
            res.function_val = fa;
            res.finalErr = 0;
            res.root = a;
            return res;
        }
 
        if (fb == 0) {
            res.converged = true;
            res.foundRoot = true;
            res.numIter = 0;
            res.function_val = fb;
            res.finalErr = 0;
            res.root = b;
            return res;
        }
 
        if (fa * fb > 0) {
            throw std::invalid_argument("initial right and left values must have opposite signs");
        }
 
        // b is always the current best estimate: |f(b)| <= |f(a)|
        if (std::fabs(fa) < std::fabs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
 
        d64 c = a;
        d64 fc = fa;
        d64 d = c; // only meaningful once a bisection step has actually been taken
        bool mflag = true;
 
        auto iqi = lagrangePolynomial();
 
        d64 error = std::fabs(b - a);
        u32 numIter = 0;
        bool converged = false;
        bool foundRoot = false;
 
        while (numIter < maxIter) {
 
            d64 s;
 
            if (fa != fc && fb != fc) {
                // inverse quadratic interpolation: interpolate x(f) through
                // (fa,a), (fb,b), (fc,c) and evaluate at f = 0
                std::array<d64, 3> fPts = {fa, fb, fc};
                std::array<d64, 3> xPts = {a, b, c};
                s = iqi(0.0, fPts, xPts);
            } else {
                // secant step
                s = b - fb * (b - a) / (fb - fa);
            }
 
            d64 bisectionMid = (3 * a + b) / 4;
            bool cond1 = (s < std::min(bisectionMid, b)) || (s > std::max(bisectionMid, b));
            bool cond2 = mflag && std::fabs(s - b) >= std::fabs(b - c) / 2;
            bool cond3 = !mflag && std::fabs(s - b) >= std::fabs(c - d) / 2;
            bool cond4 = mflag && std::fabs(b - c) < xTol;
            bool cond5 = !mflag && std::fabs(c - d) < xTol;
 
            if (cond1 || cond2 || cond3 || cond4 || cond5) {
                s = (a + b) / 2;
                mflag = true;
            } else {
                mflag = false;
            }
 
            d64 fs = func(s);
 
            error = std::fabs(s - b);
 
            d = c;
            c = b;
            fc = fb;
 
            if (fa * fs < 0) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
 
            if (std::fabs(fa) < std::fabs(fb)) {
                std::swap(a, b);
                std::swap(fa, fb);
            }
 
            ++numIter;
 
            if (std::fabs(fb) < functionTol) {
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
        res.function_val = func(b);
        res.finalErr = error;
        res.root = b;
 
        return res;
    }


} // namespace nonlin