#include <cmath>
#include <vector>
#include <span>
#include "types.hpp"
#include "linalg/vec.hpp"
#include "linalg/matrix.hpp"
#include "constants.hpp"
#include <omp.h>

namespace calculus {

namespace differentiate {
    enum class DiffScheme {
        Forward, Backward, Central
    };

    template <typename F>
    inline d64 firstDerivAt(F&& func, d64 x, d64 h = 0.01, DiffScheme scheme = DiffScheme::Central) {
        switch (scheme) {
            case DiffScheme::Central:
                return (func(x + h) - func(x - h)) / (2 * h);
                break;

            case DiffScheme::Forward:
                return (func(x + h) - func(x)) / h;
                break;
                
            case DiffScheme::Backward:
                return (func(x) - func(x - h)) / h;
                break;
        }
    }

    // second derivative uses central difference scheme
    template <typename F>
    inline d64 secondDerivAt(F&& func, d64 x, d64 h = 0.01) {
        return (func(x + h) - 2 * func(x) + func(x - h)) / (h * h);
    }

    // method to calculate the gradient of a real valued function using central difference scheme
    // and store it in a d64 vector. This is a function of a several arguments. The function 
    // needs to take a Vec<d64> as an argument and return a scalar d64. 
    // func needs to be a function that doesnt mutate shared or global state to be safe to parallelize
    template <typename F>
    inline linalg::Vec<d64> grad(F&& func, const linalg::Vec<d64>& x, d64 h = 1e-5) {
        u32 n = x.size();
        linalg::Vec<d64> gradient(n);

        #pragma omp parallel for
        for (u32 i = 0; i < n; ++i) {
            d64 raw = h * std::max(d64(1.0), std::abs(x(i)));
            d64 step = (x(i) + raw) - x(i);

            linalg::Vec<d64> xp = x;
            xp(i) = x(i) + step;
            d64 f_plus = func(xp);

            xp(i) = x(i) - step;
            d64 f_minus = func(xp);

            gradient(i) = (f_plus - f_minus) / (2.0 * step);
        }

        return gradient;
    }

    // method to calculate the hessian of a real valued function using central difference scheme
    // and store it in a d64 vector. This is a function of a several arguments. The function 
    // needs to take a Vec<d64> as an argument and return a scalar d64. 
    // func needs to be a function that doesnt mutate shared or global state to be safe to parallelize
    template <typename F>
    inline linalg::Matrix<d64> hess(F&& func, const linalg::Vec<d64>& x, d64 h = 1e-4) {
        u32 n = x.size();
        linalg::Matrix<d64> hessian(n, n);
        d64 f0 = func(x);

        std::vector<std::pair<u32, u32>> pairs;
        for (u32 i =0; i < n; ++i) {
            for (u32 j = i; j < n; ++j) {
                pairs.emplace_back(i, j);
            }
        }

        #pragma omp parallel for
        for (u32 k = 0; k < pairs.size(); ++k) {
            auto [i, j] = pairs[k];
            d64 rawi = h * std::max(d64(1.0), std::abs(x(i)));
            d64 si = (x(i) + rawi) - x(i);

            if (i == j) {
                linalg::Vec<d64> xp = x;
                xp(i) = x(i) + si;
                d64 fp = func(xp);
                xp(i) = x(i) - si;
                d64 fm = func(xp);

                hessian(i, i) = (fp - 2.0 * f0 + fm) / (si * si);
            } else {
                d64 rawj = h * std::max(d64(1.0), std::abs(x(j)));
                d64 sj = (x(j) + rawj) - x(j);
                linalg::Vec<d64> xp = x;
                xp(i) += si; xp(j) += sj; d64 f_pp = func(xp);
                xp(j) -= 2 * sj; d64 f_pm = func(xp);
                xp(i) -= 2 * si; d64 f_mm = func(xp);
                xp(j) += 2 * sj; d64 f_mp = func(xp);

                d64 hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * si * sj);
                hessian(i, j) = hij;
                hessian(j, i) = hij;
            }
        }

        return hessian;
    }

    // First order can use any scheme, but to return a vector of derivatives the same size as x a forward/backward
    // derivative evaluation is done at either end if needed. So when using a central scheme, the derivative at 
    // the first point is calculated with forward difference, and at the last point with backward difference
    std::vector<d64> differentiate(std::span<const d64> x, std::span<const d64> y, DiffScheme scheme = DiffScheme::Central);

    std::vector<d64> secondDeriv(std::span<const d64> x, std::span<const d64> y); // second derivative uses central difference scheme

} // namespace differentiate
    
} // namespace calculus